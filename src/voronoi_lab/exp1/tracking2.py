"""Hash-pinned, read-only boundary around the legacy Tracking2 ResNet inputs.

Nothing in this module imports the Tracking2 package.  The only executable
external source it can load is the exact ``tracking2.models`` file declared in
an input manifest, after its byte length and SHA-256 digest have been checked.
The experiment runners and data pipelines in that repository are deliberately
outside this boundary.
"""

from __future__ import annotations

import importlib
import io
import json
import math
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from voronoi_lab.core.hashing import sha256_bytes, sha256_file

PositiveInt = Annotated[int, Field(gt=0, strict=True)]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
StrictFloat = Annotated[float, Field(strict=True)]
NonNegativeFloat = Annotated[float, Field(ge=0, strict=True)]
PositiveFloat = Annotated[float, Field(gt=0, strict=True)]
Probability = Annotated[float, Field(ge=0, le=1, strict=True)]
StrictBool = Annotated[bool, Field(strict=True)]


def _require_true(value: bool) -> bool:
    if value is not True:
        raise ValueError("value must be true")
    return value


StrictTrue = Annotated[bool, Field(strict=True), AfterValidator(_require_true)]
VersionOne = Annotated[int, Field(ge=1, le=1, strict=True)]


class Tracking2Error(RuntimeError):
    """Base class for failures at the external Tracking2 boundary."""


class Tracking2ManifestError(Tracking2Error):
    """The input manifest is malformed or internally inconsistent."""


class ExternalInputValidationError(Tracking2Error):
    """A declared external file is absent or differs from its manifest."""


class TransplantArtifactError(Tracking2Error):
    """The legacy transplant artifact does not satisfy its declared schema."""


class OptionalTorchError(Tracking2Error):
    """The optional PyTorch dependency is unavailable or too old."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class FileReference(_StrictModel):
    """A content-addressed file path relative to the declared external root."""

    path: Path
    size_bytes: PositiveInt
    sha256: str

    @field_validator("path")
    @classmethod
    def require_safe_relative_path(cls, value: Path) -> Path:
        if value.is_absolute() or value == Path(".") or ".." in value.parts:
            raise ValueError("external file paths must be non-empty and relative to root")
        return value

    @field_validator("sha256")
    @classmethod
    def require_sha256(cls, value: str) -> str:
        if len(value) != 64 or value.lower() != value:
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters") from exc
        return value


class CheckpointReference(FileReference):
    epoch: NonNegativeInt


class ArchitectureReference(_StrictModel):
    module: Literal["tracking2.models"] = "tracking2.models"
    class_name: Literal["InstrumentedResNet18V2"] = "InstrumentedResNet18V2"
    width: PositiveInt = 64
    num_classes: PositiveInt = 10
    source: FileReference
    state_dict_tensors: PositiveInt
    state_dict_parameters: PositiveInt
    state_dict_dtype: Literal["float32"] = "float32"

    @model_validator(mode="after")
    def require_the_audited_source_path(self) -> ArchitectureReference:
        if self.source.path != Path("src/tracking2/models.py"):
            raise ValueError("the only permitted external module is src/tracking2/models.py")
        return self


class DatasetReferences(_StrictModel):
    backend: Literal["parquet"] = "parquet"
    train: FileReference
    test: FileReference

    @model_validator(mode="after")
    def require_disjoint_train_and_test_sources(self) -> DatasetReferences:
        if self.train.path == self.test.path:
            raise ValueError("training and test datasets must use distinct paths")
        if self.train.sha256 == self.test.sha256:
            raise ValueError("training and test datasets must have distinct content hashes")
        return self


class RecordedTrainingConfig(_StrictModel):
    """Configuration fields actually recorded by the legacy transplant artifact."""

    seed: NonNegativeInt
    train_size: PositiveInt
    test_size: PositiveInt
    epochs: PositiveInt
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat
    checkpoint_epochs: tuple[NonNegativeInt, ...]
    lr_milestones: tuple[PositiveInt, ...]
    width: PositiveInt
    device: str
    amp: StrictBool
    target_epoch: NonNegativeInt

    @model_validator(mode="after")
    def validate_epochs(self) -> RecordedTrainingConfig:
        if not self.checkpoint_epochs:
            raise ValueError("checkpoint_epochs cannot be empty")
        if tuple(sorted(set(self.checkpoint_epochs))) != self.checkpoint_epochs:
            raise ValueError("checkpoint_epochs must be strictly increasing")
        if tuple(sorted(set(self.lr_milestones))) != self.lr_milestones:
            raise ValueError("lr_milestones must be strictly increasing")
        if self.target_epoch not in self.checkpoint_epochs:
            raise ValueError("target_epoch must be one of checkpoint_epochs")
        if self.epochs != self.target_epoch:
            raise ValueError("this legacy artifact requires target_epoch == epochs")
        return self


class TransplantReference(_StrictModel):
    file: FileReference
    schema_version: VersionOne = 1
    experiment: Literal["resnet18_block_criticality"] = "resnet18_block_criticality"
    status: Literal["MEASURED"] = "MEASURED"


class Tracking2InputManifest(_StrictModel):
    """Strict declaration of every external byte used by the seed-0 pilot."""

    schema_version: VersionOne = 1
    name: str
    lineage_quality: Literal["exploratory_legacy"]
    lineage_note: str
    root: Path
    observed_repository_revision: str
    read_only: StrictTrue = True
    architecture: ArchitectureReference
    training: RecordedTrainingConfig
    checkpoints: tuple[CheckpointReference, ...]
    datasets: DatasetReferences
    transplant: TransplantReference

    @field_validator("name", "lineage_note")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value

    @field_validator("observed_repository_revision")
    @classmethod
    def require_full_git_revision(cls, value: str) -> str:
        if len(value) not in {40, 64} or value.lower() != value:
            raise ValueError("observed_repository_revision must be a full lowercase digest")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(
                "observed_repository_revision must be a full lowercase digest"
            ) from exc
        return value

    @model_validator(mode="after")
    def match_architecture_and_checkpoint_axes(self) -> Tracking2InputManifest:
        epochs = tuple(checkpoint.epoch for checkpoint in self.checkpoints)
        if not epochs:
            raise ValueError("checkpoints cannot be empty")
        if tuple(sorted(set(epochs))) != epochs:
            raise ValueError("checkpoints must have unique, strictly increasing epochs")
        if epochs != self.training.checkpoint_epochs:
            raise ValueError("checkpoint entries must exactly match training.checkpoint_epochs")
        if self.architecture.width != self.training.width:
            raise ValueError("architecture.width must match training.width")
        return self

    def checkpoint(self, epoch: int) -> CheckpointReference:
        for checkpoint in self.checkpoints:
            if checkpoint.epoch == epoch:
                return checkpoint
        available = ", ".join(str(item.epoch) for item in self.checkpoints)
        raise KeyError(f"checkpoint epoch {epoch} is not declared; available: {available}")


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise Tracking2ManifestError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def parse_tracking2_manifest_bytes(
    raw: bytes, *, source: str | Path = "<embedded Tracking2 manifest>"
) -> Tracking2InputManifest:
    """Parse exact manifest bytes with duplicate-key and strict-schema checks."""

    label = Path(source)
    try:
        value = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError, Tracking2ManifestError) as exc:
        if isinstance(exc, Tracking2ManifestError):
            raise
        raise Tracking2ManifestError(f"could not parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise Tracking2ManifestError(f"manifest must be a YAML mapping: {label}")
    try:
        return Tracking2InputManifest.model_validate(value)
    except ValidationError as exc:
        raise Tracking2ManifestError(f"invalid Tracking2 manifest {label}: {exc}") from exc


def load_tracking2_manifest(path: str | Path) -> Tracking2InputManifest:
    """Parse a strict manifest without touching the declared external files."""

    manifest_path = Path(path)
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise Tracking2ManifestError(f"could not read {manifest_path}: {exc}") from exc
    return parse_tracking2_manifest_bytes(raw, source=manifest_path)


@dataclass(frozen=True, slots=True)
class CutSpec:
    """Mapping between the lab's cut names and Tracking2's native model indices."""

    index: int
    name: str
    tracking2_name: str
    transplant_module_index: int
    channels: int
    height: int
    width: int
    stage: int
    block: int

    @property
    def activation_shape(self) -> tuple[int, int, int]:
        return (self.channels, self.height, self.width)

    @property
    def is_stage_end(self) -> bool:
        return self.block == 2


CUT_SPECS: tuple[CutSpec, ...] = (
    CutSpec(0, "stage1.block1", "stage1.resblk1", 1, 64, 32, 32, 1, 1),
    CutSpec(1, "stage1.block2", "stage1.resblk2", 2, 64, 32, 32, 1, 2),
    CutSpec(2, "stage2.block1", "stage2.resblk1", 3, 128, 16, 16, 2, 1),
    CutSpec(3, "stage2.block2", "stage2.resblk2", 4, 128, 16, 16, 2, 2),
    CutSpec(4, "stage3.block1", "stage3.resblk1", 5, 256, 8, 8, 3, 1),
    CutSpec(5, "stage3.block2", "stage3.resblk2", 6, 256, 8, 8, 3, 2),
    CutSpec(6, "stage4.block1", "stage4.resblk1", 7, 512, 4, 4, 4, 1),
    CutSpec(7, "stage4.block2", "stage4.resblk2", 8, 512, 4, 4, 4, 2),
)

_CUT_BY_INDEX = {cut.index: cut for cut in CUT_SPECS}
_CUT_BY_NAME = {name: cut for cut in CUT_SPECS for name in (cut.name, cut.tracking2_name)}

CutSelector: TypeAlias = int | str | CutSpec


def resolve_cut(cut: CutSelector) -> CutSpec:
    """Resolve a native index, lab name, Tracking2 name, or known ``CutSpec``."""

    if isinstance(cut, CutSpec):
        canonical = _CUT_BY_INDEX.get(cut.index)
        if canonical != cut:
            raise ValueError(f"unrecognized cut specification: {cut!r}")
        return canonical
    if isinstance(cut, bool):
        raise TypeError("a boolean is not a valid cut index")
    if isinstance(cut, int):
        try:
            return _CUT_BY_INDEX[cut]
        except KeyError as exc:
            raise ValueError("cut index must be in [0, 7]") from exc
    if isinstance(cut, str):
        try:
            return _CUT_BY_NAME[cut]
        except KeyError as exc:
            raise ValueError(f"unknown cut name: {cut!r}") from exc
    raise TypeError(f"cut must be int, str, or CutSpec, not {type(cut).__qualname__}")


class _Metrics(_StrictModel):
    loss: NonNegativeFloat
    accuracy: Probability
    error: Probability


class _TrainingMetrics(_Metrics):
    epoch: NonNegativeInt


class _LegacyIntervention(_Metrics):
    module_index: NonNegativeInt
    module: str
    source: str
    delta_loss: StrictFloat
    delta_error: StrictFloat


class _LegacyConfig(_StrictModel):
    output: str
    data_root: str
    fake_data: StrictBool
    train_size: PositiveInt
    test_size: PositiveInt
    epochs: PositiveInt
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat
    checkpoint_epochs: tuple[NonNegativeInt, ...]
    lr_milestones: tuple[PositiveInt, ...]
    width: PositiveInt
    seed: NonNegativeInt
    device: str
    amp: StrictBool


class _LegacyTransplantArtifact(_StrictModel):
    schema_version: VersionOne
    experiment: Literal["resnet18_block_criticality"]
    status: Literal["MEASURED"]
    config: _LegacyConfig
    device: str
    module_names: tuple[str, ...]
    baseline: _Metrics
    training: tuple[_TrainingMetrics, ...]
    interventions: tuple[_LegacyIntervention, ...]
    runtime_seconds: NonNegativeFloat


class TransplantRow(_StrictModel):
    """Stable internal form of one final-target residual-block transplant."""

    seed: NonNegativeInt
    target_epoch: NonNegativeInt
    cut_index: Annotated[int, Field(ge=0, le=7, strict=True)]
    cut_name: str
    tracking2_module_name: str
    transplant_module_index: Annotated[int, Field(ge=1, le=8, strict=True)]
    source_kind: Literal["random", "checkpoint"]
    source_epoch: NonNegativeInt | None
    loss: NonNegativeFloat
    accuracy: Probability
    error: Probability
    delta_loss: StrictFloat
    delta_error: StrictFloat

    @model_validator(mode="after")
    def match_source_fields(self) -> TransplantRow:
        if self.source_kind == "random" and self.source_epoch is not None:
            raise ValueError("random transplant rows cannot have source_epoch")
        if self.source_kind == "checkpoint" and self.source_epoch is None:
            raise ValueError("checkpoint transplant rows require source_epoch")
        cut = resolve_cut(self.cut_index)
        if (
            self.cut_name != cut.name
            or self.tracking2_module_name != cut.tracking2_name
            or self.transplant_module_index != cut.transplant_module_index
        ):
            raise ValueError("transplant row cut fields are inconsistent")
        return self


_TRACKING2_MODULE_NAMES = (
    "stage0",
    *(cut.tracking2_name for cut in CUT_SPECS),
    "final_linear",
)


def _strict_json_object(raw: bytes, *, path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TransplantArtifactError(f"duplicate JSON key in {path}: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransplantArtifactError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TransplantArtifactError(f"transplant artifact must be an object: {path}")
    return value


class Tracking2Adapter:
    """Read-only access to the verified model, checkpoints, and transplant table."""

    def __init__(
        self,
        manifest: Tracking2InputManifest,
        *,
        root_override: str | Path | None = None,
    ) -> None:
        self.manifest = manifest
        declared_root = Path(root_override) if root_override is not None else manifest.root
        self.root = declared_root.expanduser().resolve()

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        root_override: str | Path | None = None,
    ) -> Tracking2Adapter:
        return cls(load_tracking2_manifest(path), root_override=root_override)

    @property
    def cut_specs(self) -> tuple[CutSpec, ...]:
        return CUT_SPECS

    def _path(self, reference: FileReference) -> Path:
        path = (self.root / reference.path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ExternalInputValidationError(
                f"external path escapes declared root {self.root}: {reference.path}"
            ) from exc
        return path

    def validate_file(self, reference: FileReference) -> Path:
        """Return a path only after exact size and SHA-256 validation."""

        path = self._path(reference)
        if not path.is_file():
            raise ExternalInputValidationError(f"external input is not a file: {path}")
        observed_size = path.stat().st_size
        if observed_size != reference.size_bytes:
            raise ExternalInputValidationError(
                f"size mismatch for {path}: expected {reference.size_bytes}, "
                f"observed {observed_size}"
            )
        observed_hash = sha256_file(path)
        if observed_hash != reference.sha256:
            raise ExternalInputValidationError(
                f"SHA-256 mismatch for {path}: expected {reference.sha256}, "
                f"observed {observed_hash}"
            )
        return path

    def _validated_bytes(self, reference: FileReference) -> tuple[Path, bytes]:
        path = self._path(reference)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ExternalInputValidationError(
                f"could not read external input {path}: {exc}"
            ) from exc
        if len(raw) != reference.size_bytes:
            raise ExternalInputValidationError(
                f"size mismatch for {path}: expected {reference.size_bytes}, observed {len(raw)}"
            )
        observed_hash = sha256_bytes(raw)
        if observed_hash != reference.sha256:
            raise ExternalInputValidationError(
                f"SHA-256 mismatch for {path}: expected {reference.sha256}, "
                f"observed {observed_hash}"
            )
        return path, raw

    def validate_all(self) -> dict[str, Path]:
        """Validate every declared input without importing or executing Tracking2."""

        validated = {"model_source": self.validate_file(self.manifest.architecture.source)}
        for checkpoint in self.manifest.checkpoints:
            validated[f"checkpoint_epoch{checkpoint.epoch}"] = self.validate_file(checkpoint)
        validated["dataset_train"] = self.validate_file(self.manifest.datasets.train)
        validated["dataset_test"] = self.validate_file(self.manifest.datasets.test)
        validated["transplant"] = self.validate_file(self.manifest.transplant.file)
        return validated

    def import_verified_models(self) -> types.ModuleType:
        """Execute only the verified bytes of ``tracking2.models`` in an isolated module."""

        try:
            importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise OptionalTorchError(
                "PyTorch is required for the Tracking2 adapter; install the 'resnet' extra"
            ) from exc

        source = self.manifest.architecture.source
        path, raw = self._validated_bytes(source)
        module_name = f"_voronoi_verified_tracking2_models_{source.sha256[:16]}"
        existing = sys.modules.get(module_name)
        if existing is not None:
            if getattr(existing, "__verified_source_sha256__", None) != source.sha256:
                raise ExternalInputValidationError(
                    f"verified module cache collision for {module_name}"
                )
            return existing

        module = types.ModuleType(module_name)
        module.__file__ = str(path)
        module.__package__ = ""
        module.__verified_source_sha256__ = source.sha256
        sys.modules[module_name] = module
        try:
            exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        model_class = getattr(module, self.manifest.architecture.class_name, None)
        if model_class is None or getattr(model_class, "__module__", None) != module_name:
            sys.modules.pop(module_name, None)
            raise ExternalInputValidationError(
                f"verified source does not define {self.manifest.architecture.class_name}"
            )
        return module

    def load_model(self, epoch: int, *, device: Any = "cpu") -> Any:
        """Load one verified checkpoint with ``weights_only=True`` and strict keys."""

        checkpoint = self.manifest.checkpoint(epoch)
        _checkpoint_path, checkpoint_bytes = self._validated_bytes(checkpoint)
        models = self.import_verified_models()
        torch = importlib.import_module("torch")
        try:
            state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise OptionalTorchError(
                "installed PyTorch does not support mandatory weights_only checkpoint loading"
            ) from exc
        if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
            raise ExternalInputValidationError("checkpoint must contain a string-keyed state dict")
        if len(state) != self.manifest.architecture.state_dict_tensors:
            raise ExternalInputValidationError(
                "checkpoint tensor-count mismatch: expected "
                f"{self.manifest.architecture.state_dict_tensors}, observed {len(state)}"
            )
        if not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise ExternalInputValidationError("checkpoint state dict contains non-tensor values")
        parameter_count = sum(value.numel() for value in state.values())
        if parameter_count != self.manifest.architecture.state_dict_parameters:
            raise ExternalInputValidationError(
                "checkpoint parameter-count mismatch: expected "
                f"{self.manifest.architecture.state_dict_parameters}, observed {parameter_count}"
            )
        expected_dtype = getattr(torch, self.manifest.architecture.state_dict_dtype)
        if any(value.dtype != expected_dtype for value in state.values()):
            raise ExternalInputValidationError(
                f"checkpoint tensors must all have dtype {expected_dtype}"
            )

        model_class = getattr(models, self.manifest.architecture.class_name)
        model = model_class(
            num_classes=self.manifest.architecture.num_classes,
            width=self.manifest.architecture.width,
        )
        model.load_state_dict(state, strict=True)
        expected_names = tuple(cut.tracking2_name for cut in CUT_SPECS)
        if tuple(getattr(model, "block_names", ())) != expected_names:
            raise ExternalInputValidationError("loaded model has unexpected residual-block names")
        model.to(device)
        model.eval()
        return model

    def encode(self, model: Any, images: Any, cut: CutSelector) -> Any:
        """Return the native post-addition activation at ``cut``."""

        return model.encode_to_block(images, resolve_cut(cut).index)

    def suffix(self, model: Any, representation: Any, cut: CutSelector) -> Any:
        """Run the untouched suffix beginning immediately after ``cut``."""

        return model.forward_from_block(representation, resolve_cut(cut).index)

    def next_block(self, model: Any, representation: Any, cut: CutSelector) -> Any:
        """Apply exactly the next residual block to an intervened representation."""

        spec = resolve_cut(cut)
        if spec.index == CUT_SPECS[-1].index:
            raise ValueError("the final residual cut has no next residual block")
        return model.blocks[spec.index + 1](representation)

    def transplant_rows(self) -> tuple[TransplantRow, ...]:
        """Validate and normalize the legacy final-target transplant artifact."""

        path, raw = self._validated_bytes(self.manifest.transplant.file)
        return self.normalize_transplant_bytes(raw, source=path)

    def read_validated_bytes(self, reference: FileReference) -> bytes:
        """Return exact external bytes only after size and digest validation."""

        return self._validated_bytes(reference)[1]

    def normalize_transplant_bytes(
        self,
        raw: bytes,
        *,
        source: str | Path = "<embedded Tracking2 transplant>",
    ) -> tuple[TransplantRow, ...]:
        """Strictly normalize saved legacy bytes without reopening an external path."""

        path = Path(source)
        payload = _strict_json_object(raw, path=path)
        try:
            artifact = _LegacyTransplantArtifact.model_validate(payload)
        except ValidationError as exc:
            raise TransplantArtifactError(f"invalid transplant artifact {path}: {exc}") from exc
        self._validate_transplant_artifact(artifact, path=path)

        raw_rows = {(row.module_index, row.source): row for row in artifact.interventions}
        normalized: list[TransplantRow] = []
        source_labels = (
            "random",
            *(str(epoch) for epoch in self.manifest.training.checkpoint_epochs),
        )
        for cut in CUT_SPECS:
            for source_label in source_labels:
                row = raw_rows[(cut.transplant_module_index, source_label)]
                source_epoch = None if source_label == "random" else int(source_label)
                normalized.append(
                    TransplantRow(
                        seed=self.manifest.training.seed,
                        target_epoch=self.manifest.training.target_epoch,
                        cut_index=cut.index,
                        cut_name=cut.name,
                        tracking2_module_name=cut.tracking2_name,
                        transplant_module_index=cut.transplant_module_index,
                        source_kind="random" if source_epoch is None else "checkpoint",
                        source_epoch=source_epoch,
                        loss=row.loss,
                        accuracy=row.accuracy,
                        error=row.error,
                        delta_loss=row.delta_loss,
                        delta_error=row.delta_error,
                    )
                )
        return tuple(normalized)

    def _validate_transplant_artifact(
        self, artifact: _LegacyTransplantArtifact, *, path: Path
    ) -> None:
        expected_config: dict[str, Any] = {
            "fake_data": False,
            "train_size": self.manifest.training.train_size,
            "test_size": self.manifest.training.test_size,
            "epochs": self.manifest.training.epochs,
            "batch_size": self.manifest.training.batch_size,
            "learning_rate": self.manifest.training.learning_rate,
            "weight_decay": self.manifest.training.weight_decay,
            "checkpoint_epochs": self.manifest.training.checkpoint_epochs,
            "lr_milestones": self.manifest.training.lr_milestones,
            "width": self.manifest.training.width,
            "seed": self.manifest.training.seed,
            "device": self.manifest.training.device,
            "amp": self.manifest.training.amp,
        }
        mismatches = {
            field: {"expected": expected, "observed": getattr(artifact.config, field)}
            for field, expected in expected_config.items()
            if getattr(artifact.config, field) != expected
        }
        if mismatches:
            raise TransplantArtifactError(
                f"transplant config mismatch in {path}: "
                f"{json.dumps(mismatches, sort_keys=True, default=list)}"
            )
        if artifact.config.data_root != "data":
            raise TransplantArtifactError("legacy transplant config.data_root must be 'data'")
        if artifact.module_names != _TRACKING2_MODULE_NAMES:
            raise TransplantArtifactError("legacy transplant module_names do not match the model")

        training_epochs = tuple(row.epoch for row in artifact.training)
        if training_epochs != self.manifest.training.checkpoint_epochs:
            raise TransplantArtifactError("legacy transplant training epoch grid is incomplete")
        target_rows = [
            row for row in artifact.training if row.epoch == self.manifest.training.target_epoch
        ]
        if (
            len(target_rows) != 1
            or target_rows[0].model_dump(exclude={"epoch"}) != artifact.baseline.model_dump()
        ):
            raise TransplantArtifactError("baseline does not equal target-epoch training metrics")

        expected_sources = {
            "random",
            *(str(epoch) for epoch in self.manifest.training.checkpoint_epochs),
        }
        keys = [(row.module_index, row.source) for row in artifact.interventions]
        if len(keys) != len(set(keys)):
            raise TransplantArtifactError("legacy transplant interventions contain duplicate cells")
        expected_grid = {
            (module_index, source)
            for module_index in range(len(_TRACKING2_MODULE_NAMES))
            for source in expected_sources
        }
        if set(keys) != expected_grid:
            raise TransplantArtifactError("legacy transplant intervention grid is incomplete")

        for row in artifact.interventions:
            if row.module != _TRACKING2_MODULE_NAMES[row.module_index]:
                raise TransplantArtifactError(
                    f"module index/name mismatch at intervention index {row.module_index}"
                )
            if not math.isclose(
                row.delta_loss,
                row.loss - artifact.baseline.loss,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise TransplantArtifactError("delta_loss is inconsistent with baseline")
            if not math.isclose(
                row.delta_error,
                row.error - artifact.baseline.error,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise TransplantArtifactError("delta_error is inconsistent with baseline")


__all__ = [
    "CUT_SPECS",
    "ArchitectureReference",
    "CheckpointReference",
    "CutSpec",
    "DatasetReferences",
    "ExternalInputValidationError",
    "FileReference",
    "OptionalTorchError",
    "RecordedTrainingConfig",
    "Tracking2Adapter",
    "Tracking2Error",
    "Tracking2InputManifest",
    "Tracking2ManifestError",
    "TransplantArtifactError",
    "TransplantReference",
    "TransplantRow",
    "load_tracking2_manifest",
    "parse_tracking2_manifest_bytes",
    "resolve_cut",
]
