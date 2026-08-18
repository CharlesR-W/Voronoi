"""Hash-pinned, read-only adapter for the legacy Tracking2 VGG-19 trajectory.

The existing :mod:`voronoi_lab.exp1.tracking2` boundary intentionally supports
only the audited ResNet artifact.  This separate adapter keeps that contract
stable while admitting the user's requested non-residual control.  It executes
only the already hash-pinned ``tracking2.models`` source and loads checkpoints
with ``weights_only=True``.
"""

from __future__ import annotations

import importlib
import io
import json
import sys
import types
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from voronoi_lab.core.hashing import sha256_bytes

from .tracking2 import (
    CheckpointReference,
    DatasetReferences,
    ExternalInputValidationError,
    FileReference,
    OptionalTorchError,
    Tracking2ManifestError,
)

PositiveInt = Annotated[int, Field(gt=0, strict=True)]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
PositiveFloat = Annotated[float, Field(gt=0, strict=True)]
NonNegativeFloat = Annotated[float, Field(ge=0, strict=True)]
StrictBool = Annotated[bool, Field(strict=True)]
VersionOne = Annotated[int, Field(ge=1, le=1, strict=True)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class VGGArchitectureReference(_StrictModel):
    module: Literal["tracking2.models"] = "tracking2.models"
    class_name: Literal["InstrumentedVGG19"] = "InstrumentedVGG19"
    num_classes: PositiveInt = 10
    classifier_width: PositiveInt = 512
    width_multiplier: Annotated[float, Field(gt=0, le=1, strict=True)] = 1.0
    batch_norm: Literal[True] = True
    source: FileReference
    state_dict_tensors: PositiveInt
    state_dict_elements: PositiveInt
    dtype_tensor_counts: dict[Literal["float32", "int64"], PositiveInt]
    dtype_element_counts: dict[Literal["float32", "int64"], PositiveInt]

    @model_validator(mode="after")
    def validate_inventory(self) -> VGGArchitectureReference:
        if self.source.path != Path("src/tracking2/models.py"):
            raise ValueError("the only permitted external module is src/tracking2/models.py")
        if set(self.dtype_tensor_counts) != {"float32", "int64"}:
            raise ValueError("VGG dtype_tensor_counts must cover float32 and int64")
        if set(self.dtype_element_counts) != {"float32", "int64"}:
            raise ValueError("VGG dtype_element_counts must cover float32 and int64")
        if sum(self.dtype_tensor_counts.values()) != self.state_dict_tensors:
            raise ValueError("VGG dtype tensor counts do not sum to state_dict_tensors")
        if sum(self.dtype_element_counts.values()) != self.state_dict_elements:
            raise ValueError("VGG dtype element counts do not sum to state_dict_elements")
        return self


class VGGRecordedTrainingConfig(_StrictModel):
    seed: NonNegativeInt
    train_size: PositiveInt
    test_size: PositiveInt
    epochs: PositiveInt
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat
    checkpoint_epochs: tuple[NonNegativeInt, ...]
    lr_milestones: tuple[PositiveInt, ...]
    classifier_width: PositiveInt
    width_multiplier: Annotated[float, Field(gt=0, le=1, strict=True)]
    batch_norm: StrictBool
    device: str
    amp: StrictBool
    target_epoch: NonNegativeInt

    @model_validator(mode="after")
    def validate_schedule(self) -> VGGRecordedTrainingConfig:
        if not self.checkpoint_epochs or tuple(sorted(set(self.checkpoint_epochs))) != (
            self.checkpoint_epochs
        ):
            raise ValueError("VGG checkpoint epochs must be unique and increasing")
        if tuple(sorted(set(self.lr_milestones))) != self.lr_milestones:
            raise ValueError("VGG learning-rate milestones must be unique and increasing")
        if self.epochs != self.target_epoch or self.target_epoch not in self.checkpoint_epochs:
            raise ValueError("VGG target_epoch must equal epochs and be checkpointed")
        return self


class VGGTrainingRecordReference(_StrictModel):
    schema_version: VersionOne = 1
    experiment: Literal["vgg_checkpoint_training"] = "vgg_checkpoint_training"
    file: FileReference


class Tracking2VGGInputManifest(_StrictModel):
    schema_version: VersionOne = 1
    name: str
    lineage_quality: Literal["exploratory_legacy"]
    lineage_note: str
    root: Path
    observed_repository_revision: str
    read_only: Literal[True] = True
    architecture: VGGArchitectureReference
    training: VGGRecordedTrainingConfig
    checkpoints: tuple[CheckpointReference, ...]
    datasets: DatasetReferences
    training_record: VGGTrainingRecordReference

    @field_validator("name", "lineage_note")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("manifest text fields cannot be blank")
        return value

    @field_validator("observed_repository_revision")
    @classmethod
    def require_revision(cls, value: str) -> str:
        if len(value) not in {40, 64} or value.lower() != value:
            raise ValueError("observed_repository_revision must be a full lowercase digest")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError("observed_repository_revision must be hexadecimal") from error
        return value

    @model_validator(mode="after")
    def cross_validate(self) -> Tracking2VGGInputManifest:
        epochs = tuple(item.epoch for item in self.checkpoints)
        if epochs != self.training.checkpoint_epochs:
            raise ValueError("VGG checkpoint entries must match the training epoch axis")
        architecture = self.architecture
        training = self.training
        if (
            architecture.classifier_width != training.classifier_width
            or architecture.width_multiplier != training.width_multiplier
            or architecture.batch_norm != training.batch_norm
        ):
            raise ValueError("VGG architecture and training declarations disagree")
        return self

    def checkpoint(self, epoch: int) -> CheckpointReference:
        for item in self.checkpoints:
            if item.epoch == epoch:
                return item
        raise KeyError(f"VGG checkpoint epoch {epoch} is not declared")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
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
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def parse_tracking2_vgg_manifest_bytes(
    raw: bytes,
    *,
    source: str | Path = "<embedded Tracking2 VGG manifest>",
) -> Tracking2VGGInputManifest:
    try:
        value = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError, Tracking2ManifestError) as error:
        if isinstance(error, Tracking2ManifestError):
            raise
        raise Tracking2ManifestError(f"could not parse {source}: {error}") from error
    if not isinstance(value, dict):
        raise Tracking2ManifestError("VGG manifest must be a YAML mapping")
    try:
        return Tracking2VGGInputManifest.model_validate(value)
    except ValidationError as error:
        raise Tracking2ManifestError(f"invalid Tracking2 VGG manifest {source}: {error}") from error


def load_tracking2_vgg_manifest(path: str | Path) -> Tracking2VGGInputManifest:
    manifest_path = Path(path)
    try:
        raw = manifest_path.read_bytes()
    except OSError as error:
        raise Tracking2ManifestError(f"could not read {manifest_path}: {error}") from error
    return parse_tracking2_vgg_manifest_bytes(raw, source=manifest_path)


@dataclass(frozen=True, slots=True)
class VGGCutSpec:
    index: int
    name: str
    channels: int
    height: int
    width: int
    stage: int
    convolution: int

    @property
    def activation_shape(self) -> tuple[int, int, int]:
        return (self.channels, self.height, self.width)


def _vgg_cut_specs() -> tuple[VGGCutSpec, ...]:
    result: list[VGGCutSpec] = []
    index = 0
    spatial = 32
    for stage, (count, channels) in enumerate(
        zip((2, 2, 4, 4, 4), (64, 128, 256, 512, 512), strict=True),
        start=1,
    ):
        for convolution in range(1, count + 1):
            result.append(
                VGGCutSpec(
                    index,
                    f"stage{stage}.conv{convolution}",
                    channels,
                    spatial,
                    spatial,
                    stage,
                    convolution,
                )
            )
            index += 1
        spatial //= 2
    return tuple(result)


VGG_CUT_SPECS = _vgg_cut_specs()
_VGG_CUT_BY_INDEX = {cut.index: cut for cut in VGG_CUT_SPECS}
_VGG_CUT_BY_NAME = {cut.name: cut for cut in VGG_CUT_SPECS}
VGGCutSelector: TypeAlias = int | str | VGGCutSpec


def resolve_vgg_cut(cut: VGGCutSelector) -> VGGCutSpec:
    if isinstance(cut, VGGCutSpec):
        if _VGG_CUT_BY_INDEX.get(cut.index) != cut:
            raise ValueError("unrecognized VGG cut")
        return cut
    if isinstance(cut, bool):
        raise TypeError("a boolean is not a VGG cut index")
    if isinstance(cut, int):
        try:
            return _VGG_CUT_BY_INDEX[cut]
        except KeyError as error:
            raise ValueError("VGG cut index must be in [0, 15]") from error
    if isinstance(cut, str):
        try:
            return _VGG_CUT_BY_NAME[cut]
        except KeyError as error:
            raise ValueError(f"unknown VGG cut name: {cut!r}") from error
    raise TypeError("VGG cut must be an index, name, or VGGCutSpec")


class Tracking2VGGAdapter:
    def __init__(
        self,
        manifest: Tracking2VGGInputManifest,
        *,
        root_override: str | Path | None = None,
    ) -> None:
        self.manifest = manifest
        root = manifest.root if root_override is None else Path(root_override)
        self.root = root.expanduser().resolve()

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        root_override: str | Path | None = None,
    ) -> Tracking2VGGAdapter:
        return cls(load_tracking2_vgg_manifest(path), root_override=root_override)

    def _path(self, reference: FileReference) -> Path:
        path = (self.root / reference.path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ExternalInputValidationError("VGG external path escapes its root") from error
        return path

    def _validated_bytes(self, reference: FileReference) -> tuple[Path, bytes]:
        path = self._path(reference)
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ExternalInputValidationError(f"could not read VGG input {path}") from error
        if len(raw) != reference.size_bytes or sha256_bytes(raw) != reference.sha256:
            raise ExternalInputValidationError(
                f"VGG external input does not match manifest: {path}"
            )
        return path, raw

    def validate_file(self, reference: FileReference) -> Path:
        return self._validated_bytes(reference)[0]

    def validate_all(self) -> dict[str, Path]:
        result = {"model_source": self.validate_file(self.manifest.architecture.source)}
        for checkpoint in self.manifest.checkpoints:
            result[f"checkpoint_epoch{checkpoint.epoch}"] = self.validate_file(checkpoint)
        result["dataset_train"] = self.validate_file(self.manifest.datasets.train)
        result["dataset_test"] = self.validate_file(self.manifest.datasets.test)
        result["training_record"] = self.validate_file(self.manifest.training_record.file)
        return result

    def import_verified_models(self) -> types.ModuleType:
        try:
            importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise OptionalTorchError("PyTorch is required for the VGG adapter") from error
        source = self.manifest.architecture.source
        path, raw = self._validated_bytes(source)
        module_name = f"_voronoi_verified_tracking2_models_{source.sha256[:16]}"
        existing = sys.modules.get(module_name)
        if existing is not None:
            if getattr(existing, "__verified_source_sha256__", None) != source.sha256:
                raise ExternalInputValidationError("verified Tracking2 module cache collision")
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
            raise ExternalInputValidationError("verified source does not define InstrumentedVGG19")
        return module

    def load_model(self, epoch: int, *, device: Any = "cpu") -> Any:
        checkpoint = self.manifest.checkpoint(epoch)
        _path, raw = self._validated_bytes(checkpoint)
        torch = importlib.import_module("torch")
        try:
            state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        except TypeError as error:
            raise OptionalTorchError(
                "PyTorch weights_only checkpoint loading is required"
            ) from error
        if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
            raise ExternalInputValidationError("VGG checkpoint must be a string-keyed state dict")
        architecture = self.manifest.architecture
        if len(state) != architecture.state_dict_tensors:
            raise ExternalInputValidationError("VGG checkpoint tensor count differs from manifest")
        if not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise ExternalInputValidationError("VGG checkpoint contains non-tensor values")
        tensor_counts = Counter(str(value.dtype).removeprefix("torch.") for value in state.values())
        element_counts: Counter[str] = Counter()
        for value in state.values():
            element_counts[str(value.dtype).removeprefix("torch.")] += value.numel()
        if dict(tensor_counts) != architecture.dtype_tensor_counts:
            raise ExternalInputValidationError("VGG checkpoint dtype tensor inventory differs")
        if dict(element_counts) != architecture.dtype_element_counts:
            raise ExternalInputValidationError("VGG checkpoint dtype element inventory differs")
        models = self.import_verified_models()
        model_class = getattr(models, architecture.class_name)
        model = model_class(
            num_classes=architecture.num_classes,
            classifier_width=architecture.classifier_width,
            width_multiplier=architecture.width_multiplier,
            batch_norm=architecture.batch_norm,
        )
        model.load_state_dict(state, strict=True)
        expected_names = tuple(cut.name for cut in VGG_CUT_SPECS)
        observed_names = tuple(name for name, _module in model.intervention_modules()[:16])
        if observed_names != expected_names:
            raise ExternalInputValidationError("loaded VGG has unexpected convolution names")
        model.to(device)
        model.eval()
        return model

    def encode(self, model: Any, images: Any, cut: VGGCutSelector) -> Any:
        return model.encode_to_module(images, resolve_vgg_cut(cut).index)

    def suffix(self, model: Any, representation: Any, cut: VGGCutSelector) -> Any:
        return model.forward_from_module(representation, resolve_vgg_cut(cut).index)

    def read_training_record(self) -> dict[str, object]:
        _path, raw = self._validated_bytes(self.manifest.training_record.file)
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ExternalInputValidationError("VGG training record is invalid JSON") from error
        if not isinstance(value, dict):
            raise ExternalInputValidationError("VGG training record must be an object")
        if value.get("schema_version") != self.manifest.training_record.schema_version:
            raise ExternalInputValidationError("VGG training record schema differs from manifest")
        if value.get("experiment") != self.manifest.training_record.experiment:
            raise ExternalInputValidationError(
                "VGG training record experiment differs from manifest"
            )
        return value

    def read_validated_bytes(self, reference: FileReference) -> bytes:
        return self._validated_bytes(reference)[1]


__all__ = [
    "VGG_CUT_SPECS",
    "Tracking2VGGAdapter",
    "Tracking2VGGInputManifest",
    "VGGCutSpec",
    "load_tracking2_vgg_manifest",
    "parse_tracking2_vgg_manifest_bytes",
    "resolve_vgg_cut",
]
