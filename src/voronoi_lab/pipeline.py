"""Declarative stage DAG, shard estimates, and cache-stable stage identities."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from math import ceil
from pathlib import PurePosixPath

import numpy as np

from voronoi_lab.config import LabConfig
from voronoi_lab.core import (
    ArtifactRef,
    ArtifactStore,
    ComparisonOperator,
    GateCheck,
    GateEvaluationError,
    GateEvaluator,
    GateOverride,
    GateResult,
    GateRule,
    GateStatus,
    JSONLike,
    SeedDeriver,
    canonical_hash,
    sha256_bytes,
    thaw_json,
)
from voronoi_lab.exp1.probe_artifact import ProbeArtifactError, build_probe_bank_files
from voronoi_lab.exp1.torch_mechanics import summarize_resnet_mechanical_evidence
from voronoi_lab.exp1.tracking2 import (
    Tracking2Adapter,
    Tracking2Error,
    parse_tracking2_manifest_bytes,
)
from voronoi_lab.exp1.tracking2_vgg import parse_tracking2_vgg_manifest_bytes
from voronoi_lab.mechanical import replay_toy_geometry


class PipelineError(ValueError):
    """Raised for invalid stages, cycles, or inconsistent pipeline declarations."""


class ImplementationStatus(StrEnum):
    RUNNABLE = "RUNNABLE"
    PLANNED = "PLANNED"


ShardEstimator = Callable[[LabConfig], int]
_ARTIFACT_KIND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_STAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CONFIG_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_PAYLOAD_SCHEMA_RE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_PLATEAU_SCHEMA_VERSION = 2
_RAW_LOCAL_PLANE_JACOBIAN = "local_transition_plane_jacobian"
_RAW_ANCHOR_PLANE_JACOBIAN = "anchor_transition_plane_jacobian_by_context"
_RESIDUAL_LOCAL_PLANE_JACOBIAN = "local_residual_update_plane_jacobian"
_RESIDUAL_ANCHOR_PLANE_JACOBIAN = "anchor_residual_update_plane_jacobian_by_context"
_RAW_PLANE_ESTIMAND = "2D-plane-restricted ||DT||_F"
_RESIDUAL_PLANE_ESTIMAND = "2D-plane-restricted ||D(T-I)||_F"
_PLATEAU_CURVE_ESTIMAND = "center-and-direction median downstream-logit L2 from path base"
_ARCHITECTURE_COMPARISON_NOTE = (
    "The rows use different operators and legacy training recipes; this is a "
    "descriptive, confounded side-by-side view, not a causal architecture ablation."
)


def _validate_payload_path(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PipelineError(f"{label} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PipelineError(f"{label} is unsafe: {value!r}")
    if path.parts[0] == "manifest.json":
        raise PipelineError(f"{label} cannot use the reserved manifest.json path")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    description: str
    dependencies: tuple[str, ...] = ()
    config_paths: tuple[str, ...] = ()
    implementation: ImplementationStatus = ImplementationStatus.PLANNED
    stage_version: int = 1
    estimate_shards: ShardEstimator = lambda _config: 1
    expected_artifact_kind: str | None = None
    required_payload_paths: tuple[str, ...] = ()
    result_schema_version: int | None = None
    payload_schema_id: str | None = None
    gate_payload_path: str | None = None
    expected_gate_id: str | None = None
    gate_evidence_dependency: str | None = None
    gate_evidence_payload_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _STAGE_NAME_RE.fullmatch(self.name):
            raise PipelineError("stage names must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
        if (
            isinstance(self.stage_version, bool)
            or not isinstance(self.stage_version, int)
            or self.stage_version < 1
        ):
            raise PipelineError("stage_version must be a positive integer")
        if not isinstance(self.dependencies, Sequence) or isinstance(
            self.dependencies, (str, bytes)
        ):
            raise PipelineError("stage dependencies must be a sequence")
        dependencies = tuple(self.dependencies)
        if not all(
            isinstance(dependency, str) and _STAGE_NAME_RE.fullmatch(dependency)
            for dependency in dependencies
        ):
            raise PipelineError("stage dependencies must be valid stage names")
        object.__setattr__(self, "dependencies", dependencies)
        if len(set(self.dependencies)) != len(self.dependencies) or self.name in self.dependencies:
            raise PipelineError(f"invalid dependencies for stage {self.name}")
        if not isinstance(self.config_paths, Sequence) or isinstance(
            self.config_paths, (str, bytes)
        ):
            raise PipelineError("stage config_paths must be a sequence")
        config_paths = tuple(self.config_paths)
        if not all(
            isinstance(path, str) and _CONFIG_PATH_RE.fullmatch(path) for path in config_paths
        ):
            raise PipelineError("stage config_paths must be dotted configuration paths")
        if len(config_paths) != len(set(config_paths)):
            raise PipelineError("stage config_paths must be unique")
        object.__setattr__(self, "config_paths", config_paths)
        if not self.description.strip():
            raise PipelineError("stage descriptions cannot be blank")
        if self.expected_artifact_kind is not None and (
            not isinstance(self.expected_artifact_kind, str)
            or not _ARTIFACT_KIND_RE.fullmatch(self.expected_artifact_kind)
        ):
            raise PipelineError("expected_artifact_kind is not a valid artifact kind")
        if not isinstance(self.required_payload_paths, Sequence) or isinstance(
            self.required_payload_paths, (str, bytes)
        ):
            raise PipelineError("required_payload_paths must be a sequence")
        required_payloads = tuple(
            _validate_payload_path(path, label="required payload path")
            for path in self.required_payload_paths
        )
        if len(required_payloads) != len(set(required_payloads)):
            raise PipelineError("required payload paths must be unique")
        object.__setattr__(self, "required_payload_paths", required_payloads)
        if self.result_schema_version is not None and (
            isinstance(self.result_schema_version, bool)
            or not isinstance(self.result_schema_version, int)
            or self.result_schema_version < 1
        ):
            raise PipelineError("result_schema_version must be a positive integer")
        if self.payload_schema_id is not None and (
            not isinstance(self.payload_schema_id, str)
            or not _PAYLOAD_SCHEMA_RE.fullmatch(self.payload_schema_id)
        ):
            raise PipelineError("payload_schema_id is not a valid versioned schema identifier")
        required_json_payloads = tuple(
            path for path in required_payloads if PurePosixPath(path).suffix == ".json"
        )
        if required_json_payloads and self.result_schema_version is None:
            raise PipelineError(
                "stages with required JSON payloads must declare result_schema_version"
            )
        if self.gate_payload_path is not None:
            gate_path = _validate_payload_path(
                self.gate_payload_path,
                label="gate payload path",
            )
            if gate_path not in required_payloads:
                raise PipelineError("gate_payload_path must also be a required payload path")
            if PurePosixPath(gate_path).suffix != ".json":
                raise PipelineError("gate_payload_path must identify a JSON payload")
            object.__setattr__(self, "gate_payload_path", gate_path)
        if self.expected_gate_id is not None and (
            not isinstance(self.expected_gate_id, str)
            or not _STAGE_NAME_RE.fullmatch(self.expected_gate_id)
        ):
            raise PipelineError("expected_gate_id is not a valid gate identifier")
        if (self.gate_payload_path is None) != (self.expected_gate_id is None):
            raise PipelineError("gate_payload_path and expected_gate_id must be declared together")
        if (self.gate_evidence_dependency is None) != (self.gate_evidence_payload_path is None):
            raise PipelineError(
                "gate_evidence_dependency and gate_evidence_payload_path must be declared together"
            )
        if self.gate_evidence_dependency is not None:
            if self.gate_payload_path is None:
                raise PipelineError("gate evidence can only be declared for a gate stage")
            if not isinstance(self.gate_evidence_dependency, str) or not _STAGE_NAME_RE.fullmatch(
                self.gate_evidence_dependency
            ):
                raise PipelineError("gate_evidence_dependency must be a valid stage name")
            if self.gate_evidence_dependency not in self.dependencies:
                raise PipelineError("gate evidence must be a direct stage dependency")
            evidence_path = _validate_payload_path(
                self.gate_evidence_payload_path,  # type: ignore[arg-type]
                label="gate evidence payload path",
            )
            if PurePosixPath(evidence_path).suffix != ".json":
                raise PipelineError("gate evidence payload must identify a JSON payload")
            object.__setattr__(self, "gate_evidence_payload_path", evidence_path)

    def output_contract(self) -> dict[str, JSONLike]:
        """Return the cache-significant artifact contract for this stage."""

        return {
            "expected_artifact_kind": self.expected_artifact_kind,
            "expected_gate_id": self.expected_gate_id,
            "gate_evidence_dependency": self.gate_evidence_dependency,
            "gate_evidence_payload_path": self.gate_evidence_payload_path,
            "gate_payload_path": self.gate_payload_path,
            "payload_schema_id": self.payload_schema_id,
            "required_payload_paths": list(self.required_payload_paths),
            "result_schema_version": self.result_schema_version,
        }


@dataclass(frozen=True, slots=True)
class StagePlan:
    name: str
    description: str
    dependencies: tuple[str, ...]
    implementation: ImplementationStatus
    estimated_shards: int


class StageRegistry:
    def __init__(self, stages: Iterable[StageSpec]) -> None:
        specs = tuple(stages)
        self._stages = {stage.name: stage for stage in specs}
        if len(self._stages) != len(specs):
            raise PipelineError("stage names must be unique")
        for stage in specs:
            unknown = set(stage.dependencies) - set(self._stages)
            if unknown:
                raise PipelineError(
                    f"stage {stage.name} has unknown dependencies: {sorted(unknown)}"
                )
            if stage.gate_evidence_dependency is not None:
                producer = self._stages[stage.gate_evidence_dependency]
                if stage.gate_evidence_payload_path not in producer.required_payload_paths:
                    raise PipelineError(
                        f"stage {stage.name} gate evidence payload is not declared by "
                        f"producer {producer.name}"
                    )
        self.topological_order()

    def get(self, name: str) -> StageSpec:
        try:
            return self._stages[name]
        except KeyError as error:
            raise PipelineError(f"unknown stage: {name}") from error

    def topological_order(self, targets: Iterable[str] | None = None) -> tuple[StageSpec, ...]:
        selected = set(self._stages) if targets is None else self._closure(tuple(targets))
        temporary: set[str] = set()
        permanent: set[str] = set()
        ordered: list[StageSpec] = []

        def visit(name: str) -> None:
            if name in permanent or name not in selected:
                return
            if name in temporary:
                raise PipelineError(f"pipeline dependency cycle includes {name}")
            temporary.add(name)
            for dependency in self._stages[name].dependencies:
                visit(dependency)
            temporary.remove(name)
            permanent.add(name)
            ordered.append(self._stages[name])

        for name in self._stages:
            visit(name)
        return tuple(ordered)

    def plan(
        self, config: LabConfig, targets: Iterable[str] | None = None
    ) -> tuple[StagePlan, ...]:
        plans: list[StagePlan] = []
        for stage in self.topological_order(targets):
            shards = stage.estimate_shards(config)
            if isinstance(shards, bool) or not isinstance(shards, int) or shards < 1:
                raise PipelineError(f"stage {stage.name} returned an invalid shard estimate")
            plans.append(
                StagePlan(
                    name=stage.name,
                    description=stage.description,
                    dependencies=stage.dependencies,
                    implementation=stage.implementation,
                    estimated_shards=shards,
                )
            )
        return tuple(plans)

    def _closure(self, targets: tuple[str, ...]) -> set[str]:
        if not targets:
            raise PipelineError("at least one target stage is required")
        selected: set[str] = set()

        def add(name: str) -> None:
            stage = self.get(name)
            if name in selected:
                return
            selected.add(name)
            for dependency in stage.dependencies:
                add(dependency)

        for target in targets:
            add(target)
        return selected


@dataclass(frozen=True, slots=True)
class _SuccessfulStageValidation:
    result: GateResult | None
    referenced_artifact_ids: tuple[str, ...]


@dataclass(slots=True)
class StageValidationContext:
    """Ephemeral producer-success cache for one semantic-validation call chain.

    Entries are process-local and caller-scoped. Keys bind every input that can
    change validation semantics; failures are never inserted. Artifact bytes
    and the referenced immutable closure are checksum-verified before a cached
    success is returned. Gate artifacts remain deliberately uncached: their
    cheap replay re-enters the memoized evidence producer, which re-verifies its
    complete recorded closure before reuse.
    """

    _successes: dict[str, _SuccessfulStageValidation] = dataclass_field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def successful_validations(self) -> int:
        """Number of distinct producer validations memoized successfully."""

        return len(self._successes)

    def _key(
        self,
        reference: ArtifactRef,
        stage: StageSpec,
        *,
        config: LabConfig | None,
        gate_rule: GateRule | None,
        gate_override_authorization: Mapping[str, JSONLike] | None,
        registry: StageRegistry,
        source_identity: Mapping[str, JSONLike] | None,
    ) -> str:
        return canonical_hash(
            {
                "artifact_id": reference.artifact_id,
                "config": None if config is None else config.model_dump(mode="json"),
                "gate_override_authorization": (
                    None
                    if gate_override_authorization is None
                    else dict(gate_override_authorization)
                ),
                "gate_rule": None if gate_rule is None else gate_rule.to_dict(),
                "registry_contract": _validation_registry_contract(registry),
                "source_identity": {
                    "declared": thaw_json(reference.manifest.metadata.get("source_identity")),
                    "expected": None if source_identity is None else dict(source_identity),
                },
                "stage_contract": _validation_stage_contract(stage),
            }
        )

    def _lookup(
        self,
        key: str,
        *,
        store: ArtifactStore,
        stage_name: str,
    ) -> tuple[bool, GateResult | None]:
        success = self._successes.get(key)
        if success is None:
            return False, None
        try:
            for artifact_id in success.referenced_artifact_ids:
                store.get(artifact_id, verify=True)
        except Exception as error:
            raise PipelineError(
                f"memoized validation closure for {stage_name} is unavailable or corrupt"
            ) from error
        return True, success.result

    def _remember(
        self,
        key: str,
        result: GateResult | None,
        referenced_artifact_ids: Iterable[str],
    ) -> None:
        self._successes[key] = _SuccessfulStageValidation(
            result=result,
            referenced_artifact_ids=tuple(sorted(set(referenced_artifact_ids))),
        )


def _validation_stage_contract(stage: StageSpec) -> dict[str, JSONLike]:
    return {
        "config_paths": list(stage.config_paths),
        "dependencies": list(stage.dependencies),
        "implementation": stage.implementation.value,
        "name": stage.name,
        "output_contract": stage.output_contract(),
        "stage_version": stage.stage_version,
    }


def _validation_registry_contract(registry: StageRegistry) -> list[JSONLike]:
    return [_validation_stage_contract(stage) for stage in registry.topological_order()]


def stage_config(config: LabConfig, paths: Iterable[str]) -> dict[str, JSONLike]:
    """Select exactly the config subtrees declared as inputs to a stage."""

    root: object = config.model_dump(mode="json")
    selected: dict[str, JSONLike] = {}
    for path in paths:
        current = root
        for component in path.split("."):
            if not isinstance(current, Mapping) or component not in current:
                raise PipelineError(f"configuration path does not exist: {path}")
            current = current[component]
        selected[path] = current  # type: ignore[assignment]
    return selected


def stage_signature(
    stage: StageSpec,
    config: LabConfig,
    *,
    upstream_artifact_ids: Mapping[str, str],
    source_identity: Mapping[str, JSONLike],
) -> str:
    if set(upstream_artifact_ids) != set(stage.dependencies):
        raise PipelineError(
            f"upstream ids for {stage.name} must exactly match {stage.dependencies}"
        )
    return canonical_hash(
        {
            "config": stage_config(config, stage.config_paths),
            "output_contract": stage.output_contract(),
            "source": dict(source_identity),
            "stage": stage.name,
            "stage_version": stage.stage_version,
            "upstream_artifacts": dict(sorted(upstream_artifact_ids.items())),
        }
    )


def expected_gate_rule(stage: StageSpec | str, config: LabConfig) -> GateRule:
    """Build the frozen rule for a runnable gate from its signed threshold config."""

    name = stage.name if isinstance(stage, StageSpec) else stage
    if name == "gate.mechanical":
        thresholds = config.gates.mechanical
        return GateRule(
            gate_id="mechanical",
            description="Frozen implementation checks required before Experiment 1 plots.",
            checks=(
                GateCheck(
                    "probe_bank_determinism",
                    "probe_banks.deterministic",
                    ComparisonOperator.IS_TRUE,
                ),
                GateCheck(
                    "probe_bank_artifact",
                    "probe_banks.artifact_valid",
                    ComparisonOperator.IS_TRUE,
                ),
                GateCheck(
                    "distinct_train_test_sources",
                    "probe_banks.distinct_train_test_sources",
                    ComparisonOperator.IS_TRUE,
                ),
                GateCheck("identity", "resnet.identity_exact", ComparisonOperator.IS_TRUE),
                GateCheck(
                    "roundtrip",
                    "geometry.roundtrip.relative_rms_error",
                    ComparisonOperator.LT,
                    thresholds.roundtrip_relative_rms_max,
                ),
                GateCheck(
                    "centroid_reconstruction",
                    "geometry.centroid_reconstruction.relative_rms_error",
                    ComparisonOperator.LT,
                    thresholds.roundtrip_relative_rms_max,
                ),
                GateCheck(
                    "boundary_formula",
                    "geometry.boundary.passed",
                    ComparisonOperator.IS_TRUE,
                ),
                GateCheck(
                    "mixture_gaussian",
                    "geometry.mixture_gaussian.passed",
                    ComparisonOperator.IS_TRUE,
                ),
                GateCheck(
                    "jvp_median",
                    "resnet.jvp_median_relative_error",
                    ComparisonOperator.LT,
                    thresholds.jvp_median_relative_error_max,
                ),
                GateCheck(
                    "jvp_completion",
                    "resnet.jvp_cuts_completed",
                    ComparisonOperator.EQ,
                    len(config.experiment1.sentinel_cuts),
                ),
                GateCheck(
                    "jvp_p95",
                    "resnet.jvp_p95_relative_error",
                    ComparisonOperator.LT,
                    thresholds.jvp_p95_relative_error_max,
                ),
            ),
        )
    raise PipelineError(f"no frozen runnable gate rule is registered for {name}")


def expected_gate_override_authorization(
    stage: StageSpec | str, config: LabConfig
) -> dict[str, JSONLike] | None:
    """Return the one signed authorization applicable to a runnable gate."""

    name = stage.name if isinstance(stage, StageSpec) else stage
    authorizations = {
        "gate.mechanical": config.gates.overrides.mechanical,
    }
    if name not in authorizations:
        raise PipelineError(f"no runnable gate override slot is registered for {name}")
    authorization = authorizations[name]
    return None if authorization is None else authorization.model_dump(mode="json")


def validate_gate_result_against_rule(
    result: GateResult,
    rule: GateRule,
    *,
    gate_rule_signature: object = None,
) -> None:
    """Require serialized gate evidence to implement the exact configured rule."""

    if result.gate_id != rule.gate_id:
        raise PipelineError("gate result id does not match its configured rule")
    if result.required_passes != rule.min_passes or len(result.checks) != len(rule.checks):
        raise PipelineError("gate result check count does not match its configured rule")
    for observed, expected in zip(result.checks, rule.checks, strict=True):
        observed_contract = {
            "counts_toward_gate": observed.counts_toward_gate,
            "metric": observed.metric,
            "name": observed.name,
            "operator": observed.operator.value,
            "threshold": thaw_json(observed.threshold),
        }
        expected_contract = {
            "counts_toward_gate": expected.counts_toward_gate,
            "metric": expected.metric,
            "name": expected.name,
            "operator": expected.operator.value,
            "threshold": thaw_json(expected.threshold),
        }
        if canonical_hash(observed_contract) != canonical_hash(expected_contract):
            raise PipelineError(
                f"gate result check {observed.name!r} does not match its configured rule"
            )
    expected_signature = canonical_hash(rule.to_dict())
    if gate_rule_signature != expected_signature:
        raise PipelineError("gate artifact has no matching configured rule signature")


def _require_fields(payload: Mapping[str, object], fields: Sequence[str], *, label: str) -> None:
    missing = sorted(set(fields).difference(payload))
    if missing:
        raise PipelineError(f"{label} is missing required fields: {', '.join(missing)}")


def _as_object(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PipelineError(f"{label} must be an object")
    return value


def _as_array(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PipelineError(f"{label} must be an array")
    return value


def _as_int(value: object, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise PipelineError(f"{label} must be at least {minimum}")
    return value


def _as_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError(f"{label} must be numeric")
    return float(value)


def _as_nonnegative_number(value: object, *, label: str) -> float:
    number = _as_number(value, label=label)
    if not np.isfinite(number) or number < 0:
        raise PipelineError(f"{label} must be finite and nonnegative")
    return number


def _validate_inputs_payload(payload: Mapping[str, object], config: LabConfig | None) -> None:
    label = "Tracking2 inputs payload"
    _require_fields(
        payload,
        (
            "architecture",
            "dataset_isolation",
            "external_root",
            "lineage_note",
            "lineage_quality",
            "manifest",
            "observed_repository_revision",
            "read_only",
            "training",
            "transplant_rows",
            "validated_files",
        ),
        label=label,
    )
    if payload["read_only"] is not True:
        raise PipelineError(f"{label} must assert read_only=true")
    if config is None:
        raise PipelineError(f"{label} validation requires the signed run configuration")
    expected_root = config.inputs.tracking2.root.as_posix()
    if payload["external_root"] != expected_root:
        raise PipelineError(f"{label} external_root does not match its signed declaration")
    if not isinstance(payload["lineage_quality"], str) or not payload["lineage_quality"]:
        raise PipelineError(f"{label} lineage_quality must be nonempty")
    _as_object(payload["architecture"], label=f"{label} architecture")
    isolation = _as_object(payload["dataset_isolation"], label=f"{label} dataset_isolation")
    _require_fields(
        isolation,
        ("passed", "train_test_distinct_hashes", "train_test_distinct_paths"),
        label=f"{label} dataset_isolation",
    )
    if any(isolation[field] is not True for field in isolation):
        raise PipelineError(f"{label} must prove distinct train/test source files by path and hash")
    _as_object(payload["training"], label=f"{label} training")
    manifest = _as_object(payload["manifest"], label=f"{label} manifest")
    _require_fields(manifest, ("path", "sha256"), label=f"{label} manifest")
    if manifest["path"] != config.inputs.tracking2.manifest.as_posix():
        raise PipelineError(f"{label} manifest path does not match its signed declaration")
    if not isinstance(manifest["sha256"], str) or not _DIGEST_RE.fullmatch(manifest["sha256"]):
        raise PipelineError(f"{label} manifest digest is invalid")
    if manifest["sha256"] != config.inputs.tracking2.manifest_sha256:
        raise PipelineError(f"{label} manifest digest does not match its signed declaration")
    validated = _as_object(payload["validated_files"], label=f"{label} validated_files")
    if not validated:
        raise PipelineError(f"{label} validated_files cannot be empty")
    for name, raw in validated.items():
        entry = _as_object(raw, label=f"{label} validated_files[{name!r}]")
        _require_fields(
            entry,
            ("declared_path", "sha256", "size_bytes"),
            label=f"{label} validated_files[{name!r}]",
        )
        if not isinstance(entry["sha256"], str) or not _DIGEST_RE.fullmatch(entry["sha256"]):
            raise PipelineError(f"{label} validated file {name!r} has an invalid digest")
        _as_int(entry["size_bytes"], label=f"{label} validated file size", minimum=1)
    rows = _as_array(payload["transplant_rows"], label=f"{label} transplant_rows")
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise PipelineError(f"{label} transplant_rows must contain result objects")


def _validate_inputs_artifact(
    payload: Mapping[str, object],
    reference: ArtifactRef,
    store: ArtifactStore,
    config: LabConfig | None,
) -> None:
    label = "Tracking2 inputs artifact"
    _validate_inputs_payload(payload, config)
    if config is None:
        raise PipelineError(f"{label} validation requires the signed run configuration")
    files = {entry.path: entry for entry in reference.manifest.files}
    expected_paths = {"inputs.json", "manifest.yaml", "transplant.json"}
    if set(files) != expected_paths:
        raise PipelineError(f"{label} must preserve its exact manifest and transplant bytes")
    expected_media = {
        "inputs.json": "application/json",
        "manifest.yaml": "application/yaml",
        "transplant.json": "application/json",
    }
    if any(files[path].media_type != media for path, media in expected_media.items()):
        raise PipelineError(f"{label} payload media types are invalid")
    manifest_bytes = store.read_bytes(reference.artifact_id, "manifest.yaml")
    if sha256_bytes(manifest_bytes) != config.inputs.tracking2.manifest_sha256:
        raise PipelineError(f"{label} manifest bytes do not match the signed digest")
    try:
        manifest = parse_tracking2_manifest_bytes(manifest_bytes)
    except Tracking2Error as error:
        raise PipelineError(f"{label} embedded manifest is invalid") from error
    expected_model = f"preactivation_resnet18_v2_width{manifest.architecture.width}"
    if config.inputs.tracking2.expected_model != expected_model:
        raise PipelineError(f"{label} architecture does not match the signed model")
    if tuple(config.experiment1.checkpoints) != tuple(
        checkpoint.epoch for checkpoint in manifest.checkpoints
    ):
        raise PipelineError(f"{label} checkpoint axis does not match the signed configuration")
    banks = config.experiment1.probe_banks
    if banks.fit_train_images + banks.independent_fit_train_images > manifest.training.train_size:
        raise PipelineError(f"{label} training banks exceed the embedded training split")
    required_test = (
        max(banks.geometry_test_images, banks.intervention_test_images)
        if banks.intervention_nested_in_geometry
        else banks.geometry_test_images + banks.intervention_test_images
    )
    if required_test > manifest.training.test_size:
        raise PipelineError(f"{label} held-out banks exceed the embedded test split")
    references = {
        "model_source": manifest.architecture.source,
        **{
            f"checkpoint_epoch{checkpoint.epoch}": checkpoint for checkpoint in manifest.checkpoints
        },
        "dataset_train": manifest.datasets.train,
        "dataset_test": manifest.datasets.test,
        "transplant": manifest.transplant.file,
    }
    expected_validated_files: dict[str, JSONLike] = {
        name: {
            "declared_path": item.path.as_posix(),
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for name, item in sorted(references.items())
    }
    transplant_bytes = store.read_bytes(reference.artifact_id, "transplant.json")
    if (
        len(transplant_bytes) != manifest.transplant.file.size_bytes
        or sha256_bytes(transplant_bytes) != manifest.transplant.file.sha256
    ):
        raise PipelineError(f"{label} transplant bytes do not match the embedded manifest")
    try:
        adapter = Tracking2Adapter(manifest, root_override=config.inputs.tracking2.root)
        rows = [
            row.model_dump(mode="json")
            for row in adapter.normalize_transplant_bytes(transplant_bytes)
        ]
    except Tracking2Error as error:
        raise PipelineError(f"{label} embedded transplant is invalid") from error
    expected_payload: dict[str, JSONLike] = {
        "schema_version": 1,
        "architecture": manifest.architecture.model_dump(mode="json"),
        "dataset_isolation": {
            "passed": True,
            "train_test_distinct_hashes": (
                manifest.datasets.train.sha256 != manifest.datasets.test.sha256
            ),
            "train_test_distinct_paths": (
                manifest.datasets.train.path != manifest.datasets.test.path
            ),
        },
        "external_root": config.inputs.tracking2.root.as_posix(),
        "lineage_note": manifest.lineage_note,
        "lineage_quality": manifest.lineage_quality,
        "manifest": {
            "path": config.inputs.tracking2.manifest.as_posix(),
            "sha256": config.inputs.tracking2.manifest_sha256,
        },
        "observed_repository_revision": manifest.observed_repository_revision,
        "read_only": True,
        "training": manifest.training.model_dump(mode="json"),
        "validated_files": expected_validated_files,
        "transplant_rows": rows,
    }
    if canonical_hash(payload) != canonical_hash(expected_payload):
        raise PipelineError(f"{label} summary does not match its preserved raw evidence")


def _validate_vgg_inputs_payload(payload: Mapping[str, object], config: LabConfig | None) -> None:
    label = "Tracking2 VGG inputs payload"
    _require_fields(
        payload,
        (
            "architecture",
            "dataset_isolation",
            "external_root",
            "lineage_note",
            "lineage_quality",
            "manifest",
            "observed_repository_revision",
            "read_only",
            "training",
            "training_record",
            "validated_files",
        ),
        label=label,
    )
    if payload["read_only"] is not True:
        raise PipelineError(f"{label} must assert read_only=true")
    if config is None:
        raise PipelineError(f"{label} validation requires the signed run configuration")
    if payload["external_root"] != config.inputs.tracking2_vgg.root.as_posix():
        raise PipelineError(f"{label} external_root does not match its signed declaration")
    if payload["lineage_quality"] != "exploratory_legacy":
        raise PipelineError(f"{label} must remain explicitly exploratory legacy evidence")
    if not isinstance(payload["lineage_note"], str) or not payload["lineage_note"].strip():
        raise PipelineError(f"{label} lineage_note must be nonempty")
    _as_object(payload["architecture"], label=f"{label} architecture")
    isolation = _as_object(payload["dataset_isolation"], label=f"{label} dataset_isolation")
    _require_fields(
        isolation,
        ("passed", "train_test_distinct_hashes", "train_test_distinct_paths"),
        label=f"{label} dataset_isolation",
    )
    if any(
        isolation[field] is not True
        for field in ("passed", "train_test_distinct_hashes", "train_test_distinct_paths")
    ):
        raise PipelineError(f"{label} must prove distinct train/test source files by path and hash")
    _as_object(payload["training"], label=f"{label} training")
    record = _as_object(payload["training_record"], label=f"{label} training_record")
    _require_fields(record, ("schema_version", "experiment"), label=f"{label} training_record")
    manifest = _as_object(payload["manifest"], label=f"{label} manifest")
    _require_fields(manifest, ("path", "sha256"), label=f"{label} manifest")
    if manifest["path"] != config.inputs.tracking2_vgg.manifest.as_posix():
        raise PipelineError(f"{label} manifest path does not match its signed declaration")
    if not isinstance(manifest["sha256"], str) or not _DIGEST_RE.fullmatch(manifest["sha256"]):
        raise PipelineError(f"{label} manifest digest is invalid")
    if manifest["sha256"] != config.inputs.tracking2_vgg.manifest_sha256:
        raise PipelineError(f"{label} manifest digest does not match its signed declaration")
    validated = _as_object(payload["validated_files"], label=f"{label} validated_files")
    if not validated:
        raise PipelineError(f"{label} validated_files cannot be empty")
    for name, raw in validated.items():
        entry = _as_object(raw, label=f"{label} validated_files[{name!r}]")
        _require_fields(
            entry,
            ("declared_path", "sha256", "size_bytes"),
            label=f"{label} validated_files[{name!r}]",
        )
        if not isinstance(entry["sha256"], str) or not _DIGEST_RE.fullmatch(entry["sha256"]):
            raise PipelineError(f"{label} validated file {name!r} has an invalid digest")
        _as_int(entry["size_bytes"], label=f"{label} validated file size", minimum=1)


def _validate_vgg_inputs_artifact(
    payload: Mapping[str, object],
    reference: ArtifactRef,
    store: ArtifactStore,
    config: LabConfig | None,
) -> None:
    label = "Tracking2 VGG inputs artifact"
    _validate_vgg_inputs_payload(payload, config)
    if config is None:
        raise PipelineError(f"{label} validation requires the signed run configuration")
    files = {entry.path: entry for entry in reference.manifest.files}
    expected_paths = {"inputs.json", "manifest.yaml", "criticality.json"}
    if set(files) != expected_paths:
        raise PipelineError(f"{label} must preserve its exact manifest and criticality bytes")
    expected_media = {
        "inputs.json": "application/json",
        "manifest.yaml": "application/yaml",
        "criticality.json": "application/json",
    }
    if any(files[path].media_type != media for path, media in expected_media.items()):
        raise PipelineError(f"{label} payload media types are invalid")
    manifest_bytes = store.read_bytes(reference.artifact_id, "manifest.yaml")
    if sha256_bytes(manifest_bytes) != config.inputs.tracking2_vgg.manifest_sha256:
        raise PipelineError(f"{label} manifest bytes do not match the signed digest")
    try:
        manifest = parse_tracking2_vgg_manifest_bytes(manifest_bytes)
    except Tracking2Error as error:
        raise PipelineError(f"{label} embedded manifest is invalid") from error
    architecture = manifest.architecture
    expected_model = (
        f"vgg19_bn_classifier{architecture.classifier_width}_width{architecture.width_multiplier:g}"
    )
    if config.inputs.tracking2_vgg.expected_model != expected_model:
        raise PipelineError(f"{label} architecture does not match the signed model")
    if manifest.lineage_quality != "exploratory_legacy":
        raise PipelineError(f"{label} lineage must remain explicitly exploratory")
    if tuple(config.experiment1.checkpoints) != tuple(
        checkpoint.epoch for checkpoint in manifest.checkpoints
    ):
        raise PipelineError(f"{label} checkpoint axis does not match the signed configuration")
    banks = config.experiment1.probe_banks
    if banks.fit_train_images + banks.independent_fit_train_images > manifest.training.train_size:
        raise PipelineError(f"{label} training banks exceed the embedded training split")
    required_test = (
        max(banks.geometry_test_images, banks.intervention_test_images)
        if banks.intervention_nested_in_geometry
        else banks.geometry_test_images + banks.intervention_test_images
    )
    if required_test > manifest.training.test_size:
        raise PipelineError(f"{label} held-out banks exceed the embedded test split")
    references = {
        "model_source": manifest.architecture.source,
        **{
            f"checkpoint_epoch{checkpoint.epoch}": checkpoint for checkpoint in manifest.checkpoints
        },
        "dataset_train": manifest.datasets.train,
        "dataset_test": manifest.datasets.test,
        "training_record": manifest.training_record.file,
    }
    expected_validated_files: dict[str, JSONLike] = {
        name: {
            "declared_path": item.path.as_posix(),
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for name, item in sorted(references.items())
    }
    criticality_bytes = store.read_bytes(reference.artifact_id, "criticality.json")
    criticality_reference = manifest.training_record.file
    if (
        len(criticality_bytes) != criticality_reference.size_bytes
        or sha256_bytes(criticality_bytes) != criticality_reference.sha256
    ):
        raise PipelineError(f"{label} criticality bytes do not match the embedded manifest")
    try:
        criticality = json.loads(criticality_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PipelineError(f"{label} embedded criticality metadata is invalid JSON") from error
    if not isinstance(criticality, dict):
        raise PipelineError(f"{label} embedded criticality metadata must be an object")
    if criticality.get("schema_version") != manifest.training_record.schema_version:
        raise PipelineError(f"{label} criticality schema does not match the manifest")
    if criticality.get("experiment") != manifest.training_record.experiment:
        raise PipelineError(f"{label} criticality experiment does not match the manifest")
    expected_payload: dict[str, JSONLike] = {
        "schema_version": 1,
        "architecture": manifest.architecture.model_dump(mode="json"),
        "dataset_isolation": {
            "passed": True,
            "train_test_distinct_hashes": (
                manifest.datasets.train.sha256 != manifest.datasets.test.sha256
            ),
            "train_test_distinct_paths": (
                manifest.datasets.train.path != manifest.datasets.test.path
            ),
        },
        "external_root": config.inputs.tracking2_vgg.root.as_posix(),
        "lineage_note": manifest.lineage_note,
        "lineage_quality": "exploratory_legacy",
        "manifest": {
            "path": config.inputs.tracking2_vgg.manifest.as_posix(),
            "sha256": config.inputs.tracking2_vgg.manifest_sha256,
        },
        "observed_repository_revision": manifest.observed_repository_revision,
        "read_only": True,
        "training": manifest.training.model_dump(mode="json"),
        "training_record": criticality,
        "validated_files": expected_validated_files,
    }
    if canonical_hash(payload) != canonical_hash(expected_payload):
        raise PipelineError(f"{label} summary does not match its preserved raw evidence")
    expected_metadata = {
        "criticality_experiment": manifest.training_record.experiment,
        "input_count": len(references),
        "lineage_quality": "exploratory_legacy",
    }
    if any(
        reference.manifest.metadata.get(key) != value for key, value in expected_metadata.items()
    ):
        raise PipelineError(f"{label} provenance metadata is inconsistent")


def _validate_probe_plan_payload(payload: Mapping[str, object], declared_paths: set[str]) -> None:
    label = "probe-bank plan"
    _require_fields(
        payload,
        (
            "bootstrap",
            "cuts",
            "equal_weight_per_image",
            "intervention_nested_in_geometry",
            "max_sites_per_image",
            "roles",
            "root_seed",
            "site_coordinate_columns",
            "site_weight_rule",
        ),
        label=label,
    )
    if payload["equal_weight_per_image"] is not True:
        raise PipelineError(f"{label} must use equal image weights")
    cuts = _as_array(payload["cuts"], label=f"{label} cuts")
    if not cuts or any(not isinstance(cut, str) or not cut for cut in cuts):
        raise PipelineError(f"{label} cuts must contain names")
    roles = _as_object(payload["roles"], label=f"{label} roles")
    required_roles = {"codebook_fit", "geometry", "independent_codebook_fit", "intervention"}
    if set(roles) != required_roles:
        raise PipelineError(f"{label} roles do not match the frozen bank roles")
    for role_name, raw in roles.items():
        role = _as_object(raw, label=f"{label} role {role_name!r}")
        _require_fields(
            role,
            ("count", "index_file", "site_files", "split"),
            label=f"{label} role {role_name!r}",
        )
        _as_int(role["count"], label=f"{label} role count", minimum=1)
        index_file = role["index_file"]
        if not isinstance(index_file, str) or index_file not in declared_paths:
            raise PipelineError(f"{label} role {role_name!r} index file is not declared")
        sites = _as_object(role["site_files"], label=f"{label} role sites")
        if set(sites) != set(cuts):
            raise PipelineError(f"{label} role {role_name!r} sites do not match cuts")
        for cut_name, raw_site in sites.items():
            site = _as_object(raw_site, label=f"{label} site {cut_name!r}")
            _require_fields(site, ("activation_shape", "count", "file"), label="site entry")
            site_file = site["file"]
            if not isinstance(site_file, str) or site_file not in declared_paths:
                raise PipelineError(f"{label} site {cut_name!r} file is not declared")
        if role_name in {"geometry", "intervention"}:
            _require_fields(
                role,
                ("bootstrap_file", "bootstrap_shape"),
                label=f"{label} role {role_name!r}",
            )
            bootstrap_file = role["bootstrap_file"]
            if not isinstance(bootstrap_file, str) or bootstrap_file not in declared_paths:
                raise PipelineError(f"{label} role {role_name!r} bootstrap is not declared")


def _validate_probe_artifact(
    payload: Mapping[str, object],
    declared_paths: set[str],
    reference: ArtifactRef,
    store: ArtifactStore,
    config: LabConfig | None,
) -> None:
    label = "probe-bank artifact"
    if config is None:
        raise PipelineError(f"{label} validation requires the signed run configuration")
    upstreams = thaw_json(reference.manifest.metadata.get("upstream_artifacts"))
    if not isinstance(upstreams, dict) or set(upstreams) != {"inputs.tracking2"}:
        raise PipelineError(f"{label} must bind its exact Tracking2 input artifact")
    input_id = upstreams["inputs.tracking2"]
    if not isinstance(input_id, str):
        raise PipelineError(f"{label} input artifact id is invalid")
    try:
        input_reference = store.get(input_id, verify=True)
        input_payload = _as_object(
            store.read_json(input_id, "inputs.json"), label=f"{label} input dependency"
        )
    except Exception as error:
        raise PipelineError(f"{label} input dependency is unavailable or corrupt") from error
    if input_reference.manifest.kind != "stage/inputs-tracking2":
        raise PipelineError(f"{label} input dependency has the wrong artifact kind")
    _validate_inputs_artifact(input_payload, input_reference, store, config)
    _validate_probe_plan_payload(payload, declared_paths)
    try:
        expected_files, _ = build_probe_bank_files(config, input_payload)
    except (ProbeArtifactError, ValueError) as error:
        raise PipelineError(f"{label} cannot be deterministically replayed") from error
    if declared_paths != set(expected_files) or any(
        store.read_bytes(reference.artifact_id, path) != expected
        for path, expected in expected_files.items()
    ):
        raise PipelineError(f"{label} does not match deterministic replay")


def _validate_mechanical_payload(
    payload: Mapping[str, object],
    config: LabConfig | None,
    reference: ArtifactRef,
    store: ArtifactStore,
) -> None:
    label = "mechanical result"
    _require_fields(
        payload,
        ("geometry", "probe_banks", "protocol", "resnet", "warnings"),
        label=label,
    )
    if config is None:
        raise PipelineError(f"{label} validation requires the signed run configuration")
    upstreams = thaw_json(reference.manifest.metadata.get("upstream_artifacts"))
    if not isinstance(upstreams, dict) or set(upstreams) != {
        "inputs.tracking2",
        "exp1.probe_banks",
    }:
        raise PipelineError(f"{label} must bind its exact input and probe artifacts")
    input_id = upstreams["inputs.tracking2"]
    probe_id = upstreams["exp1.probe_banks"]
    if not isinstance(input_id, str) or not isinstance(probe_id, str):
        raise PipelineError(f"{label} upstream artifact ids are invalid")
    input_reference = store.get(input_id, verify=True)
    if input_reference.manifest.kind != "stage/inputs-tracking2":
        raise PipelineError(f"{label} input dependency has the wrong artifact kind")
    input_payload = _as_object(
        store.read_json(input_id, "inputs.json"), label=f"{label} input dependency"
    )
    _validate_inputs_payload(input_payload, config)
    probe_reference = store.get(probe_id, verify=True)
    if probe_reference.manifest.kind != "stage/exp1-probe-banks":
        raise PipelineError(f"{label} probe dependency has the wrong artifact kind")
    probe_payload = _as_object(
        store.read_json(probe_id, "plan.json"), label=f"{label} probe dependency"
    )
    probe_paths = {entry.path for entry in probe_reference.manifest.files}
    _validate_probe_artifact(
        probe_payload,
        probe_paths,
        probe_reference,
        store,
        config,
    )
    expected_protocol = config.experiment1.mechanical_protocol.model_dump(mode="json")
    if payload["protocol"] != expected_protocol:
        raise PipelineError(f"{label} protocol does not match the signed configuration")
    expected_geometry = replay_toy_geometry(
        config.protocol.root_seed,
        rms_epsilon=config.experiment1.state_metric.rms_epsilon,
    )
    if canonical_hash(payload["geometry"]) != canonical_hash(expected_geometry):
        raise PipelineError(f"{label} geometry does not match deterministic replay")
    geometry = _as_object(payload["geometry"], label=f"{label} geometry")
    _require_fields(
        geometry,
        ("boundary", "centroid_reconstruction", "mixture_gaussian", "roundtrip"),
        label=f"{label} geometry",
    )
    for component, metric in (
        ("roundtrip", "relative_rms_error"),
        ("centroid_reconstruction", "relative_rms_error"),
    ):
        values = _as_object(geometry[component], label=f"{label} {component}")
        _as_nonnegative_number(values.get(metric), label=f"{label} {component}.{metric}")
    for component in ("boundary", "mixture_gaussian"):
        values = _as_object(geometry[component], label=f"{label} {component}")
        if not isinstance(values.get("passed"), bool):
            raise PipelineError(f"{label} {component}.passed must be boolean")
    probes = _as_object(payload["probe_banks"], label=f"{label} probe_banks")
    _require_fields(
        probes,
        ("artifact_valid", "deterministic", "distinct_train_test_sources"),
        label=f"{label} probe_banks",
    )
    if any(probes[field] is not True for field in probes):
        raise PipelineError(f"{label} probe-bank evidence must be independently verified true")
    resnet = _as_object(payload["resnet"], label=f"{label} resnet")
    _require_fields(
        resnet,
        (
            "actual_device",
            "identity_exact",
            "identity_logits_by_cut",
            "identity_max_absolute_error",
            "identity_per_cut",
            "jvp_by_cut",
            "jvp_cuts_completed",
            "jvp_failures",
            "jvp_median_relative_error",
            "jvp_p95_relative_error",
        ),
        label=f"{label} resnet",
    )
    if resnet["actual_device"] != "cpu":
        raise PipelineError(f"{label} ResNet mechanics must record actual_device='cpu'")
    if resnet["identity_exact"] is not None and not isinstance(resnet["identity_exact"], bool):
        raise PipelineError(f"{label} identity_exact must be boolean or null")
    jvp_completed = _as_int(
        resnet["jvp_cuts_completed"], label=f"{label} jvp_cuts_completed", minimum=0
    )
    identity_logits = _as_object(
        resnet["identity_logits_by_cut"], label=f"{label} identity_logits_by_cut"
    )
    identity_by_cut = _as_object(resnet["identity_per_cut"], label=f"{label} identity_per_cut")
    jvp_by_cut = _as_object(resnet["jvp_by_cut"], label=f"{label} jvp_by_cut")
    jvp_failures = _as_object(resnet["jvp_failures"], label=f"{label} jvp_failures")
    jvp_values = (
        resnet["jvp_median_relative_error"],
        resnet["jvp_p95_relative_error"],
    )
    identity_available = resnet["identity_exact"] is not None
    if identity_available:
        if set(identity_logits) != set(config.experiment1.cuts):
            raise PipelineError(f"{label} identity cuts do not match the signed cut axis")
        if set(identity_by_cut) != set(config.experiment1.cuts):
            raise PipelineError(f"{label} identity errors do not match the signed cut axis")
    elif identity_logits or identity_by_cut or resnet["identity_max_absolute_error"] is not None:
        raise PipelineError(f"{label} unavailable identity check must not claim metrics")
    if set(jvp_by_cut) & set(jvp_failures):
        raise PipelineError(f"{label} JVP successes and failures overlap")
    jvp_outcomes = set(jvp_by_cut) | set(jvp_failures)
    if jvp_outcomes and jvp_outcomes != set(config.experiment1.sentinel_cuts):
        raise PipelineError(f"{label} JVP outcomes do not cover the signed sentinel axis")
    if any(not isinstance(reason, str) or not reason for reason in jvp_failures.values()):
        raise PipelineError(f"{label} JVP failures must contain non-empty messages")
    architecture = _as_object(
        input_payload.get("architecture"), label=f"{label} input architecture"
    )
    expected_values = config.experiment1.mechanical_protocol.input_batch_size * _as_int(
        architecture.get("num_classes"),
        label=f"{label} input architecture.num_classes",
        minimum=1,
    )
    identity_evidence: dict[str, Mapping[str, object]] = {}
    for cut_name, raw in identity_logits.items():
        cut = _as_object(raw, label=f"{label} identity logits {cut_name!r}")
        if set(cut) != {"full_logits", "split_logits"}:
            raise PipelineError(f"{label} identity logit evidence fields are invalid")
        for field in ("full_logits", "split_logits"):
            values = _as_array(cut[field], label=f"{label} identity logits {cut_name!r}.{field}")
            if len(values) != expected_values:
                raise PipelineError(f"{label} identity logit vector has the wrong flattened size")
        identity_evidence[cut_name] = cut
    jvp_evidence: dict[str, Mapping[str, object]] = {}
    expected_epsilon = (
        expected_protocol["jvp_epsilon_float64"]
        if config.runtime.dtype == "float64"
        else expected_protocol["jvp_epsilon_float32"]
    )
    for cut_name, raw in jvp_by_cut.items():
        cut = _as_object(raw, label=f"{label} JVP cut {cut_name!r}")
        if set(cut) != {
            "automatic_jvp",
            "epsilon",
            "finite_difference_jvp",
            "relative_error",
        }:
            raise PipelineError(f"{label} JVP evidence fields are invalid")
        if _as_nonnegative_number(cut["epsilon"], label=f"{label} JVP epsilon") != float(
            expected_epsilon
        ):
            raise PipelineError(f"{label} JVP epsilon does not match the signed protocol")
        for field in ("automatic_jvp", "finite_difference_jvp"):
            values = _as_array(cut[field], label=f"{label} JVP {cut_name!r}.{field}")
            if len(values) != expected_values:
                raise PipelineError(f"{label} JVP vector has the wrong flattened size")
        jvp_evidence[cut_name] = cut
    try:
        summary = summarize_resnet_mechanical_evidence(
            identity_evidence,
            jvp_evidence,
            identity_cuts=config.experiment1.cuts,
            jvp_cuts=config.experiment1.sentinel_cuts,
            denominator_floor=config.experiment1.mechanical_protocol.denominator_floor,
        )
    except (TypeError, ValueError) as error:
        raise PipelineError(f"{label} raw ResNet evidence is invalid: {error}") from error
    claimed_identity_errors = {
        cut_name: _as_nonnegative_number(value, label=f"{label} identity error")
        for cut_name, value in identity_by_cut.items()
    }
    if claimed_identity_errors != summary.identity_per_cut:
        raise PipelineError(f"{label} identity errors disagree with saved raw logits")
    if resnet["identity_exact"] is not summary.identity_exact:
        raise PipelineError(f"{label} identity summary disagrees with saved raw logits")
    if summary.identity_max_absolute_error is None:
        if resnet["identity_max_absolute_error"] is not None:
            raise PipelineError(f"{label} identity summary disagrees with saved raw logits")
    elif (
        _as_nonnegative_number(
            resnet["identity_max_absolute_error"],
            label=f"{label} identity_max_absolute_error",
        )
        != summary.identity_max_absolute_error
    ):
        raise PipelineError(f"{label} identity summary disagrees with saved raw logits")
    claimed_jvp_errors = {
        cut_name: _as_nonnegative_number(
            _as_object(raw, label=f"{label} JVP cut {cut_name!r}")["relative_error"],
            label=f"{label} JVP relative error",
        )
        for cut_name, raw in jvp_by_cut.items()
    }
    if claimed_jvp_errors != summary.jvp_relative_error_by_cut:
        raise PipelineError(f"{label} JVP errors disagree with saved raw outputs")
    if jvp_completed != summary.jvp_cuts_completed:
        raise PipelineError(f"{label} JVP completion disagrees with saved raw outputs")
    if summary.jvp_median_relative_error is None:
        if jvp_values != (None, None):
            raise PipelineError(f"{label} JVP aggregates require every sentinel cut")
    elif (
        _as_nonnegative_number(jvp_values[0], label=f"{label} JVP median")
        != summary.jvp_median_relative_error
        or _as_nonnegative_number(jvp_values[1], label=f"{label} JVP p95")
        != summary.jvp_p95_relative_error
    ):
        raise PipelineError(f"{label} JVP aggregates disagree with saved raw outputs")
    warnings = _as_array(payload["warnings"], label=f"{label} warnings")
    if any(not isinstance(warning, str) for warning in warnings):
        raise PipelineError(f"{label} warnings must contain strings")


def _validate_mock_report_payload(
    payload: Mapping[str, object], reference: ArtifactRef, store: ArtifactStore
) -> None:
    try:
        from voronoi_lab.reporting.payload import ReportPayload

        parsed = ReportPayload.model_validate(payload)
    except Exception as error:
        raise PipelineError("mock report payload does not satisfy ReportPayload v1") from error
    if parsed.mode != "mockup":
        raise PipelineError("report.build may publish only a MOCKUP payload")
    files = {entry.path: entry for entry in reference.manifest.files}
    if set(files) != {"report.html", "report_payload.json", "spec.md"}:
        raise PipelineError(
            "mock report artifact must contain exactly HTML, payload JSON, and specification"
        )
    if files["report.html"].media_type != "text/html; charset=utf-8":
        raise PipelineError("mock report HTML has the wrong media type")
    if files["spec.md"].media_type != "text/markdown; charset=utf-8":
        raise PipelineError("mock report specification has the wrong media type")
    try:
        document = store.read_bytes(reference.artifact_id, "report.html").decode("utf-8")
        specification = store.read_bytes(reference.artifact_id, "spec.md").decode("utf-8")
    except UnicodeDecodeError as error:
        raise PipelineError("mock report text payload is not valid UTF-8") from error
    if not document.lstrip().lower().startswith("<!doctype html>") or "<html" not in document:
        raise PipelineError("mock report payload is not an HTML document")
    if "MOCKUP · SCHEMATIC VALUES · NOT EMPIRICAL EVIDENCE" not in document:
        raise PipelineError("mock report HTML is missing its visible MOCKUP warning")
    payload_digest = canonical_hash(parsed.model_dump(mode="json"))
    marker = f'<meta name="voronoi-report-payload-sha256" content="{payload_digest}">'
    if document.count(marker) != 1:
        raise PipelineError("mock report HTML is not bound to its saved payload")
    try:
        from voronoi_lab.reporting.builder import _assert_offline, render_report

        _assert_offline(document)
    except ValueError as error:
        raise PipelineError("mock report HTML is not self-contained") from error
    try:
        expected_document = render_report(parsed, specification)
    except ValueError as error:
        raise PipelineError("mock report cannot be deterministically reconstructed") from error
    if document != expected_document:
        raise PipelineError(
            "mock report HTML does not exactly match its saved payload and specification"
        )


def _artifact_file_map(reference: ArtifactRef) -> dict[str, object]:
    return {entry.path: entry for entry in reference.manifest.files}


def _validate_inventory_file_record(
    raw: object,
    *,
    label: str,
    expected_path: str,
    declared_files: Mapping[str, object],
) -> None:
    record = _as_object(raw, label=label)
    _require_fields(record, ("path", "size_bytes", "sha256"), label=label)
    if record["path"] != expected_path:
        raise PipelineError(f"{label} path does not match its semantic location")
    size = _as_int(record["size_bytes"], label=f"{label} size", minimum=1)
    digest = record["sha256"]
    if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
        raise PipelineError(f"{label} digest is invalid")
    entry = declared_files.get(expected_path)
    if (
        entry is None
        or getattr(entry, "size", None) != size
        or getattr(entry, "sha256", None) != digest
    ):
        raise PipelineError(f"{label} disagrees with the immutable artifact manifest")


def _validate_numeric_npz(
    raw: bytes,
    *,
    label: str,
    expected_inventory: Mapping[str, object] | None = None,
) -> None:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if not archive.files:
                raise PipelineError(f"{label} cannot be empty")
            if expected_inventory is not None and set(archive.files) != set(expected_inventory):
                raise PipelineError(f"{label} array set differs from metadata")
            for name in archive.files:
                value = archive[name]
                if value.dtype.hasobject or value.size == 0:
                    raise PipelineError(f"{label} array {name!r} is empty or object-valued")
                if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
                    raise PipelineError(f"{label} array {name!r} is non-finite")
                if expected_inventory is not None:
                    row = _as_object(expected_inventory[name], label=f"{label} inventory {name!r}")
                    _require_fields(row, ("dtype", "shape"), label=f"{label} inventory")
                    if row["dtype"] != str(value.dtype) or list(value.shape) != list(
                        _as_array(row["shape"], label=f"{label} shape")
                    ):
                        raise PipelineError(f"{label} array {name!r} differs from metadata")
    except PipelineError:
        raise
    except (OSError, ValueError, EOFError) as error:
        raise PipelineError(f"{label} is not a valid pickle-free NPZ payload") from error


def _validate_synthetic_plateau_task_artifact(
    inventory: Mapping[str, object],
    reference: ArtifactRef,
    store: ArtifactStore,
    config: LabConfig | None,
) -> None:
    label = "synthetic plateau task artifact"
    if config is None:
        raise PipelineError(f"{label} validation requires the signed configuration")
    _require_fields(
        inventory,
        (
            "schema_version",
            "task",
            "architecture",
            "dataset_config",
            "model_config",
            "training_config",
            "state_schema",
            "dataset_file",
            "metrics_file",
            "checkpoints",
        ),
        label=label,
    )
    if (
        inventory["task"] != "three_class_2d_gaussian_mixture"
        or inventory["architecture"] != "normalization_free_residual_mlp"
    ):
        raise PipelineError(f"{label} declares an unsupported task or architecture")
    declared = _artifact_file_map(reference)
    _validate_inventory_file_record(
        inventory["dataset_file"],
        label=f"{label} dataset_file",
        expected_path="dataset.npz",
        declared_files=declared,
    )
    _validate_inventory_file_record(
        inventory["metrics_file"],
        label=f"{label} metrics_file",
        expected_path="training_progress.npz",
        declared_files=declared,
    )
    checkpoints = _as_array(inventory["checkpoints"], label=f"{label} checkpoints")
    epochs = tuple(config.experiment1.checkpoints)
    if len(checkpoints) != len(epochs):
        raise PipelineError(f"{label} checkpoint count differs from config")
    expected_paths = {"inventory.json", "dataset.npz", "training_progress.npz"}
    for expected_epoch, raw_row in zip(epochs, checkpoints, strict=True):
        row = _as_object(raw_row, label=f"{label} checkpoint")
        _require_fields(
            row,
            ("epoch", "file", "train_accuracy", "test_accuracy"),
            label=f"{label} checkpoint",
        )
        if row["epoch"] != expected_epoch:
            raise PipelineError(f"{label} checkpoint axis differs from config")
        path = f"checkpoints/epoch_{expected_epoch:05d}.npz"
        _validate_inventory_file_record(
            row["file"],
            label=f"{label} checkpoint {expected_epoch}",
            expected_path=path,
            declared_files=declared,
        )
        expected_paths.add(path)
        _validate_numeric_npz(
            store.read_bytes(reference.artifact_id, path),
            label=f"{label} checkpoint {expected_epoch}",
        )
    if set(declared) != expected_paths:
        raise PipelineError(f"{label} file set differs from its inventory")
    declared_task = config.experiment1.synthetic_plateau_task
    dataset_config = _as_object(inventory["dataset_config"], label=f"{label} dataset config")
    model_config = _as_object(inventory["model_config"], label=f"{label} model config")
    training_config = _as_object(inventory["training_config"], label=f"{label} training config")
    expected_dataset = {
        "seed": SeedDeriver(config.protocol.root_seed, ("exp1", "synthetic_task", "v1")).derive(
            "dataset", bits=32
        ),
        "train_samples_per_class": declared_task.train_samples_per_class,
        "test_samples_per_class": declared_task.test_samples_per_class,
        "radius": declared_task.class_radius,
        "standard_deviation": declared_task.noise_standard_deviation,
    }
    expected_model = {
        "input_dim": declared_task.input_dimensions,
        "width": declared_task.hidden_width,
        "blocks": declared_task.residual_blocks,
        "classes": declared_task.classes,
    }
    expected_training = {
        "seed": SeedDeriver(config.protocol.root_seed, ("exp1", "synthetic_task", "v1")).derive(
            "training", bits=32
        ),
        "epochs": declared_task.epochs,
        "checkpoint_epochs": list(epochs),
        "batch_size": declared_task.batch_size,
        "learning_rate": declared_task.learning_rate,
        "momentum": declared_task.momentum,
        "weight_decay": declared_task.weight_decay,
    }
    if any(
        canonical_hash(observed) != canonical_hash(expected)
        for observed, expected in (
            (dataset_config, expected_dataset),
            (model_config, expected_model),
            (training_config, expected_training),
        )
    ):
        raise PipelineError(f"{label} configuration differs from the signed protocol")
    _validate_numeric_npz(
        store.read_bytes(reference.artifact_id, "dataset.npz"), label=f"{label} dataset"
    )
    _validate_numeric_npz(
        store.read_bytes(reference.artifact_id, "training_progress.npz"),
        label=f"{label} training progress",
    )


def _validate_plateau_checkpoint_plane_contract(
    raw_arrays: bytes,
    metadata: Mapping[str, object],
    *,
    residual_identity: bool,
    label: str,
) -> None:
    """Validate the v2 raw/adjusted plane fields and their display selection."""

    expected_display = {
        "local_array": (
            _RESIDUAL_LOCAL_PLANE_JACOBIAN if residual_identity else _RAW_LOCAL_PLANE_JACOBIAN
        ),
        "anchor_array": (
            _RESIDUAL_ANCHOR_PLANE_JACOBIAN if residual_identity else _RAW_ANCHOR_PLANE_JACOBIAN
        ),
        "estimand": _RESIDUAL_PLANE_ESTIMAND if residual_identity else _RAW_PLANE_ESTIMAND,
        "selection": (
            "residual_update_for_residual_transition"
            if residual_identity
            else "raw_transition_for_nonresidual_transition"
        ),
    }
    if metadata.get("residual_identity") is not residual_identity:
        raise PipelineError(f"{label} residual-identity declaration is inconsistent")
    display = _as_object(metadata.get("plane_jacobian_display"), label=f"{label} display")
    if canonical_hash(display) != canonical_hash(expected_display):
        raise PipelineError(f"{label} plane Jacobian display selection is inconsistent")

    try:
        with np.load(io.BytesIO(raw_arrays), allow_pickle=False) as archive:
            names = set(archive.files)
            common_required = {
                _RAW_LOCAL_PLANE_JACOBIAN,
                _RAW_ANCHOR_PLANE_JACOBIAN,
                "local_axis",
                "local_transition_sites",
                "anchor_axis",
                "anchor_transition_sites_by_context",
                "anchor_vectors",
            }
            if not common_required.issubset(names):
                raise PipelineError(f"{label} cannot replay both raw transition plane fields")
            local_raw = np.asarray(archive[_RAW_LOCAL_PLANE_JACOBIAN]).copy()
            anchor_raw = np.asarray(archive[_RAW_ANCHOR_PLANE_JACOBIAN]).copy()
            local_axis = np.asarray(archive["local_axis"]).copy()
            local_transition = np.asarray(archive["local_transition_sites"]).copy()
            anchor_axis = np.asarray(archive["anchor_axis"]).copy()
            anchor_transition = np.asarray(archive["anchor_transition_sites_by_context"]).copy()
            anchor_vectors = np.asarray(archive["anchor_vectors"]).copy()
            if (
                local_raw.ndim != 4
                or local_raw.shape[0] != 2
                or anchor_raw.ndim != 3
                or anchor_raw.shape[0] != 3
            ):
                raise PipelineError(f"{label} raw plane Jacobian fields have invalid shapes")
            residual_names = {
                _RESIDUAL_LOCAL_PLANE_JACOBIAN,
                _RESIDUAL_ANCHOR_PLANE_JACOBIAN,
            }
            if not residual_identity:
                if names & residual_names:
                    raise PipelineError(
                        f"{label} non-residual transition must not declare residual-update fields"
                    )
                local_adjusted = None
                anchor_adjusted = None
                local_grid = None
                anchor_grid = None
            else:
                residual_required = {
                    *residual_names,
                    "local_grid_vectors",
                    "anchor_grid_vectors",
                }
                if not residual_required.issubset(names):
                    raise PipelineError(
                        f"{label} residual transition is missing replayable adjusted fields"
                    )
                local_adjusted = np.asarray(archive[_RESIDUAL_LOCAL_PLANE_JACOBIAN]).copy()
                anchor_adjusted = np.asarray(archive[_RESIDUAL_ANCHOR_PLANE_JACOBIAN]).copy()
                local_grid = np.asarray(archive["local_grid_vectors"]).copy()
                anchor_grid = np.asarray(archive["anchor_grid_vectors"]).copy()
                if (
                    local_adjusted.shape != local_raw.shape
                    or anchor_adjusted.shape != anchor_raw.shape
                ):
                    raise PipelineError(
                        f"{label} raw and residual-adjusted plane fields must align"
                    )
    except PipelineError:
        raise
    except (OSError, ValueError, EOFError, TypeError) as error:
        raise PipelineError(f"{label} arrays are not a valid pickle-free NPZ") from error

    from voronoi_lab.exp1.surface_geometry import (
        ThreeAnchorSlice,
        plane_pullback_jacobian_frobenius,
    )

    robust_scale = metadata.get("robust_activation_scale")
    if isinstance(robust_scale, bool) or not isinstance(robust_scale, (int, float)):
        raise PipelineError(f"{label} robust activation scale is invalid")
    try:
        expected_local_raw = np.empty_like(local_raw)
        for kind in range(local_raw.shape[0]):
            for center in range(local_raw.shape[1]):
                expected_local_raw[kind, center] = plane_pullback_jacobian_frobenius(
                    local_transition[kind, center],
                    local_axis,
                    local_axis,
                    first_scale=float(robust_scale),
                    second_scale=float(robust_scale),
                )
        anchor_plane = ThreeAnchorSlice.from_anchors(*anchor_vectors)
        expected_anchor_raw = np.empty_like(anchor_raw)
        for context in range(anchor_raw.shape[0]):
            expected_anchor_raw[context] = plane_pullback_jacobian_frobenius(
                anchor_transition[:, :, context],
                anchor_axis,
                anchor_axis,
                first_scale=anchor_plane.alpha_scale,
                second_scale=anchor_plane.beta_scale,
            )
    except (TypeError, ValueError) as error:
        raise PipelineError(f"{label} raw transition plane replay failed") from error
    if not np.allclose(local_raw, expected_local_raw, rtol=1e-5, atol=2e-6):
        raise PipelineError(f"{label} local raw transition plane field does not replay")
    if not np.allclose(anchor_raw, expected_anchor_raw, rtol=1e-5, atol=2e-6):
        raise PipelineError(f"{label} anchor raw transition plane field does not replay")
    if not residual_identity:
        return

    assert local_adjusted is not None
    assert anchor_adjusted is not None
    assert local_grid is not None
    assert anchor_grid is not None
    expected_local_adjusted = np.empty_like(local_adjusted)
    for kind in range(local_adjusted.shape[0]):
        for center in range(local_adjusted.shape[1]):
            expected_local_adjusted[kind, center] = plane_pullback_jacobian_frobenius(
                local_transition[kind, center] - local_grid[kind, center],
                local_axis,
                local_axis,
                first_scale=float(robust_scale),
                second_scale=float(robust_scale),
            )
    try:
        expected_anchor_adjusted = np.empty_like(anchor_adjusted)
        for context in range(anchor_adjusted.shape[0]):
            expected_anchor_adjusted[context] = plane_pullback_jacobian_frobenius(
                anchor_transition[:, :, context] - anchor_grid,
                anchor_axis,
                anchor_axis,
                first_scale=anchor_plane.alpha_scale,
                second_scale=anchor_plane.beta_scale,
            )
    except (TypeError, ValueError) as error:
        raise PipelineError(f"{label} residual-adjusted plane replay failed") from error
    if not np.allclose(local_adjusted, expected_local_adjusted, rtol=1e-5, atol=2e-6):
        raise PipelineError(f"{label} local residual-adjusted plane field does not replay")
    if not np.allclose(
        anchor_adjusted,
        expected_anchor_adjusted,
        rtol=1e-5,
        atol=2e-6,
    ):
        raise PipelineError(f"{label} anchor residual-adjusted plane field does not replay")


def _validate_plateau_collection_artifact(
    summary: Mapping[str, object],
    reference: ArtifactRef,
    store: ArtifactStore,
    config: LabConfig | None,
) -> None:
    label = "plateau collection artifact"
    if config is None:
        raise PipelineError(f"{label} validation requires the signed configuration")
    _require_fields(
        summary,
        (
            "task",
            "architecture",
            "checkpoint_rows",
            "checkpoint_epochs",
            "source_separation",
            "task_artifact_id",
            "jacobian_display_contract",
        ),
        label=label,
    )
    if summary["checkpoint_epochs"] != list(config.experiment1.checkpoints):
        raise PipelineError(f"{label} checkpoint axis differs from config")
    expected_display = {
        "protocol_version": config.experiment1.plateau_protocol.protocol_version,
        "local_source_array": _RESIDUAL_LOCAL_PLANE_JACOBIAN,
        "anchor_source_array": _RESIDUAL_ANCHOR_PLANE_JACOBIAN,
        "estimand": _RESIDUAL_PLANE_ESTIMAND,
    }
    if canonical_hash(summary["jacobian_display_contract"]) != canonical_hash(expected_display):
        raise PipelineError(f"{label} Jacobian display contract differs from protocol v2")
    rows = _as_array(summary["checkpoint_rows"], label=f"{label} checkpoint rows")
    if len(rows) != len(config.experiment1.checkpoints):
        raise PipelineError(f"{label} checkpoint row count differs from config")
    declared = _artifact_file_map(reference)
    expected_paths = {"summary.json"}
    for epoch, raw_row in zip(config.experiment1.checkpoints, rows, strict=True):
        row = _as_object(raw_row, label=f"{label} checkpoint row")
        _require_fields(row, ("epoch", "arrays", "metadata", "inventory"), label=label)
        prefix = f"checkpoints/epoch_{epoch:05d}"
        expected = {
            "epoch": epoch,
            "arrays": f"{prefix}/arrays.npz",
            "metadata": f"{prefix}/metadata.json",
            "inventory": f"{prefix}/inventory.json",
        }
        if canonical_hash(row) != canonical_hash(expected):
            raise PipelineError(f"{label} checkpoint row has inconsistent paths")
        expected_paths.update((expected["arrays"], expected["metadata"], expected["inventory"]))
        metadata = store.read_json(reference.artifact_id, expected["metadata"])
        inventory = store.read_json(reference.artifact_id, expected["inventory"])
        if not isinstance(metadata, Mapping) or not isinstance(inventory, Mapping):
            raise PipelineError(f"{label} child metadata/inventory must be objects")
        if (
            metadata.get("schema_version") != _PLATEAU_SCHEMA_VERSION
            or metadata.get("epoch") != epoch
        ):
            raise PipelineError(f"{label} child metadata has an invalid epoch or schema")
        array_inventory = _as_object(
            metadata.get("array_inventory"), label=f"{label} array inventory"
        )
        raw_arrays = store.read_bytes(reference.artifact_id, expected["arrays"])
        raw_metadata = store.read_bytes(reference.artifact_id, expected["metadata"])
        inventory_files = _as_object(inventory.get("files"), label=f"{label} byte inventory")
        for name, data in (("arrays.npz", raw_arrays), ("metadata.json", raw_metadata)):
            item = _as_object(inventory_files.get(name), label=f"{label} {name}")
            _require_fields(item, ("sha256", "size_bytes"), label=f"{label} {name}")
            if item["sha256"] != sha256_bytes(data) or item["size_bytes"] != len(data):
                raise PipelineError(f"{label} {name} byte inventory is inconsistent")
        _validate_numeric_npz(
            raw_arrays,
            label=f"{label} epoch {epoch} arrays",
            expected_inventory=array_inventory,
        )
        _validate_plateau_checkpoint_plane_contract(
            raw_arrays,
            metadata,
            residual_identity=True,
            label=f"{label} epoch {epoch}",
        )
    if set(declared) != expected_paths:
        raise PipelineError(f"{label} file set differs from checkpoint rows")
    upstreams = reference.manifest.metadata.get("upstream_artifacts")
    if not isinstance(upstreams, Mapping) or summary["task_artifact_id"] != upstreams.get(
        "exp1.synthetic_task"
    ):
        raise PipelineError(f"{label} is not bound to its synthetic task artifact")


def _validate_plateau_checkpoint_child(
    child: ArtifactRef,
    store: ArtifactStore,
    *,
    architecture: str,
    epoch: int,
) -> None:
    label = f"CIFAR plateau {architecture} epoch {epoch} shard"
    if child.manifest.kind != "shard/exp1-plateau-cifar-checkpoint":
        raise PipelineError(f"{label} has the wrong artifact kind")
    files = {entry.path: entry for entry in child.manifest.files}
    if set(files) != {"arrays.npz", "metadata.json", "inventory.json"}:
        raise PipelineError(f"{label} must contain arrays, metadata, and byte inventory")
    if files["arrays.npz"].media_type != "application/x-npz" or any(
        files[name].media_type != "application/json" for name in ("metadata.json", "inventory.json")
    ):
        raise PipelineError(f"{label} has invalid payload media types")
    metadata = store.read_json(child.artifact_id, "metadata.json")
    inventory = store.read_json(child.artifact_id, "inventory.json")
    if not isinstance(metadata, Mapping) or not isinstance(inventory, Mapping):
        raise PipelineError(f"{label} metadata and inventory must be objects")
    expected_architecture = {
        "resnet": "tracking2_resnet18_v2_width64",
        "vgg": "tracking2_vgg19_bn_width1_classifier512",
    }[architecture]
    if (
        metadata.get("schema_version") != _PLATEAU_SCHEMA_VERSION
        or metadata.get("epoch") != epoch
        or metadata.get("architecture") != expected_architecture
    ):
        raise PipelineError(f"{label} metadata identity is inconsistent")
    if (
        child.manifest.metadata.get("architecture") != architecture
        or child.manifest.metadata.get("epoch") != epoch
        or child.manifest.metadata.get("result_schema_version") != _PLATEAU_SCHEMA_VERSION
    ):
        raise PipelineError(f"{label} immutable shard coordinates are inconsistent")
    arrays = store.read_bytes(child.artifact_id, "arrays.npz")
    metadata_bytes = store.read_bytes(child.artifact_id, "metadata.json")
    inventory_files = _as_object(inventory.get("files"), label=f"{label} byte inventory")
    for name, data in (("arrays.npz", arrays), ("metadata.json", metadata_bytes)):
        row = _as_object(inventory_files.get(name), label=f"{label} {name}")
        _require_fields(row, ("sha256", "size_bytes"), label=f"{label} {name}")
        if row["sha256"] != sha256_bytes(data) or row["size_bytes"] != len(data):
            raise PipelineError(f"{label} {name} byte inventory is inconsistent")
    array_inventory = _as_object(metadata.get("array_inventory"), label=f"{label} array inventory")
    _validate_numeric_npz(
        arrays,
        label=f"{label} arrays",
        expected_inventory=array_inventory,
    )
    _validate_plateau_checkpoint_plane_contract(
        arrays,
        metadata,
        residual_identity=architecture == "resnet",
        label=label,
    )


def _validate_plateau_cifar_artifact(
    summary: Mapping[str, object],
    reference: ArtifactRef,
    store: ArtifactStore,
    config: LabConfig | None,
) -> None:
    label = "CIFAR plateau collection artifact"
    if config is None:
        raise PipelineError(f"{label} validation requires the signed configuration")
    _require_fields(
        summary,
        (
            "task",
            "architectures",
            "checkpoint_epochs",
            "checkpoint_rows",
            "reducer_artifact_id",
            "train_bank",
            "test_bank",
            "lineage_scope",
            "confounds",
            "source_separation",
            "jacobian_display_contract",
        ),
        label=label,
    )
    if summary["task"] != "cifar10" or summary["architectures"] != ["resnet", "vgg"]:
        raise PipelineError(f"{label} task or architecture axis is invalid")
    if summary["checkpoint_epochs"] != list(config.experiment1.checkpoints):
        raise PipelineError(f"{label} checkpoint axis differs from config")
    expected_display = {
        "protocol_version": config.experiment1.plateau_protocol.protocol_version,
        "resnet": {
            "local_source_array": _RESIDUAL_LOCAL_PLANE_JACOBIAN,
            "anchor_source_array": _RESIDUAL_ANCHOR_PLANE_JACOBIAN,
            "estimand": _RESIDUAL_PLANE_ESTIMAND,
        },
        "vgg": {
            "local_source_array": _RAW_LOCAL_PLANE_JACOBIAN,
            "anchor_source_array": _RAW_ANCHOR_PLANE_JACOBIAN,
            "estimand": _RAW_PLANE_ESTIMAND,
        },
        "comparison_scope": "descriptive_confounded_side_by_side_with_row_specific_operators",
    }
    if canonical_hash(summary["jacobian_display_contract"]) != canonical_hash(expected_display):
        raise PipelineError(f"{label} Jacobian display contract differs from protocol v2")
    if set(_artifact_file_map(reference)) != {"summary.json"}:
        raise PipelineError(f"{label} parent must contain only its reducer summary")
    rows = _as_array(summary["checkpoint_rows"], label=f"{label} checkpoint rows")
    expected_coordinates = tuple(
        (architecture, epoch)
        for architecture in ("resnet", "vgg")
        for epoch in config.experiment1.checkpoints
    )
    if len(rows) != len(expected_coordinates):
        raise PipelineError(f"{label} checkpoint row count differs from config")
    child_ids: list[str] = []
    for (architecture, epoch), raw_row in zip(expected_coordinates, rows, strict=True):
        row = _as_object(raw_row, label=f"{label} checkpoint row")
        _require_fields(
            row,
            ("architecture", "epoch", "artifact_id", "arrays", "metadata", "inventory"),
            label=f"{label} checkpoint row",
        )
        if (
            row["architecture"] != architecture
            or row["epoch"] != epoch
            or row["arrays"] != "arrays.npz"
            or row["metadata"] != "metadata.json"
            or row["inventory"] != "inventory.json"
        ):
            raise PipelineError(f"{label} checkpoint row coordinates are inconsistent")
        artifact_id = row["artifact_id"]
        if not isinstance(artifact_id, str) or not _DIGEST_RE.fullmatch(artifact_id):
            raise PipelineError(f"{label} checkpoint artifact id is invalid")
        child = store.get(artifact_id, verify=True)
        _validate_plateau_checkpoint_child(
            child,
            store,
            architecture=architecture,
            epoch=epoch,
        )
        child_ids.append(artifact_id)
    if len(set(child_ids)) != len(child_ids):
        raise PipelineError(f"{label} checkpoint artifacts must be unique")
    reducer_id = summary["reducer_artifact_id"]
    if not isinstance(reducer_id, str) or not _DIGEST_RE.fullmatch(reducer_id):
        raise PipelineError(f"{label} reducer artifact id is invalid")
    reducer = store.get(reducer_id, verify=True)
    if reducer.manifest.kind != "shards/reducer-manifest":
        raise PipelineError(f"{label} reducer artifact has the wrong kind")
    for bank_name, split in (("train_bank", "train"), ("test_bank", "test")):
        bank = _as_object(summary[bank_name], label=f"{label} {bank_name}")
        required = ["bank_id", "image_ids", "tensor_sha256", "source_sha256", "recipe"]
        if split == "test":
            required.append("labels")
        _require_fields(bank, required, label=f"{label} {bank_name}")
        for digest_name in ("bank_id", "tensor_sha256", "source_sha256"):
            digest = bank[digest_name]
            if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
                raise PipelineError(f"{label} {bank_name} {digest_name} is invalid")
        image_ids = _as_array(bank["image_ids"], label=f"{label} {bank_name} image ids")
        if not image_ids or len(set(image_ids)) != len(image_ids):
            raise PipelineError(f"{label} {bank_name} image ids must be nonempty and unique")
        if split == "test" and len(_as_array(bank["labels"], label=f"{label} test labels")) != len(
            image_ids
        ):
            raise PipelineError(f"{label} test labels must align with image ids")
    confounds = _as_array(summary["confounds"], label=f"{label} confounds")
    if not confounds or any(not isinstance(item, str) or not item for item in confounds):
        raise PipelineError(f"{label} must preserve its exploratory comparison caveats")


def _animation_npz_member(
    store: ArtifactStore,
    artifact_id: str,
    path: str,
    member: str,
    *,
    label: str,
) -> np.ndarray:
    """Read one finite numeric animation source without enabling pickle."""

    try:
        raw = store.read_bytes(artifact_id, path)
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if member not in archive.files:
                raise PipelineError(f"{label} is missing NPZ member {member!r}")
            array = np.asarray(archive[member]).copy()
    except PipelineError:
        raise
    except (OSError, ValueError, EOFError, TypeError) as error:
        raise PipelineError(f"{label} is not a valid pickle-free NPZ") from error
    if array.dtype.kind not in "biuf" or array.size == 0 or not np.all(np.isfinite(array)):
        raise PipelineError(f"{label} member {member!r} must be finite and numeric")
    return array


def _expected_computed_scale_metadata(
    data_range: tuple[float, float],
    *,
    padded: bool = False,
    include_zero: bool = False,
) -> dict[str, float | str | bool]:
    """Replay the renderer's no-override global scale contract exactly."""

    data_min, data_max = data_range
    if padded:
        scale_min = min(data_min, 0.0) if include_zero else data_min
        scale_max = max(data_max, 0.0) if include_zero else data_max
        span = scale_max - scale_min
        if span <= 0.0:
            span = max(abs(scale_min), 1.0)
        padding = 0.05 * span
        display_min = scale_min - padding
        display_max = scale_max + padding
        source = "computed_global_with_padding"
    elif data_min < data_max:
        display_min, display_max = data_min, data_max
        source = "computed_global"
    else:
        padding = max(abs(data_min) * 0.05, 1.0)
        display_min, display_max = data_min - padding, data_max + padding
        source = "computed_global_constant_padding"
    return {
        "display_min": display_min,
        "display_max": display_max,
        "data_min": data_min,
        "data_max": data_max,
        "source": source,
        "clips_data": display_min > data_min or display_max < data_max,
    }


def _validate_animation_image_bundle(
    reference: ArtifactRef,
    store: ArtifactStore,
    row: Mapping[str, object],
    *,
    checkpoints: tuple[int, ...],
    expected_name: str,
    expected_kind: str,
    expected_task: str,
    expected_architectures: tuple[str, ...],
    timing: tuple[int, int, int],
    expected_bundle_contract: Mapping[str, object],
    expected_scalar_range: tuple[float, float],
    expected_x_coordinates: np.ndarray,
    expected_y_coordinates: np.ndarray,
    expected_curve_x_range: tuple[float, float] | None = None,
    expected_curve_y_range: tuple[float, float] | None = None,
    architecture_maxima: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Validate image bytes, decoded timing, and fixed-scale renderer metadata."""

    from PIL import Image, ImageSequence

    label = f"plateau animation bundle {expected_name}"
    _require_fields(
        row,
        ("name", "task", "architectures", "animation_kind", "gif", "final_png", "metadata"),
        label=label,
    )
    if row["name"] != expected_name or row["animation_kind"] != expected_kind:
        raise PipelineError(f"{label} identity is inconsistent")
    if row["task"] != expected_task or row["architectures"] != list(expected_architectures):
        raise PipelineError(f"{label} task or architecture identity is inconsistent")
    for key, expected in expected_bundle_contract.items():
        if canonical_hash(row.get(key)) != canonical_hash(expected):
            raise PipelineError(f"{label} {key} differs from its source-field contract")
    expected_paths = {
        "gif": f"animations/{expected_name}.gif",
        "final_png": f"animations/{expected_name}_final.png",
        "metadata": f"animations/{expected_name}_metadata.json",
    }
    if any(row[name] != path for name, path in expected_paths.items()):
        raise PipelineError(f"{label} paths are inconsistent")
    declared = _artifact_file_map(reference)
    expected_media = {
        "gif": "image/gif",
        "final_png": "image/png",
        "metadata": "application/json",
    }
    for name, path in expected_paths.items():
        entry = declared.get(path)
        if entry is None or entry.media_type != expected_media[name]:
            raise PipelineError(f"{label} has an invalid {name} payload declaration")

    metadata_value = store.read_json(reference.artifact_id, expected_paths["metadata"])
    metadata = _as_object(metadata_value, label=f"{label} metadata")
    _require_fields(
        metadata,
        (
            "schema_version",
            "animation_kind",
            "canvas",
            "input_checkpoints",
            "rendered_frame_count",
            "rendered_frames",
            "timing",
            "scales",
            "labels",
            "files",
            "gif_badge",
            "layout_fixed_across_frames",
            "checkpoint_panels_synchronized",
        ),
        label=f"{label} metadata",
    )
    checkpoint_labels = [f"epoch {epoch}" for epoch in checkpoints]
    if (
        metadata["schema_version"] != _PLATEAU_SCHEMA_VERSION
        or metadata["animation_kind"] != expected_kind
        or metadata["input_checkpoints"] != checkpoint_labels
        or metadata["rendered_frame_count"] != len(checkpoints) + 1
    ):
        raise PipelineError(f"{label} metadata identity or checkpoint axis is inconsistent")
    if any(
        metadata[name] is not True
        for name in (
            "gif_badge",
            "layout_fixed_across_frames",
            "checkpoint_panels_synchronized",
        )
    ):
        raise PipelineError(f"{label} must preserve the fixed synchronized GIF contract")
    canvas = _as_object(metadata["canvas"], label=f"{label} canvas")
    if canvas != {"width": 1440, "height": 900}:
        raise PipelineError(f"{label} canvas differs from the declared fixed layout")
    files = _as_object(metadata["files"], label=f"{label} files")
    if files != {
        "gif": PurePosixPath(expected_paths["gif"]).name,
        "final_png": PurePosixPath(expected_paths["final_png"]).name,
        "metadata": PurePosixPath(expected_paths["metadata"]).name,
    }:
        raise PipelineError(f"{label} renderer file metadata is inconsistent")
    orientation, checkpoint, conclusion = timing
    expected_durations = [orientation] + [checkpoint] * (len(checkpoints) - 1) + [conclusion]
    expected_roles = ["orientation"] + ["checkpoint"] * (len(checkpoints) - 1) + ["conclusion"]
    rendered = _as_array(metadata["rendered_frames"], label=f"{label} rendered frames")
    if len(rendered) != len(expected_durations):
        raise PipelineError(f"{label} has an invalid rendered frame count")
    for index, (raw_frame, role, duration) in enumerate(
        zip(rendered, expected_roles, expected_durations, strict=True)
    ):
        frame = _as_object(raw_frame, label=f"{label} rendered frame")
        source_index = min(index, len(checkpoints) - 1)
        if frame != {
            "index": index,
            "checkpoint": checkpoint_labels[source_index],
            "role": role,
            "duration_ms": duration,
        }:
            raise PipelineError(f"{label} rendered schedule is inconsistent")
    timing_metadata = _as_object(metadata["timing"], label=f"{label} timing")
    if timing_metadata != {
        "orientation_ms": orientation,
        "checkpoint_ms": checkpoint,
        "conclusion_ms": conclusion,
        "loop": 0,
    }:
        raise PipelineError(f"{label} timing metadata differs from config")
    scales = _as_object(metadata["scales"], label=f"{label} scales")
    labels = _as_object(metadata["labels"], label=f"{label} labels")
    x_coordinates = np.asarray(expected_x_coordinates, dtype=np.float64)
    y_coordinates = np.asarray(expected_y_coordinates, dtype=np.float64)
    if x_coordinates.ndim != 1 or y_coordinates.ndim != 1:
        raise PipelineError(f"{label} source coordinate axes are invalid")
    extent = [
        float(x_coordinates[0]),
        float(x_coordinates[-1]),
        float(y_coordinates[0]),
        float(y_coordinates[-1]),
    ]
    if expected_kind == "real_fake_scalar_fields":
        if expected_curve_x_range is None or expected_curve_y_range is None:
            raise PipelineError(f"{label} is missing source-derived curve ranges")
        expected_scales = {
            "heatmap": _expected_computed_scale_metadata(expected_scalar_range),
            "curve_x": _expected_computed_scale_metadata(
                expected_curve_x_range,
                padded=True,
            ),
            "curve_y": _expected_computed_scale_metadata(
                expected_curve_y_range,
                padded=True,
                include_zero=True,
            ),
            "heatmap_extent": extent,
            "heatmap_x": x_coordinates.tolist(),
            "heatmap_y": y_coordinates.tolist(),
            "array_coordinate_contract": "row_zero_is_minimum_y",
        }
        if canonical_hash(scales) != canonical_hash(expected_scales):
            raise PipelineError(
                f"{label} scales or heatmap coordinates are not exactly source-derived"
            )
        expected_estimands = {
            "heatmap": expected_bundle_contract["scalar_estimand"],
            "curve": expected_bundle_contract["curve_estimand"],
        }
        if canonical_hash(metadata.get("estimands")) != canonical_hash(expected_estimands):
            raise PipelineError(f"{label} renderer estimands are inconsistent")
        expected_label = (
            "2D ‖D(T-I)‖F"
            if expected_bundle_contract["scalar_estimand"] == _RESIDUAL_PLANE_ESTIMAND
            else "2D ‖DT‖F"
        )
        if (
            labels.get("scalar") != expected_label
            or labels.get("heatmap_classification") != "NEW HYBRID"
        ):
            raise PipelineError(f"{label} visible scalar label is inconsistent")
    else:
        assert architecture_maxima is not None
        expected_scales = {
            "rgb_cells": {
                "display_min": 0.0,
                "display_max": 1.0,
                "source": "caller_prepared_with_declared_global_channel_maxima",
                "normalization_scope": "fixed_across_all_checkpoints_within_architecture",
                "comparability_contract": (
                    "per_architecture_channel_normalization_not_absolute_cross_architecture_scale"
                ),
                "resnet_channel_maxima": architecture_maxima[0].astype(np.float64).tolist(),
                "vgg_channel_maxima": architecture_maxima[1].astype(np.float64).tolist(),
            },
            "jacobian": _expected_computed_scale_metadata(expected_scalar_range),
            "plane_extent": extent,
            "alpha_coordinates": x_coordinates.tolist(),
            "beta_coordinates": y_coordinates.tolist(),
            "array_coordinate_contract": "row_zero_is_minimum_y",
        }
        if canonical_hash(scales) != canonical_hash(expected_scales):
            raise PipelineError(
                f"{label} scales or plane coordinates are not exactly source-derived"
            )
        expected_estimands = expected_bundle_contract["jacobian_estimands"]
        if canonical_hash(metadata.get("jacobian_estimands")) != canonical_hash(expected_estimands):
            raise PipelineError(f"{label} renderer Jacobian estimands are inconsistent")
        if (
            labels.get("jacobian") != "2D ‖·‖F"
            or labels.get("rgb") != "SOURCE ANALOGUE · THREE-ANCHOR RGB"
            or labels.get("resnet_jacobian") != "NEW HYBRID · D(T-I)"
            or labels.get("vgg_jacobian") != "NEW HYBRID · DT"
            or metadata.get("rgb_estimand")
            != (
                "three frozen-context downstream-logit L2 distances encoded as "
                "per-architecture/channel-normalized RGB; not discovered or stable cells"
            )
            or metadata.get("comparison_note") != _ARCHITECTURE_COMPARISON_NOTE
        ):
            raise PipelineError(f"{label} row-specific labels or comparison note are invalid")

    gif_bytes = store.read_bytes(reference.artifact_id, expected_paths["gif"])
    try:
        with Image.open(io.BytesIO(gif_bytes)) as gif:
            decoded_durations = [
                int(frame.info.get("duration", 0)) for frame in ImageSequence.Iterator(gif)
            ]
            decoded_sizes = {frame.size for frame in ImageSequence.Iterator(gif)}
            if (
                gif.format != "GIF"
                or gif.n_frames != len(expected_durations)
                or gif.size != (1440, 900)
                or decoded_durations != expected_durations
                or decoded_sizes != {(1440, 900)}
                or int(gif.info.get("loop", -1)) != 0
            ):
                raise PipelineError(f"{label} GIF readback differs from its metadata")
        with Image.open(
            io.BytesIO(store.read_bytes(reference.artifact_id, expected_paths["final_png"]))
        ) as image:
            if image.format != "PNG" or image.size != (1440, 900):
                raise PipelineError(f"{label} final PNG has an invalid format or canvas")
    except PipelineError:
        raise
    except (OSError, ValueError, EOFError) as error:
        raise PipelineError(f"{label} contains an unreadable image") from error


def _validate_plateau_animation_artifact(
    summary: Mapping[str, object],
    reference: ArtifactRef,
    store: ArtifactStore,
    config: LabConfig | None,
) -> None:
    """Validate source binding, global normalization, and decoded GIF contracts."""

    label = "plateau animation artifact"
    if config is None:
        raise PipelineError(f"{label} validation requires the signed configuration")
    _require_fields(
        summary,
        (
            "task",
            "checkpoint_epochs",
            "source_artifacts",
            "timing_ms",
            "bundles",
            "normalization",
            "jacobian_display_contract",
            "source_separation",
            "nonclaims",
        ),
        label=label,
    )
    checkpoints = tuple(config.experiment1.checkpoints)
    if summary["task"] != "experiment1_activation_geometry_animations" or summary[
        "checkpoint_epochs"
    ] != list(checkpoints):
        raise PipelineError(f"{label} task or checkpoint axis is inconsistent")
    upstreams = thaw_json(reference.manifest.metadata.get("upstream_artifacts"))
    if not isinstance(upstreams, dict) or set(upstreams) != {
        "exp1.plateau.synthetic",
        "exp1.plateau.cifar",
    }:
        raise PipelineError(f"{label} must bind exactly its two collection artifacts")
    sources = _as_object(summary["source_artifacts"], label=f"{label} source artifacts")
    if sources != upstreams:
        raise PipelineError(f"{label} source summary differs from immutable upstream binding")

    synthetic_id = upstreams["exp1.plateau.synthetic"]
    if not isinstance(synthetic_id, str) or not _DIGEST_RE.fullmatch(synthetic_id):
        raise PipelineError(f"{label} synthetic source artifact id is invalid")
    synthetic_reference = store.get(synthetic_id, verify=True)
    if synthetic_reference.manifest.kind != "stage/exp1-plateau-synthetic":
        raise PipelineError(f"{label} synthetic source has the wrong artifact kind")
    synthetic_value = store.read_json(synthetic_id, "summary.json")
    synthetic_summary = _as_object(synthetic_value, label=f"{label} synthetic source summary")
    if synthetic_summary.get("schema_version") != _PLATEAU_SCHEMA_VERSION or synthetic_summary.get(
        "checkpoint_epochs"
    ) != list(checkpoints):
        raise PipelineError(f"{label} synthetic source checkpoint axis is inconsistent")
    expected_synthetic_display = {
        "protocol_version": config.experiment1.plateau_protocol.protocol_version,
        "local_source_array": _RESIDUAL_LOCAL_PLANE_JACOBIAN,
        "anchor_source_array": _RESIDUAL_ANCHOR_PLANE_JACOBIAN,
        "estimand": _RESIDUAL_PLANE_ESTIMAND,
    }
    if canonical_hash(synthetic_summary.get("jacobian_display_contract")) != canonical_hash(
        expected_synthetic_display
    ):
        raise PipelineError(f"{label} synthetic source display contract is inconsistent")
    scalar_ranges: dict[str, list[float]] = {
        "synthetic_real_fake": [float("inf"), float("-inf")],
        "cifar_resnet_real_fake": [float("inf"), float("-inf")],
        "cifar_vgg_real_fake": [float("inf"), float("-inf")],
        "cifar_architecture_cells": [float("inf"), float("-inf")],
    }
    curve_x_ranges: dict[str, list[float]] = {
        name: [float("inf"), float("-inf")]
        for name in (
            "synthetic_real_fake",
            "cifar_resnet_real_fake",
            "cifar_vgg_real_fake",
        )
    }
    curve_y_ranges = {name: list(bounds) for name, bounds in curve_x_ranges.items()}
    coordinate_axes: dict[str, np.ndarray] = {}

    def update_range(name: str, field: np.ndarray) -> None:
        bounds = scalar_ranges[name]
        bounds[0] = min(bounds[0], float(field.min()))
        bounds[1] = max(bounds[1], float(field.max()))

    def update_curve_ranges(name: str, coordinates: np.ndarray, curves: np.ndarray) -> None:
        for ranges, values in (
            (curve_x_ranges, coordinates),
            (curve_y_ranges, curves),
        ):
            bounds = ranges[name]
            bounds[0] = min(bounds[0], float(values.min()))
            bounds[1] = max(bounds[1], float(values.max()))

    def bind_coordinate_axis(name: str, values: np.ndarray) -> None:
        axis = np.asarray(values, dtype=np.float64)
        if (
            axis.ndim != 1
            or len(axis) < 2
            or not np.all(np.isfinite(axis))
            or np.any(np.diff(axis) <= 0.0)
        ):
            raise PipelineError(f"{label} {name} coordinate axis is invalid")
        existing = coordinate_axes.get(name)
        if existing is None:
            coordinate_axes[name] = axis.copy()
        elif not np.array_equal(existing, axis):
            raise PipelineError(f"{label} {name} coordinate axis changes across checkpoints")

    synthetic_rows = _as_array(
        synthetic_summary.get("checkpoint_rows"),
        label=f"{label} synthetic source rows",
    )
    if len(synthetic_rows) != len(checkpoints):
        raise PipelineError(f"{label} synthetic source row count is inconsistent")
    for epoch, raw_row in zip(checkpoints, synthetic_rows, strict=True):
        row = _as_object(raw_row, label=f"{label} synthetic source row")
        path = f"checkpoints/epoch_{epoch:05d}/arrays.npz"
        if row.get("epoch") != epoch or row.get("arrays") != path:
            raise PipelineError(f"{label} synthetic source coordinates are inconsistent")
        residual_field = _animation_npz_member(
            store,
            synthetic_id,
            path,
            _RESIDUAL_LOCAL_PLANE_JACOBIAN,
            label=f"{label} synthetic epoch {epoch}",
        )
        _animation_npz_member(
            store,
            synthetic_id,
            path,
            _RAW_LOCAL_PLANE_JACOBIAN,
            label=f"{label} synthetic epoch {epoch} raw field",
        )
        if residual_field.ndim != 4 or residual_field.shape[0] != 2:
            raise PipelineError(f"{label} synthetic residual plane field has an invalid shape")
        local_axis = _animation_npz_member(
            store,
            synthetic_id,
            path,
            "local_axis",
            label=f"{label} synthetic epoch {epoch} local axis",
        )
        bind_coordinate_axis("synthetic_real_fake", local_axis)
        if residual_field.shape[2:] != (len(local_axis), len(local_axis)):
            raise PipelineError(f"{label} synthetic local field does not align with its axis")
        update_range(
            "synthetic_real_fake",
            np.maximum(residual_field, 0.0).mean(axis=1),
        )
        path_axis = _animation_npz_member(
            store,
            synthetic_id,
            path,
            "path_coefficients",
            label=f"{label} synthetic epoch {epoch} path axis",
        )
        path_response = _animation_npz_member(
            store,
            synthetic_id,
            path,
            "path_response_l2",
            label=f"{label} synthetic epoch {epoch} path response",
        )
        if (
            path_axis.ndim != 1
            or path_response.ndim != 4
            or path_response.shape[0] != 2
            or path_response.shape[-1] != len(path_axis)
        ):
            raise PipelineError(f"{label} synthetic path profile has an invalid shape")
        bind_coordinate_axis("synthetic_real_fake_curve", path_axis)
        update_curve_ranges(
            "synthetic_real_fake",
            path_axis,
            np.median(np.maximum(path_response, 0.0), axis=(1, 2)),
        )

    cifar_id = upstreams["exp1.plateau.cifar"]
    if not isinstance(cifar_id, str) or not _DIGEST_RE.fullmatch(cifar_id):
        raise PipelineError(f"{label} CIFAR source artifact id is invalid")
    cifar_reference = store.get(cifar_id, verify=True)
    if cifar_reference.manifest.kind != "stage/exp1-plateau-cifar":
        raise PipelineError(f"{label} CIFAR source has the wrong artifact kind")
    cifar_value = store.read_json(cifar_id, "summary.json")
    cifar_summary = _as_object(cifar_value, label=f"{label} CIFAR source summary")
    if cifar_summary.get("schema_version") != _PLATEAU_SCHEMA_VERSION:
        raise PipelineError(f"{label} CIFAR source schema is inconsistent")
    expected_cifar_display = {
        "protocol_version": config.experiment1.plateau_protocol.protocol_version,
        "resnet": {
            "local_source_array": _RESIDUAL_LOCAL_PLANE_JACOBIAN,
            "anchor_source_array": _RESIDUAL_ANCHOR_PLANE_JACOBIAN,
            "estimand": _RESIDUAL_PLANE_ESTIMAND,
        },
        "vgg": {
            "local_source_array": _RAW_LOCAL_PLANE_JACOBIAN,
            "anchor_source_array": _RAW_ANCHOR_PLANE_JACOBIAN,
            "estimand": _RAW_PLANE_ESTIMAND,
        },
        "comparison_scope": "descriptive_confounded_side_by_side_with_row_specific_operators",
    }
    if canonical_hash(cifar_summary.get("jacobian_display_contract")) != canonical_hash(
        expected_cifar_display
    ):
        raise PipelineError(f"{label} CIFAR source display contract is inconsistent")
    rows = _as_array(cifar_summary.get("checkpoint_rows"), label=f"{label} CIFAR rows")
    expected_coordinates = tuple(
        (architecture, epoch) for architecture in ("resnet", "vgg") for epoch in checkpoints
    )
    if len(rows) != len(expected_coordinates):
        raise PipelineError(f"{label} CIFAR source row count is inconsistent")
    raw_maxima: dict[str, np.ndarray] = {
        "resnet": np.zeros(3, dtype=np.float64),
        "vgg": np.zeros(3, dtype=np.float64),
    }
    for (architecture, epoch), raw_row in zip(expected_coordinates, rows, strict=True):
        row = _as_object(raw_row, label=f"{label} CIFAR source row")
        if row.get("architecture") != architecture or row.get("epoch") != epoch:
            raise PipelineError(f"{label} CIFAR source coordinates are inconsistent")
        artifact_id = row.get("artifact_id")
        if not isinstance(artifact_id, str) or not _DIGEST_RE.fullmatch(artifact_id):
            raise PipelineError(f"{label} CIFAR child artifact id is invalid")
        field = _animation_npz_member(
            store,
            artifact_id,
            "arrays.npz",
            "anchor_output_distances",
            label=f"{label} {architecture} epoch {epoch}",
        )
        if field.ndim != 3 or field.shape[-1] != 3 or np.any(field < -1e-12):
            raise PipelineError(f"{label} source anchor distances have an invalid shape")
        local_axis = _animation_npz_member(
            store,
            artifact_id,
            "arrays.npz",
            "local_axis",
            label=f"{label} {architecture} epoch {epoch} local axis",
        )
        anchor_axis = _animation_npz_member(
            store,
            artifact_id,
            "arrays.npz",
            "anchor_axis",
            label=f"{label} {architecture} epoch {epoch} anchor axis",
        )
        bind_coordinate_axis(f"cifar_{architecture}_real_fake", local_axis)
        bind_coordinate_axis("cifar_architecture_cells", anchor_axis)
        if field.shape[:2] != (len(anchor_axis), len(anchor_axis)):
            raise PipelineError(f"{label} anchor distances do not align with their axis")
        raw_maxima[architecture] = np.maximum(
            raw_maxima[architecture],
            np.maximum(field, 0.0).max(axis=(0, 1)),
        )
        local_name = (
            _RESIDUAL_LOCAL_PLANE_JACOBIAN
            if architecture == "resnet"
            else _RAW_LOCAL_PLANE_JACOBIAN
        )
        anchor_name = (
            _RESIDUAL_ANCHOR_PLANE_JACOBIAN
            if architecture == "resnet"
            else _RAW_ANCHOR_PLANE_JACOBIAN
        )
        local_field = _animation_npz_member(
            store,
            artifact_id,
            "arrays.npz",
            local_name,
            label=f"{label} {architecture} epoch {epoch} local field",
        )
        anchor_field = _animation_npz_member(
            store,
            artifact_id,
            "arrays.npz",
            anchor_name,
            label=f"{label} {architecture} epoch {epoch} anchor field",
        )
        if architecture == "resnet":
            _animation_npz_member(
                store,
                artifact_id,
                "arrays.npz",
                _RAW_LOCAL_PLANE_JACOBIAN,
                label=f"{label} ResNet epoch {epoch} retained raw local field",
            )
            _animation_npz_member(
                store,
                artifact_id,
                "arrays.npz",
                _RAW_ANCHOR_PLANE_JACOBIAN,
                label=f"{label} ResNet epoch {epoch} retained raw anchor field",
            )
        if local_field.ndim != 4 or local_field.shape[0] != 2:
            raise PipelineError(f"{label} selected local plane field has an invalid shape")
        if anchor_field.ndim != 3 or anchor_field.shape[0] != 3:
            raise PipelineError(f"{label} selected anchor plane field has an invalid shape")
        if local_field.shape[2:] != (len(local_axis), len(local_axis)):
            raise PipelineError(f"{label} selected local field does not align with its axis")
        if anchor_field.shape[1:] != (len(anchor_axis), len(anchor_axis)):
            raise PipelineError(f"{label} selected anchor field does not align with its axis")
        if np.any(local_field < -1e-12) or np.any(anchor_field < -1e-12):
            raise PipelineError(f"{label} selected Jacobian fields must be nonnegative")
        update_range(
            f"cifar_{architecture}_real_fake",
            np.maximum(local_field, 0.0).mean(axis=1),
        )
        update_range(
            "cifar_architecture_cells",
            np.maximum(anchor_field, 0.0).mean(axis=0),
        )
        path_axis = _animation_npz_member(
            store,
            artifact_id,
            "arrays.npz",
            "path_coefficients",
            label=f"{label} {architecture} epoch {epoch} path axis",
        )
        path_response = _animation_npz_member(
            store,
            artifact_id,
            "arrays.npz",
            "path_response_l2",
            label=f"{label} {architecture} epoch {epoch} path response",
        )
        if (
            path_axis.ndim != 1
            or path_response.ndim != 4
            or path_response.shape[0] != 2
            or path_response.shape[-1] != len(path_axis)
        ):
            raise PipelineError(f"{label} CIFAR path profile has an invalid shape")
        bind_coordinate_axis(f"cifar_{architecture}_real_fake_curve", path_axis)
        update_curve_ranges(
            f"cifar_{architecture}_real_fake",
            path_axis,
            np.median(np.maximum(path_response, 0.0), axis=(1, 2)),
        )
    if any(np.any(value <= 0.0) for value in raw_maxima.values()):
        raise PipelineError(f"{label} source anchor distance channels need positive range")

    normalization = _as_object(summary["normalization"], label=f"{label} normalization")
    expected_normalization = {
        "real_fake_scalar_scales": "global within each bundle across all checkpoints",
        "real_fake_curve_scales": "global within each bundle across all checkpoints",
        "cell_rgb_raw_field": "anchor_output_distances",
        "cell_rgb_transform": "clip(1 - distance / channel_max, 0, 1)",
        "cell_rgb_scope": "per architecture and anchor channel across every checkpoint",
        "resnet_channel_maxima": raw_maxima["resnet"].tolist(),
        "vgg_channel_maxima": raw_maxima["vgg"].tolist(),
        "jacobian_scale": (
            "one numerical display scale across row-specific estimands and all checkpoints; "
            "not an equivalent-operator or causal comparison"
        ),
    }
    if canonical_hash(normalization) != canonical_hash(expected_normalization):
        raise PipelineError(f"{label} normalization is not derived from raw source distances")

    protocol = config.experiment1.plateau_protocol
    expected_animation_display = {
        "protocol_version": protocol.protocol_version,
        "configured_selection": protocol.animation_jacobian_selection,
        "synthetic": {
            "source_array": _RESIDUAL_LOCAL_PLANE_JACOBIAN,
            "estimand": _RESIDUAL_PLANE_ESTIMAND,
        },
        "resnet": {
            "local_source_array": _RESIDUAL_LOCAL_PLANE_JACOBIAN,
            "anchor_source_array": _RESIDUAL_ANCHOR_PLANE_JACOBIAN,
            "estimand": _RESIDUAL_PLANE_ESTIMAND,
        },
        "vgg": {
            "local_source_array": _RAW_LOCAL_PLANE_JACOBIAN,
            "anchor_source_array": _RAW_ANCHOR_PLANE_JACOBIAN,
            "estimand": _RAW_PLANE_ESTIMAND,
        },
    }
    if canonical_hash(summary["jacobian_display_contract"]) != canonical_hash(
        expected_animation_display
    ):
        raise PipelineError(f"{label} selected Jacobian fields differ from signed protocol")
    timing = (
        protocol.orientation_frame_ms,
        protocol.intermediate_frame_ms,
        protocol.final_frame_ms,
    )
    if summary["timing_ms"] != {
        "orientation": timing[0],
        "checkpoint": timing[1],
        "conclusion": timing[2],
    }:
        raise PipelineError(f"{label} timing differs from config")
    bundles = _as_array(summary["bundles"], label=f"{label} bundles")
    expected_bundles = (
        (
            "synthetic_real_fake",
            "real_fake_scalar_fields",
            "three_class_gaussian_mixture",
            ("normalization_free_residual_mlp",),
            {
                "scalar_source_array": _RESIDUAL_LOCAL_PLANE_JACOBIAN,
                "scalar_estimand": _RESIDUAL_PLANE_ESTIMAND,
                "scalar_selection": "residual_update_for_residual_transition",
                "curve_estimand": _PLATEAU_CURVE_ESTIMAND,
            },
        ),
        (
            "cifar_resnet_real_fake",
            "real_fake_scalar_fields",
            "cifar10",
            ("tracking2_resnet18_v2_width64",),
            {
                "scalar_source_array": _RESIDUAL_LOCAL_PLANE_JACOBIAN,
                "scalar_estimand": _RESIDUAL_PLANE_ESTIMAND,
                "scalar_selection": "residual_update_for_residual_transition",
                "curve_estimand": _PLATEAU_CURVE_ESTIMAND,
            },
        ),
        (
            "cifar_vgg_real_fake",
            "real_fake_scalar_fields",
            "cifar10",
            ("tracking2_vgg19_bn_width1_classifier512",),
            {
                "scalar_source_array": _RAW_LOCAL_PLANE_JACOBIAN,
                "scalar_estimand": _RAW_PLANE_ESTIMAND,
                "scalar_selection": "raw_transition_for_nonresidual_transition",
                "curve_estimand": _PLATEAU_CURVE_ESTIMAND,
            },
        ),
        (
            "cifar_architecture_cells",
            "architecture_cells_and_jacobians",
            "cifar10",
            (
                "tracking2_resnet18_v2_width64",
                "tracking2_vgg19_bn_width1_classifier512",
            ),
            {
                "jacobian_source_arrays": {
                    "resnet": _RESIDUAL_ANCHOR_PLANE_JACOBIAN,
                    "vgg": _RAW_ANCHOR_PLANE_JACOBIAN,
                },
                "jacobian_estimands": {
                    "resnet": _RESIDUAL_PLANE_ESTIMAND,
                    "vgg": _RAW_PLANE_ESTIMAND,
                },
                "comparison_scope": (
                    "descriptive_confounded_side_by_side_with_row_specific_operators"
                ),
            },
        ),
    )
    if len(bundles) != len(expected_bundles):
        raise PipelineError(f"{label} must contain exactly four animation bundles")
    for raw_row, (name, kind, task, architectures, bundle_contract) in zip(
        bundles,
        expected_bundles,
        strict=True,
    ):
        row = _as_object(raw_row, label=f"{label} bundle")
        _validate_animation_image_bundle(
            reference,
            store,
            row,
            checkpoints=checkpoints,
            expected_name=name,
            expected_kind=kind,
            expected_task=task,
            expected_architectures=architectures,
            timing=timing,
            expected_bundle_contract=bundle_contract,
            expected_scalar_range=tuple(scalar_ranges[name]),
            expected_x_coordinates=coordinate_axes[name],
            expected_y_coordinates=coordinate_axes[name],
            expected_curve_x_range=(
                tuple(curve_x_ranges[name]) if kind == "real_fake_scalar_fields" else None
            ),
            expected_curve_y_range=(
                tuple(curve_y_ranges[name]) if kind == "real_fake_scalar_fields" else None
            ),
            architecture_maxima=(raw_maxima["resnet"], raw_maxima["vgg"])
            if kind == "architecture_cells_and_jacobians"
            else None,
        )
    expected_paths = {"summary.json"}
    for name, _kind, _task, _architectures, _contract in expected_bundles:
        expected_paths.update(
            {
                f"animations/{name}.gif",
                f"animations/{name}_final.png",
                f"animations/{name}_metadata.json",
            }
        )
    if set(_artifact_file_map(reference)) != expected_paths:
        raise PipelineError(f"{label} file set differs from its four declared bundles")


def _validate_named_payload_schema(
    schema_id: str | None,
    payloads: Mapping[str, Mapping[str, object]],
    *,
    declared_paths: set[str],
    reference: ArtifactRef,
    store: ArtifactStore,
    config: LabConfig | None,
) -> None:
    if schema_id is None or schema_id == "gate-result-v1":
        return
    if schema_id == "tracking2-inputs-v1":
        _validate_inputs_artifact(payloads["inputs.json"], reference, store, config)
    elif schema_id == "tracking2-vgg-inputs-v1":
        _validate_vgg_inputs_artifact(payloads["inputs.json"], reference, store, config)
    elif schema_id == "synthetic-plateau-task-v1":
        _validate_synthetic_plateau_task_artifact(
            payloads["inventory.json"], reference, store, config
        )
    elif schema_id == "plateau-collection-v2":
        _validate_plateau_collection_artifact(payloads["summary.json"], reference, store, config)
    elif schema_id == "plateau-cifar-collection-v2":
        _validate_plateau_cifar_artifact(payloads["summary.json"], reference, store, config)
    elif schema_id == "plateau-animation-v2":
        _validate_plateau_animation_artifact(payloads["summary.json"], reference, store, config)
    elif schema_id == "probe-plan-v1":
        _validate_probe_artifact(payloads["plan.json"], declared_paths, reference, store, config)
    elif schema_id == "mechanical-result-v1":
        _validate_mechanical_payload(payloads["mechanical.json"], config, reference, store)
    elif schema_id == "mock-report-v1":
        _validate_mock_report_payload(payloads["report_payload.json"], reference, store)
    else:
        raise PipelineError(f"unknown payload schema id: {schema_id}")


def _expected_gate_result_from_bound_evidence(
    reference: ArtifactRef,
    stage: StageSpec,
    store: ArtifactStore,
    *,
    config: LabConfig,
    rule: GateRule,
    override_authorization: Mapping[str, JSONLike] | None,
    registry: StageRegistry,
    source_identity: Mapping[str, JSONLike] | None,
    validation_context: StageValidationContext,
) -> GateResult:
    """Re-evaluate a runnable gate from its verified, content-addressed producer."""

    dependency = stage.gate_evidence_dependency
    payload_path = stage.gate_evidence_payload_path
    if dependency is None or payload_path is None:
        raise PipelineError(f"gate {stage.name} has no declared evidence binding")

    upstreams = thaw_json(reference.manifest.metadata.get("upstream_artifacts"))
    if not isinstance(upstreams, dict) or set(upstreams) != set(stage.dependencies):
        raise PipelineError(
            f"artifact for {stage.name} must bind exactly its declared upstream artifacts"
        )
    evidence_artifact_id = upstreams.get(dependency)
    if not isinstance(evidence_artifact_id, str) or not _DIGEST_RE.fullmatch(evidence_artifact_id):
        raise PipelineError(
            f"artifact for {stage.name} has no valid bound evidence artifact for {dependency}"
        )
    try:
        evidence_reference = store.get(evidence_artifact_id, verify=True)
    except Exception as error:
        raise PipelineError(
            f"artifact for {stage.name} has unavailable or corrupt bound evidence"
        ) from error

    try:
        evidence_stage = registry.get(dependency)
    except PipelineError as error:
        raise PipelineError(
            f"gate {stage.name} declares an unknown evidence producer {dependency}"
        ) from error
    if payload_path not in evidence_stage.required_payload_paths:
        raise PipelineError(f"gate {stage.name} evidence payload is not declared by {dependency}")
    # A content hash alone proves byte identity, not that those bytes implement
    # the producer's scientific contract. Re-run that contract before using the
    # payload as gate evidence.
    validate_stage_output(
        evidence_reference,
        evidence_stage,
        store,
        config=config,
        registry=registry,
        source_identity=source_identity,
        validation_context=validation_context,
    )
    try:
        observations = store.read_json(evidence_artifact_id, payload_path)
    except Exception as error:
        raise PipelineError(
            f"artifact for {stage.name} cannot read its bound evidence payload"
        ) from error
    if not isinstance(observations, Mapping):
        raise PipelineError(f"artifact for {stage.name} bound evidence must be an object")

    evaluator = GateEvaluator()
    try:
        natural = evaluator.evaluate(rule, observations)
        reason = (
            None
            if override_authorization is None or natural.status is GateStatus.PASS
            else override_authorization.get("reason")
        )
        if reason is not None and not isinstance(reason, str):
            raise PipelineError(f"artifact for {stage.name} has an invalid signed override reason")
        return (
            natural
            if reason is None
            else evaluator.evaluate(
                rule,
                observations,
                override_reason=reason,
            )
        )
    except GateEvaluationError as error:
        raise PipelineError(
            f"artifact for {stage.name} cannot evaluate its bound evidence"
        ) from error


def _effective_gate_validation_inputs(
    stage: StageSpec,
    *,
    gate_rule: GateRule | None,
    gate_override_authorization: Mapping[str, JSONLike] | None,
    config: LabConfig | None,
) -> tuple[GateRule | None, Mapping[str, JSONLike] | None]:
    effective_rule = gate_rule
    effective_authorization = gate_override_authorization
    if stage.gate_evidence_dependency is None or config is None:
        return effective_rule, effective_authorization
    configured_rule = expected_gate_rule(stage, config)
    if gate_rule is not None and canonical_hash(gate_rule.to_dict()) != canonical_hash(
        configured_rule.to_dict()
    ):
        raise PipelineError(
            f"artifact for {stage.name} received a gate rule inconsistent with config"
        )
    configured_authorization = expected_gate_override_authorization(stage, config)
    if gate_override_authorization is not None and canonical_hash(
        dict(gate_override_authorization)
    ) != canonical_hash(configured_authorization):
        raise PipelineError(
            f"artifact for {stage.name} received override authorization inconsistent with config"
        )
    return configured_rule, configured_authorization


def _validation_referenced_artifact_ids(
    reference: ArtifactRef,
    stage: StageSpec,
    json_payloads: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    referenced: set[str] = set()
    upstreams = thaw_json(reference.manifest.metadata.get("upstream_artifacts"))
    if isinstance(upstreams, Mapping):
        referenced.update(
            artifact_id
            for artifact_id in upstreams.values()
            if isinstance(artifact_id, str) and _DIGEST_RE.fullmatch(artifact_id)
        )
    if stage.payload_schema_id == "plateau-cifar-collection-v2":
        payload = json_payloads.get("summary.json")
        if payload is not None:
            reducer_id = payload.get("reducer_artifact_id")
            if isinstance(reducer_id, str) and _DIGEST_RE.fullmatch(reducer_id):
                referenced.add(reducer_id)
            rows = payload.get("checkpoint_rows")
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                referenced.update(
                    artifact_id
                    for row in rows
                    if isinstance(row, Mapping)
                    for artifact_id in (row.get("artifact_id"),)
                    if isinstance(artifact_id, str) and _DIGEST_RE.fullmatch(artifact_id)
                )
    return tuple(sorted(referenced))


def validate_stage_output(
    reference: ArtifactRef,
    stage: StageSpec,
    store: ArtifactStore,
    *,
    gate_rule: GateRule | None = None,
    gate_override_authorization: Mapping[str, JSONLike] | None = None,
    config: LabConfig | None = None,
    registry: StageRegistry | None = None,
    source_identity: Mapping[str, JSONLike] | None = None,
    validation_context: StageValidationContext | None = None,
) -> GateResult | None:
    """Validate one resolved artifact against its stage's declared output contract.

    The function intentionally has no execution-state dependency, so callers can
    apply the same checks to fresh outputs, cross-run cache hits, and already
    completed run records.  Gate stages additionally return a fully reconstructed
    :class:`GateResult`; non-gate stages return ``None``. An explicitly shared
    ``validation_context`` memoizes successes only within the caller's current
    validation chain.
    """

    try:
        reference = store.get(reference.artifact_id, verify=True)
    except Exception as error:
        raise PipelineError(f"artifact for {stage.name} is unavailable or corrupt") from error
    effective_registry = DEFAULT_STAGES if registry is None else registry
    effective_rule, effective_authorization = _effective_gate_validation_inputs(
        stage,
        gate_rule=gate_rule,
        gate_override_authorization=gate_override_authorization,
        config=config,
    )
    context = StageValidationContext() if validation_context is None else validation_context
    validation_key: str | None = None
    if stage.gate_payload_path is None:
        validation_key = context._key(
            reference,
            stage,
            config=config,
            gate_rule=effective_rule,
            gate_override_authorization=effective_authorization,
            registry=effective_registry,
            source_identity=source_identity,
        )
        cached, cached_result = context._lookup(
            validation_key,
            store=store,
            stage_name=stage.name,
        )
        if cached:
            return cached_result

    expected_kind = stage.expected_artifact_kind
    if expected_kind is not None and reference.manifest.kind != expected_kind:
        raise PipelineError(
            f"artifact for {stage.name} has kind {reference.manifest.kind!r}; "
            f"expected {expected_kind!r}"
        )

    declared_files = {entry.path: entry for entry in reference.manifest.files}
    declared_paths = set(declared_files)
    missing = sorted(set(stage.required_payload_paths) - declared_paths)
    if missing:
        raise PipelineError(
            f"artifact for {stage.name} is missing required payloads: {', '.join(missing)}"
        )

    expected_schema = stage.result_schema_version
    if expected_schema is not None:
        observed_schema = reference.manifest.metadata.get("result_schema_version")
        if type(observed_schema) is not int or observed_schema != expected_schema:
            raise PipelineError(
                f"artifact for {stage.name} has result_schema_version "
                f"{observed_schema!r}; expected {expected_schema}"
            )

    json_payloads: dict[str, Mapping[str, object]] = {}
    for path in stage.required_payload_paths:
        if PurePosixPath(path).suffix != ".json":
            continue
        if declared_files[path].media_type != "application/json":
            raise PipelineError(
                f"artifact for {stage.name} declares JSON payload {path!r} with "
                f"media type {declared_files[path].media_type!r}"
            )
        try:
            payload = store.read_json(reference.artifact_id, path)
        except Exception as exc:
            raise PipelineError(
                f"artifact for {stage.name} contains an invalid JSON payload: {path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise PipelineError(
                f"artifact for {stage.name} JSON payload {path!r} must be an object"
            )
        json_payloads[path] = payload
        if path != stage.gate_payload_path:
            payload_schema = payload.get("schema_version")
            if type(payload_schema) is not int or payload_schema != expected_schema:
                raise PipelineError(
                    f"artifact for {stage.name} JSON payload {path!r} has schema_version "
                    f"{payload_schema!r}; expected {expected_schema}"
                )

    _validate_named_payload_schema(
        stage.payload_schema_id,
        json_payloads,
        declared_paths=declared_paths,
        reference=reference,
        store=store,
        config=config,
    )

    referenced_artifact_ids = _validation_referenced_artifact_ids(
        reference,
        stage,
        json_payloads,
    )

    if stage.gate_payload_path is None:
        assert validation_key is not None
        context._remember(validation_key, None, referenced_artifact_ids)
        return None
    try:
        result = GateResult.from_dict(json_payloads[stage.gate_payload_path])
    except Exception as exc:
        if isinstance(exc, PipelineError):
            raise
        raise PipelineError(
            f"artifact for {stage.name} contains an invalid GateResult payload"
        ) from exc

    if result.gate_id != stage.expected_gate_id:
        raise PipelineError(
            f"artifact for {stage.name} contains gate_id {result.gate_id!r}; "
            f"expected {stage.expected_gate_id!r}"
        )
    if effective_rule is not None:
        validate_gate_result_against_rule(
            result,
            effective_rule,
            gate_rule_signature=reference.manifest.metadata.get("gate_rule_signature"),
        )

    observed_authorization = thaw_json(reference.manifest.metadata.get("override_authorization"))
    expected_authorization = (
        None if effective_authorization is None else dict(effective_authorization)
    )
    if canonical_hash(observed_authorization) != canonical_hash(expected_authorization):
        raise PipelineError(
            f"artifact for {stage.name} has override authorization inconsistent with config"
        )
    if result.override_reason is not None:
        if expected_authorization is None:
            raise PipelineError(
                f"artifact for {stage.name} is overridden without configured authorization"
            )
        if result.override_reason != expected_authorization.get("reason"):
            raise PipelineError(
                f"artifact for {stage.name} override reason differs from authorization"
            )

    raw_inherited_lineage = reference.manifest.metadata.get("inherited_gate_overrides", ())
    if not isinstance(raw_inherited_lineage, Sequence) or isinstance(
        raw_inherited_lineage, (str, bytes)
    ):
        raise PipelineError(f"artifact for {stage.name} inherited_gate_overrides must be an array")
    try:
        inherited_lineage = tuple(
            GateOverride.from_dict(thaw_json(item))  # type: ignore[arg-type]
            for item in raw_inherited_lineage
        )
    except (GateEvaluationError, TypeError) as exc:
        raise PipelineError(
            f"artifact for {stage.name} has invalid inherited gate override metadata"
        ) from exc
    payload_inherited_lineage = tuple(
        item for item in result.override_lineage if item.gate_id != result.gate_id
    )
    if inherited_lineage != payload_inherited_lineage:
        raise PipelineError(
            f"artifact for {stage.name} gate payload does not preserve inherited override lineage"
        )
    if inherited_lineage and result.status.value == "PASS":
        raise PipelineError(
            f"artifact for {stage.name} cannot report PASS with inherited gate overrides"
        )

    expected_gate_metadata: dict[str, object] = {
        "gate_id": stage.expected_gate_id,
        "gate_status": result.status.value,
        "natural_status": result.natural_status.value,
    }
    mismatches = [
        key
        for key, expected in expected_gate_metadata.items()
        if reference.manifest.metadata.get(key) != expected
    ]
    if mismatches:
        raise PipelineError(
            f"artifact for {stage.name} has gate metadata inconsistent with its payload: "
            + ", ".join(mismatches)
        )
    if stage.gate_evidence_dependency is not None:
        if config is None:
            raise PipelineError(
                f"artifact for {stage.name} evidence validation requires the signed config"
            )
        assert effective_rule is not None
        expected_result = _expected_gate_result_from_bound_evidence(
            reference,
            stage,
            store,
            config=config,
            rule=effective_rule,
            override_authorization=effective_authorization,
            registry=effective_registry,
            source_identity=source_identity,
            validation_context=context,
        )
        if canonical_hash(result.to_dict()) != canonical_hash(expected_result.to_dict()):
            raise PipelineError(
                f"artifact for {stage.name} gate result does not match its bound upstream evidence"
            )
    return result


def _activation_shards(config: LabConfig) -> int:
    banks = config.experiment1.probe_banks
    image_shards_per_checkpoint_cut = sum(
        ceil(images / config.runtime.shard_images)
        for images in (
            banks.fit_train_images,
            banks.independent_fit_train_images,
            banks.geometry_test_images,
            banks.intervention_test_images,
        )
    )
    return (
        len(config.experiment1.checkpoints)
        * len(config.experiment1.cuts)
        * len(config.experiment1.input_recipes)
        * image_shards_per_checkpoint_cut
    )


def _codebook_shards(config: LabConfig) -> int:
    return (
        len(config.experiment1.checkpoints)
        * len(config.experiment1.cuts)
        * len(config.experiment1.state_metric.sensitivity)
        * len(config.experiment1.codebooks.k_values)
    )


def _boundary_shards(config: LabConfig) -> int:
    return (
        len(config.experiment1.checkpoints)
        * len(config.experiment1.sentinel_cuts)
        * len(config.experiment1.boundary_paths.directions)
        * len(config.experiment1.input_recipes)
        * ceil(
            config.experiment1.probe_banks.intervention_test_images / config.runtime.shard_images
        )
    )


DEFAULT_STAGES = StageRegistry(
    (
        StageSpec(
            "inputs.tracking2",
            "Validate every hash-pinned external model, checkpoint, dataset, and transplant input.",
            config_paths=(
                "inputs.tracking2",
                "experiment1.checkpoints",
                "experiment1.probe_banks",
            ),
            implementation=ImplementationStatus.RUNNABLE,
            expected_artifact_kind="stage/inputs-tracking2",
            required_payload_paths=("inputs.json", "manifest.yaml", "transplant.json"),
            result_schema_version=1,
            payload_schema_id="tracking2-inputs-v1",
        ),
        StageSpec(
            "inputs.tracking2_vgg",
            "Validate every hash-pinned exploratory VGG model, checkpoint, dataset, "
            "and criticality input.",
            config_paths=(
                "inputs.tracking2_vgg",
                "experiment1.checkpoints",
                "experiment1.probe_banks",
            ),
            implementation=ImplementationStatus.RUNNABLE,
            expected_artifact_kind="stage/inputs-tracking2-vgg",
            required_payload_paths=("inputs.json", "manifest.yaml", "criticality.json"),
            result_schema_version=1,
            payload_schema_id="tracking2-vgg-inputs-v1",
        ),
        StageSpec(
            "exp1.synthetic_task",
            "Train the deterministic three-class residual-MLP checkpoint trajectory.",
            config_paths=(
                "protocol.root_seed",
                "experiment1.checkpoints",
                "experiment1.synthetic_plateau_task",
            ),
            implementation=ImplementationStatus.RUNNABLE,
            expected_artifact_kind="stage/exp1-synthetic-task",
            required_payload_paths=("inventory.json",),
            result_schema_version=1,
            payload_schema_id="synthetic-plateau-task-v1",
        ),
        StageSpec(
            "exp1.plateau.synthetic",
            "Collect replayable real/fake paths, three-anchor fields, and Jacobian diagnostics "
            "for the synthetic residual trajectory.",
            dependencies=("exp1.synthetic_task",),
            config_paths=(
                "protocol.root_seed",
                "experiment1.checkpoints",
                "experiment1.synthetic_plateau_task.intervention_block",
                "experiment1.plateau_protocol",
            ),
            implementation=ImplementationStatus.RUNNABLE,
            estimate_shards=lambda config: len(config.experiment1.checkpoints),
            expected_artifact_kind="stage/exp1-plateau-synthetic",
            required_payload_paths=("summary.json",),
            result_schema_version=2,
            payload_schema_id="plateau-collection-v2",
        ),
        StageSpec(
            "exp1.probe_banks",
            "Freeze train/test image identities and image-level bootstrap namespaces.",
            dependencies=("inputs.tracking2",),
            config_paths=(
                "protocol.root_seed",
                "experiment1.cuts",
                "experiment1.probe_banks",
                "experiment1.bootstrap",
            ),
            implementation=ImplementationStatus.RUNNABLE,
            expected_artifact_kind="stage/exp1-probe-banks",
            required_payload_paths=("plan.json",),
            result_schema_version=1,
            payload_schema_id="probe-plan-v1",
        ),
        StageSpec(
            "exp1.plateau.cifar",
            "Collect resumable checkpoint shards for matched ResNet/VGG CIFAR activation "
            "paths, three-anchor fields, and Jacobian diagnostics.",
            dependencies=("inputs.tracking2", "inputs.tracking2_vgg", "exp1.probe_banks"),
            config_paths=(
                "protocol.root_seed",
                "runtime.device",
                "runtime.dtype",
                "experiment1.checkpoints",
                "experiment1.input_recipes",
                "experiment1.plateau_protocol",
            ),
            implementation=ImplementationStatus.RUNNABLE,
            estimate_shards=lambda config: 2 * len(config.experiment1.checkpoints),
            expected_artifact_kind="stage/exp1-plateau-cifar",
            required_payload_paths=("summary.json",),
            result_schema_version=2,
            payload_schema_id="plateau-cifar-collection-v2",
        ),
        StageSpec(
            "exp1.plateau.animations",
            "Render fixed-scale, source-bound GIFs for real/fake geometry and synchronized "
            "ResNet/VGG three-anchor cells plus Jacobian diagnostics.",
            dependencies=("exp1.plateau.synthetic", "exp1.plateau.cifar"),
            config_paths=(
                "experiment1.checkpoints",
                "experiment1.plateau_protocol.protocol_version",
                "experiment1.plateau_protocol.plane_jacobian_fields",
                "experiment1.plateau_protocol.animation_jacobian_selection",
                "experiment1.plateau_protocol.global_animation_scales",
                "experiment1.plateau_protocol.orientation_frame_ms",
                "experiment1.plateau_protocol.intermediate_frame_ms",
                "experiment1.plateau_protocol.final_frame_ms",
            ),
            implementation=ImplementationStatus.RUNNABLE,
            expected_artifact_kind="stage/exp1-plateau-animations",
            required_payload_paths=("summary.json",),
            result_schema_version=2,
            payload_schema_id="plateau-animation-v2",
        ),
        StageSpec(
            "exp1.mechanical",
            "Run metric round-trip, identity, JVP, boundary, null, and generator checks.",
            dependencies=("inputs.tracking2", "exp1.probe_banks"),
            config_paths=(
                "protocol.root_seed",
                "runtime.dtype",
                "experiment1.cuts",
                "experiment1.sentinel_cuts",
                "experiment1.mechanical_protocol",
                "experiment1.state_metric.rms_epsilon",
            ),
            implementation=ImplementationStatus.RUNNABLE,
            expected_artifact_kind="stage/exp1-mechanical",
            required_payload_paths=("mechanical.json",),
            result_schema_version=1,
            payload_schema_id="mechanical-result-v1",
        ),
        StageSpec(
            "gate.mechanical",
            "Evaluate the frozen Experiment 1 mechanical tolerances.",
            dependencies=("exp1.mechanical", "exp1.probe_banks"),
            config_paths=("gates.mechanical", "gates.overrides.mechanical"),
            implementation=ImplementationStatus.RUNNABLE,
            expected_artifact_kind="gate/mechanical",
            required_payload_paths=("gate.json",),
            result_schema_version=1,
            payload_schema_id="gate-result-v1",
            gate_payload_path="gate.json",
            expected_gate_id="mechanical",
            gate_evidence_dependency="exp1.mechanical",
            gate_evidence_payload_path="mechanical.json",
        ),
        StageSpec(
            "exp1.activations",
            "Extract immutable activation shards for every checkpoint, cut, and bank.",
            dependencies=("inputs.tracking2", "exp1.probe_banks", "gate.mechanical"),
            config_paths=(
                "runtime",
                "experiment1.checkpoints",
                "experiment1.cuts",
                "experiment1.input_recipes",
            ),
            estimate_shards=_activation_shards,
        ),
        StageSpec(
            "exp1.codebooks",
            "Fit metric-specific codebooks without mixing standardized and native geometry.",
            dependencies=("exp1.activations",),
            config_paths=("experiment1.state_metric", "experiment1.codebooks"),
            estimate_shards=_codebook_shards,
        ),
        StageSpec(
            "exp1.static_geometry",
            "Measure held-out distortion, occupancy, stability, margins, diagnostics, and nulls.",
            dependencies=("exp1.codebooks", "exp1.activations"),
            config_paths=("experiment1",),
            estimate_shards=_codebook_shards,
        ),
        StageSpec(
            "exp1.boundary_paths",
            "Measure aligned empirical-chord and off-cloud functional paths at sentinel cuts.",
            dependencies=("exp1.codebooks", "exp1.activations"),
            config_paths=(
                "runtime.shard_images",
                "experiment1.sentinel_cuts",
                "experiment1.input_recipes",
                "experiment1.probe_banks.intervention_test_images",
                "experiment1.boundary_paths",
            ),
            estimate_shards=_boundary_shards,
        ),
        StageSpec(
            "gate.coarse",
            "Evaluate held-out geometry and shifted-boundary-null evidence.",
            dependencies=("exp1.static_geometry", "exp1.boundary_paths"),
            config_paths=("gates.coarse", "gates.overrides.coarse"),
        ),
        StageSpec(
            "exp1.snapping_recovery",
            "Run fixed-assignment snapping, matched controls, and finite next-block recovery.",
            dependencies=("gate.coarse", "exp1.codebooks", "exp1.activations"),
            config_paths=("experiment1.snapping",),
        ),
        StageSpec(
            "gate.functional",
            "Evaluate snapping benefit, finite gain, and clean-cell recovery.",
            dependencies=("exp1.snapping_recovery", "gate.coarse"),
            config_paths=("gates.functional", "gates.overrides.functional"),
        ),
        StageSpec(
            "exp1.transplant_join",
            "Normalize and join same-seed transplant damage to gate-supported "
            "exploratory measurements.",
            dependencies=(
                "gate.functional",
                "inputs.tracking2",
                "exp1.static_geometry",
                "exp1.snapping_recovery",
            ),
            config_paths=("inputs.tracking2",),
        ),
        StageSpec(
            "exp1.confirmation",
            "Run the frozen geometry, boundary, snapping, and recovery protocol on three seeds.",
            dependencies=("gate.functional", "inputs.tracking2"),
            config_paths=("experiment1",),
        ),
        StageSpec(
            "gate.confirmation",
            "Evaluate training-seed replication without treating image bootstraps as replicates.",
            dependencies=("exp1.confirmation", "gate.functional"),
            config_paths=("gates.confirmation", "gates.overrides.confirmation"),
        ),
        StageSpec(
            "report.build",
            "Build the deterministic self-contained MOCKUP report; real reports require a saved "
            "immutable payload outside this zero-dependency stage.",
            config_paths=("report",),
            implementation=ImplementationStatus.RUNNABLE,
            expected_artifact_kind="report/mockup",
            required_payload_paths=("report.html", "report_payload.json", "spec.md"),
            result_schema_version=1,
            payload_schema_id="mock-report-v1",
        ),
    )
)
