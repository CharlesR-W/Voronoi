"""Declarative stage DAG, shard estimates, and cache-stable stage identities."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
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
from voronoi_lab.mechanical import replay_synthetic_invariants, replay_toy_geometry


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
                GateCheck(
                    "generator_twirl_support",
                    "synthetic_invariants.passed",
                    ComparisonOperator.IS_TRUE,
                ),
            ),
        )
    if name == "gate.synthetic_exact":
        thresholds = config.gates.synthetic
        return GateRule(
            gate_id="synthetic_exact",
            description="Noiseless exhaustive recovery prerequisite for sampled recovery.",
            checks=(
                GateCheck(
                    "instance_count",
                    "instances",
                    ComparisonOperator.EQ,
                    thresholds.noiseless_instances,
                ),
                GateCheck(
                    "exact_tuple_recovery",
                    "exact_tuple_recovery_fraction",
                    ComparisonOperator.GE,
                    thresholds.exact_tuple_recovery_fraction_min,
                ),
                GateCheck(
                    "support_component_error",
                    "worst_support_error",
                    ComparisonOperator.LT,
                    thresholds.relative_support_error_max,
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
        "gate.synthetic_exact": config.gates.overrides.synthetic_exact,
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
        ("geometry", "probe_banks", "protocol", "resnet", "synthetic_invariants", "warnings"),
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
    expected_invariants = replay_synthetic_invariants(config.protocol.root_seed)
    if canonical_hash(payload["synthetic_invariants"]) != canonical_hash(expected_invariants):
        raise PipelineError(f"{label} synthetic invariants do not match deterministic replay")
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
    invariants = _as_object(payload["synthetic_invariants"], label=f"{label} synthetic_invariants")
    if not isinstance(invariants.get("passed"), bool):
        raise PipelineError(f"{label} synthetic invariant status must be boolean")
    warnings = _as_array(payload["warnings"], label=f"{label} warnings")
    if any(not isinstance(warning, str) for warning in warnings):
        raise PipelineError(f"{label} warnings must contain strings")


def _validate_exact_payload(
    payload: Mapping[str, object],
    reference: ArtifactRef,
    store: ArtifactStore,
    config: LabConfig | None,
) -> None:
    label = "synthetic exact result"
    _require_fields(
        payload,
        (
            "aggregate",
            "exact_instances",
            "exact_tuple_recovery_fraction",
            "factor_sizes",
            "instance_results",
            "instances",
            "numeric_dtype",
            "ordered_instance_artifact_ids",
            "protocol_version",
            "reducer_artifact_id",
            "worst_support_error",
        ),
        label=label,
    )
    instances = _as_int(payload["instances"], label=f"{label} instances", minimum=1)
    exact_instances = _as_int(
        payload["exact_instances"], label=f"{label} exact_instances", minimum=0
    )
    if exact_instances > instances:
        raise PipelineError(f"{label} exact_instances exceeds instances")
    fraction = _as_number(
        payload["exact_tuple_recovery_fraction"], label=f"{label} recovery fraction"
    )
    if abs(fraction - exact_instances / instances) > 1e-15:
        raise PipelineError(f"{label} recovery fraction is inconsistent")
    if payload["numeric_dtype"] != "float64":
        raise PipelineError(f"{label} must record float64 numeric_dtype")
    if config is None:
        raise PipelineError(f"{label} validation requires the signed run configuration")
    experiment = config.experiment2
    protocol = experiment.exact_protocol
    instance_rows = _as_array(payload["instance_results"], label=f"{label} instance_results")
    artifact_ids = _as_array(
        payload["ordered_instance_artifact_ids"], label=f"{label} instance artifact ids"
    )
    if len(instance_rows) != instances or len(artifact_ids) != instances:
        raise PipelineError(f"{label} instance evidence count is inconsistent")
    if len(set(artifact_ids)) != instances or any(
        not isinstance(artifact_id, str) or not _DIGEST_RE.fullmatch(artifact_id)
        for artifact_id in artifact_ids
    ):
        raise PipelineError(f"{label} instance artifact ids are invalid")
    from voronoi_lab.synthetic.runner import (
        OracleExhaustiveInstanceResult,
        run_oracle_exhaustive_instance,
        summarize_oracle_exhaustive_instances,
    )

    instance_fields = set(OracleExhaustiveInstanceResult.__dataclass_fields__)
    reconstructed: list[OracleExhaustiveInstanceResult] = []
    for expected_index, (raw_row, artifact_id) in enumerate(
        zip(instance_rows, artifact_ids, strict=True)
    ):
        row = _as_object(raw_row, label=f"{label} instance {expected_index}")
        if set(row) != instance_fields:
            raise PipelineError(f"{label} instance {expected_index} has invalid fields")
        if row["instance_index"] != expected_index:
            raise PipelineError(f"{label} instance ordering is inconsistent")
        for integer_field in ("instance_index", "seed", "best_tie_count", "evaluated_labelings"):
            _as_int(row[integer_field], label=f"{label} {integer_field}", minimum=0)
        if not isinstance(row["exact"], bool):
            raise PipelineError(f"{label} instance exact must be boolean")
        for numeric_field in (
            "train_support_error",
            "heldout_support_error",
            "train_objective",
            "heldout_objective",
            "heldout_oracle_objective",
        ):
            _as_nonnegative_number(row[numeric_field], label=f"{label} {numeric_field}")
        excess = _as_number(
            row["heldout_excess_objective"], label=f"{label} heldout_excess_objective"
        )
        if not np.isfinite(excess):
            raise PipelineError(f"{label} heldout_excess_objective must be finite")
        for sequence_field in (
            "observed_primitive_names",
            "observed_generator_family",
            "realized_normalized_support_order_spectrum",
            "train_primitive_indices",
            "heldout_primitive_indices",
            "truth_labeling",
            "selected_labeling",
        ):
            _as_array(row[sequence_field], label=f"{label} {sequence_field}")
        for value_field, hash_field in (
            ("observed_generator_family", "observed_generator_family_hash"),
            (
                "realized_normalized_support_order_spectrum",
                "realized_normalized_support_order_spectrum_hash",
            ),
            ("selected_labeling", "selected_labeling_hash"),
            ("truth_labeling", "truth_labeling_hash"),
        ):
            if row[hash_field] != canonical_hash(row[value_field]):
                raise PipelineError(
                    f"{label} instance {expected_index} has inconsistent {hash_field}"
                )
        try:
            reconstructed_row = OracleExhaustiveInstanceResult(**row)  # type: ignore[arg-type]
        except TypeError as error:
            raise PipelineError(
                f"{label} instance {expected_index} cannot be reconstructed"
            ) from error
        try:
            replayed_row = run_oracle_exhaustive_instance(
                seed=config.protocol.root_seed,
                instance_index=expected_index,
                factor_sizes=experiment.oracle_factor_sizes,
                train_primitives=experiment.train_primitives,
                heldout_primitives=experiment.heldout_primitives,
                density=experiment.generator_density,
                unary_weight=experiment.unary_weight,
                rho=protocol.rho,
                delta=protocol.delta,
                support_policy=protocol.support_policy,
                random_relabel=protocol.random_relabel,
                penalties=experiment.support_penalties,
                max_states=protocol.max_states,
                generator_rate_shape=protocol.generator_rate_shape,
                generator_connectivity_policy=protocol.generator_connectivity_policy,
                generator_normalization=protocol.generator_normalization,
                exhaustive_tie_atol=protocol.exhaustive_tie_atol,
                exhaustive_tie_rtol=protocol.exhaustive_tie_rtol,
            )
        except (TypeError, ValueError) as error:
            raise PipelineError(f"{label} instance {expected_index} cannot be replayed") from error
        if canonical_hash(asdict(replayed_row)) != canonical_hash(row):
            raise PipelineError(
                f"{label} instance {expected_index} does not match deterministic replay"
            )
        reconstructed.append(reconstructed_row)
        child = store.get(artifact_id, verify=True)  # type: ignore[arg-type]
        if child.manifest.kind != "shard/exp2-exact-instance":
            raise PipelineError(f"{label} instance {expected_index} has the wrong artifact kind")
        child_payload = store.read_json(child.artifact_id, "instance.json")
        child_object = _as_object(child_payload, label=f"{label} instance shard")
        if child_object.get("schema_version") != 1 or canonical_hash(
            child_object.get("instance")
        ) != canonical_hash(row):
            raise PipelineError(f"{label} instance shard content is inconsistent")

    reducer_id = payload["reducer_artifact_id"]
    if not isinstance(reducer_id, str) or not _DIGEST_RE.fullmatch(reducer_id):
        raise PipelineError(f"{label} reducer artifact id is invalid")
    if reference.manifest.metadata.get("reducer_artifact_id") != reducer_id:
        raise PipelineError(f"{label} reducer metadata is inconsistent")
    reducer = store.get(reducer_id, verify=True)
    if reducer.manifest.kind != "shards/reducer-manifest":
        raise PipelineError(f"{label} reducer has the wrong artifact kind")
    reducer_payload = _as_object(
        store.read_json(reducer.artifact_id, "shards.json"), label=f"{label} reducer"
    )
    ordered = _as_array(reducer_payload.get("ordered_shards"), label=f"{label} reducer shards")
    reducer_ids = [
        _as_object(entry, label=f"{label} reducer entry").get("artifact_id") for entry in ordered
    ]
    if list(artifact_ids) != reducer_ids:
        raise PipelineError(f"{label} reducer ordering is inconsistent")

    try:
        smoke = summarize_oracle_exhaustive_instances(
            reconstructed,
            seed=config.protocol.root_seed,
            factor_sizes=experiment.oracle_factor_sizes,
            train_primitives=experiment.train_primitives,
            heldout_primitives=experiment.heldout_primitives,
            density=experiment.generator_density,
            unary_weight=experiment.unary_weight,
            rho=protocol.rho,
            delta=protocol.delta,
            support_policy=protocol.support_policy,
            random_relabel=protocol.random_relabel,
            penalties=experiment.support_penalties,
            max_states=protocol.max_states,
            generator_rate_shape=protocol.generator_rate_shape,
            generator_connectivity_policy=protocol.generator_connectivity_policy,
            generator_normalization=protocol.generator_normalization,
            exhaustive_tie_atol=protocol.exhaustive_tie_atol,
            exhaustive_tie_rtol=protocol.exhaustive_tie_rtol,
        )
    except (TypeError, ValueError) as error:
        raise PipelineError(f"{label} rows do not satisfy the signed protocol") from error
    aggregate: dict[str, JSONLike] = {
        "evaluated_labelings": smoke.evaluated_labelings,
        "exact_instances": smoke.exact_instances,
        "exact_tuple_recovery_fraction": smoke.exact_instances / smoke.instances,
        "max_best_objective": smoke.max_best_objective,
        "max_heldout_excess_objective": smoke.max_heldout_excess_objective,
        "worst_support_error": smoke.worst_support_error,
        "worst_train_support_error": smoke.worst_train_support_error,
    }
    expected_payload: dict[str, JSONLike] = {
        "schema_version": 1,
        **asdict(smoke),
        "exact_tuple_recovery_fraction": smoke.exact_instances / smoke.instances,
        "numeric_dtype": "float64",
        "aggregate": aggregate,
        "ordered_instance_artifact_ids": list(artifact_ids),
        "reducer_artifact_id": reducer_id,
    }
    if canonical_hash(payload) != canonical_hash(expected_payload):
        raise PipelineError(f"{label} aggregate does not match its signed instance evidence")


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
    elif schema_id == "probe-plan-v1":
        _validate_probe_artifact(payloads["plan.json"], declared_paths, reference, store, config)
    elif schema_id == "mechanical-result-v1":
        _validate_mechanical_payload(payloads["mechanical.json"], config, reference, store)
    elif schema_id == "synthetic-exact-result-v1":
        _validate_exact_payload(payloads["exact.json"], reference, store, config)
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
    if stage.payload_schema_id == "synthetic-exact-result-v1":
        payload = json_payloads.get("exact.json")
        if payload is not None:
            reducer_id = payload.get("reducer_artifact_id")
            if isinstance(reducer_id, str) and _DIGEST_RE.fullmatch(reducer_id):
                referenced.add(reducer_id)
            instance_ids = payload.get("ordered_instance_artifact_ids")
            if isinstance(instance_ids, Sequence) and not isinstance(instance_ids, (str, bytes)):
                referenced.update(
                    artifact_id
                    for artifact_id in instance_ids
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


def _synthetic_exact_shards(config: LabConfig) -> int:
    return config.experiment2.exact_instances


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
            "exp2.exact",
            "Run the narrow tiny-state oracle/exhaustive optimization subgate; sampled and "
            "symmetry-limit evidence remain separate.",
            # Worker count is execution policy recorded by the run receipt, not
            # scientific identity; sequential and threaded runs share shards.
            config_paths=(
                "protocol.root_seed",
                "experiment2.oracle_factor_sizes",
                "experiment2.train_primitives",
                "experiment2.heldout_primitives",
                "experiment2.generator_density",
                "experiment2.unary_weight",
                "experiment2.exact_instances",
                "experiment2.support_penalties",
                "experiment2.exact_protocol",
            ),
            implementation=ImplementationStatus.RUNNABLE,
            estimate_shards=_synthetic_exact_shards,
            expected_artifact_kind="stage/exp2-exact",
            required_payload_paths=("exact.json",),
            result_schema_version=1,
            payload_schema_id="synthetic-exact-result-v1",
        ),
        StageSpec(
            "gate.synthetic_exact",
            "Evaluate exact tuple recovery and support-component error.",
            dependencies=("exp2.exact",),
            config_paths=(
                "gates.synthetic.noiseless_instances",
                "gates.synthetic.exact_tuple_recovery_fraction_min",
                "gates.synthetic.relative_support_error_max",
                "gates.overrides.synthetic_exact",
            ),
            implementation=ImplementationStatus.RUNNABLE,
            expected_artifact_kind="gate/synthetic-exact",
            required_payload_paths=("gate.json",),
            result_schema_version=1,
            payload_schema_id="gate-result-v1",
            gate_payload_path="gate.json",
            expected_gate_id="synthetic_exact",
            gate_evidence_dependency="exp2.exact",
            gate_evidence_payload_path="exact.json",
        ),
        StageSpec(
            "exp2.sampled",
            "Run held-out sampled recovery, interaction/symmetry sweeps, baselines, and nulls.",
            dependencies=("gate.synthetic_exact",),
            config_paths=("experiment2",),
        ),
        StageSpec(
            "gate.synthetic",
            "Evaluate the sampled recovery and calibrated false-positive rules.",
            dependencies=("exp2.sampled", "gate.synthetic_exact"),
            config_paths=("gates.synthetic", "gates.overrides.synthetic"),
        ),
        StageSpec(
            "real.transitions",
            "Estimate same-grid rectangular real cell transitions on held-out contexts.",
            dependencies=("gate.functional", "exp1.codebooks", "exp1.activations"),
            config_paths=("experiment1",),
        ),
        StageSpec(
            "real.algebra",
            "Run occupied-subspace intertwiners and held-out factor compression.",
            dependencies=("real.transitions", "gate.synthetic", "gate.functional"),
            config_paths=("experiment1", "experiment2"),
        ),
        StageSpec(
            "gate.real_algebra",
            "Evaluate calibrated real-factor compression against synthetic and null controls.",
            dependencies=("real.algebra", "gate.synthetic", "gate.functional"),
            config_paths=("gates.real_algebra", "gates.overrides.real_algebra"),
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
