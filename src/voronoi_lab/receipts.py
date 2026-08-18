"""Immutable, database-independent receipts for successful run requests.

The SQLite index is a discovery and execution surface, not the durable audit
record.  A receipt snapshots one successful dependency-closed request into the
content-addressed artifact store and can be verified later with only that store.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from voronoi_lab.config import LabConfig
from voronoi_lab.core import (
    ArtifactRef,
    ArtifactStore,
    GateEvaluationError,
    GateEvaluator,
    GateOverride,
    GateResult,
    GateRule,
    JSONLike,
    JSONValue,
    Provenance,
    RunIndex,
    StageAttemptEvent,
    StageRecord,
    StageState,
    canonical_hash,
    thaw_json,
)
from voronoi_lab.pipeline import (
    DEFAULT_STAGES,
    ImplementationStatus,
    PipelineError,
    StageRegistry,
    StageSpec,
    StageValidationContext,
    expected_gate_override_authorization,
    expected_gate_rule,
    stage_config,
    validate_gate_result_against_rule,
    validate_stage_output,
)
from voronoi_lab.sharding import (
    ShardKey,
    ShardSpec,
    ShardValidationError,
    validate_shard_artifact,
)

RECEIPT_SCHEMA_VERSION = 1
REGISTRY_CONTRACT_SCHEMA_VERSION = 2
_SUPPORTED_REGISTRY_CONTRACT_SCHEMA_VERSIONS = frozenset({1, 2})
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_EVENT_KINDS = frozenset({"CLAIMED", "RECLAIMED", "FINISHED"})
_ATTEMPT_HISTORY_STATUSES = frozenset({"COMPLETE", "UNAVAILABLE_PRE_V3"})


class ReceiptError(RuntimeError):
    """Raised when a receipt cannot be published or independently verified."""


@dataclass(frozen=True, slots=True)
class VerifiedRunReceipt:
    """A verified receipt and all immutable objects it directly audits."""

    reference: ArtifactRef
    payload: Mapping[str, JSONLike]
    provenance_reference: ArtifactRef
    stage_artifacts: tuple[ArtifactRef, ...]
    auxiliary_artifacts: tuple[ArtifactRef, ...]
    integrity_validation: str
    config_schema_version: int
    provenance_schema_version: int
    config_compatibility: str
    provenance_compatibility: str
    source_compatibility: str
    registry_compatibility: str
    semantic_validation: str
    semantic_validation_reason: str | None

    @property
    def artifact_id(self) -> str:
        return self.reference.artifact_id


@dataclass(frozen=True, slots=True)
class _EmbeddedStageContract:
    spec: StageSpec
    output_contract: Mapping[str, JSONLike]
    selected_stage_config: Mapping[str, JSONLike]
    gate_rule: GateRule | None
    gate_override_authorization: Mapping[str, JSONLike] | None


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise ReceiptError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _exact_object(value: object, keys: set[str], *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ReceiptError(f"{label} keys must be exactly {sorted(keys)}")
    return value


def _representation_version(
    value: object,
    *,
    key: str,
    label: str,
) -> int:
    if not isinstance(value, Mapping):
        raise ReceiptError(f"{label} must be a JSON object")
    version = value.get(key)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ReceiptError(f"{label} {key} must be a positive integer")
    return version


def _stage_contract_payload(stage: StageSpec, config: LabConfig) -> dict[str, JSONLike]:
    gate_rule = (
        None if stage.gate_payload_path is None else expected_gate_rule(stage, config).to_dict()
    )
    gate_authorization = (
        None
        if stage.gate_payload_path is None
        else expected_gate_override_authorization(stage, config)
    )
    return {
        "config_paths": list(stage.config_paths),
        "dependencies": list(stage.dependencies),
        "expected_gate_override_authorization": gate_authorization,
        "expected_gate_rule": gate_rule,
        "expected_gate_rule_signature": (None if gate_rule is None else canonical_hash(gate_rule)),
        "name": stage.name,
        "output_contract": stage.output_contract(),
        "selected_stage_config": stage_config(config, stage.config_paths),
        "stage_version": stage.stage_version,
    }


def _raw_stage_config(config: Mapping[str, object], paths: Sequence[str]) -> dict[str, JSONLike]:
    selected: dict[str, JSONLike] = {}
    for path in paths:
        current: object = config
        for component in path.split("."):
            if not isinstance(current, Mapping) or component not in current:
                raise ReceiptError(f"embedded config lacks historical stage path {path!r}")
            current = current[component]
        selected[path] = current  # type: ignore[assignment]
    return selected


def _registry_contract_payload(
    registry: StageRegistry,
    config: LabConfig,
    targets: Sequence[str],
) -> dict[str, JSONLike]:
    stages = [
        _stage_contract_payload(stage, config) for stage in registry.topological_order(targets)
    ]
    body: dict[str, JSONLike] = {
        "schema_version": REGISTRY_CONTRACT_SCHEMA_VERSION,
        "stages": stages,
    }
    return {**body, "sha256": canonical_hash(body)}


def _parse_registry_contract(
    raw: object,
    *,
    config: Mapping[str, object],
    requested_targets: Sequence[str],
    stage_names: Sequence[str],
) -> tuple[tuple[_EmbeddedStageContract, ...], str]:
    contract = _exact_object(
        raw,
        {"schema_version", "sha256", "stages"},
        label="receipt registry contract",
    )
    if (
        type(contract["schema_version"]) is not int
        or contract["schema_version"] not in _SUPPORTED_REGISTRY_CONTRACT_SCHEMA_VERSIONS
    ):
        raise ReceiptError("unsupported receipt registry-contract schema")
    contract_schema_version = contract["schema_version"]
    contract_hash = _digest(contract["sha256"], label="receipt registry contract sha256")
    expected_hash = canonical_hash(
        {
            "schema_version": contract["schema_version"],
            "stages": contract["stages"],
        }
    )
    if contract_hash != expected_hash:
        raise ReceiptError("receipt registry-contract hash is inconsistent")
    raw_stages = contract["stages"]
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ReceiptError("receipt registry contract must contain stages")
    if len(raw_stages) != len(stage_names):
        raise ReceiptError("receipt registry contract and stage closure differ")

    stage_keys = {
        "config_paths",
        "dependencies",
        "expected_gate_override_authorization",
        "expected_gate_rule",
        "expected_gate_rule_signature",
        "name",
        "output_contract",
        "selected_stage_config",
        "stage_version",
    }
    output_keys = {
        "expected_artifact_kind",
        "expected_gate_id",
        "gate_evidence_dependency",
        "gate_evidence_payload_path",
        "gate_payload_path",
        "payload_schema_id",
        "required_payload_paths",
        "result_schema_version",
    }
    if contract_schema_version == 1:
        output_keys -= {
            "gate_evidence_dependency",
            "gate_evidence_payload_path",
        }
    parsed: list[_EmbeddedStageContract] = []
    seen: set[str] = set()
    for ordinal, raw_stage in enumerate(raw_stages):
        stage = _exact_object(
            raw_stage,
            stage_keys,
            label=f"receipt registry stage {ordinal}",
        )
        name = stage["name"]
        dependencies = stage["dependencies"]
        config_paths = stage["config_paths"]
        output = _exact_object(
            stage["output_contract"],
            output_keys,
            label=f"receipt registry stage {name!r} output contract",
        )
        required_payload_paths = output["required_payload_paths"]
        if (
            not isinstance(name, str)
            or name != stage_names[ordinal]
            or not isinstance(dependencies, list)
            or not all(isinstance(value, str) for value in dependencies)
            or not isinstance(config_paths, list)
            or not all(isinstance(value, str) for value in config_paths)
            or not isinstance(required_payload_paths, list)
            or not all(isinstance(value, str) for value in required_payload_paths)
        ):
            raise ReceiptError(f"receipt registry stage {ordinal} has invalid declarations")
        try:
            spec = StageSpec(
                name=name,
                description=f"Embedded historical contract for {name}",
                dependencies=tuple(dependencies),
                config_paths=tuple(config_paths),
                implementation=ImplementationStatus.RUNNABLE,
                stage_version=stage["stage_version"],  # type: ignore[arg-type]
                expected_artifact_kind=output["expected_artifact_kind"],  # type: ignore[arg-type]
                required_payload_paths=tuple(required_payload_paths),
                result_schema_version=output["result_schema_version"],  # type: ignore[arg-type]
                payload_schema_id=output["payload_schema_id"],  # type: ignore[arg-type]
                gate_payload_path=output["gate_payload_path"],  # type: ignore[arg-type]
                expected_gate_id=output["expected_gate_id"],  # type: ignore[arg-type]
                gate_evidence_dependency=output.get("gate_evidence_dependency"),  # type: ignore[arg-type]
                gate_evidence_payload_path=output.get("gate_evidence_payload_path"),  # type: ignore[arg-type]
            )
        except (PipelineError, TypeError, ValueError) as error:
            raise ReceiptError(
                f"receipt registry stage {name!r} is not a valid stage contract"
            ) from error
        if name in seen or any(dependency not in seen for dependency in spec.dependencies):
            raise ReceiptError("receipt registry stages are not a unique topological order")
        seen.add(name)
        selected = stage["selected_stage_config"]
        if not isinstance(selected, dict) or canonical_hash(selected) != canonical_hash(
            _raw_stage_config(config, spec.config_paths)
        ):
            raise ReceiptError(f"receipt registry stage {name} selected config is inconsistent")

        raw_rule = stage["expected_gate_rule"]
        raw_rule_signature = stage["expected_gate_rule_signature"]
        raw_authorization = stage["expected_gate_override_authorization"]
        if spec.gate_payload_path is None:
            if any(
                value is not None for value in (raw_rule, raw_rule_signature, raw_authorization)
            ):
                raise ReceiptError(f"non-gate stage {name} embeds gate policy")
            gate_rule = None
            authorization = None
        else:
            try:
                gate_rule = GateRule.from_dict(raw_rule)
            except (GateEvaluationError, TypeError, ValueError) as error:
                raise ReceiptError(f"receipt registry stage {name} gate rule is invalid") from error
            rule_signature = _digest(
                raw_rule_signature,
                label=f"receipt registry stage {name} gate rule signature",
            )
            if rule_signature != canonical_hash(gate_rule.to_dict()):
                raise ReceiptError(f"receipt registry stage {name} gate rule hash is invalid")
            if gate_rule.gate_id != spec.expected_gate_id:
                raise ReceiptError(f"receipt registry stage {name} gate identity is inconsistent")
            if raw_authorization is not None and not isinstance(raw_authorization, dict):
                raise ReceiptError(
                    f"receipt registry stage {name} override authorization is invalid"
                )
            override_paths = [
                path for path in spec.config_paths if path.startswith("gates.overrides.")
            ]
            if len(override_paths) != 1 or canonical_hash(raw_authorization) != canonical_hash(
                selected[override_paths[0]]
            ):
                raise ReceiptError(
                    f"receipt registry stage {name} override authorization is not signed config"
                )
            authorization = raw_authorization  # type: ignore[assignment]
        parsed.append(
            _EmbeddedStageContract(
                spec,
                output,  # type: ignore[arg-type]
                selected,
                gate_rule,
                authorization,
            )
        )

    by_name = {item.spec.name: item.spec for item in parsed}
    for item in parsed:
        dependency = item.spec.gate_evidence_dependency
        evidence_path = item.spec.gate_evidence_payload_path
        if dependency is None or evidence_path is None:
            continue
        producer = by_name.get(dependency)
        if producer is None or evidence_path not in producer.required_payload_paths:
            raise ReceiptError(
                f"receipt registry stage {item.spec.name} has invalid gate evidence binding"
            )
    closure: set[str] = set()

    def add(name: str) -> None:
        if name not in by_name:
            raise ReceiptError(f"receipt target {name!r} is absent from its embedded contract")
        if name in closure:
            return
        closure.add(name)
        for dependency in by_name[name].dependencies:
            add(dependency)

    for target in requested_targets:
        add(target)
    if closure != set(stage_names):
        raise ReceiptError("receipt stages do not equal the embedded target closure")
    return tuple(parsed), contract_hash


def _embedded_stage_signature(
    contract: _EmbeddedStageContract,
    *,
    source_identity: Mapping[str, JSONLike],
    upstream_artifact_ids: Mapping[str, str],
) -> str:
    spec = contract.spec
    if set(upstream_artifact_ids) != set(spec.dependencies):
        raise ReceiptError(f"receipt stage {spec.name} has the wrong dependency set")
    return canonical_hash(
        {
            "config": dict(contract.selected_stage_config),
            "output_contract": dict(contract.output_contract),
            "source": dict(source_identity),
            "stage": spec.name,
            "stage_version": spec.stage_version,
            "upstream_artifacts": dict(sorted(upstream_artifact_ids.items())),
        }
    )


def _validate_embedded_output(
    store: ArtifactStore,
    reference: ArtifactRef,
    contract: _EmbeddedStageContract,
    dependencies: Mapping[str, ArtifactRef],
) -> GateResult | None:
    """Apply the historical structural contract without consulting a live registry."""

    spec = contract.spec
    if (
        spec.expected_artifact_kind is not None
        and reference.manifest.kind != spec.expected_artifact_kind
    ):
        raise ReceiptError(
            f"receipt stage {spec.name} artifact kind violates its embedded contract"
        )
    declared_files = {entry.path: entry for entry in reference.manifest.files}
    missing = sorted(set(spec.required_payload_paths) - set(declared_files))
    if missing:
        raise ReceiptError(
            f"receipt stage {spec.name} is missing embedded required payloads: "
            + ", ".join(missing)
        )
    if spec.result_schema_version is not None and (
        type(reference.manifest.metadata.get("result_schema_version")) is not int
        or reference.manifest.metadata.get("result_schema_version") != spec.result_schema_version
    ):
        raise ReceiptError(
            f"receipt stage {spec.name} result schema violates its embedded contract"
        )
    payloads: dict[str, Mapping[str, object]] = {}
    for path in spec.required_payload_paths:
        if not path.endswith(".json"):
            continue
        entry = declared_files[path]
        if entry.media_type != "application/json":
            raise ReceiptError(f"receipt stage {spec.name} JSON payload has the wrong media type")
        try:
            raw = store.read_json(reference.artifact_id, path)
        except Exception as error:
            raise ReceiptError(
                f"receipt stage {spec.name} required JSON payload is unreadable"
            ) from error
        if not isinstance(raw, Mapping):
            raise ReceiptError(f"receipt stage {spec.name} required JSON payload is not an object")
        payloads[path] = raw
        if path != spec.gate_payload_path and (
            type(raw.get("schema_version")) is not int
            or raw.get("schema_version") != spec.result_schema_version
        ):
            raise ReceiptError(
                f"receipt stage {spec.name} payload schema violates its embedded contract"
            )
    if spec.gate_payload_path is None:
        return None
    try:
        result = GateResult.from_dict(payloads[spec.gate_payload_path])
    except Exception as error:
        raise ReceiptError(f"receipt stage {spec.name} gate payload is invalid") from error
    if result.gate_id != spec.expected_gate_id or contract.gate_rule is None:
        raise ReceiptError(f"receipt stage {spec.name} gate identity is inconsistent")
    try:
        validate_gate_result_against_rule(
            result,
            contract.gate_rule,
            gate_rule_signature=reference.manifest.metadata.get("gate_rule_signature"),
        )
    except PipelineError as error:
        raise ReceiptError(
            f"receipt stage {spec.name} violates its trusted output contract (embedded gate rule)"
        ) from error
    observed_authorization = thaw_json(reference.manifest.metadata.get("override_authorization"))
    expected_authorization = (
        None
        if contract.gate_override_authorization is None
        else dict(contract.gate_override_authorization)
    )
    if canonical_hash(observed_authorization) != canonical_hash(expected_authorization):
        raise ReceiptError(
            f"receipt stage {spec.name} override authorization violates its embedded contract"
        )
    if result.override_reason is not None and (
        expected_authorization is None
        or result.override_reason != expected_authorization.get("reason")
    ):
        raise ReceiptError(
            f"receipt stage {spec.name} override is not authorized by its embedded contract"
        )
    raw_inherited = reference.manifest.metadata.get("inherited_gate_overrides", ())
    if not isinstance(raw_inherited, (list, tuple)):
        raise ReceiptError(f"receipt stage {spec.name} inherited overrides are invalid")
    try:
        inherited = tuple(GateOverride.from_dict(thaw_json(item)) for item in raw_inherited)
    except (GateEvaluationError, TypeError, ValueError) as error:
        raise ReceiptError(f"receipt stage {spec.name} inherited overrides are invalid") from error
    payload_inherited = tuple(
        item for item in result.override_lineage if item.gate_id != result.gate_id
    )
    if inherited != payload_inherited or (inherited and result.status.value == "PASS"):
        raise ReceiptError(f"receipt stage {spec.name} inherited override lineage is inconsistent")
    expected_metadata = {
        "gate_id": spec.expected_gate_id,
        "gate_status": result.status.value,
        "natural_status": result.natural_status.value,
    }
    if any(
        reference.manifest.metadata.get(name) != value for name, value in expected_metadata.items()
    ):
        raise ReceiptError(f"receipt stage {spec.name} gate metadata is inconsistent")
    evidence_dependency = spec.gate_evidence_dependency
    evidence_path = spec.gate_evidence_payload_path
    if evidence_dependency is not None and evidence_path is not None:
        evidence_reference = dependencies.get(evidence_dependency)
        if evidence_reference is None:
            raise ReceiptError(
                f"receipt stage {spec.name} lacks its embedded gate evidence dependency"
            )
        try:
            observations = store.read_json(evidence_reference.artifact_id, evidence_path)
        except Exception as error:
            raise ReceiptError(
                f"receipt stage {spec.name} bound gate evidence is unreadable"
            ) from error
        if not isinstance(observations, Mapping):
            raise ReceiptError(f"receipt stage {spec.name} bound gate evidence must be an object")
        evaluator = GateEvaluator()
        try:
            natural = evaluator.evaluate(contract.gate_rule, observations)
            reason = (
                None
                if expected_authorization is None or natural.status.value == "PASS"
                else expected_authorization.get("reason")
            )
            if reason is not None and not isinstance(reason, str):
                raise ReceiptError(f"receipt stage {spec.name} embedded override reason is invalid")
            expected_result = (
                natural
                if reason is None
                else evaluator.evaluate(
                    contract.gate_rule,
                    observations,
                    override_reason=reason,
                )
            )
        except GateEvaluationError as error:
            raise ReceiptError(
                f"receipt stage {spec.name} cannot evaluate its embedded gate evidence"
            ) from error
        if canonical_hash(result.to_dict()) != canonical_hash(expected_result.to_dict()):
            raise ReceiptError(
                f"receipt stage {spec.name} result disagrees with embedded bound evidence"
            )
    return result


def _current_config_compatibility(
    raw_config: object,
) -> tuple[LabConfig | None, str]:
    try:
        config = LabConfig.model_validate(raw_config)
    except Exception:
        return None, "CURRENT_MODEL_INCOMPATIBLE"
    if canonical_hash(config.model_dump(mode="json")) != canonical_hash(raw_config):
        return None, "CURRENT_CANONICAL_ROUNDTRIP_INCOMPATIBLE"
    return config, "CURRENT_COMPATIBLE"


def _current_provenance_compatibility(
    raw_provenance: object,
    *,
    historical_source_identity: Mapping[str, object],
) -> str:
    try:
        provenance = Provenance.from_dict(raw_provenance)
        roundtrip_matches = canonical_hash(provenance.to_dict()) == canonical_hash(raw_provenance)
        source_matches = canonical_hash(provenance.source_identity) == canonical_hash(
            historical_source_identity
        )
    except Exception:
        return "CURRENT_MODEL_INCOMPATIBLE"
    if not roundtrip_matches:
        return "CURRENT_CANONICAL_ROUNDTRIP_INCOMPATIBLE"
    if not source_matches:
        return "CURRENT_SOURCE_IDENTITY_INCOMPATIBLE"
    return "CURRENT_COMPATIBLE"


def _registry_compatibility(
    registry: StageRegistry | None,
    *,
    config: LabConfig | None,
    targets: Sequence[str],
    embedded_contract_hash: str,
    source_compatibility: str,
) -> tuple[str, str, str | None, Mapping[str, StageSpec]]:
    if registry is None:
        return "NOT_CHECKED", "SKIPPED", "no current registry requested", {}
    if config is None:
        return (
            "NOT_CHECKED_CONFIG_INCOMPATIBLE",
            "SKIPPED",
            "embedded config cannot canonical-roundtrip through the current model",
            {},
        )
    try:
        current_contract = _registry_contract_payload(registry, config, targets)
        current_hash = current_contract["sha256"]
        current_specs = {stage.name: stage for stage in registry.topological_order(targets)}
    except (PipelineError, TypeError, ValueError):
        return (
            "DRIFTED",
            "SKIPPED",
            "current registry cannot reconstruct the embedded target closure",
            {},
        )
    if current_hash != embedded_contract_hash:
        return (
            "DRIFTED",
            "SKIPPED",
            "embedded registry contract differs from the current registry",
            {},
        )
    if source_compatibility == "NOT_CHECKED":
        return (
            "MATCHED",
            "SKIPPED",
            "current source identity was not supplied for semantic replay",
            {},
        )
    if source_compatibility != "MATCHED":
        return (
            "MATCHED",
            "SKIPPED",
            "current source identity differs from the receipt source identity",
            {},
        )
    return "MATCHED", "PASSED", None, current_specs


def _attempt_event_payload(event: StageAttemptEvent) -> dict[str, JSONLike]:
    return {
        "attempt": event.attempt,
        "created_at": event.created_at,
        "event_kind": event.event_kind,
        "event_sequence": event.sequence,
        "message": event.message,
        "metadata": thaw_json(event.metadata),
        "owner_token": event.owner_token,
        "state": event.state.value,
    }


def _attempt_history_status(attempts: int, history: Sequence[StageAttemptEvent]) -> str:
    if attempts > 0 and not history:
        return "UNAVAILABLE_PRE_V3"
    return "COMPLETE"


def _verify_attempt_history(
    raw: object,
    *,
    history_status: object,
    attempts: int,
    final_state: str,
    final_message: object,
    final_metadata: object,
    label: str,
) -> None:
    if not isinstance(raw, list):
        raise ReceiptError(f"{label} attempt_history must be an array")
    if history_status not in _ATTEMPT_HISTORY_STATUSES:
        raise ReceiptError(f"{label} attempt_history_status is invalid")
    if history_status == "UNAVAILABLE_PRE_V3":
        if attempts < 1 or raw:
            raise ReceiptError(f"{label} unavailable legacy history is inconsistent")
        return
    if attempts == 0:
        if raw:
            raise ReceiptError(f"{label} zero-attempt record has attempt events")
        return
    if not raw:
        raise ReceiptError(f"{label} has attempts but no immutable attempt history")
    keys = {
        "attempt",
        "created_at",
        "event_kind",
        "event_sequence",
        "message",
        "metadata",
        "owner_token",
        "state",
    }
    active_attempt: int | None = None
    latest_attempt = 0
    final_event: dict[str, object] | None = None
    for ordinal, raw_event in enumerate(raw, start=1):
        event = _exact_object(raw_event, keys, label=f"{label} attempt event {ordinal}")
        attempt = event["attempt"]
        event_kind = event["event_kind"]
        state = event["state"]
        owner_token = event["owner_token"]
        message = event["message"]
        metadata = event["metadata"]
        created_at = event["created_at"]
        if (
            event["event_sequence"] != ordinal
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt < 1
            or event_kind not in _ATTEMPT_EVENT_KINDS
            or not isinstance(owner_token, str)
            or not owner_token
            or (message is not None and (not isinstance(message, str) or not message.strip()))
            or not isinstance(metadata, dict)
            or not isinstance(created_at, str)
        ):
            raise ReceiptError(f"{label} attempt event {ordinal} is invalid")
        try:
            timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ReceiptError(f"{label} attempt event {ordinal} timestamp is invalid") from error
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ReceiptError(f"{label} attempt event {ordinal} timestamp lacks a timezone")
        if event_kind == "CLAIMED":
            if (
                state != StageState.RUNNING.value
                or active_attempt is not None
                or attempt != latest_attempt + 1
                or message is not None
            ):
                raise ReceiptError(f"{label} claim event sequence is inconsistent")
            active_attempt = attempt
            latest_attempt = attempt
        elif event_kind == "RECLAIMED":
            if (
                state != StageState.RUNNING.value
                or active_attempt is None
                or attempt != latest_attempt + 1
                or not isinstance(message, str)
            ):
                raise ReceiptError(f"{label} reclaim event sequence is inconsistent")
            active_attempt = attempt
            latest_attempt = attempt
        else:
            if (
                state
                not in {
                    StageState.COMPLETED.value,
                    StageState.FAILED.value,
                    StageState.BLOCKED.value,
                }
                or active_attempt != attempt
            ):
                raise ReceiptError(f"{label} finish event sequence is inconsistent")
            active_attempt = None
        final_event = event
    assert final_event is not None
    if (
        active_attempt is not None
        or latest_attempt != attempts
        or final_event["event_kind"] != "FINISHED"
        or final_event["state"] != final_state
        or final_event["message"] != final_message
        or canonical_hash(final_event["metadata"]) != canonical_hash(final_metadata)
    ):
        raise ReceiptError(f"{label} final attempt event disagrees with its stage record")


def _provenance_reference(
    store: ArtifactStore,
    artifact_id: str,
    *,
    expected_run_id: str,
) -> ArtifactRef:
    try:
        reference = store.get(artifact_id, verify=True)
    except Exception as error:
        raise ReceiptError(
            f"provenance artifact {artifact_id} is unavailable or corrupt"
        ) from error
    if reference.manifest.kind != "run/provenance":
        raise ReceiptError(f"provenance artifact {artifact_id} has the wrong kind")
    if reference.manifest.metadata.get("run_id") != expected_run_id:
        raise ReceiptError(f"provenance artifact {artifact_id} belongs to another run")
    if set(reference.manifest.metadata) != {
        "config_hash",
        "run_id",
        "source_identity",
    }:
        raise ReceiptError(f"provenance artifact {artifact_id} has invalid metadata keys")
    files = {entry.path: entry for entry in reference.manifest.files}
    if set(files) != {"config.json", "provenance.json"} or any(
        entry.media_type != "application/json" for entry in files.values()
    ):
        raise ReceiptError(f"provenance artifact {artifact_id} has invalid payload inventory")
    try:
        saved_config = store.read_json(artifact_id, "config.json")
        saved_provenance = store.read_json(artifact_id, "provenance.json")
    except Exception as error:
        raise ReceiptError(f"provenance artifact {artifact_id} has invalid JSON") from error
    _representation_version(saved_config, key="schema_version", label="saved config")
    _representation_version(
        saved_provenance,
        key="provenance_schema_version",
        label="saved provenance",
    )
    config_hash = _digest(
        reference.manifest.metadata.get("config_hash"),
        label=f"provenance artifact {artifact_id} config_hash",
    )
    source_identity = reference.manifest.metadata.get("source_identity")
    if not isinstance(source_identity, Mapping) or canonical_hash(saved_config) != config_hash:
        raise ReceiptError(f"provenance artifact {artifact_id} has inconsistent identity")
    return reference


def _gate_payload(store: ArtifactStore, reference: ArtifactRef) -> dict[str, JSONValue] | None:
    paths = {entry.path for entry in reference.manifest.files}
    if "gate.json" not in paths:
        return None
    try:
        raw = store.read_json(reference.artifact_id, "gate.json")
        result = GateResult.from_dict(raw)
    except Exception as error:
        raise ReceiptError(f"gate artifact {reference.artifact_id} is invalid") from error
    metadata_checks = {
        "gate_id": result.gate_id,
        "gate_status": result.status.value,
        "natural_status": result.natural_status.value,
    }
    inconsistent = [
        name
        for name, expected in metadata_checks.items()
        if name in reference.manifest.metadata and reference.manifest.metadata.get(name) != expected
    ]
    if inconsistent:
        raise ReceiptError(
            f"gate artifact {reference.artifact_id} has inconsistent metadata: "
            + ", ".join(inconsistent)
        )
    return result.to_dict()


def _override_lineage(
    reference: ArtifactRef,
    gate: Mapping[str, JSONLike] | None,
) -> list[JSONValue]:
    raw: object
    if gate is not None:
        raw = gate.get("override_lineage")
    else:
        raw = thaw_json(reference.manifest.metadata.get("inherited_gate_overrides", ()))
    if not isinstance(raw, list):
        raise ReceiptError("stage override lineage must be a JSON array")
    return raw  # type: ignore[return-value]


def _inherited_gate_overrides(
    store: ArtifactStore,
    dependencies: Mapping[str, ArtifactRef],
) -> list[JSONLike]:
    lineage: list[JSONLike] = []
    seen: set[str] = set()

    def add(entries: object, *, label: str) -> None:
        if not isinstance(entries, (list, tuple)):
            raise ReceiptError(f"{label} gate override lineage must be an array")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ReceiptError(f"{label} gate override lineage contains a non-object")
            reason = entry.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ReceiptError(f"{label} gate override lineage has no reason")
            digest = canonical_hash(entry)
            if digest not in seen:
                lineage.append(entry)  # type: ignore[arg-type]
                seen.add(digest)

    for name, reference in sorted(dependencies.items()):
        add(reference.manifest.metadata.get("inherited_gate_overrides", ()), label=name)
        if (
            name.startswith("gate.")
            and reference.manifest.metadata.get("gate_status") == "OVERRIDDEN"
        ):
            payload = store.read_json(reference.artifact_id, "gate.json")
            if not isinstance(payload, Mapping):
                raise ReceiptError(f"{name} gate payload must be an object")
            add(payload.get("override_lineage"), label=name)
    return lineage


def _snapshot_stage(
    store: ArtifactStore,
    record: StageRecord,
    *,
    attempt_history: Sequence[StageAttemptEvent],
    ordinal: int,
    current_provenance_id: str,
) -> dict[str, JSONLike]:
    if record.state is not StageState.COMPLETED or record.artifact_id is None:
        raise ReceiptError(f"receipt stage {record.stage_name} is not complete")
    try:
        reference = store.get(record.artifact_id, verify=True)
    except Exception as error:
        raise ReceiptError(
            f"stage artifact {record.artifact_id} is unavailable or corrupt"
        ) from error
    metadata = reference.manifest.metadata
    if metadata.get("stage") != record.stage_name:
        raise ReceiptError(f"stage artifact {record.artifact_id} has the wrong stage identity")
    if metadata.get("stage_signature") != record.stage_signature:
        raise ReceiptError(f"stage artifact {record.artifact_id} has the wrong signature")
    upstream = thaw_json(metadata.get("upstream_artifacts"))
    if not isinstance(upstream, dict) or not all(
        isinstance(name, str) and isinstance(artifact_id, str)
        for name, artifact_id in upstream.items()
    ):
        raise ReceiptError(f"stage artifact {record.artifact_id} has invalid upstream lineage")
    record_metadata = thaw_json(record.metadata)
    if not isinstance(record_metadata, dict):
        raise ReceiptError(f"stage record {record.stage_name} has invalid metadata")
    cache_hit = record_metadata.get("cache_hit")
    producer_run_id = record_metadata.get("producer_run_id")
    producer_provenance_id = record_metadata.get("producer_provenance_artifact_id")
    if not isinstance(cache_hit, bool):
        raise ReceiptError(f"stage record {record.stage_name} has no cache decision")
    if not isinstance(producer_run_id, str):
        raise ReceiptError(f"stage record {record.stage_name} has no cache producer run")
    producer_provenance_id = _digest(
        producer_provenance_id,
        label=f"stage {record.stage_name} producer provenance",
    )
    if record_metadata.get("run_provenance_artifact_id") != current_provenance_id:
        raise ReceiptError(f"stage record {record.stage_name} has wrong run provenance")
    _provenance_reference(
        store,
        producer_provenance_id,
        expected_run_id=producer_run_id,
    )
    gate = _gate_payload(store, reference)
    return {
        "artifact_id": reference.artifact_id,
        "artifact_kind": reference.manifest.kind,
        "attempt_history": [_attempt_event_payload(event) for event in attempt_history],
        "attempt_history_status": _attempt_history_status(record.attempts, attempt_history),
        "attempts": record.attempts,
        "cache": {
            "hit": cache_hit,
            "producer_provenance_artifact_id": producer_provenance_id,
            "producer_run_id": producer_run_id,
        },
        "gate": gate,
        "message": record.message,
        "ordinal": ordinal,
        "override_authorization": thaw_json(metadata.get("override_authorization")),
        "override_lineage": _override_lineage(reference, gate),
        "record_metadata": record_metadata,
        "stage_name": record.stage_name,
        "stage_signature": record.stage_signature,
        "state": record.state.value,
        "upstream_artifacts": upstream,
    }


def _snapshot_auxiliary_stage(
    store: ArtifactStore,
    record: StageRecord,
    *,
    attempt_history: Sequence[StageAttemptEvent],
    ordinal: int,
) -> dict[str, JSONLike]:
    if record.state is not StageState.COMPLETED or record.artifact_id is None:
        raise ReceiptError(f"auxiliary stage {record.stage_name} is not complete")
    try:
        reference = store.get(record.artifact_id, verify=True)
    except Exception as error:
        raise ReceiptError(
            f"auxiliary artifact {record.artifact_id} is unavailable or corrupt"
        ) from error
    record_metadata = thaw_json(record.metadata)
    if not isinstance(record_metadata, dict):
        raise ReceiptError(f"auxiliary stage {record.stage_name} has invalid metadata")
    parent_stage = record_metadata.get("parent_stage")
    cache_hit = record_metadata.get("cache_hit")
    producer_run_id = record_metadata.get("producer_run_id")
    producer_provenance_id = record_metadata.get("producer_provenance_artifact_id")
    record_provenance_id = _digest(
        record_metadata.get("run_provenance_artifact_id"),
        label=f"auxiliary stage {record.stage_name} run provenance",
    )
    if not isinstance(parent_stage, str) or not isinstance(cache_hit, bool):
        raise ReceiptError(f"auxiliary stage {record.stage_name} has invalid parent/cache data")
    if not isinstance(producer_run_id, str):
        raise ReceiptError(f"auxiliary stage {record.stage_name} has no producer run")
    producer_provenance_id = _digest(
        producer_provenance_id,
        label=f"auxiliary stage {record.stage_name} producer provenance",
    )
    _provenance_reference(store, record_provenance_id, expected_run_id=record.run_id)
    _provenance_reference(
        store,
        producer_provenance_id,
        expected_run_id=producer_run_id,
    )
    return {
        "artifact_id": reference.artifact_id,
        "artifact_kind": reference.manifest.kind,
        "attempt_history": [_attempt_event_payload(event) for event in attempt_history],
        "attempt_history_status": _attempt_history_status(record.attempts, attempt_history),
        "attempts": record.attempts,
        "cache": {
            "hit": cache_hit,
            "producer_provenance_artifact_id": producer_provenance_id,
            "producer_run_id": producer_run_id,
        },
        "message": record.message,
        "ordinal": ordinal,
        "parent_stage": parent_stage,
        "record_run_id": record.run_id,
        "record_metadata": record_metadata,
        "row_name": record.stage_name,
        "stage_signature": record.stage_signature,
        "state": record.state.value,
    }


def _verify_auxiliary_stages(
    store: ArtifactStore,
    raw_records: object,
    *,
    main_stage_entries: Mapping[str, Mapping[str, object]],
    source_identity: Mapping[str, JSONLike],
) -> tuple[
    tuple[ArtifactRef, ...],
    dict[str, list[tuple[ShardSpec, ArtifactRef]]],
]:
    if not isinstance(raw_records, list):
        raise ReceiptError("receipt auxiliary_stage_records must be an array")
    keys = {
        "artifact_id",
        "artifact_kind",
        "attempt_history",
        "attempt_history_status",
        "attempts",
        "cache",
        "message",
        "ordinal",
        "parent_stage",
        "record_metadata",
        "record_run_id",
        "row_name",
        "stage_signature",
        "state",
    }
    artifacts: list[ArtifactRef] = []
    by_parent: dict[str, list[tuple[ShardSpec, ArtifactRef]]] = {}
    seen_rows: set[str] = set()
    seen_artifacts: set[str] = set()
    parent_order = {name: ordinal for ordinal, name in enumerate(main_stage_entries)}
    observed_sort_keys: list[tuple[int, str]] = []
    for ordinal, raw_record in enumerate(raw_records):
        record = _exact_object(raw_record, keys, label=f"auxiliary stage {ordinal}")
        row_name = record["row_name"]
        parent_stage = record["parent_stage"]
        record_run_id = record["record_run_id"]
        if (
            not isinstance(row_name, str)
            or not row_name
            or row_name in seen_rows
            or not isinstance(parent_stage, str)
            or parent_stage not in main_stage_entries
            or not isinstance(record_run_id, str)
            or not record_run_id
        ):
            raise ReceiptError("auxiliary stage row/parent identity is invalid")
        if record["ordinal"] != ordinal or record["state"] != StageState.COMPLETED.value:
            raise ReceiptError(f"auxiliary stage {row_name} has invalid order or state")
        attempts = record["attempts"]
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ReceiptError(f"auxiliary stage {row_name} has invalid attempts")
        if record["message"] is not None and not isinstance(record["message"], str):
            raise ReceiptError(f"auxiliary stage {row_name} has invalid message")
        signature = _digest(
            record["stage_signature"], label=f"auxiliary stage {row_name} signature"
        )
        artifact_id = _digest(
            record["artifact_id"], label=f"auxiliary stage {row_name} artifact_id"
        )
        if artifact_id in seen_artifacts:
            raise ReceiptError("auxiliary stage artifacts must be unique")
        seen_rows.add(row_name)
        seen_artifacts.add(artifact_id)
        observed_sort_keys.append((parent_order[parent_stage], row_name))
        try:
            reference = store.get(artifact_id, verify=True)
        except Exception as error:
            raise ReceiptError(
                f"auxiliary stage artifact {artifact_id} is unavailable or corrupt"
            ) from error
        if reference.manifest.kind != record["artifact_kind"]:
            raise ReceiptError(f"auxiliary stage {row_name} has inconsistent artifact kind")
        metadata = record["record_metadata"]
        if not isinstance(metadata, dict):
            raise ReceiptError(f"auxiliary stage {row_name} metadata must be an object")
        cache = _exact_object(
            record["cache"],
            {"hit", "producer_provenance_artifact_id", "producer_run_id"},
            label=f"auxiliary stage {row_name} cache",
        )
        producer_run_id = cache["producer_run_id"]
        if not isinstance(cache["hit"], bool) or not isinstance(producer_run_id, str):
            raise ReceiptError(f"auxiliary stage {row_name} cache lineage is invalid")
        producer_provenance_id = _digest(
            cache["producer_provenance_artifact_id"],
            label=f"auxiliary stage {row_name} producer provenance",
        )
        record_provenance_id = _digest(
            metadata.get("run_provenance_artifact_id"),
            label=f"auxiliary stage {row_name} run provenance",
        )
        if (
            metadata.get("cache_hit") != cache["hit"]
            or metadata.get("producer_run_id") != producer_run_id
            or metadata.get("producer_provenance_artifact_id") != producer_provenance_id
        ):
            raise ReceiptError(f"auxiliary stage {row_name} cache record is inconsistent")
        _provenance_reference(
            store,
            record_provenance_id,
            expected_run_id=record_run_id,
        )
        _provenance_reference(
            store,
            producer_provenance_id,
            expected_run_id=producer_run_id,
        )
        _verify_attempt_history(
            record["attempt_history"],
            history_status=record["attempt_history_status"],
            attempts=attempts,
            final_state=record["state"],  # type: ignore[arg-type]
            final_message=record["message"],
            final_metadata=metadata,
            label=f"auxiliary stage {row_name}",
        )
        try:
            shard_key = ShardKey(metadata["shard_key"])  # type: ignore[arg-type]
            spec = ShardSpec(
                parent_stage=metadata["parent_stage"],  # type: ignore[arg-type]
                parent_stage_signature=metadata["parent_stage_signature"],  # type: ignore[arg-type]
                key=shard_key,
                artifact_kind=record["artifact_kind"],  # type: ignore[arg-type]
                stage_config=metadata["stage_config"],  # type: ignore[arg-type]
                source_identity=metadata["source_identity"],  # type: ignore[arg-type]
                upstream_artifacts=metadata["upstream_artifacts"],  # type: ignore[arg-type]
                shard_version=metadata["shard_version"],  # type: ignore[arg-type]
            )
            validate_shard_artifact(reference, spec)
        except (KeyError, ShardValidationError, TypeError, ValueError) as error:
            raise ReceiptError(f"auxiliary stage {row_name} has invalid shard identity") from error
        parent_signature = main_stage_entries[parent_stage]["stage_signature"]
        record_identity_mismatches = [
            name
            for name, expected in spec.artifact_metadata.items()
            if canonical_hash(metadata.get(name)) != canonical_hash(expected)
        ]
        if (
            spec.row_name != row_name
            or spec.signature != signature
            or spec.parent_stage_signature != parent_signature
            or canonical_hash(spec.source_identity) != canonical_hash(source_identity)
            or record_identity_mismatches
        ):
            raise ReceiptError(f"auxiliary stage {row_name} shard lineage is inconsistent")
        artifacts.append(reference)
        by_parent.setdefault(parent_stage, []).append((spec, reference))
    if observed_sort_keys != sorted(observed_sort_keys):
        raise ReceiptError("auxiliary stage records are not in canonical order")
    return tuple(artifacts), by_parent


def publish_run_receipt(
    store: ArtifactStore,
    index: RunIndex,
    *,
    run_id: str,
    config: LabConfig,
    source_identity: Mapping[str, JSONLike],
    requested_targets: Sequence[str],
    ordered_stage_names: Sequence[str],
    registry: StageRegistry = DEFAULT_STAGES,
    validation_context: StageValidationContext | None = None,
) -> ArtifactRef:
    """Publish and append one successful run snapshot, then verify it from artifacts."""

    targets = tuple(requested_targets)
    stage_names = tuple(ordered_stage_names)
    if not targets or not stage_names:
        raise ReceiptError("a receipt requires targets and an ordered stage closure")
    if len(targets) != len(set(targets)):
        raise ReceiptError("receipt requested targets must be unique")
    if len(stage_names) != len(set(stage_names)):
        raise ReceiptError("ordered receipt stages must be unique")
    try:
        expected_stage_names = tuple(stage.name for stage in registry.topological_order(targets))
    except PipelineError as error:
        raise ReceiptError("receipt requested targets are not registered") from error
    if stage_names != expected_stage_names:
        raise ReceiptError("ordered receipt stages do not equal the requested target closure")
    run = index.get_run(run_id)
    if run is None or run.provenance_artifact_id is None:
        raise ReceiptError(f"run {run_id!r} has no immutable provenance")
    config_value = config.model_dump(mode="json")
    if canonical_hash(config_value) != run.config_hash:
        raise ReceiptError(f"run {run_id!r} configuration does not match its index identity")
    registered_source = run.metadata.get("source_identity")
    if canonical_hash(registered_source) != canonical_hash(source_identity):
        raise ReceiptError(f"run {run_id!r} source identity does not match its index identity")
    provenance = _provenance_reference(
        store,
        run.provenance_artifact_id,
        expected_run_id=run_id,
    )
    saved_config = store.read_json(provenance.artifact_id, "config.json")
    saved_provenance = store.read_json(provenance.artifact_id, "provenance.json")
    if canonical_hash(saved_config) != run.config_hash:
        raise ReceiptError(f"run {run_id!r} provenance contains a different configuration")
    if not isinstance(saved_provenance, dict):
        raise ReceiptError(f"run {run_id!r} provenance payload must be an object")
    current_config, config_compatibility = _current_config_compatibility(saved_config)
    provenance_compatibility = _current_provenance_compatibility(
        saved_provenance,
        historical_source_identity=source_identity,
    )
    if (
        current_config is None
        or config_compatibility != "CURRENT_COMPATIBLE"
        or provenance_compatibility != "CURRENT_COMPATIBLE"
    ):
        raise ReceiptError(f"run {run_id!r} cannot publish from non-current provenance")

    registry_contract = _registry_contract_payload(registry, config, targets)
    registry_contract_hash = _digest(registry_contract["sha256"], label="registry contract sha256")
    stages: list[dict[str, JSONLike]] = []
    main_records: dict[str, StageRecord] = {}
    for ordinal, stage_name in enumerate(stage_names):
        record = index.get_stage(run_id, stage_name)
        if record is None:
            raise ReceiptError(f"run {run_id!r} has no stage record for {stage_name}")
        main_records[stage_name] = record
        stages.append(
            _snapshot_stage(
                store,
                record,
                attempt_history=index.list_stage_attempts(run_id, stage_name),
                ordinal=ordinal,
                current_provenance_id=provenance.artifact_id,
            )
        )
    stage_name_set = set(stage_names)
    parent_ordinals = {name: ordinal for ordinal, name in enumerate(stage_names)}
    auxiliary_records = [
        record
        for record in index.list_stages(run_id)
        if record.stage_name not in stage_name_set
        and record.metadata.get("parent_stage") in stage_name_set
    ]
    auxiliary_by_artifact: dict[str, StageRecord] = {}
    for record in auxiliary_records:
        if record.artifact_id is not None:
            auxiliary_by_artifact.setdefault(record.artifact_id, record)
    auxiliary_records = list(auxiliary_by_artifact.values())
    auxiliary_records.sort(
        key=lambda record: (
            parent_ordinals[str(record.metadata.get("parent_stage"))],
            record.stage_name,
        )
    )
    auxiliary_stages = [
        _snapshot_auxiliary_stage(
            store,
            record,
            attempt_history=index.list_stage_attempts(record.run_id, record.stage_name),
            ordinal=ordinal,
        )
        for ordinal, record in enumerate(auxiliary_records)
    ]

    payload: dict[str, JSONLike] = {
        "auxiliary_stage_records": auxiliary_stages,
        "config": {
            "artifact_id": provenance.artifact_id,
            "payload_path": "config.json",
            "sha256": run.config_hash,
            "value": config_value,
        },
        "provenance": {
            "artifact_id": provenance.artifact_id,
            "value": saved_provenance,
        },
        "registry_contract": registry_contract,
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "requested_targets": list(targets),
        "run_id": run_id,
        "source_identity": dict(source_identity),
        "stages": stages,
    }
    reference = store.put_json(
        payload,
        filename="receipt.json",
        kind="run/receipt",
        metadata={
            "config_hash": run.config_hash,
            "auxiliary_stage_count": len(auxiliary_stages),
            "provenance_artifact_id": provenance.artifact_id,
            "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
            "registry_contract_hash": registry_contract_hash,
            "run_id": run_id,
            "stage_count": len(stages),
        },
    )
    verify_run_receipt(
        store,
        reference.artifact_id,
        registry=registry,
        current_source_identity=source_identity,
        require_current_semantics=True,
        validation_context=validation_context,
    )
    index.append_receipt(
        run_id,
        artifact_id=reference.artifact_id,
        requested_targets=targets,
    )
    return reference


def verify_run_receipt(
    store: ArtifactStore,
    artifact_id: str,
    *,
    registry: StageRegistry | None = DEFAULT_STAGES,
    current_source_identity: Mapping[str, JSONLike] | None = None,
    require_current_semantics: bool = False,
    validation_context: StageValidationContext | None = None,
) -> VerifiedRunReceipt:
    """Verify immutable history, optionally requiring current-model compatibility.

    Integrity and the embedded historical contract are always verified first.
    Current config/provenance models, a supplied registry, and an explicitly
    recaptured ``current_source_identity`` form a compatibility layer whose
    status is returned separately. ``require_current_semantics`` upgrades any
    skipped, failed, or incompatible current layer to a verification failure.
    """

    semantic_context = (
        StageValidationContext() if validation_context is None else validation_context
    )
    try:
        reference = store.get(artifact_id, verify=True)
    except Exception as error:
        raise ReceiptError(f"receipt artifact {artifact_id} is unavailable or corrupt") from error
    if reference.manifest.kind != "run/receipt":
        raise ReceiptError(f"artifact {artifact_id} is not a run receipt")
    if {entry.path for entry in reference.manifest.files} != {"receipt.json"}:
        raise ReceiptError("run receipt must contain exactly receipt.json")
    try:
        raw = store.read_json(reference.artifact_id, "receipt.json")
    except Exception as error:
        raise ReceiptError("run receipt payload is invalid JSON") from error
    payload = _exact_object(
        raw,
        {
            "auxiliary_stage_records",
            "config",
            "provenance",
            "receipt_schema_version",
            "registry_contract",
            "requested_targets",
            "run_id",
            "source_identity",
            "stages",
        },
        label="run receipt",
    )
    if (
        type(payload["receipt_schema_version"]) is not int
        or payload["receipt_schema_version"] != RECEIPT_SCHEMA_VERSION
    ):
        raise ReceiptError("unsupported run receipt schema")
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise ReceiptError("run receipt run_id must be a non-empty string")
    source_identity = payload["source_identity"]
    if not isinstance(source_identity, dict):
        raise ReceiptError("run receipt source_identity must be an object")
    targets = payload["requested_targets"]
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(target, str) and target for target in targets)
    ):
        raise ReceiptError("run receipt requested_targets must be a non-empty string array")
    if len(targets) != len(set(targets)):
        raise ReceiptError("run receipt requested_targets must be unique")

    config = _exact_object(
        payload["config"],
        {"artifact_id", "payload_path", "sha256", "value"},
        label="receipt config",
    )
    config_hash = _digest(config["sha256"], label="receipt config sha256")
    config_artifact_id = _digest(config["artifact_id"], label="receipt config artifact_id")
    if config["payload_path"] != "config.json":
        raise ReceiptError("receipt config payload_path must be config.json")
    config_schema_version = _representation_version(
        config["value"],
        key="schema_version",
        label="receipt embedded config",
    )
    if canonical_hash(config["value"]) != config_hash:
        raise ReceiptError("run receipt embedded configuration hash does not match")
    provenance = _exact_object(
        payload["provenance"], {"artifact_id", "value"}, label="receipt provenance"
    )
    provenance_id = _digest(provenance["artifact_id"], label="receipt provenance artifact_id")
    if config_artifact_id != provenance_id:
        raise ReceiptError("receipt config and provenance must share the registered artifact")
    provenance_reference = _provenance_reference(
        store,
        provenance_id,
        expected_run_id=run_id,
    )
    provenance_metadata = provenance_reference.manifest.metadata
    expected_provenance_metadata: dict[str, object] = {
        "config_hash": config_hash,
        "run_id": run_id,
        "source_identity": source_identity,
    }
    if any(
        canonical_hash(provenance_metadata.get(name)) != canonical_hash(value)
        for name, value in expected_provenance_metadata.items()
    ):
        raise ReceiptError("run receipt provenance metadata is inconsistent")
    saved_config = store.read_json(provenance_id, "config.json")
    saved_provenance = store.read_json(provenance_id, "provenance.json")
    if canonical_hash(saved_config) != config_hash or canonical_hash(
        saved_config
    ) != canonical_hash(config["value"]):
        raise ReceiptError("run receipt configuration disagrees with provenance")
    if canonical_hash(saved_provenance) != canonical_hash(provenance["value"]):
        raise ReceiptError("run receipt embedded provenance disagrees with its artifact")
    provenance_schema_version = _representation_version(
        provenance["value"],
        key="provenance_schema_version",
        label="receipt embedded provenance",
    )
    validated_config, config_compatibility = _current_config_compatibility(config["value"])
    provenance_compatibility = _current_provenance_compatibility(
        provenance["value"],
        historical_source_identity=source_identity,
    )
    source_compatibility = (
        "NOT_CHECKED"
        if current_source_identity is None
        else (
            "MATCHED"
            if canonical_hash(current_source_identity) == canonical_hash(source_identity)
            else "INCOMPATIBLE"
        )
    )

    raw_stages = payload["stages"]
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ReceiptError("run receipt stages must be a non-empty array")
    stage_names = tuple(
        stage.get("stage_name") if isinstance(stage, dict) else None for stage in raw_stages
    )
    if not all(isinstance(name, str) and name for name in stage_names):
        raise ReceiptError("run receipt stage names are invalid")
    embedded_contracts, registry_contract_hash = _parse_registry_contract(
        payload["registry_contract"],
        config=config["value"],  # type: ignore[arg-type]
        requested_targets=targets,
        stage_names=stage_names,  # type: ignore[arg-type]
    )
    (
        registry_compatibility,
        semantic_validation,
        semantic_validation_reason,
        compatible_specs,
    ) = _registry_compatibility(
        registry,
        config=validated_config,
        targets=targets,
        embedded_contract_hash=registry_contract_hash,
        source_compatibility=source_compatibility,
    )
    contracts_by_name = {contract.spec.name: contract for contract in embedded_contracts}
    stage_artifacts: list[ArtifactRef] = []
    prior_artifacts: dict[str, str] = {}
    prior_references: dict[str, ArtifactRef] = {}
    seen_artifacts: set[str] = set()
    stage_keys = {
        "artifact_id",
        "artifact_kind",
        "attempt_history",
        "attempt_history_status",
        "attempts",
        "cache",
        "gate",
        "message",
        "ordinal",
        "override_authorization",
        "override_lineage",
        "record_metadata",
        "stage_name",
        "stage_signature",
        "state",
        "upstream_artifacts",
    }
    for ordinal, raw_stage in enumerate(raw_stages):
        stage = _exact_object(raw_stage, stage_keys, label=f"receipt stage {ordinal}")
        stage_name = stage["stage_name"]
        if not isinstance(stage_name, str) or not stage_name or stage_name in prior_artifacts:
            raise ReceiptError("receipt stage names must be non-empty and unique")
        if stage["ordinal"] != ordinal or stage["state"] != StageState.COMPLETED.value:
            raise ReceiptError(f"receipt stage {stage_name} has invalid order or state")
        attempts = stage["attempts"]
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
            raise ReceiptError(f"receipt stage {stage_name} has invalid attempts")
        if stage["message"] is not None and not isinstance(stage["message"], str):
            raise ReceiptError(f"receipt stage {stage_name} has invalid message")
        recorded_signature = _digest(
            stage["stage_signature"], label=f"receipt stage {stage_name} signature"
        )
        stage_artifact_id = _digest(
            stage["artifact_id"], label=f"receipt stage {stage_name} artifact_id"
        )
        if stage_artifact_id in seen_artifacts:
            raise ReceiptError("receipt stage artifacts must be unique")
        seen_artifacts.add(stage_artifact_id)
        try:
            stage_reference = store.get(stage_artifact_id, verify=True)
        except Exception as error:
            raise ReceiptError(
                f"receipt stage artifact {stage_artifact_id} is unavailable or corrupt"
            ) from error
        if stage_reference.manifest.kind != stage["artifact_kind"]:
            raise ReceiptError(f"receipt stage {stage_name} artifact kind is inconsistent")
        artifact_metadata = stage_reference.manifest.metadata
        upstream = stage["upstream_artifacts"]
        if not isinstance(upstream, dict) or canonical_hash(
            artifact_metadata.get("upstream_artifacts")
        ) != canonical_hash(upstream):
            raise ReceiptError(f"receipt stage {stage_name} upstream lineage is inconsistent")
        for dependency, dependency_id in upstream.items():
            if not isinstance(dependency, str) or prior_artifacts.get(dependency) != dependency_id:
                raise ReceiptError(
                    f"receipt stage {stage_name} references a missing or later dependency"
                )
        contract = contracts_by_name[stage_name]
        stage_spec = contract.spec
        expected_signature = _embedded_stage_signature(
            contract,
            upstream_artifact_ids=upstream,  # type: ignore[arg-type]
            source_identity=source_identity,  # type: ignore[arg-type]
        )
        dependencies = {name: prior_references[name] for name in upstream}
        expected_identity: dict[str, JSONLike] = {
            "inherited_gate_overrides": _inherited_gate_overrides(store, dependencies),
            "source_identity": source_identity,  # type: ignore[dict-item]
            "stage": stage_name,
            "stage_config": dict(contract.selected_stage_config),
            "stage_signature": expected_signature,
            "stage_version": stage_spec.stage_version,
            "upstream_artifacts": upstream,  # type: ignore[dict-item]
        }
        identity_mismatches = [
            name
            for name, expected in expected_identity.items()
            if canonical_hash(artifact_metadata.get(name)) != canonical_hash(expected)
        ]
        if recorded_signature != expected_signature:
            identity_mismatches.append("recorded stage_signature")
        if identity_mismatches:
            raise ReceiptError(
                f"receipt stage {stage_name} artifact identity is inconsistent: "
                + ", ".join(identity_mismatches)
            )
        validated_gate = _validate_embedded_output(
            store,
            stage_reference,
            contract,
            dependencies,
        )
        compatible_spec = compatible_specs.get(stage_name)
        if compatible_spec is not None and semantic_validation == "PASSED":
            assert registry is not None
            try:
                compatibility_gate = validate_stage_output(
                    stage_reference,
                    compatible_spec,
                    store,
                    gate_rule=contract.gate_rule,
                    gate_override_authorization=contract.gate_override_authorization,
                    config=validated_config,
                    registry=registry,
                    source_identity=current_source_identity,
                    validation_context=semantic_context,
                )
            except PipelineError as error:
                semantic_validation = "FAILED"
                semantic_validation_reason = (
                    f"current semantic replay failed for {stage_name}: {error}"
                )
            else:
                if canonical_hash(
                    None if compatibility_gate is None else compatibility_gate.to_dict()
                ) != canonical_hash(None if validated_gate is None else validated_gate.to_dict()):
                    semantic_validation = "FAILED"
                    semantic_validation_reason = (
                        f"current semantic replay disagrees with embedded gate {stage_name}"
                    )

        record_metadata = stage["record_metadata"]
        cache = _exact_object(
            stage["cache"],
            {"hit", "producer_provenance_artifact_id", "producer_run_id"},
            label=f"receipt stage {stage_name} cache",
        )
        if not isinstance(record_metadata, dict) or not isinstance(cache["hit"], bool):
            raise ReceiptError(f"receipt stage {stage_name} cache lineage is invalid")
        producer_run_id = cache["producer_run_id"]
        if not isinstance(producer_run_id, str) or not producer_run_id:
            raise ReceiptError(f"receipt stage {stage_name} producer run is invalid")
        producer_provenance_id = _digest(
            cache["producer_provenance_artifact_id"],
            label=f"receipt stage {stage_name} producer provenance",
        )
        if (
            record_metadata.get("cache_hit") != cache["hit"]
            or record_metadata.get("producer_run_id") != producer_run_id
            or record_metadata.get("producer_provenance_artifact_id") != producer_provenance_id
            or record_metadata.get("run_provenance_artifact_id") != provenance_id
        ):
            raise ReceiptError(f"receipt stage {stage_name} record cache lineage is inconsistent")
        _verify_attempt_history(
            stage["attempt_history"],
            history_status=stage["attempt_history_status"],
            attempts=attempts,
            final_state=stage["state"],  # type: ignore[arg-type]
            final_message=stage["message"],
            final_metadata=record_metadata,
            label=f"receipt stage {stage_name}",
        )
        _provenance_reference(
            store,
            producer_provenance_id,
            expected_run_id=producer_run_id,
        )

        actual_gate = _gate_payload(store, stage_reference)
        if validated_gate is not None and canonical_hash(
            validated_gate.to_dict()
        ) != canonical_hash(actual_gate):
            raise ReceiptError(f"receipt stage {stage_name} gate rule is inconsistent")
        if canonical_hash(actual_gate) != canonical_hash(stage["gate"]):
            raise ReceiptError(f"receipt stage {stage_name} gate result is inconsistent")
        expected_overrides = _override_lineage(stage_reference, actual_gate)
        if canonical_hash(expected_overrides) != canonical_hash(stage["override_lineage"]):
            raise ReceiptError(f"receipt stage {stage_name} override lineage is inconsistent")
        if canonical_hash(stage_reference.manifest.metadata.get("override_authorization")) != (
            canonical_hash(stage["override_authorization"])
        ):
            raise ReceiptError(f"receipt stage {stage_name} override authorization is inconsistent")
        prior_artifacts[stage_name] = stage_artifact_id
        prior_references[stage_name] = stage_reference
        stage_artifacts.append(stage_reference)

    main_stage_entries = {
        entry["stage_name"]: entry
        for entry in raw_stages
        if isinstance(entry, dict) and isinstance(entry.get("stage_name"), str)
    }
    auxiliary_artifacts, _ = _verify_auxiliary_stages(
        store,
        payload["auxiliary_stage_records"],
        main_stage_entries=main_stage_entries,
        source_identity=source_identity,  # type: ignore[arg-type]
    )
    referenced_auxiliary = list(auxiliary_artifacts)
    expected_receipt_metadata: dict[str, object] = {
        "auxiliary_stage_count": len(auxiliary_artifacts),
        "config_hash": config_hash,
        "provenance_artifact_id": provenance_id,
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "registry_contract_hash": registry_contract_hash,
        "run_id": run_id,
        "stage_count": len(stage_artifacts),
    }
    if set(reference.manifest.metadata) != set(expected_receipt_metadata) or any(
        canonical_hash(reference.manifest.metadata.get(name)) != canonical_hash(value)
        for name, value in expected_receipt_metadata.items()
    ):
        raise ReceiptError("run receipt manifest metadata is inconsistent")
    frozen_payload = thaw_json(payload)  # detach from the JSON decoder before returning
    assert isinstance(frozen_payload, dict)
    if require_current_semantics and (
        semantic_validation != "PASSED"
        or config_compatibility != "CURRENT_COMPATIBLE"
        or provenance_compatibility != "CURRENT_COMPATIBLE"
    ):
        reason = semantic_validation_reason or (
            "embedded config/provenance is incompatible with current models"
        )
        raise ReceiptError(f"current semantic compatibility required: {reason}")
    return VerifiedRunReceipt(
        reference=reference,
        payload=frozen_payload,
        provenance_reference=provenance_reference,
        stage_artifacts=tuple(stage_artifacts),
        auxiliary_artifacts=tuple(referenced_auxiliary),
        integrity_validation="PASSED",
        config_schema_version=config_schema_version,
        provenance_schema_version=provenance_schema_version,
        config_compatibility=config_compatibility,
        provenance_compatibility=provenance_compatibility,
        source_compatibility=source_compatibility,
        registry_compatibility=registry_compatibility,
        semantic_validation=semantic_validation,
        semantic_validation_reason=semantic_validation_reason,
    )


__all__ = [
    "RECEIPT_SCHEMA_VERSION",
    "REGISTRY_CONTRACT_SCHEMA_VERSION",
    "ReceiptError",
    "VerifiedRunReceipt",
    "publish_run_receipt",
    "verify_run_receipt",
]
