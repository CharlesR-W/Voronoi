from __future__ import annotations

import sqlite3

import pytest

import voronoi_lab.pipeline as pipeline_module
import voronoi_lab.receipts as receipts_module
from voronoi_lab.config import GateOverrideAuthorization, LabConfig
from voronoi_lab.core import (
    GateCheck,
    GateEvaluator,
    GateRule,
    Provenance,
    ProvenanceError,
    RunIndex,
    StageState,
    canonical_hash,
    canonical_json_bytes,
    thaw_json,
)
from voronoi_lab.execution import ExecutionError, ExperimentRunner, artifact_metadata
from voronoi_lab.pipeline import (
    DEFAULT_STAGES,
    ImplementationStatus,
    PipelineError,
    StageRegistry,
    StageSpec,
    StageValidationContext,
    expected_gate_rule,
    stage_signature,
    validate_stage_output,
)
from voronoi_lab.receipts import ReceiptError, verify_run_receipt
from voronoi_lab.stage_handlers import handle_exp2_exact, handle_gate_synthetic_exact


def _registry() -> StageRegistry:
    return StageRegistry(
        (
            StageSpec(
                "first",
                "first fixture stage",
                implementation=ImplementationStatus.RUNNABLE,
            ),
            StageSpec(
                "second",
                "second fixture stage",
                dependencies=("first",),
                implementation=ImplementationStatus.RUNNABLE,
            ),
        )
    )


def _handlers(calls: list[str]):
    def first(context, dependencies):
        calls.append("first")
        return context.store.put_json(
            {"value": 1},
            kind="fixture/first",
            metadata=artifact_metadata(context, dependencies),
        )

    def second(context, dependencies):
        calls.append("second")
        return context.store.put_json(
            {"value": 2},
            kind="fixture/second",
            metadata=artifact_metadata(context, dependencies),
        )

    return {"first": first, "second": second}


def test_receipt_verifies_after_index_loss_and_detects_referenced_tampering(tmp_path) -> None:
    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=_registry(),
        handlers=_handlers([]),
        run_id="durable",
    )
    stage = runner.run(["first"])["first"]
    receipt_id = runner.receipt_artifact_id
    assert receipt_id is not None

    runner.index.path.unlink()
    verified = verify_run_receipt(runner.store, receipt_id, registry=runner.registry)
    assert verified.payload["run_id"] == "durable"
    assert verified.payload["requested_targets"] == ["first"]
    assert [entry["stage_name"] for entry in verified.payload["stages"]] == ["first"]

    payload_path = stage.payload_path("data.json")
    payload_path.chmod(0o600)
    payload_path.write_text('{"value":999}', encoding="utf-8")
    with pytest.raises(ReceiptError, match="unavailable or corrupt"):
        verify_run_receipt(runner.store, receipt_id, registry=runner.registry)


def test_repeat_and_incremental_runs_append_deterministic_receipts(tmp_path) -> None:
    calls: list[str] = []
    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=_registry(),
        handlers=_handlers(calls),
        run_id="incremental",
    )

    runner.run(["first"])
    first_receipt = runner.receipt_artifact_id
    runner.run(["first"])
    repeated_receipt = runner.receipt_artifact_id
    runner.run(["second"])
    incremental_receipt = runner.receipt_artifact_id

    assert first_receipt == repeated_receipt
    assert incremental_receipt != first_receipt
    assert calls == ["first", "second"]
    records = runner.index.list_receipts("incremental")
    assert [record.sequence for record in records] == [1, 2, 3]
    assert [record.artifact_id for record in records] == [
        first_receipt,
        repeated_receipt,
        incremental_receipt,
    ]
    incremental = verify_run_receipt(runner.store, incremental_receipt, registry=runner.registry)
    assert incremental.payload["requested_targets"] == ["second"]
    assert [entry["stage_name"] for entry in incremental.payload["stages"]] == [
        "first",
        "second",
    ]

    with pytest.raises(ExecutionError, match="must be unique"):
        runner.run(["first", "first"])
    assert len(runner.index.list_receipts("incremental")) == 3


def test_receipt_preserves_cross_run_cache_producer_lineage(tmp_path) -> None:
    calls: list[str] = []
    left = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=_registry(),
        handlers=_handlers(calls),
        run_id="producer",
    )
    left.run(["first"])
    left_run = left.index.get_run("producer")
    assert left_run is not None

    right = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=_registry(),
        handlers=_handlers(calls),
        run_id="consumer",
    )
    right.run(["first"])
    assert calls == ["first"]
    receipt_id = right.receipt_artifact_id
    assert receipt_id is not None
    verified = verify_run_receipt(right.store, receipt_id, registry=right.registry)
    stage = verified.payload["stages"][0]
    assert stage["cache"] == {
        "hit": True,
        "producer_provenance_artifact_id": left_run.provenance_artifact_id,
        "producer_run_id": "producer",
    }


def test_receipt_embeds_gate_result_and_override_lineage(tmp_path) -> None:
    registry = StageRegistry(
        (
            StageSpec(
                "gate.fixture",
                "fixture gate",
                implementation=ImplementationStatus.RUNNABLE,
            ),
        )
    )

    def gate_handler(context, dependencies):
        result = GateEvaluator().evaluate(
            GateRule(
                gate_id="fixture",
                checks=(GateCheck("required", "required", "is_true"),),
            ),
            {"required": False},
            override_reason="receipt test authorization",
        )
        return context.store.put_json(
            result.to_dict(),
            filename="gate.json",
            kind="gate/fixture",
            metadata={
                **artifact_metadata(context, dependencies),
                "gate_id": result.gate_id,
                "gate_status": result.status.value,
                "natural_status": result.natural_status.value,
            },
        )

    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers={"gate.fixture": gate_handler},
        run_id="gate-receipt",
    )
    runner.run(["gate.fixture"])
    assert runner.receipt_artifact_id is not None
    stage = verify_run_receipt(
        runner.store,
        runner.receipt_artifact_id,
        registry=runner.registry,
    ).payload["stages"][0]
    assert stage["gate"]["status"] == "OVERRIDDEN"
    assert stage["gate"]["natural_status"] == "FAIL"
    assert stage["override_lineage"][0]["reason"] == "receipt test authorization"


def test_exact_receipt_verifies_shard_records_and_reducer(tmp_path) -> None:
    raw = LabConfig().model_dump(mode="json")
    raw["runtime"]["workers"] = 2
    raw["experiment2"]["exact_instances"] = 2
    raw["experiment2"]["oracle_factor_sizes"] = [2, 2]
    raw["experiment2"]["train_primitives"] = 4
    raw["experiment2"]["heldout_primitives"] = 2
    raw["experiment2"]["exact_protocol"]["max_states"] = 4
    raw["gates"]["synthetic"]["noiseless_instances"] = 2
    config = LabConfig.model_validate(raw)
    runner = ExperimentRunner(
        config,
        project_root=tmp_path,
        registry=DEFAULT_STAGES,
        handlers={"exp2.exact": handle_exp2_exact},
        run_id="exact-receipt",
    )

    runner.run(["exp2.exact"])

    assert runner.receipt_artifact_id is not None
    verified = verify_run_receipt(runner.store, runner.receipt_artifact_id)
    auxiliary = verified.payload["auxiliary_stage_records"]
    assert len(auxiliary) == 2
    assert all(record["parent_stage"] == "exp2.exact" for record in auxiliary)
    assert len(verified.auxiliary_artifacts) == 3  # two shards plus reducer manifest


def test_runner_gate_and_receipt_replay_exact_once_per_validation_chain(
    tmp_path,
    monkeypatch,
) -> None:
    raw = LabConfig().model_dump(mode="json")
    raw["runtime"]["workers"] = 0
    raw["experiment2"]["exact_instances"] = 1
    raw["experiment2"]["oracle_factor_sizes"] = [2, 2]
    raw["experiment2"]["train_primitives"] = 4
    raw["experiment2"]["heldout_primitives"] = 2
    raw["experiment2"]["exact_protocol"]["max_states"] = 4
    raw["gates"]["synthetic"]["noiseless_instances"] = 1
    config = LabConfig.model_validate(raw)
    calls = 0
    original = pipeline_module._validate_exact_payload

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "_validate_exact_payload", counted)
    runner = ExperimentRunner(
        config,
        project_root=tmp_path,
        registry=DEFAULT_STAGES,
        handlers={
            "exp2.exact": handle_exp2_exact,
            "gate.synthetic_exact": handle_gate_synthetic_exact,
        },
        run_id="exact-validation-memo",
    )
    outputs = runner.run(["gate.synthetic_exact"])
    assert calls == 1

    context = StageValidationContext()
    exact = outputs["exp2.exact"]
    gate = outputs["gate.synthetic_exact"]
    validate_stage_output(
        exact,
        DEFAULT_STAGES.get("exp2.exact"),
        runner.store,
        config=config,
        registry=DEFAULT_STAGES,
        source_identity=runner.source_identity,
        validation_context=context,
    )
    validate_stage_output(
        gate,
        DEFAULT_STAGES.get("gate.synthetic_exact"),
        runner.store,
        config=config,
        registry=DEFAULT_STAGES,
        source_identity=runner.source_identity,
        validation_context=context,
    )
    assert calls == 2

    exact_payload = runner.store.read_json(exact.artifact_id, "exact.json")
    shard_id = exact_payload["ordered_instance_artifact_ids"][0]
    shard = runner.store.get(shard_id)
    shard_path = shard.payload_path("instance.json")
    shard_path.chmod(0o600)
    shard_path.write_bytes(shard_path.read_bytes() + b"\n")
    with pytest.raises(PipelineError, match="memoized validation closure"):
        validate_stage_output(
            gate,
            DEFAULT_STAGES.get("gate.synthetic_exact"),
            runner.store,
            config=config,
            registry=DEFAULT_STAGES,
            source_identity=runner.source_identity,
            validation_context=context,
        )
    assert calls == 2


def test_receipt_binds_contracted_gate_override_authorization(tmp_path) -> None:
    authorization = GateOverrideAuthorization(
        target_gate="mechanical",
        reason="receipt test diagnostic continuation",
        authorized_by="receipt-test",
        recorded_at="2026-08-17T12:00:00-07:00",
    )
    base = LabConfig()
    overrides = base.gates.overrides.model_copy(update={"mechanical": authorization})
    config = base.model_copy(
        update={"gates": base.gates.model_copy(update={"overrides": overrides})}
    )
    registry = StageRegistry(
        (
            StageSpec(
                "gate.mechanical",
                "contracted fixture gate",
                config_paths=("gates.mechanical", "gates.overrides.mechanical"),
                implementation=ImplementationStatus.RUNNABLE,
                expected_artifact_kind="gate/mechanical",
                required_payload_paths=("gate.json",),
                result_schema_version=1,
                gate_payload_path="gate.json",
                expected_gate_id="mechanical",
            ),
        )
    )

    def gate_handler(context, dependencies):
        rule = expected_gate_rule(context.stage_spec, context.config)
        result = GateEvaluator().evaluate(
            rule,
            {},
            override_reason=authorization.reason,
        )
        return context.store.put_json(
            result.to_dict(),
            filename="gate.json",
            kind="gate/mechanical",
            metadata={
                **artifact_metadata(context, dependencies),
                "gate_id": result.gate_id,
                "gate_rule_signature": canonical_hash(rule.to_dict()),
                "gate_status": result.status.value,
                "natural_status": result.natural_status.value,
                "override_authorization": authorization.model_dump(mode="json"),
                "result_schema_version": 1,
            },
        )

    runner = ExperimentRunner(
        config,
        project_root=tmp_path,
        registry=registry,
        handlers={"gate.mechanical": gate_handler},
        run_id="authorized-receipt",
    )
    runner.run(["gate.mechanical"])
    assert runner.receipt_artifact_id is not None
    verified = verify_run_receipt(
        runner.store,
        runner.receipt_artifact_id,
        registry=registry,
    )
    gate = verified.payload["stages"][0]["gate"]
    assert gate["status"] == "OVERRIDDEN"
    assert gate["override_lineage"][0]["reason"] == authorization.reason
    assert verified.payload["stages"][0]["override_authorization"] == (
        authorization.model_dump(mode="json")
    )

    wrong_rule = GateRule(
        gate_id="mechanical",
        checks=(GateCheck("self_asserted", "self_asserted", "is_true"),),
    )
    wrong_result = GateEvaluator().evaluate(wrong_rule, {"self_asserted": True})
    original_stage = verified.payload["stages"][0]
    original_reference = verified.stage_artifacts[0]
    wrong_metadata = thaw_json(original_reference.manifest.metadata)
    wrong_metadata.update(
        {
            "gate_rule_signature": canonical_hash(wrong_rule.to_dict()),
            "gate_status": wrong_result.status.value,
            "natural_status": wrong_result.natural_status.value,
        }
    )
    wrong_gate = runner.store.put_json(
        wrong_result.to_dict(),
        filename="gate.json",
        kind="gate/mechanical",
        metadata=wrong_metadata,
    )
    forged_payload = runner.store.read_json(verified.artifact_id, "receipt.json")
    forged_payload["stages"][0]["artifact_id"] = wrong_gate.artifact_id
    forged_payload["stages"][0]["gate"] = wrong_result.to_dict()
    forged_receipt = runner.store.put_json(
        forged_payload,
        filename="receipt.json",
        kind="run/receipt",
        metadata=thaw_json(verified.reference.manifest.metadata),
    )
    assert original_stage["artifact_id"] != wrong_gate.artifact_id
    with pytest.raises(ReceiptError, match="trusted output contract"):
        verify_run_receipt(runner.store, forged_receipt.artifact_id, registry=registry)


def test_receipt_preserves_reclaim_history_after_retry_and_index_loss(tmp_path) -> None:
    registry = _registry()
    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers=_handlers([]),
        run_id="recovered",
    )
    signature = stage_signature(
        registry.get("first"),
        runner.config,
        upstream_artifact_ids={},
        source_identity=runner.source_identity,
    )
    runner.index.claim_stage(
        "recovered",
        "first",
        signature,
        owner_token="dead-worker",
        metadata={"phase": "before-crash"},
    )
    runner.index.reclaim_stage(
        "recovered",
        "first",
        owner_token="recovery-worker",
        reason="worker host was terminated",
    )
    runner.index.finish_stage(
        "recovered",
        "first",
        owner_token="recovery-worker",
        state=StageState.FAILED,
        message="recovery audit complete",
        metadata={"phase": "recovered"},
    )

    runner.run(["first"])

    assert runner.receipt_artifact_id is not None
    receipt_id = runner.receipt_artifact_id
    history = runner.index.list_stage_attempts("recovered", "first")
    assert [event.event_kind for event in history] == [
        "CLAIMED",
        "RECLAIMED",
        "FINISHED",
        "CLAIMED",
        "FINISHED",
    ]
    assert history[1].message == "reclaimed: worker host was terminated"
    runner.index.path.unlink()
    verified = verify_run_receipt(runner.store, receipt_id, registry=registry)
    embedded = verified.payload["stages"][0]["attempt_history"]
    assert embedded[1]["message"] == "reclaimed: worker host was terminated"
    assert embedded[-1]["state"] == "COMPLETED"
    forged = runner.store.read_json(receipt_id, "receipt.json")
    forged["stages"][0]["attempt_history"][1]["event_sequence"] = 99
    forged_receipt = runner.store.put_json(
        forged,
        filename="receipt.json",
        kind="run/receipt",
        metadata=thaw_json(verified.reference.manifest.metadata),
    )
    with pytest.raises(ReceiptError, match="attempt event"):
        verify_run_receipt(runner.store, forged_receipt.artifact_id, registry=registry)


def test_receipt_marks_unrecoverable_pre_v3_attempt_history(tmp_path) -> None:
    registry = _registry()
    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers=_handlers([]),
        run_id="legacy-attempt-history",
    )
    runner.run(["first"])
    with sqlite3.connect(runner.index.path) as connection:
        connection.execute("DROP TABLE stage_attempt_events")
        connection.execute("PRAGMA user_version = 2")
    runner.index = RunIndex(runner.index.path)

    runner.run(["first"])

    assert runner.receipt_artifact_id is not None
    verified = verify_run_receipt(
        runner.store,
        runner.receipt_artifact_id,
        registry=registry,
    )
    stage = verified.payload["stages"][0]
    assert stage["attempts"] == 1
    assert stage["attempt_history"] == []
    assert stage["attempt_history_status"] == "UNAVAILABLE_PRE_V3"


def test_receipt_uses_embedded_contract_across_registry_evolution(tmp_path) -> None:
    original = _registry()
    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=original,
        handlers=_handlers([]),
        run_id="historical-contract",
    )
    runner.run(["second"])
    assert runner.receipt_artifact_id is not None

    description_only = StageRegistry(
        (
            StageSpec(
                "first",
                "new prose that is deliberately not scientific identity",
                implementation=ImplementationStatus.RUNNABLE,
            ),
            StageSpec(
                "second",
                "new prose",
                dependencies=("first",),
                implementation=ImplementationStatus.RUNNABLE,
            ),
        )
    )
    description_verified = verify_run_receipt(
        runner.store,
        runner.receipt_artifact_id,
        registry=description_only,
        current_source_identity=runner.source_identity,
    )
    assert description_verified.registry_compatibility == "MATCHED"
    assert description_verified.semantic_validation == "PASSED"

    evolved = StageRegistry(
        (
            StageSpec(
                "first",
                "first v2",
                implementation=ImplementationStatus.RUNNABLE,
                stage_version=2,
            ),
            StageSpec(
                "second",
                "second no longer depends on first",
                implementation=ImplementationStatus.RUNNABLE,
            ),
        )
    )
    historical = verify_run_receipt(
        runner.store,
        runner.receipt_artifact_id,
        registry=evolved,
    )
    assert historical.registry_compatibility == "DRIFTED"
    assert historical.semantic_validation == "SKIPPED"
    assert [stage["name"] for stage in historical.payload["registry_contract"]["stages"]] == [
        "first",
        "second",
    ]

    forged = runner.store.read_json(runner.receipt_artifact_id, "receipt.json")
    forged["registry_contract"]["stages"][0]["stage_version"] = 99
    forged_receipt = runner.store.put_json(
        forged,
        filename="receipt.json",
        kind="run/receipt",
        metadata=thaw_json(historical.reference.manifest.metadata),
    )
    with pytest.raises(ReceiptError, match="registry-contract hash"):
        verify_run_receipt(runner.store, forged_receipt.artifact_id, registry=evolved)


def test_receipt_integrity_survives_current_config_and_provenance_model_evolution(
    tmp_path,
) -> None:
    registry = _registry()
    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers=_handlers([]),
        run_id="model-evolution-producer",
    )
    runner.run(["first"])
    assert runner.receipt_artifact_id is not None
    original = runner.store.read_json(runner.receipt_artifact_id, "receipt.json")
    original_provenance_id = original["provenance"]["artifact_id"]

    historical_config = thaw_json(original["config"]["value"])
    historical_config.pop("report")
    historical_provenance = thaw_json(original["provenance"]["value"])
    historical_provenance["provenance_schema_version"] = 2
    historical_provenance["historical_capture_policy"] = "provenance-v2"
    current_roundtrip = LabConfig.model_validate(historical_config).model_dump(mode="json")
    assert canonical_hash(current_roundtrip) != canonical_hash(historical_config)
    with pytest.raises(ProvenanceError):
        Provenance.from_dict(historical_provenance)

    historical_run_id = "historical-model-representation"
    historical_config_hash = canonical_hash(historical_config)
    historical_provenance_reference = runner.store.put_files(
        {
            "config.json": canonical_json_bytes(historical_config),
            "provenance.json": canonical_json_bytes(historical_provenance),
        },
        kind="run/provenance",
        metadata={
            "config_hash": historical_config_hash,
            "run_id": historical_run_id,
            "source_identity": original["source_identity"],
        },
        media_types={
            "config.json": "application/json",
            "provenance.json": "application/json",
        },
    )

    historical_receipt = thaw_json(original)
    historical_receipt["run_id"] = historical_run_id
    historical_receipt["config"] = {
        "artifact_id": historical_provenance_reference.artifact_id,
        "payload_path": "config.json",
        "sha256": historical_config_hash,
        "value": historical_config,
    }
    historical_receipt["provenance"] = {
        "artifact_id": historical_provenance_reference.artifact_id,
        "value": historical_provenance,
    }
    stage = historical_receipt["stages"][0]
    stage["cache"]["hit"] = True
    stage["cache"]["producer_provenance_artifact_id"] = original_provenance_id
    stage["cache"]["producer_run_id"] = "model-evolution-producer"
    stage_metadata = stage["record_metadata"]
    stage_metadata["cache_hit"] = True
    stage_metadata["producer_provenance_artifact_id"] = original_provenance_id
    stage_metadata["producer_run_id"] = "model-evolution-producer"
    stage_metadata["run_provenance_artifact_id"] = historical_provenance_reference.artifact_id
    stage["attempt_history"][-1]["metadata"] = thaw_json(stage_metadata)
    historical_reference = runner.store.put_json(
        historical_receipt,
        filename="receipt.json",
        kind="run/receipt",
        metadata={
            **thaw_json(runner.store.get(runner.receipt_artifact_id).manifest.metadata),
            "config_hash": historical_config_hash,
            "provenance_artifact_id": historical_provenance_reference.artifact_id,
            "run_id": historical_run_id,
        },
    )

    verified = verify_run_receipt(
        runner.store,
        historical_reference.artifact_id,
        registry=registry,
    )
    assert verified.integrity_validation == "PASSED"
    assert verified.config_schema_version == 1
    assert verified.provenance_schema_version == 2
    assert verified.config_compatibility == "CURRENT_CANONICAL_ROUNDTRIP_INCOMPATIBLE"
    assert verified.provenance_compatibility == "CURRENT_MODEL_INCOMPATIBLE"
    assert verified.registry_compatibility == "NOT_CHECKED_CONFIG_INCOMPATIBLE"
    assert verified.semantic_validation == "SKIPPED"
    assert "cannot canonical-roundtrip" in verified.semantic_validation_reason
    with pytest.raises(ReceiptError, match="current semantic compatibility required"):
        verify_run_receipt(
            runner.store,
            historical_reference.artifact_id,
            registry=registry,
            require_current_semantics=True,
        )

    tampered = runner.store.read_json(historical_reference.artifact_id, "receipt.json")
    tampered["config"]["value"]["protocol"]["root_seed"] += 1
    tampered_reference = runner.store.put_json(
        tampered,
        filename="receipt.json",
        kind="run/receipt",
        metadata=thaw_json(historical_reference.manifest.metadata),
    )
    with pytest.raises(ReceiptError, match="configuration hash"):
        verify_run_receipt(runner.store, tampered_reference.artifact_id, registry=registry)


def test_current_source_or_dependency_drift_does_not_invalidate_receipt_integrity(
    tmp_path,
    monkeypatch,
) -> None:
    registry = _registry()
    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers=_handlers([]),
        run_id="semantic-replay-drift",
    )
    runner.run(["first"])
    assert runner.receipt_artifact_id is not None

    source_drift = verify_run_receipt(
        runner.store,
        runner.receipt_artifact_id,
        registry=registry,
        current_source_identity={"environment": "different"},
    )
    assert source_drift.integrity_validation == "PASSED"
    assert source_drift.source_compatibility == "INCOMPATIBLE"
    assert source_drift.semantic_validation == "SKIPPED"
    assert "differs" in source_drift.semantic_validation_reason

    def fail_current_replay(*_args, **_kwargs):
        raise receipts_module.PipelineError("dependency-version replay drift")

    monkeypatch.setattr(receipts_module, "validate_stage_output", fail_current_replay)
    dependency_drift = verify_run_receipt(
        runner.store,
        runner.receipt_artifact_id,
        registry=registry,
        current_source_identity=runner.source_identity,
    )
    assert dependency_drift.integrity_validation == "PASSED"
    assert dependency_drift.source_compatibility == "MATCHED"
    assert dependency_drift.semantic_validation == "FAILED"
    assert "dependency-version replay drift" in dependency_drift.semantic_validation_reason
    with pytest.raises(ReceiptError, match="current semantic compatibility required"):
        verify_run_receipt(
            runner.store,
            runner.receipt_artifact_id,
            registry=registry,
            current_source_identity=runner.source_identity,
            require_current_semantics=True,
        )


def test_receipt_verifies_registry_contract_v1_after_contract_schema_expansion(
    tmp_path,
) -> None:
    registry = _registry()
    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers=_handlers([]),
        run_id="registry-contract-v1",
    )
    result = runner.run(["first"])["first"]
    assert runner.receipt_artifact_id is not None
    receipt = runner.store.read_json(runner.receipt_artifact_id, "receipt.json")
    contract = receipt["registry_contract"]
    contract["schema_version"] = 1
    output_contract = contract["stages"][0]["output_contract"]
    output_contract.pop("gate_evidence_dependency")
    output_contract.pop("gate_evidence_payload_path")
    contract["sha256"] = canonical_hash({"schema_version": 1, "stages": contract["stages"]})
    old_signature = canonical_hash(
        {
            "config": contract["stages"][0]["selected_stage_config"],
            "output_contract": output_contract,
            "source": receipt["source_identity"],
            "stage": "first",
            "stage_version": 1,
            "upstream_artifacts": {},
        }
    )
    old_metadata = thaw_json(result.manifest.metadata)
    old_metadata["stage_signature"] = old_signature
    old_result = runner.store.put_json(
        runner.store.read_json(result.artifact_id, "data.json"),
        kind=result.manifest.kind,
        metadata=old_metadata,
    )
    receipt["stages"][0]["artifact_id"] = old_result.artifact_id
    receipt["stages"][0]["stage_signature"] = old_signature
    old_receipt = runner.store.put_json(
        receipt,
        filename="receipt.json",
        kind="run/receipt",
        metadata={
            **thaw_json(runner.receipt_reference.manifest.metadata),
            "registry_contract_hash": contract["sha256"],
        },
    )

    verified = verify_run_receipt(
        runner.store,
        old_receipt.artifact_id,
        registry=registry,
    )
    assert verified.integrity_validation == "PASSED"
    assert verified.registry_compatibility == "DRIFTED"
    assert verified.semantic_validation == "SKIPPED"


def test_cached_exact_receipt_recovers_producer_shard_closure_without_index(tmp_path) -> None:
    raw = LabConfig().model_dump(mode="json")
    raw["runtime"]["workers"] = 0
    raw["experiment2"]["exact_instances"] = 2
    raw["experiment2"]["oracle_factor_sizes"] = [2, 2]
    raw["experiment2"]["train_primitives"] = 4
    raw["experiment2"]["heldout_primitives"] = 2
    raw["experiment2"]["exact_protocol"]["max_states"] = 4
    raw["gates"]["synthetic"]["noiseless_instances"] = 2
    config = LabConfig.model_validate(raw)
    calls: list[str] = []

    def exact(context, dependencies):
        calls.append(context.run_id)
        return handle_exp2_exact(context, dependencies)

    producer = ExperimentRunner(
        config,
        project_root=tmp_path,
        registry=DEFAULT_STAGES,
        handlers={"exp2.exact": exact},
        run_id="exact-producer",
    )
    producer.run(["exp2.exact"])
    producer_run = producer.index.get_run("exact-producer")
    assert producer_run is not None
    consumer = ExperimentRunner(
        config,
        project_root=tmp_path,
        registry=DEFAULT_STAGES,
        handlers={"exp2.exact": exact},
        run_id="exact-consumer",
    )
    consumer.run(["exp2.exact"])

    assert calls == ["exact-producer"]
    assert [stage.stage_name for stage in consumer.index.list_stages("exact-consumer")] == [
        "exp2.exact"
    ]
    assert consumer.receipt_artifact_id is not None
    receipt_id = consumer.receipt_artifact_id
    verified = verify_run_receipt(consumer.store, receipt_id)
    auxiliary = verified.payload["auxiliary_stage_records"]
    assert len(auxiliary) == 2
    assert {record["record_run_id"] for record in auxiliary} == {"exact-producer"}
    assert {record["record_metadata"]["run_provenance_artifact_id"] for record in auxiliary} == {
        producer_run.provenance_artifact_id
    }
    consumer.index.path.unlink()
    verified_without_index = verify_run_receipt(consumer.store, receipt_id)
    assert len(verified_without_index.auxiliary_artifacts) == 3
