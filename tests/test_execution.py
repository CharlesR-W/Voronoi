from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Event

import pytest

import voronoi_lab.execution as execution_module
from voronoi_lab.config import LabConfig
from voronoi_lab.core import (
    GateCheck,
    GateEvaluator,
    GateRule,
    StageClaimError,
    StageState,
    canonical_hash,
    canonical_json_bytes,
)
from voronoi_lab.execution import (
    ExecutionError,
    ExperimentRunner,
    artifact_metadata,
    load_verified_run_identity,
)
from voronoi_lab.pipeline import (
    ImplementationStatus,
    StageRegistry,
    StageSpec,
    expected_gate_rule,
    stage_signature,
)


def _fixture_registry() -> StageRegistry:
    return StageRegistry(
        (
            StageSpec(
                "fixture",
                "fixture stage",
                config_paths=("protocol.root_seed", "experiment2.oracle_factor_sizes"),
                implementation=ImplementationStatus.RUNNABLE,
            ),
        )
    )


def _gated_registry() -> StageRegistry:
    return StageRegistry(
        (
            StageSpec(
                "gate.fixture",
                "fixture gate",
                implementation=ImplementationStatus.RUNNABLE,
            ),
            StageSpec(
                "after-gate",
                "must obey fixture gate",
                dependencies=("gate.fixture",),
                implementation=ImplementationStatus.RUNNABLE,
            ),
            StageSpec(
                "after-after",
                "must retain inherited gate lineage",
                dependencies=("after-gate",),
                implementation=ImplementationStatus.RUNNABLE,
            ),
        )
    )


def _register_forged_provenance_run(
    runner: ExperimentRunner,
    *,
    run_id: str,
    provenance_payload: object,
    artifact_source_identity=None,
) -> None:
    config_payload = runner.config.model_dump(mode="json")
    config_hash = canonical_hash(config_payload)
    source_identity = (
        runner.source_identity if artifact_source_identity is None else artifact_source_identity
    )
    reference = runner.store.put_files(
        {
            "config.json": canonical_json_bytes(config_payload),
            "provenance.json": canonical_json_bytes(provenance_payload),
        },
        kind="run/provenance",
        metadata={
            "config_hash": config_hash,
            "run_id": run_id,
            "source_identity": source_identity,
        },
        media_types={
            "config.json": "application/json",
            "provenance.json": "application/json",
        },
    )
    runner.index.register_run(
        run_id,
        config_hash=config_hash,
        provenance_artifact_id=reference.artifact_id,
        metadata={
            "mode": runner.config.protocol.mode,
            "source_identity": runner.source_identity,
        },
    )


def _gate_handler(*, override_reason: str | None = None):
    def handler(context, dependencies):
        rule = GateRule(
            gate_id="fixture",
            checks=(GateCheck("required", "required", "is_true"),),
        )
        result = GateEvaluator().evaluate(
            rule,
            {"required": False},
            override_reason=override_reason,
        )
        return context.store.put_json(
            result.to_dict(),
            filename="gate.json",
            kind="gate/fixture",
            metadata={
                **artifact_metadata(context, dependencies),
                "gate_status": result.status.value,
            },
        )

    return handler


def test_runner_reuses_verified_content_cache_across_run_ids(tmp_path) -> None:
    registry = _fixture_registry()
    calls = 0

    def handler(context, dependencies):
        nonlocal calls
        calls += 1
        return context.store.put_json(
            {"value": 7},
            kind="fixture",
            metadata=artifact_metadata(context, dependencies),
        )

    config = LabConfig()
    left = ExperimentRunner(
        config,
        project_root=tmp_path,
        registry=registry,
        handlers={"fixture": handler},
        run_id="left",
    )
    first = left.run(["fixture"])["fixture"]
    second = left.run(["fixture"])["fixture"]
    assert first.artifact_id == second.artifact_id
    assert calls == 1

    right = ExperimentRunner(
        config,
        project_root=tmp_path,
        registry=registry,
        handlers={"fixture": handler},
        run_id="right",
    )
    third = right.run(["fixture"])["fixture"]
    assert third.artifact_id == first.artifact_id
    assert calls == 1
    assert third.manifest.metadata["stage"] == "fixture"
    assert "run_id" not in third.manifest.metadata
    assert "producer_run_id" not in third.manifest.metadata
    consumed = right.index.get_stage("right", "fixture")
    assert consumed is not None
    assert consumed.metadata["cache_hit"] is True
    assert consumed.metadata["producer_run_id"] == "left"
    right_run = right.index.get_run("right")
    assert right_run is not None
    assert consumed.metadata["run_provenance_artifact_id"] == right_run.provenance_artifact_id


def test_stage_preflight_runs_before_cross_run_cache_consumption(tmp_path) -> None:
    registry = _fixture_registry()
    external_state = {"digest": "expected"}
    calls = 0

    def handler(context, dependencies):
        nonlocal calls
        calls += 1
        return context.store.put_json(
            {"value": 29},
            kind="fixture",
            metadata=artifact_metadata(context, dependencies),
        )

    def preflight(_context):
        if external_state["digest"] != "expected":
            raise ValueError("external input digest changed")

    handler.__voronoi_stage_preflight__ = preflight  # type: ignore[attr-defined]
    producer = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers={"fixture": handler},
        run_id="preflight-producer",
    )
    producer.run(["fixture"])
    assert calls == 1

    external_state["digest"] = "changed"
    consumer = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers={"fixture": handler},
        run_id="preflight-consumer",
    )
    with pytest.raises(ExecutionError, match="external input digest changed"):
        consumer.run(["fixture"])
    assert calls == 1
    assert consumer.index.get_stage("preflight-consumer", "fixture") is None


def test_runner_validates_gate_evidence_through_its_custom_registry(tmp_path) -> None:
    evidence_stage = StageSpec(
        "custom.mechanical_evidence",
        "custom registry evidence producer",
        implementation=ImplementationStatus.RUNNABLE,
        expected_artifact_kind="stage/custom-mechanical-evidence",
        required_payload_paths=("observations.json",),
        result_schema_version=1,
    )
    gate_stage = StageSpec(
        "gate.mechanical",
        "custom-registry mechanical gate",
        dependencies=(evidence_stage.name,),
        config_paths=("gates.mechanical", "gates.overrides.mechanical"),
        implementation=ImplementationStatus.RUNNABLE,
        expected_artifact_kind="gate/custom-mechanical",
        required_payload_paths=("gate.json",),
        result_schema_version=1,
        payload_schema_id="gate-result-v1",
        gate_payload_path="gate.json",
        expected_gate_id="mechanical",
        gate_evidence_dependency=evidence_stage.name,
        gate_evidence_payload_path="observations.json",
    )
    registry = StageRegistry((evidence_stage, gate_stage))

    def evidence_handler(context, dependencies):
        assert not dependencies
        observations = {
            "schema_version": 1,
            "probe_banks": {
                "artifact_valid": True,
                "deterministic": True,
                "distinct_train_test_sources": True,
            },
            "geometry": {
                "roundtrip": {"relative_rms_error": 0.0},
                "centroid_reconstruction": {"relative_rms_error": 0.0},
                "boundary": {"passed": True},
                "mixture_gaussian": {"passed": True},
            },
            "resnet": {
                "identity_exact": True,
                "jvp_cuts_completed": len(context.config.experiment1.sentinel_cuts),
                "jvp_median_relative_error": 0.0,
                "jvp_p95_relative_error": 0.0,
            },
            "synthetic_invariants": {"passed": True},
        }
        return context.store.put_json(
            observations,
            filename="observations.json",
            kind="stage/custom-mechanical-evidence",
            metadata={
                **artifact_metadata(context, dependencies),
                "result_schema_version": 1,
            },
        )

    def gate_handler(context, dependencies):
        observations = context.store.read_json(
            dependencies[evidence_stage.name].artifact_id,
            "observations.json",
        )
        rule = expected_gate_rule(context.stage_spec, context.config)
        result = GateEvaluator().evaluate(rule, observations)
        return context.store.put_json(
            result.to_dict(),
            filename="gate.json",
            kind="gate/custom-mechanical",
            metadata={
                **artifact_metadata(context, dependencies),
                "gate_id": result.gate_id,
                "gate_rule_signature": canonical_hash(rule.to_dict()),
                "gate_status": result.status.value,
                "natural_status": result.natural_status.value,
                "override_authorization": None,
                "result_schema_version": 1,
            },
        )

    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers={
            evidence_stage.name: evidence_handler,
            gate_stage.name: gate_handler,
        },
        run_id="custom-gate-registry",
    )
    artifacts = runner.run([gate_stage.name])

    gate = runner.store.read_json(artifacts[gate_stage.name].artifact_id, "gate.json")
    assert gate["status"] == "PASS"


def test_failed_stage_is_retryable_and_preserves_attempt_count(tmp_path) -> None:
    calls = 0

    def handler(context, dependencies):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected failure")
        return context.store.put_json(
            {"value": 11},
            kind="fixture",
            metadata=artifact_metadata(context, dependencies),
        )

    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=_fixture_registry(),
        handlers={"fixture": handler},
        run_id="retry",
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        runner.run(["fixture"])
    failed = runner.index.get_stage("retry", "fixture")
    assert failed is not None
    assert failed.state is StageState.FAILED
    assert failed.owner_token is None
    assert failed.attempts == 1

    runner.run(["fixture"])
    completed = runner.index.get_stage("retry", "fixture")
    assert completed is not None
    assert completed.state is StageState.COMPLETED
    assert completed.owner_token is None
    assert completed.attempts == 2


def test_keyboard_interrupt_releases_stage_claim_for_retry(tmp_path) -> None:
    def handler(_context, _dependencies):
        raise KeyboardInterrupt("injected interrupt")

    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=_fixture_registry(),
        handlers={"fixture": handler},
        run_id="interrupted",
    )
    with pytest.raises(KeyboardInterrupt, match="injected interrupt"):
        runner.run(["fixture"])

    record = runner.index.get_stage("interrupted", "fixture")
    assert record is not None
    assert record.state is StageState.FAILED
    assert record.owner_token is None
    assert record.metadata["interrupted"] is True


def test_invalid_run_id_is_rejected_before_creating_a_lock_path(tmp_path) -> None:
    escaped_lock = tmp_path / "runs" / "escaped.lock"
    with pytest.raises(Exception, match="run_id must match"):
        ExperimentRunner(
            LabConfig(),
            project_root=tmp_path,
            registry=_fixture_registry(),
            handlers={},
            run_id="../escaped",
        )
    assert not escaped_lock.exists()


def test_runner_never_steals_an_active_stage_claim(tmp_path) -> None:
    registry = _fixture_registry()
    config = LabConfig()
    runner = ExperimentRunner(
        config,
        project_root=tmp_path,
        registry=registry,
        handlers={"fixture": lambda _context, _dependencies: pytest.fail("must not run")},
        run_id="claimed",
    )
    spec = registry.get("fixture")
    signature = stage_signature(
        spec,
        config,
        upstream_artifact_ids={},
        source_identity=runner.source_identity,
    )
    runner.index.claim_stage(
        "claimed",
        "fixture",
        signature,
        owner_token="other-worker",
    )

    with pytest.raises(StageClaimError, match="another worker"):
        runner.run(["fixture"])
    record = runner.index.get_stage("claimed", "fixture")
    assert record is not None
    assert record.state is StageState.RUNNING
    assert record.owner_token == "other-worker"


def test_handler_artifact_must_carry_stage_identity(tmp_path) -> None:
    def handler(context, _dependencies):
        return context.store.put_json({"value": 3}, kind="fixture", metadata={})

    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=_fixture_registry(),
        handlers={"fixture": handler},
        run_id="bad-metadata",
    )
    with pytest.raises(RuntimeError, match="incompatible stage identity fields"):
        runner.run(["fixture"])
    record = runner.index.get_stage("bad-metadata", "fixture")
    assert record is not None
    assert record.state is StageState.FAILED


def test_runner_rejects_verified_but_semantically_wrong_cache_entry(tmp_path) -> None:
    registry = _fixture_registry()
    config = LabConfig()
    calls = 0

    def handler(context, dependencies):
        nonlocal calls
        calls += 1
        return context.store.put_json(
            {"value": 5},
            kind="fixture",
            metadata=artifact_metadata(context, dependencies),
        )

    runner = ExperimentRunner(
        config,
        project_root=tmp_path,
        registry=registry,
        handlers={"fixture": handler},
        run_id="wrong-cache",
    )
    spec = registry.get("fixture")
    signature = stage_signature(
        spec,
        config,
        upstream_artifact_ids={},
        source_identity=runner.source_identity,
    )
    unrelated = runner.store.put_json(
        {"value": "unrelated"},
        kind="fixture",
        metadata={"stage": "somewhere.else", "stage_signature": signature},
    )
    runner.index.cache_store(
        signature,
        stage_name="fixture",
        artifact_id=unrelated.artifact_id,
    )

    with pytest.raises(ExecutionError, match="incompatible stage identity fields"):
        runner.run(["fixture"])
    assert calls == 0
    assert runner.index.cache_lookup(signature) is None


def test_bad_cache_producer_lineage_can_be_recomputed_and_re_elected(tmp_path) -> None:
    registry = _fixture_registry()
    calls = 0

    def handler(context, dependencies):
        nonlocal calls
        calls += 1
        return context.store.put_json(
            {"value": 31},
            kind="fixture",
            metadata=artifact_metadata(context, dependencies),
        )

    producer = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers={"fixture": handler},
        run_id="lineage-producer",
    )
    artifact = producer.run(["fixture"])["fixture"]
    stage = registry.get("fixture")
    signature = stage_signature(
        stage,
        producer.config,
        upstream_artifact_ids={},
        source_identity=producer.source_identity,
    )
    original = producer.index.cache_lookup(signature)
    assert original is not None
    assert producer.index.cache_forget(
        signature,
        expected_generation=original.generation,
        reason="inject invalid producer lineage",
    )
    invalid = producer.index.cache_store(
        signature,
        stage_name="fixture",
        artifact_id=artifact.artifact_id,
        metadata={
            "producer_provenance_artifact_id": "a" * 64,
            "producer_run_id": "missing-producer",
        },
    )

    consumer = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers={"fixture": handler},
        run_id="lineage-consumer",
    )
    with pytest.raises(ExecutionError, match="missing-producer"):
        consumer.run(["fixture"])
    assert consumer.index.cache_lookup(signature) is None
    assert calls == 1

    recovered = consumer.run(["fixture"])["fixture"]
    assert recovered.artifact_id == artifact.artifact_id
    assert calls == 2
    elected = consumer.index.cache_lookup(signature)
    assert elected is not None
    assert elected.generation == invalid.generation + 1
    assert elected.metadata["producer_run_id"] == "lineage-consumer"
    history = consumer.index.list_cache_elections(signature)
    assert [event.event_kind for event in history] == [
        "ELECTED",
        "DEACTIVATED",
        "ELECTED",
        "DEACTIVATED",
        "ELECTED",
    ]
    assert history[-2].reason == "cache first-publisher provenance validation failed"


def test_stale_cache_validator_cannot_deactivate_newer_election(
    tmp_path,
    monkeypatch,
) -> None:
    registry = _fixture_registry()

    def handler(context, dependencies):
        return context.store.put_json(
            {"value": 37},
            kind="fixture",
            metadata=artifact_metadata(context, dependencies),
        )

    producer = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers={"fixture": handler},
        run_id="slow-validator-producer",
    )
    producer.run(["fixture"])
    stage = registry.get("fixture")
    signature = stage_signature(
        stage,
        producer.config,
        upstream_artifact_ids={},
        source_identity=producer.source_identity,
    )
    observed = producer.index.cache_lookup(signature)
    assert observed is not None
    consumer = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers={"fixture": handler},
        run_id="slow-validator-consumer",
    )
    entered = Event()
    release = Event()
    original_validator = execution_module._validate_artifact_identity

    def slow_validator(reference, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=5)
            raise ExecutionError("stale cache validation failed")
        return original_validator(reference, **kwargs)

    monkeypatch.setattr(execution_module, "_validate_artifact_identity", slow_validator)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(consumer.run, ["fixture"])
        assert entered.wait(timeout=5)
        assert consumer.index.cache_forget(
            signature,
            expected_generation=observed.generation,
            reason="concurrent recovery",
        )
        replacement = consumer.index.cache_store(
            signature,
            stage_name=observed.stage_name,
            artifact_id=observed.artifact_id,
            metadata=observed.metadata,
        )
        release.set()
        with pytest.raises(ExecutionError, match="stale cache validation failed"):
            future.result(timeout=5)

    current = consumer.index.cache_lookup(signature)
    assert current == replacement
    assert current.generation == observed.generation + 1
    assert [event.event_kind for event in consumer.index.list_cache_elections(signature)][-2:] == [
        "DEACTIVATED",
        "ELECTED",
    ]


def test_concurrent_cache_misses_publish_one_cache_invariant_artifact(tmp_path) -> None:
    registry = _fixture_registry()
    barrier = Barrier(2)

    def run_one(run_id: str):
        def handler(context, dependencies):
            barrier.wait(timeout=5)
            return context.store.put_json(
                {"value": 17},
                kind="fixture",
                metadata=artifact_metadata(context, dependencies),
            )

        runner = ExperimentRunner(
            LabConfig(),
            project_root=tmp_path,
            registry=registry,
            handlers={"fixture": handler},
            run_id=run_id,
        )
        artifact = runner.run(["fixture"])["fixture"]
        record = runner.index.get_stage(run_id, "fixture")
        assert record is not None
        return artifact, record

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_one, run_id) for run_id in ("parallel-a", "parallel-b")]
        results = [future.result(timeout=15) for future in futures]

    artifacts = [artifact for artifact, _record in results]
    records = [record for _artifact, record in results]
    assert artifacts[0].artifact_id == artifacts[1].artifact_id
    assert all(record.state is StageState.COMPLETED for record in records)
    assert all(record.metadata["cache_hit"] is False for record in records)
    producer_ids = {record.metadata["producer_run_id"] for record in records}
    assert len(producer_ids) == 1
    assert producer_ids <= {"parallel-a", "parallel-b"}


def test_concurrent_workers_attach_to_one_canonical_run_registration(tmp_path) -> None:
    barrier = Barrier(2)

    def construct_runner():
        barrier.wait(timeout=5)
        runner = ExperimentRunner(
            LabConfig(),
            project_root=tmp_path,
            registry=_fixture_registry(),
            handlers={},
            run_id="same-run",
        )
        record = runner.index.get_run("same-run")
        assert record is not None
        return record

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(construct_runner) for _ in range(2)]
        records = [future.result(timeout=15) for future in futures]

    assert records[0].config_hash == records[1].config_hash
    assert records[0].provenance_artifact_id == records[1].provenance_artifact_id


def test_existing_run_requires_its_verified_provenance_artifact(tmp_path) -> None:
    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=_fixture_registry(),
        handlers={},
        run_id="provenance-check",
    )
    run = runner.index.get_run("provenance-check")
    assert run is not None and run.provenance_artifact_id is not None
    provenance_path = runner.store.get(run.provenance_artifact_id).path
    provenance_path.rename(tmp_path / "detached-provenance")

    with pytest.raises(ExecutionError, match="provenance artifact is unavailable"):
        ExperimentRunner(
            LabConfig(),
            project_root=tmp_path,
            registry=_fixture_registry(),
            handlers={},
            run_id="provenance-check",
        )


def test_verified_run_rejects_empty_and_source_mismatched_provenance_payloads(
    tmp_path,
) -> None:
    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=_fixture_registry(),
        handlers={},
        run_id="provenance-source",
    )
    _register_forged_provenance_run(
        runner,
        run_id="empty-provenance",
        provenance_payload={},
    )
    with pytest.raises(ExecutionError, match="saved provenance payload is invalid"):
        load_verified_run_identity(runner.store, runner.index, "empty-provenance")

    mismatched = replace(
        runner._captured_provenance,
        packages={"injected-package": "1.0"},
    )
    _register_forged_provenance_run(
        runner,
        run_id="mismatched-provenance",
        provenance_payload=mismatched.to_dict(),
    )
    with pytest.raises(ExecutionError, match="does not match registered source identity"):
        load_verified_run_identity(runner.store, runner.index, "mismatched-provenance")


def test_source_drift_before_stage_resolution_prevents_handler_execution(
    tmp_path,
    monkeypatch,
) -> None:
    called = False

    def handler(_context, _dependencies):
        nonlocal called
        called = True
        pytest.fail("handler must not run after source drift")

    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=_fixture_registry(),
        handlers={"fixture": handler},
        run_id="pre-handler-drift",
    )
    drifted = replace(
        runner._captured_provenance,
        packages={"drifted-package": "1.0"},
    )
    monkeypatch.setattr(execution_module, "capture_provenance", lambda **_kwargs: drifted)

    with pytest.raises(ExecutionError, match="source identity drift detected before resolving"):
        runner.run(["fixture"])
    assert not called
    assert runner.index.get_stage("pre-handler-drift", "fixture") is None


def test_non_git_project_source_drift_prevents_cache_or_handler_use(tmp_path) -> None:
    package = tmp_path / "src" / "voronoi_lab"
    package.mkdir(parents=True)
    source = package / "fixture.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    asset = package / "assets" / "runtime.js"
    asset.parent.mkdir()
    asset.write_text("const value = 1;\n", encoding="utf-8")
    called = False

    def handler(_context, _dependencies):
        nonlocal called
        called = True
        pytest.fail("handler must not run after unpacked source drift")

    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=_fixture_registry(),
        handlers={"fixture": handler},
        run_id="non-git-source-drift",
    )
    asset.write_text("const value = 2;\n", encoding="utf-8")

    with pytest.raises(ExecutionError, match="source identity drift detected before resolving"):
        runner.run(["fixture"])
    assert not called
    assert runner.index.get_stage("non-git-source-drift", "fixture") is None


def test_git_mode_provenance_also_hashes_imported_and_project_source(
    tmp_path,
    monkeypatch,
) -> None:
    local_package = tmp_path / "src" / "voronoi_lab"
    local_package.mkdir(parents=True)
    local_source = local_package / "local_fixture.py"
    local_source.write_text("VALUE = 1\n", encoding="utf-8")
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_capture_provenance(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(execution_module, "capture_provenance", fake_capture_provenance)
    runner = object.__new__(ExperimentRunner)
    runner.project_root = tmp_path
    runner._provenance_root = tmp_path

    assert runner._capture_source_provenance() is sentinel
    assert captured["repo_root"] == tmp_path
    inputs = captured["input_files"]
    assert isinstance(inputs, dict)
    assert any(name.endswith("voronoi_lab/execution.py") for name in inputs)
    assert inputs["source/project/voronoi_lab/local_fixture.py"] == local_source


def test_source_drift_during_handler_fails_stage_before_cache_publication(
    tmp_path,
    monkeypatch,
) -> None:
    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=_fixture_registry(),
        handlers={},
        run_id="during-handler-drift",
    )
    baseline = runner._captured_provenance
    drifted = replace(baseline, packages={"drifted-package": "1.0"})
    snapshots = iter((baseline, baseline, drifted))
    monkeypatch.setattr(
        execution_module,
        "capture_provenance",
        lambda **_kwargs: next(snapshots),
    )

    def handler(context, dependencies):
        return context.store.put_json(
            {"value": 19},
            kind="fixture",
            metadata=artifact_metadata(context, dependencies),
        )

    runner.handlers["fixture"] = handler
    stage = runner.registry.get("fixture")
    signature = stage_signature(
        stage,
        runner.config,
        upstream_artifact_ids={},
        source_identity=runner.source_identity,
    )
    with pytest.raises(
        ExecutionError,
        match="source identity drift detected immediately after handler",
    ):
        runner.run(["fixture"])

    record = runner.index.get_stage("during-handler-drift", "fixture")
    assert record is not None
    assert record.state is StageState.FAILED
    assert runner.index.cache_lookup(signature) is None


def test_source_drift_before_cache_consumption_does_not_complete_stage(
    tmp_path,
    monkeypatch,
) -> None:
    registry = _fixture_registry()

    def handler(context, dependencies):
        return context.store.put_json(
            {"value": 23},
            kind="fixture",
            metadata=artifact_metadata(context, dependencies),
        )

    producer = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers={"fixture": handler},
        run_id="cache-producer",
    )
    producer.run(["fixture"])
    consumer = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=registry,
        handlers={"fixture": lambda *_args: pytest.fail("cache should satisfy stage")},
        run_id="cache-consumer-drift",
    )
    baseline = consumer._captured_provenance
    drifted = replace(baseline, packages={"drifted-package": "1.0"})
    snapshots = iter((baseline, drifted))
    monkeypatch.setattr(
        execution_module,
        "capture_provenance",
        lambda **_kwargs: next(snapshots),
    )

    with pytest.raises(
        ExecutionError,
        match="source identity drift detected before consuming cache",
    ):
        consumer.run(["fixture"])
    assert consumer.index.get_stage("cache-consumer-drift", "fixture") is None


def test_failed_gate_dependency_records_blocked_stage_without_calling_handler(tmp_path) -> None:
    called = False

    def after_gate(_context, _dependencies):
        nonlocal called
        called = True
        pytest.fail("blocked handler must not run")

    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=_gated_registry(),
        handlers={"gate.fixture": _gate_handler(), "after-gate": after_gate},
        run_id="blocked-gate",
    )
    with pytest.raises(ExecutionError, match=r"gate\.fixture:FAIL"):
        runner.run(["after-gate"])
    assert not called
    record = runner.index.get_stage("blocked-gate", "after-gate")
    assert record is not None
    assert record.state is StageState.BLOCKED
    assert record.metadata["gate_blockers"] == ("gate.fixture:FAIL",)


def test_explicit_gate_override_allows_downstream_and_retains_lineage(tmp_path) -> None:
    def after_gate(context, dependencies):
        return context.store.put_json(
            {"continued": True},
            kind="fixture",
            metadata=artifact_metadata(context, dependencies),
        )

    runner = ExperimentRunner(
        LabConfig(),
        project_root=tmp_path,
        registry=_gated_registry(),
        handlers={
            "gate.fixture": _gate_handler(override_reason="diagnostic continuation"),
            "after-gate": after_gate,
            "after-after": after_gate,
        },
        run_id="overridden-gate",
    )
    final_reference = runner.run(["after-after"])["after-after"]
    gate_reference = runner.run(["gate.fixture"])["gate.fixture"]
    gate_payload = runner.store.read_json(gate_reference.artifact_id, "gate.json")
    assert gate_payload["status"] == "OVERRIDDEN"
    assert gate_payload["override_lineage"][0]["reason"] == "diagnostic continuation"
    assert final_reference.manifest.metadata["inherited_gate_overrides"][0]["reason"] == (
        "diagnostic continuation"
    )
    record = runner.index.get_stage("overridden-gate", "after-gate")
    assert record is not None
    assert record.state is StageState.COMPLETED
