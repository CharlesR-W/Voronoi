from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event

import numpy as np
import pytest

import voronoi_lab.sharding as sharding_module
from voronoi_lab.core import ArtifactStore, RunIndex, StageClaimError, StageState
from voronoi_lab.sharding import (
    SHARD_EXECUTION_SCHEMA_VERSION,
    ShardExecutor,
    ShardKey,
    ShardReducer,
    ShardReductionError,
    ShardSpec,
    ShardValidationError,
    plan_image_chunks,
)

CONFIG_HASH = "a" * 64
PARENT_SIGNATURE = "b" * 64
UPSTREAM_ID = "c" * 64


def _infrastructure(tmp_path: Path, *run_ids: str) -> tuple[ArtifactStore, RunIndex]:
    store = ArtifactStore(tmp_path / "artifacts")
    index = RunIndex(tmp_path / "index.sqlite")
    for run_id in run_ids or ("run-1",):
        provenance = store.put_json(
            {"run_id": run_id},
            kind="run/provenance",
            metadata={"run_id": run_id},
        )
        index.register_run(
            run_id,
            config_hash=CONFIG_HASH,
            provenance_artifact_id=provenance.artifact_id,
        )
    return store, index


def _provenance_id(index: RunIndex, run_id: str) -> str:
    run = index.get_run(run_id)
    assert run is not None and run.provenance_artifact_id is not None
    return run.provenance_artifact_id


def _spec(
    shard: int,
    *,
    parent_stage: str = "exp1.activations",
    source_revision: str = "source-a",
) -> ShardSpec:
    return ShardSpec(
        parent_stage=parent_stage,
        parent_stage_signature=PARENT_SIGNATURE,
        key=ShardKey(
            {
                "bank": "geometry_test",
                "checkpoint": 20,
                "cut": "stage2.block1.post_add",
                "image_chunk": shard,
            }
        ),
        artifact_kind="exp1/activation-shard",
        stage_config={"runtime.shard_images": 64},
        source_identity={"git_commit": source_revision},
        upstream_artifacts={"exp1.probe_banks": UPSTREAM_ID},
    )


def _publish_payload(context, value: int = 1):
    return context.store.put_json(
        {"value": value},
        kind=context.spec.artifact_kind,
        metadata=context.artifact_metadata({"value_count": 1}),
    )


def test_shard_key_and_spec_signatures_are_canonical_and_strict() -> None:
    left_key = ShardKey({"checkpoint": 20, "bank": "test"})
    right_key = ShardKey({"bank": "test", "checkpoint": 20})
    assert left_key.signature == right_key.signature
    assert left_key.to_dict() == {"bank": "test", "checkpoint": 20}

    left = _spec(0)
    same = ShardSpec(
        parent_stage=left.parent_stage,
        parent_stage_signature=left.parent_stage_signature,
        key=ShardKey(dict(reversed(list(left.key.to_dict().items())))),
        artifact_kind=left.artifact_kind,
        stage_config={"runtime.shard_images": 64},
        source_identity={"git_commit": "source-a"},
        upstream_artifacts={"exp1.probe_banks": UPSTREAM_ID},
    )
    assert left.signature == same.signature
    assert left.row_name == same.row_name
    assert replace(left, source_identity={"git_commit": "source-b"}).signature != left.signature
    assert replace(left, source_identity={"git_commit": "source-b"}).row_name == left.row_name

    with pytest.raises(ShardValidationError, match="must not be empty"):
        ShardKey({})
    with pytest.raises(ShardValidationError, match="coordinate names"):
        ShardKey({"not semantic!": 1})
    with pytest.raises(ShardValidationError, match="finite JSON"):
        replace(left, stage_config={"bad": float("nan")})
    with pytest.raises(ShardValidationError, match="SHA-256"):
        replace(left, upstream_artifacts={"exp1.probe_banks": "not-a-digest"})


def test_image_chunk_planner_honors_size_and_binds_exact_ordered_ids() -> None:
    chunks = plan_image_chunks(
        np.asarray([9, 2, 17, 4, 31], dtype=np.int64),
        shard_images=2,
        coordinates={"bank": "geometry", "checkpoint": 5, "cut": "cut3"},
    )
    again = plan_image_chunks(
        [9, 2, 17, 4, 31],
        shard_images=2,
        coordinates={"cut": "cut3", "checkpoint": 5, "bank": "geometry"},
    )

    assert [(chunk.start, chunk.stop, chunk.image_ids) for chunk in chunks] == [
        (0, 2, (9, 2)),
        (2, 4, (17, 4)),
        (4, 5, (31,)),
    ]
    assert [chunk.key.signature for chunk in chunks] == [chunk.key.signature for chunk in again]
    assert chunks[0].key.to_dict()["image_chunk"] == {
        "image_ids": [9, 2],
        "ordinal": 0,
        "start": 0,
        "stop": 2,
    }
    assert plan_image_chunks([], shard_images=8) == ()
    with pytest.raises(ShardValidationError, match="positive integer"):
        plan_image_chunks([1], shard_images=0)
    with pytest.raises(ShardValidationError, match="unique"):
        plan_image_chunks([1, 1], shard_images=1)
    with pytest.raises(ShardValidationError, match="reserved"):
        plan_image_chunks([1], shard_images=1, coordinates={"image_chunk": "mine"})


def test_completed_shard_resumes_without_rerunning_handler(tmp_path: Path) -> None:
    store, index = _infrastructure(tmp_path, "run-1")
    executor = ShardExecutor(store, index, "run-1", owner_token="worker-1")
    spec = _spec(0)
    calls = 0

    def handler(context):
        nonlocal calls
        calls += 1
        return _publish_payload(context)

    first = executor.execute(spec, handler)
    second = executor.execute(spec, handler)

    assert first.artifact_id == second.artifact_id
    assert calls == 1
    record = index.get_stage("run-1", spec.row_name)
    assert record is not None
    assert record.state is StageState.COMPLETED
    assert record.attempts == 1
    assert record.metadata["shard_key_signature"] == spec.key.signature
    assert record.metadata["cache_hit"] is False
    assert record.metadata["producer_run_id"] == "run-1"
    assert record.metadata["producer_provenance_artifact_id"] == _provenance_id(index, "run-1")
    assert record.metadata["run_provenance_artifact_id"] == _provenance_id(index, "run-1")
    assert [stage.stage_name for stage in index.list_stages("run-1")] == [spec.row_name]


def test_verified_cache_reuse_crosses_runs_but_bad_cache_is_rejected(tmp_path: Path) -> None:
    store, index = _infrastructure(tmp_path, "run-1", "run-2")
    spec = _spec(0)
    first = ShardExecutor(store, index, "run-1", owner_token="worker-1").execute(
        spec, _publish_payload
    )

    def must_not_run(_context):
        raise AssertionError("verified cache should avoid the handler")

    second = ShardExecutor(store, index, "run-2", owner_token="worker-2").execute(
        spec, must_not_run
    )
    assert first.artifact_id == second.artifact_id
    cached_record = index.get_stage("run-2", spec.row_name)
    assert cached_record is not None
    assert cached_record.message == "verified shard cache hit"
    assert cached_record.metadata["cache_hit"] is True
    assert cached_record.metadata["producer_run_id"] == "run-1"
    assert cached_record.metadata["producer_provenance_artifact_id"] == _provenance_id(
        index, "run-1"
    )
    assert cached_record.metadata["run_provenance_artifact_id"] == _provenance_id(index, "run-2")
    producer_record = index.get_stage("run-1", spec.row_name)
    assert producer_record is not None and producer_record.metadata["cache_hit"] is False
    cache = index.cache_lookup(spec.signature)
    assert cache is not None
    assert cache.metadata["producer_run_id"] == "run-1"
    assert cache.metadata["producer_provenance_artifact_id"] == _provenance_id(index, "run-1")
    assert "producer_run_id" not in first.manifest.metadata
    assert "run_provenance_artifact_id" not in first.manifest.metadata

    bad_store, bad_index = _infrastructure(tmp_path / "bad", "run-bad")
    bad_spec = _spec(1)
    bad_ref = bad_store.put_json(
        {"value": 0}, kind=bad_spec.artifact_kind, metadata={"wrong": True}
    )
    bad_index.cache_store(
        bad_spec.signature,
        stage_name=bad_spec.row_name,
        artifact_id=bad_ref.artifact_id,
        metadata={
            "parent_stage": bad_spec.parent_stage,
            "parent_stage_signature": bad_spec.parent_stage_signature,
            "producer_provenance_artifact_id": _provenance_id(bad_index, "run-bad"),
            "producer_run_id": "run-bad",
            "shard_execution_schema_version": SHARD_EXECUTION_SCHEMA_VERSION,
            "shard_key_signature": bad_spec.key.signature,
        },
    )
    with pytest.raises(ShardValidationError, match="identity fields"):
        ShardExecutor(bad_store, bad_index, "run-bad").execute(bad_spec, must_not_run)
    assert bad_index.cache_lookup(bad_spec.signature) is None


def test_cache_rejects_inconsistent_first_publisher_provenance(tmp_path: Path) -> None:
    store, index = _infrastructure(tmp_path, "producer", "consumer")
    spec = _spec(0)
    reference = store.put_json(
        {"value": 1},
        kind=spec.artifact_kind,
        metadata=spec.artifact_metadata,
    )
    index.cache_store(
        spec.signature,
        stage_name=spec.row_name,
        artifact_id=reference.artifact_id,
        metadata={
            "parent_stage": spec.parent_stage,
            "parent_stage_signature": spec.parent_stage_signature,
            "producer_provenance_artifact_id": "d" * 64,
            "producer_run_id": "producer",
            "shard_execution_schema_version": SHARD_EXECUTION_SCHEMA_VERSION,
            "shard_key_signature": spec.key.signature,
        },
    )

    with pytest.raises(ShardValidationError, match="first-publisher provenance"):
        ShardExecutor(store, index, "consumer").execute(spec, _publish_payload)
    assert index.cache_lookup(spec.signature) is None


def test_stale_shard_validator_cannot_deactivate_recovery_election(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store, index = _infrastructure(tmp_path, "producer", "consumer")
    spec = _spec(0)
    ShardExecutor(store, index, "producer").execute(spec, _publish_payload)
    observed = index.cache_lookup(spec.signature)
    assert observed is not None
    entered = Event()
    release = Event()
    original_validator = sharding_module.validate_shard_artifact

    def slow_validator(reference, candidate):
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=5)
            raise ShardValidationError("stale shard cache validation failed")
        return original_validator(reference, candidate)

    monkeypatch.setattr(sharding_module, "validate_shard_artifact", slow_validator)
    executor = ShardExecutor(store, index, "consumer")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(executor.execute, spec, _publish_payload)
        assert entered.wait(timeout=5)
        assert index.cache_forget(
            spec.signature,
            expected_generation=observed.generation,
            reason="concurrent shard recovery",
        )
        replacement = index.cache_store(
            spec.signature,
            stage_name=observed.stage_name,
            artifact_id=observed.artifact_id,
            metadata=observed.metadata,
        )
        release.set()
        with pytest.raises(ShardValidationError, match="stale shard cache validation failed"):
            future.result(timeout=5)

    assert index.cache_lookup(spec.signature) == replacement
    assert replacement.generation == observed.generation + 1


def test_concurrent_publishers_preserve_one_first_publisher_lineage(tmp_path: Path) -> None:
    store, index = _infrastructure(tmp_path, "parallel-a", "parallel-b")
    spec = _spec(0)
    barrier = Barrier(2)

    def run_one(run_id: str):
        def publish(context):
            barrier.wait(timeout=5)
            return _publish_payload(context, value=17)

        executor = ShardExecutor(
            store,
            RunIndex(index.path),
            run_id,
            owner_token=f"worker-{run_id}",
        )
        artifact = executor.execute(spec, publish)
        record = index.get_stage(run_id, spec.row_name)
        assert record is not None
        return artifact, record

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_one, run_id) for run_id in ("parallel-a", "parallel-b")]
        results = [future.result(timeout=15) for future in futures]

    artifacts = [artifact for artifact, _record in results]
    records = [record for _artifact, record in results]
    assert artifacts[0].artifact_id == artifacts[1].artifact_id
    assert all(record.metadata["cache_hit"] is False for record in records)
    producer_ids = {record.metadata["producer_run_id"] for record in records}
    assert len(producer_ids) == 1
    producer_id = producer_ids.pop()
    assert producer_id in {"parallel-a", "parallel-b"}
    producer_provenance = _provenance_id(index, producer_id)
    assert all(
        record.metadata["producer_provenance_artifact_id"] == producer_provenance
        for record in records
    )
    for run_id, record in zip(("parallel-a", "parallel-b"), records, strict=True):
        assert record.metadata["run_provenance_artifact_id"] == _provenance_id(index, run_id)
    cache = index.cache_lookup(spec.signature)
    assert cache is not None
    assert cache.metadata["producer_run_id"] == producer_id
    assert cache.metadata["producer_provenance_artifact_id"] == producer_provenance


def test_concurrent_worker_cannot_steal_a_running_shard(tmp_path: Path) -> None:
    store, index = _infrastructure(tmp_path, "run-1")
    spec = _spec(0)
    entered = Event()
    release = Event()

    def slow_handler(context):
        entered.set()
        assert release.wait(timeout=5)
        return _publish_payload(context)

    first = ShardExecutor(store, index, "run-1", owner_token="worker-1")
    second = ShardExecutor(store, RunIndex(index.path), "run-1", owner_token="worker-2")
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(first.execute, spec, slow_handler)
        assert entered.wait(timeout=5)
        with pytest.raises(StageClaimError, match="another worker"):
            second.execute(spec, _publish_payload)
        release.set()
        reference = future.result(timeout=5)

    assert store.verify(reference.artifact_id).artifact_id == reference.artifact_id
    record = index.get_stage("run-1", spec.row_name)
    assert record is not None and record.attempts == 1


def test_partial_failure_resumes_only_failed_shard(tmp_path: Path) -> None:
    store, index = _infrastructure(tmp_path, "run-1")
    completed_spec = _spec(0)
    failed_spec = _spec(1)
    first_worker = ShardExecutor(store, index, "run-1", owner_token="worker-1")
    completed = first_worker.execute(completed_spec, _publish_payload)

    def fail(_context):
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated"):
        first_worker.execute(failed_spec, fail)
    failed = index.get_stage("run-1", failed_spec.row_name)
    assert failed is not None and failed.state is StageState.FAILED

    completed_calls = 0

    def completed_must_not_repeat(context):
        nonlocal completed_calls
        completed_calls += 1
        return _publish_payload(context)

    second_worker = ShardExecutor(store, index, "run-1", owner_token="worker-2")
    assert (
        second_worker.execute(completed_spec, completed_must_not_repeat).artifact_id
        == completed.artifact_id
    )
    retried = second_worker.execute(failed_spec, _publish_payload)

    assert completed_calls == 0
    assert store.verify(retried.artifact_id).artifact_id == retried.artifact_id
    retried_record = index.get_stage("run-1", failed_spec.row_name)
    assert retried_record is not None
    assert retried_record.state is StageState.COMPLETED
    assert retried_record.attempts == 2


def test_keyboard_interrupt_marks_shard_failed_before_reraising(tmp_path: Path) -> None:
    store, index = _infrastructure(tmp_path, "run-1")
    spec = _spec(0)
    executor = ShardExecutor(store, index, "run-1")

    def interrupt(_context):
        raise KeyboardInterrupt("operator interrupt")

    with pytest.raises(KeyboardInterrupt, match="operator interrupt"):
        executor.execute(spec, interrupt)
    failed = index.get_stage("run-1", spec.row_name)
    assert failed is not None
    assert failed.state is StageState.FAILED
    assert failed.message == "KeyboardInterrupt: operator interrupt"
    assert failed.metadata["cache_hit"] is False
    assert failed.metadata["run_provenance_artifact_id"] == _provenance_id(index, "run-1")

    executor.execute(spec, _publish_payload)
    completed = index.get_stage("run-1", spec.row_name)
    assert completed is not None
    assert completed.state is StageState.COMPLETED
    assert completed.attempts == 2


def test_shard_execution_requires_current_run_provenance(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    index = RunIndex(tmp_path / "index.sqlite")
    index.register_run("run-1", config_hash=CONFIG_HASH)

    with pytest.raises(ShardValidationError, match="no provenance artifact"):
        ShardExecutor(store, index, "run-1").execute(_spec(0), _publish_payload)


def test_crash_reclaim_is_explicit_and_requires_a_reason(tmp_path: Path) -> None:
    store, index = _infrastructure(tmp_path, "run-1")
    spec = _spec(0)
    index.claim_stage(
        "run-1",
        spec.row_name,
        spec.signature,
        owner_token="dead-worker",
        metadata=spec.artifact_metadata,
    )
    replacement = ShardExecutor(store, index, "run-1", owner_token="replacement-worker")

    with pytest.raises(ShardValidationError, match="non-empty"):
        replacement.reclaim(spec, reason="")
    reclaimed = replacement.reclaim(spec, reason="dead worker process was confirmed absent")
    assert reclaimed.owner_token == "replacement-worker"
    assert reclaimed.attempts == 2
    assert reclaimed.message == "reclaimed: dead worker process was confirmed absent"

    reference = replacement.execute(spec, _publish_payload)
    assert store.verify(reference.artifact_id).artifact_id == reference.artifact_id


def test_handler_must_publish_bound_json_metadata(tmp_path: Path) -> None:
    store, index = _infrastructure(tmp_path, "run-1")
    spec = _spec(0)
    executor = ShardExecutor(store, index, "run-1")

    def unbound(context):
        return context.store.put_json(
            {"value": 1}, kind=context.spec.artifact_kind, metadata={"unbound": True}
        )

    with pytest.raises(ShardValidationError, match="identity fields"):
        executor.execute(spec, unbound)
    record = index.get_stage("run-1", spec.row_name)
    assert record is not None and record.state is StageState.FAILED

    context_metadata = spec.artifact_metadata
    assert context_metadata["upstream_artifacts"] == {"exp1.probe_banks": UPSTREAM_ID}


def test_reducer_publishes_stable_ordered_exact_manifest(tmp_path: Path) -> None:
    store, index = _infrastructure(tmp_path, "run-1")
    specs = (_spec(2), _spec(0), _spec(1))
    executor = ShardExecutor(store, index, "run-1")
    artifacts = [executor.execute(spec, _publish_payload) for spec in specs]

    reducer = ShardReducer(store, index, "run-1")
    first = reducer.publish(specs)
    second = reducer.publish(specs)
    payload = store.read_json(first.reference.artifact_id, "shards.json")

    assert first.reference.artifact_id == second.reference.artifact_id
    assert [reference.artifact_id for reference in first.shard_artifacts] == [
        reference.artifact_id for reference in artifacts
    ]
    assert [entry["artifact_id"] for entry in payload["ordered_shards"]] == [
        reference.artifact_id for reference in artifacts
    ]
    assert [entry["ordinal"] for entry in payload["ordered_shards"]] == [0, 1, 2]
    assert payload["reducer_signature"] == first.reducer_signature

    reordered = reducer.publish(tuple(reversed(specs)))
    assert reordered.reducer_signature != first.reducer_signature
    assert reordered.reference.artifact_id != first.reference.artifact_id


def test_reducer_refuses_missing_failed_duplicate_and_unexpected_shards(
    tmp_path: Path,
) -> None:
    store, index = _infrastructure(tmp_path, "run-1")
    first, second, unexpected = _spec(0), _spec(1), _spec(2)
    executor = ShardExecutor(store, index, "run-1", owner_token="worker-1")
    executor.execute(first, _publish_payload)
    reducer = ShardReducer(store, index, "run-1")

    with pytest.raises(ShardReductionError, match="missing"):
        reducer.publish((first, second))
    with pytest.raises(ShardReductionError, match="duplicates"):
        reducer.publish((first, first))

    def fail(_context):
        raise RuntimeError("failure")

    with pytest.raises(RuntimeError):
        executor.execute(second, fail)
    with pytest.raises(ShardReductionError, match="not complete"):
        reducer.publish((first, second))

    executor.execute(second, _publish_payload)
    executor.execute(unexpected, _publish_payload)
    with pytest.raises(ShardReductionError, match="unexpected"):
        reducer.publish((first, second))


def test_reducer_refuses_mixed_parent_input_identity(tmp_path: Path) -> None:
    store, index = _infrastructure(tmp_path, "run-1")
    reducer = ShardReducer(store, index, "run-1")
    with pytest.raises(ShardReductionError, match="one parent/input identity"):
        reducer.publish((_spec(0), _spec(1, source_revision="source-b")))
