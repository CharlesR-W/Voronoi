from __future__ import annotations

import sqlite3

import pytest

from voronoi_lab.core import (
    RunIndex,
    RunIndexConflictError,
    RunIndexValidationError,
    StageClaimError,
    StageState,
)

CONFIG_HASH = "a" * 64
PROVENANCE_ID = "b" * 64
STAGE_SIGNATURE = "c" * 64
ARTIFACT_ID = "d" * 64
OTHER_ARTIFACT_ID = "e" * 64


def test_register_run_is_persistent_idempotent_and_identity_strict(tmp_path) -> None:
    path = tmp_path / "run-index.sqlite3"
    index = RunIndex(path)
    first = index.register_run(
        "pilot-001",
        config_hash=CONFIG_HASH,
        provenance_artifact_id=PROVENANCE_ID,
        metadata={"mode": "exploratory"},
    )
    second = index.register_run(
        "pilot-001",
        config_hash=CONFIG_HASH,
        provenance_artifact_id=PROVENANCE_ID,
        metadata={"mode": "exploratory"},
    )

    assert first == second
    assert RunIndex(path).get_run("pilot-001") == first
    with pytest.raises(RunIndexConflictError, match="different identity"):
        index.register_run("pilot-001", config_hash="f" * 64)


def test_stage_state_tracks_attempts_and_completed_identity(tmp_path) -> None:
    index = RunIndex(tmp_path / "index.sqlite3")
    index.register_run("run-1", config_hash=CONFIG_HASH)

    pending = index.set_stage("run-1", "static.geometry", STAGE_SIGNATURE, StageState.PENDING)
    running = index.claim_stage("run-1", "static.geometry", STAGE_SIGNATURE, owner_token="worker-1")
    failed = index.finish_stage(
        "run-1",
        "static.geometry",
        owner_token="worker-1",
        state=StageState.FAILED,
        message="worker interrupted",
    )
    retried = index.claim_stage("run-1", "static.geometry", STAGE_SIGNATURE, owner_token="worker-2")
    completed = index.finish_stage(
        "run-1",
        "static.geometry",
        owner_token="worker-2",
        state=StageState.COMPLETED,
        artifact_id=ARTIFACT_ID,
        metadata={"shards": 8},
    )

    assert pending.attempts == 0
    assert running.attempts == 1
    assert failed.attempts == 1
    assert retried.attempts == 2
    assert completed.attempts == 2
    assert completed.artifact_id == ARTIFACT_ID
    assert (
        index.set_stage(
            "run-1",
            "static.geometry",
            STAGE_SIGNATURE,
            StageState.COMPLETED,
            artifact_id=ARTIFACT_ID,
            metadata={"shards": 8},
        )
        == completed
    )

    with pytest.raises(RunIndexConflictError, match="cannot be mutated"):
        index.set_stage(
            "run-1",
            "static.geometry",
            STAGE_SIGNATURE,
            StageState.COMPLETED,
            artifact_id=OTHER_ARTIFACT_ID,
            metadata={"shards": 8},
        )


def test_stage_validation_blocks_invalid_transitions_and_missing_runs(tmp_path) -> None:
    index = RunIndex(tmp_path / "index.sqlite3")
    index.register_run("run-1", config_hash=CONFIG_HASH)

    with pytest.raises(RunIndexValidationError, match="registered"):
        index.set_stage("missing", "stage", STAGE_SIGNATURE, StageState.PENDING)
    with pytest.raises(RunIndexValidationError, match="must reference"):
        index.set_stage("run-1", "stage", STAGE_SIGNATURE, StageState.COMPLETED)
    with pytest.raises(RunIndexValidationError, match="only a completed"):
        index.set_stage(
            "run-1",
            "stage",
            STAGE_SIGNATURE,
            StageState.RUNNING,
            artifact_id=ARTIFACT_ID,
        )

    index.set_stage("run-1", "stage", STAGE_SIGNATURE, StageState.PENDING)
    index.set_stage("run-1", "stage", STAGE_SIGNATURE, StageState.SKIPPED)
    with pytest.raises(RunIndexValidationError, match="invalid stage transition"):
        index.set_stage(
            "run-1",
            "stage",
            STAGE_SIGNATURE,
            StageState.COMPLETED,
            artifact_id=ARTIFACT_ID,
        )
    with pytest.raises(RunIndexConflictError, match="change signature"):
        index.claim_stage("run-1", "stage", "9" * 64, owner_token="worker")


def test_cache_election_is_idempotent_and_inactive_entry_can_be_replaced(tmp_path) -> None:
    index = RunIndex(tmp_path / "index.sqlite3")
    first = index.cache_store(
        STAGE_SIGNATURE,
        stage_name="static.geometry",
        artifact_id=ARTIFACT_ID,
        metadata={"schema": 1},
    )
    second = index.cache_store(
        STAGE_SIGNATURE,
        stage_name="static.geometry",
        artifact_id=ARTIFACT_ID,
        metadata={"schema": 1},
    )

    assert first == second
    assert index.cache_lookup(STAGE_SIGNATURE) == first
    concurrent_equivalent = index.cache_store(
        STAGE_SIGNATURE,
        stage_name="static.geometry",
        artifact_id=ARTIFACT_ID,
        metadata={"schema": 1, "producer": "later-equivalent-worker"},
    )
    assert concurrent_equivalent.metadata == {"schema": 1}
    with pytest.raises(RunIndexConflictError, match="already maps"):
        index.cache_store(
            STAGE_SIGNATURE,
            stage_name="static.geometry",
            artifact_id=OTHER_ARTIFACT_ID,
            metadata={"schema": 1},
        )

    assert first.generation == 1
    assert [event.event_kind for event in index.list_cache_elections(STAGE_SIGNATURE)] == [
        "ELECTED"
    ]
    assert index.cache_forget(
        STAGE_SIGNATURE,
        expected_generation=first.generation,
        reason="artifact verification failed",
    )
    assert not index.cache_forget(STAGE_SIGNATURE)
    assert index.cache_lookup(STAGE_SIGNATURE) is None
    replaced = index.cache_store(
        STAGE_SIGNATURE,
        stage_name="static.geometry",
        artifact_id=OTHER_ARTIFACT_ID,
        metadata={"schema": 2, "producer": "recovery-worker"},
    )
    assert replaced.artifact_id == OTHER_ARTIFACT_ID
    assert replaced.metadata == {"schema": 2, "producer": "recovery-worker"}
    assert replaced.generation == 2
    assert replaced.created_at == first.created_at
    assert not index.cache_forget(STAGE_SIGNATURE, expected_generation=first.generation)
    assert index.cache_lookup(STAGE_SIGNATURE) == replaced

    history = index.list_cache_elections(STAGE_SIGNATURE)
    assert [(event.event_kind, event.generation, event.active) for event in history] == [
        ("ELECTED", 1, True),
        ("DEACTIVATED", 1, False),
        ("ELECTED", 2, True),
    ]
    assert history[1].reason == "artifact verification failed"
    assert history[-1].artifact_id == OTHER_ARTIFACT_ID
    assert history[-1].metadata["producer"] == "recovery-worker"


def test_stage_listing_is_stable(tmp_path) -> None:
    index = RunIndex(tmp_path / "index.sqlite3")
    index.register_run("run-1", config_hash=CONFIG_HASH)
    index.set_stage("run-1", "zeta", "1" * 64, StageState.PENDING)
    index.set_stage("run-1", "alpha", "2" * 64, StageState.PENDING)

    assert [record.stage_name for record in index.list_stages("run-1")] == ["alpha", "zeta"]


def test_cached_stage_can_be_recorded_complete_without_a_worker_attempt(tmp_path) -> None:
    index = RunIndex(tmp_path / "index.sqlite3")
    index.register_run("run-1", config_hash=CONFIG_HASH)
    index.set_stage("run-1", "cached", STAGE_SIGNATURE, StageState.PENDING)

    completed = index.set_stage(
        "run-1",
        "cached",
        STAGE_SIGNATURE,
        StageState.COMPLETED,
        artifact_id=ARTIFACT_ID,
        message="cache hit",
    )

    assert completed.state is StageState.COMPLETED
    assert completed.attempts == 0


def test_stage_claim_is_exclusive_and_reclaim_is_explicit(tmp_path) -> None:
    index = RunIndex(tmp_path / "index.sqlite3")
    index.register_run("run-1", config_hash=CONFIG_HASH)
    first = index.claim_stage("run-1", "paths", STAGE_SIGNATURE, owner_token="worker-1")

    assert first.owner_token == "worker-1"
    assert first.attempts == 1
    with pytest.raises(StageClaimError, match="another worker"):
        index.claim_stage("run-1", "paths", STAGE_SIGNATURE, owner_token="worker-2")
    with pytest.raises(StageClaimError, match="does not own"):
        index.finish_stage("run-1", "paths", owner_token="worker-2", state=StageState.FAILED)

    reclaimed = index.reclaim_stage(
        "run-1", "paths", owner_token="worker-2", reason="worker-1 process exited"
    )
    assert reclaimed.owner_token == "worker-2"
    assert reclaimed.attempts == 2
    assert reclaimed.message == "reclaimed: worker-1 process exited"
    completed = index.finish_stage(
        "run-1",
        "paths",
        owner_token="worker-2",
        state=StageState.COMPLETED,
        artifact_id=ARTIFACT_ID,
    )
    assert completed.owner_token is None


def test_legacy_set_stage_running_uses_an_exclusive_implicit_owner(tmp_path) -> None:
    path = tmp_path / "index.sqlite3"
    first = RunIndex(path)
    second = RunIndex(path)
    first.register_run("run-1", config_hash=CONFIG_HASH)

    claimed = first.set_stage("run-1", "stage", STAGE_SIGNATURE, StageState.RUNNING)

    assert claimed.owner_token is not None
    with pytest.raises(StageClaimError, match="another worker"):
        second.set_stage("run-1", "stage", STAGE_SIGNATURE, StageState.RUNNING)
    failed = first.set_stage("run-1", "stage", STAGE_SIGNATURE, StageState.FAILED, message="failed")
    assert failed.owner_token is None
    history = first.list_stage_attempts("run-1", "stage")
    assert [event.event_kind for event in history] == ["CLAIMED", "FINISHED"]
    assert history[-1].message == "failed"


def test_receipt_history_is_append_only_and_allows_repeat_content(tmp_path) -> None:
    index = RunIndex(tmp_path / "index.sqlite3")
    index.register_run("run-1", config_hash=CONFIG_HASH)

    first = index.append_receipt(
        "run-1",
        artifact_id=ARTIFACT_ID,
        requested_targets=("fixture",),
    )
    repeated = index.append_receipt(
        "run-1",
        artifact_id=ARTIFACT_ID,
        requested_targets=("fixture",),
    )
    incremental = index.append_receipt(
        "run-1",
        artifact_id=OTHER_ARTIFACT_ID,
        requested_targets=("fixture", "gate.fixture"),
    )

    assert [record.sequence for record in index.list_receipts("run-1")] == [1, 2, 3]
    assert first.artifact_id == repeated.artifact_id
    assert incremental.requested_targets == ("fixture", "gate.fixture")
    assert index.latest_receipt("run-1") == incremental
    assert index.get_receipt("run-1", 2) == repeated
    with pytest.raises(RunIndexValidationError, match="must not be empty"):
        index.append_receipt("run-1", artifact_id=ARTIFACT_ID, requested_targets=())


def test_attempt_history_preserves_reclaim_reason_through_retry(tmp_path) -> None:
    index = RunIndex(tmp_path / "index.sqlite3")
    index.register_run("run-1", config_hash=CONFIG_HASH)
    index.claim_stage(
        "run-1",
        "paths",
        STAGE_SIGNATURE,
        owner_token="worker-1",
        metadata={"batch": 1},
    )
    index.reclaim_stage(
        "run-1",
        "paths",
        owner_token="worker-2",
        reason="worker-1 process exited",
    )
    index.finish_stage(
        "run-1",
        "paths",
        owner_token="worker-2",
        state=StageState.FAILED,
        message="recovery audit complete",
        metadata={"batch": 1, "recovery": True},
    )
    index.claim_stage("run-1", "paths", STAGE_SIGNATURE, owner_token="worker-3")
    index.finish_stage(
        "run-1",
        "paths",
        owner_token="worker-3",
        state=StageState.COMPLETED,
        artifact_id=ARTIFACT_ID,
        metadata={"batch": 1, "recovery": True},
    )

    history = index.list_stage_attempts("run-1", "paths")
    observed = [
        (event.sequence, event.attempt, event.event_kind, event.state.value) for event in history
    ]
    assert observed == [
        (1, 1, "CLAIMED", "RUNNING"),
        (2, 2, "RECLAIMED", "RUNNING"),
        (3, 2, "FINISHED", "FAILED"),
        (4, 3, "CLAIMED", "RUNNING"),
        (5, 3, "FINISHED", "COMPLETED"),
    ]
    assert history[1].message == "reclaimed: worker-1 process exited"
    assert history[1].owner_token == "worker-2"
    assert history[-1].metadata == {"batch": 1, "recovery": True}


def test_v1_index_migrates_to_v4_without_losing_runs_or_faking_cache_history(
    tmp_path,
) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                config_hash TEXT NOT NULL,
                provenance_artifact_id TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE run_stages (
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                stage_name TEXT NOT NULL,
                stage_signature TEXT NOT NULL,
                state TEXT NOT NULL,
                owner_token TEXT,
                artifact_id TEXT,
                attempts INTEGER NOT NULL,
                message TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, stage_name)
            );
            CREATE INDEX run_stages_signature_idx ON run_stages(stage_signature);
            CREATE TABLE stage_cache (
                stage_signature TEXT PRIMARY KEY,
                stage_name TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO stage_cache VALUES (
                'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
                'legacy-stage',
                'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd',
                '{"producer":"legacy"}',
                1,
                '2026-08-16T00:00:00Z',
                '2026-08-16T00:00:00Z'
            );
            INSERT INTO runs VALUES (
                'legacy-run',
                'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                NULL,
                '{}',
                '2026-08-16T00:00:00Z',
                '2026-08-16T00:00:00Z'
            );
            INSERT INTO run_stages VALUES (
                'legacy-run',
                'legacy-stage',
                'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
                'COMPLETED',
                NULL,
                'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd',
                1,
                NULL,
                '{}',
                '2026-08-16T00:00:00Z',
                '2026-08-16T00:00:00Z'
            );
            PRAGMA user_version = 1;
            """
        )

    index = RunIndex(path)

    assert index.get_run("legacy-run") is not None
    assert index.get_stage("legacy-run", "legacy-stage") is not None
    assert index.list_stage_attempts("legacy-run", "legacy-stage") == ()
    receipt = index.append_receipt(
        "legacy-run",
        artifact_id=ARTIFACT_ID,
        requested_targets=("fixture",),
    )
    assert receipt.sequence == 1
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'stage_attempt_events'"
        ).fetchone() == ("stage_attempt_events",)
    legacy_cache = index.cache_lookup(STAGE_SIGNATURE)
    assert legacy_cache is not None
    assert legacy_cache.generation == 1
    cache_history = index.list_cache_elections(STAGE_SIGNATURE)
    assert len(cache_history) == 1
    assert cache_history[0].event_kind == "UNAVAILABLE_PRE_V4"
    assert cache_history[0].reason == "cache election history predates schema v4"


def test_v2_index_migrates_to_v4_without_losing_receipts(tmp_path) -> None:
    path = tmp_path / "legacy-v2.sqlite3"
    index = RunIndex(path)
    index.register_run("legacy-v2", config_hash=CONFIG_HASH)
    receipt = index.append_receipt(
        "legacy-v2", artifact_id=ARTIFACT_ID, requested_targets=("fixture",)
    )
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE stage_attempt_events")
        connection.execute("PRAGMA user_version = 2")

    migrated = RunIndex(path)

    assert migrated.latest_receipt("legacy-v2") == receipt
    migrated.claim_stage("legacy-v2", "fixture", STAGE_SIGNATURE, owner_token="post-migration")
    assert migrated.list_stage_attempts("legacy-v2", "fixture")[0].event_kind == "CLAIMED"
