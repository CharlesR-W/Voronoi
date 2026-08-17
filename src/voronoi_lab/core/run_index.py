"""Mutable SQLite index for runs, stage state, and immutable artifact cache keys."""

from __future__ import annotations

import fcntl
import json
import re
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from .hashing import JSONLike, canonical_json_text, freeze_json

INDEX_SCHEMA_VERSION = 4
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_EVENT_KINDS = frozenset({"CLAIMED", "RECLAIMED", "FINISHED"})
_CACHE_ELECTION_EVENT_KINDS = frozenset({"ELECTED", "DEACTIVATED", "UNAVAILABLE_PRE_V4"})


class RunIndexError(RuntimeError):
    """Base class for run-index failures."""


class RunIndexConflictError(RunIndexError):
    """Raised when a supposedly stable mapping changes without explicit replacement."""


class RunIndexValidationError(RunIndexError):
    """Raised for invalid identifiers, digests, metadata, or state transitions."""


class StageClaimError(RunIndexConflictError):
    """Raised when another worker owns a running stage."""


class StageState(StrEnum):
    """Mutable execution state; completed artifacts remain immutable elsewhere."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"


_ALLOWED_TRANSITIONS: dict[StageState, frozenset[StageState]] = {
    StageState.PENDING: frozenset(
        {
            StageState.PENDING,
            StageState.RUNNING,
            StageState.COMPLETED,
            StageState.FAILED,
            StageState.BLOCKED,
            StageState.SKIPPED,
        }
    ),
    StageState.RUNNING: frozenset(
        {StageState.RUNNING, StageState.COMPLETED, StageState.FAILED, StageState.BLOCKED}
    ),
    StageState.FAILED: frozenset(
        {StageState.FAILED, StageState.RUNNING, StageState.BLOCKED, StageState.SKIPPED}
    ),
    StageState.BLOCKED: frozenset({StageState.BLOCKED, StageState.RUNNING, StageState.SKIPPED}),
    StageState.SKIPPED: frozenset({StageState.SKIPPED, StageState.RUNNING}),
    StageState.COMPLETED: frozenset({StageState.COMPLETED}),
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise RunIndexValidationError(f"{label} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}}")
    return value


def _validate_digest(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise RunIndexValidationError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _metadata_text(metadata: Mapping[str, JSONLike] | None) -> str:
    return canonical_json_text({} if metadata is None else metadata)


def _metadata_from_text(value: str) -> Mapping[str, JSONLike]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise RunIndexError("stored metadata is not a JSON object")
    frozen = freeze_json(decoded)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    config_hash: str
    provenance_artifact_id: str | None
    metadata: Mapping[str, JSONLike]
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class StageRecord:
    run_id: str
    stage_name: str
    stage_signature: str
    state: StageState
    owner_token: str | None
    artifact_id: str | None
    attempts: int
    message: str | None
    metadata: Mapping[str, JSONLike]
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class CacheRecord:
    stage_signature: str
    stage_name: str
    artifact_id: str
    metadata: Mapping[str, JSONLike]
    generation: int
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class CacheElectionEvent:
    stage_signature: str
    event_sequence: int
    generation: int
    event_kind: str
    active: bool
    stage_name: str
    artifact_id: str
    metadata: Mapping[str, JSONLike]
    reason: str | None
    created_at: str

    @property
    def sequence(self) -> int:
        """Compatibility shorthand matching other append-only index events."""

        return self.event_sequence


@dataclass(frozen=True, slots=True)
class ReceiptRecord:
    run_id: str
    sequence: int
    artifact_id: str
    requested_targets: tuple[str, ...]
    created_at: str


@dataclass(frozen=True, slots=True)
class StageAttemptEvent:
    run_id: str
    stage_name: str
    event_sequence: int
    attempt: int
    event_kind: str
    state: StageState
    owner_token: str
    message: str | None
    metadata: Mapping[str, JSONLike]
    created_at: str

    @property
    def sequence(self) -> int:
        """Compatibility shorthand matching receipt sequence terminology."""

        return self.event_sequence


class RunIndex:
    """Concurrency-safe local index; payload bytes live in :class:`ArtifactStore`."""

    def __init__(self, path: str | Path, *, timeout: float = 30.0) -> None:
        self.path = Path(path)
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self._implicit_owner_token = f"index-{uuid.uuid4().hex}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def register_run(
        self,
        run_id: str,
        *,
        config_hash: str,
        provenance_artifact_id: str | None = None,
        metadata: Mapping[str, JSONLike] | None = None,
    ) -> RunRecord:
        """Register a run idempotently, rejecting changed immutable identity fields."""

        _validate_identifier(run_id, label="run_id")
        _validate_digest(config_hash, label="config_hash")
        if provenance_artifact_id is not None:
            _validate_digest(provenance_artifact_id, label="provenance_artifact_id")
        metadata_json = _metadata_text(metadata)
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, config_hash, provenance_artifact_id, metadata_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        config_hash,
                        provenance_artifact_id,
                        metadata_json,
                        now,
                        now,
                    ),
                )
            else:
                stable_existing = (
                    existing["config_hash"],
                    existing["provenance_artifact_id"],
                    existing["metadata_json"],
                )
                stable_requested = (config_hash, provenance_artifact_id, metadata_json)
                if stable_existing != stable_requested:
                    raise RunIndexConflictError(
                        f"run {run_id!r} is already registered with different identity data"
                    )
        record = self.get_run(run_id)
        assert record is not None
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        _validate_identifier(run_id, label="run_id")
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else self._run_from_row(row)

    def set_stage(
        self,
        run_id: str,
        stage_name: str,
        stage_signature: str,
        state: StageState | str,
        *,
        artifact_id: str | None = None,
        message: str | None = None,
        metadata: Mapping[str, JSONLike] | None = None,
    ) -> StageRecord:
        """Create or transition a stage state with strict completion identity."""

        _validate_identifier(run_id, label="run_id")
        _validate_identifier(stage_name, label="stage_name")
        _validate_digest(stage_signature, label="stage_signature")
        try:
            normalized_state = StageState(state)
        except ValueError as exc:
            raise RunIndexValidationError(f"unknown stage state: {state!r}") from exc
        if artifact_id is not None:
            _validate_digest(artifact_id, label="artifact_id")
        if normalized_state is StageState.COMPLETED and artifact_id is None:
            raise RunIndexValidationError("a completed stage must reference an artifact")
        if normalized_state is not StageState.COMPLETED and artifact_id is not None:
            raise RunIndexValidationError("only a completed stage may reference an artifact")
        if message is not None and (not isinstance(message, str) or not message.strip()):
            raise RunIndexValidationError("message must be a non-empty string when supplied")
        metadata_json = _metadata_text(metadata)
        now = _utc_now()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
                is None
            ):
                raise RunIndexValidationError(f"run is not registered: {run_id}")
            existing = connection.execute(
                "SELECT * FROM run_stages WHERE run_id = ? AND stage_name = ?",
                (run_id, stage_name),
            ).fetchone()
            changed = False
            if existing is None:
                attempts = 1 if normalized_state is StageState.RUNNING else 0
                owner_token = (
                    self._implicit_owner_token if normalized_state is StageState.RUNNING else None
                )
                connection.execute(
                    """
                    INSERT INTO run_stages (
                        run_id, stage_name, stage_signature, state, owner_token, artifact_id,
                        attempts, message, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        stage_name,
                        stage_signature,
                        normalized_state.value,
                        owner_token,
                        artifact_id,
                        attempts,
                        message,
                        metadata_json,
                        now,
                        now,
                    ),
                )
                if normalized_state is StageState.RUNNING:
                    assert owner_token is not None
                    self._insert_attempt_event(
                        connection,
                        run_id=run_id,
                        stage_name=stage_name,
                        attempt=attempts,
                        event_kind="CLAIMED",
                        state=StageState.RUNNING,
                        owner_token=owner_token,
                        message=None,
                        metadata_json=metadata_json,
                        created_at=now,
                    )
                changed = True
            else:
                old_state = StageState(existing["state"])
                if old_state is StageState.RUNNING:
                    if existing["owner_token"] != self._implicit_owner_token:
                        raise StageClaimError(
                            f"stage {run_id}/{stage_name} is owned by another worker"
                        )
                    if normalized_state is StageState.RUNNING:
                        return self._stage_from_row(existing)
                if existing["stage_signature"] != stage_signature:
                    raise RunIndexConflictError(
                        f"stage {run_id}/{stage_name} cannot change signature in place"
                    )
                if normalized_state not in _ALLOWED_TRANSITIONS[old_state]:
                    raise RunIndexValidationError(
                        f"invalid stage transition {old_state.value} -> {normalized_state.value}"
                    )
                if old_state is StageState.COMPLETED:
                    stable_existing = (
                        existing["artifact_id"],
                        existing["message"],
                        existing["metadata_json"],
                    )
                    stable_requested = (artifact_id, message, metadata_json)
                    if stable_existing != stable_requested:
                        raise RunIndexConflictError(
                            f"completed stage {run_id}/{stage_name} cannot be mutated"
                        )
                else:
                    attempts = existing["attempts"]
                    if (
                        normalized_state is StageState.RUNNING
                        and old_state is not StageState.RUNNING
                    ):
                        attempts += 1
                    owner_token = (
                        self._implicit_owner_token
                        if normalized_state is StageState.RUNNING
                        else None
                    )
                    connection.execute(
                        """
                        UPDATE run_stages
                        SET state = ?, owner_token = ?, artifact_id = ?, attempts = ?,
                            message = ?,
                            metadata_json = ?, updated_at = ?
                        WHERE run_id = ? AND stage_name = ?
                        """,
                        (
                            normalized_state.value,
                            owner_token,
                            artifact_id,
                            attempts,
                            message,
                            metadata_json,
                            now,
                            run_id,
                            stage_name,
                        ),
                    )
                    if normalized_state is StageState.RUNNING:
                        assert owner_token is not None
                        self._insert_attempt_event(
                            connection,
                            run_id=run_id,
                            stage_name=stage_name,
                            attempt=attempts,
                            event_kind="CLAIMED",
                            state=StageState.RUNNING,
                            owner_token=owner_token,
                            message=None,
                            metadata_json=metadata_json,
                            created_at=now,
                        )
                    elif old_state is StageState.RUNNING:
                        previous_owner = existing["owner_token"]
                        assert isinstance(previous_owner, str)
                        self._insert_attempt_event(
                            connection,
                            run_id=run_id,
                            stage_name=stage_name,
                            attempt=attempts,
                            event_kind="FINISHED",
                            state=normalized_state,
                            owner_token=previous_owner,
                            message=message,
                            metadata_json=metadata_json,
                            created_at=now,
                        )
                    changed = True
            if changed:
                connection.execute("UPDATE runs SET updated_at = ? WHERE run_id = ?", (now, run_id))
        record = self.get_stage(run_id, stage_name)
        assert record is not None
        return record

    def claim_stage(
        self,
        run_id: str,
        stage_name: str,
        stage_signature: str,
        *,
        owner_token: str,
        metadata: Mapping[str, JSONLike] | None = None,
    ) -> StageRecord:
        """Atomically claim a resumable stage for one worker."""

        _validate_identifier(run_id, label="run_id")
        _validate_identifier(stage_name, label="stage_name")
        _validate_digest(stage_signature, label="stage_signature")
        _validate_identifier(owner_token, label="owner_token")
        requested_metadata = None if metadata is None else _metadata_text(metadata)
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
                is None
            ):
                raise RunIndexValidationError(f"run is not registered: {run_id}")
            existing = connection.execute(
                "SELECT * FROM run_stages WHERE run_id = ? AND stage_name = ?",
                (run_id, stage_name),
            ).fetchone()
            if existing is None:
                metadata_json = "{}" if requested_metadata is None else requested_metadata
                attempt = 1
                connection.execute(
                    """
                    INSERT INTO run_stages (
                        run_id, stage_name, stage_signature, state, owner_token, artifact_id,
                        attempts, message, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, NULL, 1, NULL, ?, ?, ?)
                    """,
                    (
                        run_id,
                        stage_name,
                        stage_signature,
                        StageState.RUNNING.value,
                        owner_token,
                        metadata_json,
                        now,
                        now,
                    ),
                )
            else:
                if existing["stage_signature"] != stage_signature:
                    raise RunIndexConflictError(
                        f"stage {run_id}/{stage_name} cannot change signature in place"
                    )
                old_state = StageState(existing["state"])
                if old_state is StageState.COMPLETED:
                    raise RunIndexConflictError(
                        f"completed stage {run_id}/{stage_name} cannot be claimed"
                    )
                if old_state is StageState.RUNNING:
                    if existing["owner_token"] != owner_token:
                        raise StageClaimError(
                            f"stage {run_id}/{stage_name} is already claimed by another worker"
                        )
                    return self._stage_from_row(existing)
                metadata_json = (
                    existing["metadata_json"] if requested_metadata is None else requested_metadata
                )
                attempt = int(existing["attempts"]) + 1
                connection.execute(
                    """
                    UPDATE run_stages
                    SET state = ?, owner_token = ?, artifact_id = NULL,
                        attempts = attempts + 1, message = NULL,
                        metadata_json = ?, updated_at = ?
                    WHERE run_id = ? AND stage_name = ?
                    """,
                    (
                        StageState.RUNNING.value,
                        owner_token,
                        metadata_json,
                        now,
                        run_id,
                        stage_name,
                    ),
                )
            self._insert_attempt_event(
                connection,
                run_id=run_id,
                stage_name=stage_name,
                attempt=attempt,
                event_kind="CLAIMED",
                state=StageState.RUNNING,
                owner_token=owner_token,
                message=None,
                metadata_json=metadata_json,
                created_at=now,
            )
            connection.execute("UPDATE runs SET updated_at = ? WHERE run_id = ?", (now, run_id))
        record = self.get_stage(run_id, stage_name)
        assert record is not None
        return record

    def finish_stage(
        self,
        run_id: str,
        stage_name: str,
        *,
        owner_token: str,
        state: StageState | str,
        artifact_id: str | None = None,
        message: str | None = None,
        metadata: Mapping[str, JSONLike] | None = None,
    ) -> StageRecord:
        """Finish a claimed stage using an owner-token compare-and-swap."""

        _validate_identifier(run_id, label="run_id")
        _validate_identifier(stage_name, label="stage_name")
        _validate_identifier(owner_token, label="owner_token")
        try:
            final_state = StageState(state)
        except ValueError as exc:
            raise RunIndexValidationError(f"unknown stage state: {state!r}") from exc
        if final_state not in {StageState.COMPLETED, StageState.FAILED, StageState.BLOCKED}:
            raise RunIndexValidationError("claimed stages may finish COMPLETED, FAILED, or BLOCKED")
        if final_state is StageState.COMPLETED:
            if artifact_id is None:
                raise RunIndexValidationError("a completed stage must reference an artifact")
            _validate_digest(artifact_id, label="artifact_id")
        elif artifact_id is not None:
            raise RunIndexValidationError("only a completed stage may reference an artifact")
        if message is not None and (not isinstance(message, str) or not message.strip()):
            raise RunIndexValidationError("message must be a non-empty string when supplied")
        requested_metadata = None if metadata is None else _metadata_text(metadata)
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM run_stages WHERE run_id = ? AND stage_name = ?",
                (run_id, stage_name),
            ).fetchone()
            if existing is None or StageState(existing["state"]) is not StageState.RUNNING:
                raise StageClaimError(f"stage {run_id}/{stage_name} is not currently claimed")
            if existing["owner_token"] != owner_token:
                raise StageClaimError(f"worker does not own claimed stage {run_id}/{stage_name}")
            metadata_json = (
                existing["metadata_json"] if requested_metadata is None else requested_metadata
            )
            connection.execute(
                """
                UPDATE run_stages
                SET state = ?, owner_token = NULL, artifact_id = ?, message = ?,
                    metadata_json = ?, updated_at = ?
                WHERE run_id = ? AND stage_name = ? AND owner_token = ?
                """,
                (
                    final_state.value,
                    artifact_id,
                    message,
                    metadata_json,
                    now,
                    run_id,
                    stage_name,
                    owner_token,
                ),
            )
            self._insert_attempt_event(
                connection,
                run_id=run_id,
                stage_name=stage_name,
                attempt=int(existing["attempts"]),
                event_kind="FINISHED",
                state=final_state,
                owner_token=owner_token,
                message=message,
                metadata_json=metadata_json,
                created_at=now,
            )
            connection.execute("UPDATE runs SET updated_at = ? WHERE run_id = ?", (now, run_id))
        record = self.get_stage(run_id, stage_name)
        assert record is not None
        return record

    def reclaim_stage(
        self,
        run_id: str,
        stage_name: str,
        *,
        owner_token: str,
        reason: str,
    ) -> StageRecord:
        """Explicitly take over a crashed worker's claim, preserving a reason."""

        _validate_identifier(run_id, label="run_id")
        _validate_identifier(stage_name, label="stage_name")
        _validate_identifier(owner_token, label="owner_token")
        if not isinstance(reason, str) or not reason.strip():
            raise RunIndexValidationError("reclaim reason must be a non-empty string")
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM run_stages WHERE run_id = ? AND stage_name = ?",
                (run_id, stage_name),
            ).fetchone()
            if existing is None or StageState(existing["state"]) is not StageState.RUNNING:
                raise StageClaimError(f"stage {run_id}/{stage_name} is not currently claimed")
            if existing["owner_token"] == owner_token:
                raise StageClaimError("reclaim requires a new owner token")
            reclaim_message = f"reclaimed: {reason.strip()}"
            attempt = int(existing["attempts"]) + 1
            connection.execute(
                """
                UPDATE run_stages
                SET owner_token = ?, attempts = attempts + 1, message = ?, updated_at = ?
                WHERE run_id = ? AND stage_name = ?
                """,
                (owner_token, reclaim_message, now, run_id, stage_name),
            )
            self._insert_attempt_event(
                connection,
                run_id=run_id,
                stage_name=stage_name,
                attempt=attempt,
                event_kind="RECLAIMED",
                state=StageState.RUNNING,
                owner_token=owner_token,
                message=reclaim_message,
                metadata_json=existing["metadata_json"],
                created_at=now,
            )
            connection.execute("UPDATE runs SET updated_at = ? WHERE run_id = ?", (now, run_id))
        record = self.get_stage(run_id, stage_name)
        assert record is not None
        return record

    def get_stage(self, run_id: str, stage_name: str) -> StageRecord | None:
        _validate_identifier(run_id, label="run_id")
        _validate_identifier(stage_name, label="stage_name")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM run_stages WHERE run_id = ? AND stage_name = ?",
                (run_id, stage_name),
            ).fetchone()
        return None if row is None else self._stage_from_row(row)

    def list_stages(self, run_id: str) -> tuple[StageRecord, ...]:
        _validate_identifier(run_id, label="run_id")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM run_stages WHERE run_id = ? ORDER BY stage_name", (run_id,)
            ).fetchall()
        return tuple(self._stage_from_row(row) for row in rows)

    def list_stage_attempts(self, run_id: str, stage_name: str) -> tuple[StageAttemptEvent, ...]:
        """Return the immutable attempt-event history for one stage."""

        _validate_identifier(run_id, label="run_id")
        _validate_identifier(stage_name, label="stage_name")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM stage_attempt_events "
                "WHERE run_id = ? AND stage_name = ? ORDER BY event_sequence",
                (run_id, stage_name),
            ).fetchall()
        return tuple(self._attempt_event_from_row(row) for row in rows)

    def append_receipt(
        self,
        run_id: str,
        *,
        artifact_id: str,
        requested_targets: Sequence[str],
    ) -> ReceiptRecord:
        """Append one immutable receipt pointer to a run's audit history.

        Repeated snapshots may intentionally reference the same content-addressed
        artifact.  The monotonically increasing sequence records that each
        successful ``run`` call emitted a receipt without making receipt bytes
        depend on mutable database state or wall-clock time.
        """

        _validate_identifier(run_id, label="run_id")
        _validate_digest(artifact_id, label="receipt artifact_id")
        targets = tuple(requested_targets)
        if not targets:
            raise RunIndexValidationError("receipt requested_targets must not be empty")
        for target in targets:
            _validate_identifier(target, label="receipt target")
        targets_json = canonical_json_text(targets)
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
                is None
            ):
                raise RunIndexValidationError(f"run is not registered: {run_id}")
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(receipt_sequence), 0) + 1 "
                    "FROM run_receipts WHERE run_id = ?",
                    (run_id,),
                ).fetchone()[0]
            )
            connection.execute(
                """
                INSERT INTO run_receipts (
                    run_id, receipt_sequence, artifact_id, requested_targets_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, sequence, artifact_id, targets_json, now),
            )
        record = self.get_receipt(run_id, sequence)
        assert record is not None
        return record

    def get_receipt(self, run_id: str, sequence: int) -> ReceiptRecord | None:
        _validate_identifier(run_id, label="run_id")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise RunIndexValidationError("receipt sequence must be a positive integer")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM run_receipts WHERE run_id = ? AND receipt_sequence = ?",
                (run_id, sequence),
            ).fetchone()
        return None if row is None else self._receipt_from_row(row)

    def list_receipts(self, run_id: str) -> tuple[ReceiptRecord, ...]:
        _validate_identifier(run_id, label="run_id")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM run_receipts WHERE run_id = ? ORDER BY receipt_sequence",
                (run_id,),
            ).fetchall()
        return tuple(self._receipt_from_row(row) for row in rows)

    def latest_receipt(self, run_id: str) -> ReceiptRecord | None:
        _validate_identifier(run_id, label="run_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM run_receipts WHERE run_id = ? "
                "ORDER BY receipt_sequence DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return None if row is None else self._receipt_from_row(row)

    def cache_store(
        self,
        stage_signature: str,
        *,
        stage_name: str,
        artifact_id: str,
        metadata: Mapping[str, JSONLike] | None = None,
    ) -> CacheRecord:
        """Elect an immutable artifact for a deterministic stage signature.

        An active election remains first-publisher-wins.  Once explicitly
        deactivated, a corrected recomputation may atomically replace both the
        artifact and its producer metadata.  Every election transition is kept
        in the append-only cache history.
        """

        _validate_digest(stage_signature, label="stage_signature")
        _validate_identifier(stage_name, label="stage_name")
        _validate_digest(artifact_id, label="artifact_id")
        metadata_json = _metadata_text(metadata)
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM stage_cache WHERE stage_signature = ?", (stage_signature,)
            ).fetchone()
            if existing is None:
                generation = 1
                connection.execute(
                    """
                    INSERT INTO stage_cache (
                        stage_signature, stage_name, artifact_id, metadata_json,
                        generation, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stage_signature,
                        stage_name,
                        artifact_id,
                        metadata_json,
                        generation,
                        now,
                        now,
                    ),
                )
                self._insert_cache_election_event(
                    connection,
                    stage_signature=stage_signature,
                    generation=generation,
                    event_kind="ELECTED",
                    active=True,
                    stage_name=stage_name,
                    artifact_id=artifact_id,
                    metadata_json=metadata_json,
                    reason=None,
                    created_at=now,
                )
            else:
                if existing["active"]:
                    stable_existing = (existing["stage_name"], existing["artifact_id"])
                    stable_requested = (stage_name, artifact_id)
                    if stable_existing != stable_requested:
                        raise RunIndexConflictError(
                            f"cache signature {stage_signature} already maps to another artifact"
                        )
                    # Metadata describes the first successful publisher in this election,
                    # not cache identity. Concurrent equivalent publishers preserve it.
                else:
                    if existing["stage_name"] != stage_name:
                        raise RunIndexConflictError(
                            f"cache signature {stage_signature} belongs to another stage"
                        )
                    generation = int(existing["generation"]) + 1
                    connection.execute(
                        """
                        UPDATE stage_cache
                        SET artifact_id = ?, metadata_json = ?, active = 1,
                            generation = ?, updated_at = ?
                        WHERE stage_signature = ?
                        """,
                        (artifact_id, metadata_json, generation, now, stage_signature),
                    )
                    self._insert_cache_election_event(
                        connection,
                        stage_signature=stage_signature,
                        generation=generation,
                        event_kind="ELECTED",
                        active=True,
                        stage_name=stage_name,
                        artifact_id=artifact_id,
                        metadata_json=metadata_json,
                        reason=None,
                        created_at=now,
                    )
            elected = connection.execute(
                "SELECT * FROM stage_cache WHERE stage_signature = ? AND active = 1",
                (stage_signature,),
            ).fetchone()
            assert elected is not None
            record = self._cache_from_row(elected)
        return record

    def cache_lookup(self, stage_signature: str) -> CacheRecord | None:
        _validate_digest(stage_signature, label="stage_signature")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM stage_cache WHERE stage_signature = ? AND active = 1",
                (stage_signature,),
            ).fetchone()
        return None if row is None else self._cache_from_row(row)

    def cache_forget(
        self,
        stage_signature: str,
        *,
        expected_generation: int | None = None,
        reason: str | None = None,
    ) -> bool:
        """Deactivate one cache election without deleting immutable artifacts.

        ``expected_generation`` provides compare-and-swap semantics for callers
        acting on a previously read record, so they cannot deactivate a newer
        recovery election that won a concurrent race.
        """

        _validate_digest(stage_signature, label="stage_signature")
        if expected_generation is not None and (
            isinstance(expected_generation, bool)
            or not isinstance(expected_generation, int)
            or expected_generation < 1
        ):
            raise RunIndexValidationError("expected cache generation must be positive")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise RunIndexValidationError("cache deactivation reason must be non-empty")
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM stage_cache WHERE stage_signature = ? AND active = 1",
                (stage_signature,),
            ).fetchone()
            if existing is None or (
                expected_generation is not None
                and int(existing["generation"]) != expected_generation
            ):
                return False
            connection.execute(
                "UPDATE stage_cache SET active = 0, updated_at = ? "
                "WHERE stage_signature = ? AND active = 1 AND generation = ?",
                (now, stage_signature, existing["generation"]),
            )
            self._insert_cache_election_event(
                connection,
                stage_signature=stage_signature,
                generation=int(existing["generation"]),
                event_kind="DEACTIVATED",
                active=False,
                stage_name=existing["stage_name"],
                artifact_id=existing["artifact_id"],
                metadata_json=existing["metadata_json"],
                reason=None if reason is None else reason.strip(),
                created_at=now,
            )
        return True

    def list_cache_elections(self, stage_signature: str) -> tuple[CacheElectionEvent, ...]:
        """Return the immutable election/deactivation history for a signature."""

        _validate_digest(stage_signature, label="stage_signature")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cache_election_events "
                "WHERE stage_signature = ? ORDER BY event_sequence",
                (stage_signature,),
            ).fetchall()
        return tuple(self._cache_election_event_from_row(row) for row in rows)

    def _initialize(self) -> None:
        lock_path = self.path.with_name(f"{self.path.name}.init.lock")
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                with self._connect() as connection:
                    current_version = connection.execute("PRAGMA user_version").fetchone()[0]
                    if current_version == INDEX_SCHEMA_VERSION:
                        return
                    if current_version not in {0, 1, 2, 3}:
                        raise RunIndexError(
                            f"unsupported run-index schema {current_version}; "
                            f"expected {INDEX_SCHEMA_VERSION}"
                        )
                    connection.execute("PRAGMA journal_mode = WAL")
                    if current_version == 0:
                        connection.executescript(
                            """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    config_hash TEXT NOT NULL,
                    provenance_artifact_id TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_stages (
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

                CREATE INDEX IF NOT EXISTS run_stages_signature_idx
                    ON run_stages(stage_signature);

                CREATE TABLE IF NOT EXISTS stage_cache (
                    stage_signature TEXT PRIMARY KEY,
                    stage_name TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    generation INTEGER NOT NULL DEFAULT 1 CHECK (generation > 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
                        )
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS run_receipts (
                            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                            receipt_sequence INTEGER NOT NULL CHECK (receipt_sequence > 0),
                            artifact_id TEXT NOT NULL,
                            requested_targets_json TEXT NOT NULL,
                            created_at TEXT NOT NULL,
                            PRIMARY KEY (run_id, receipt_sequence)
                        )
                        """
                    )
                    connection.execute(
                        "CREATE INDEX IF NOT EXISTS run_receipts_artifact_idx "
                        "ON run_receipts(artifact_id)"
                    )
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS stage_attempt_events (
                            run_id TEXT NOT NULL,
                            stage_name TEXT NOT NULL,
                            event_sequence INTEGER NOT NULL CHECK (event_sequence > 0),
                            attempt INTEGER NOT NULL CHECK (attempt > 0),
                            event_kind TEXT NOT NULL,
                            state TEXT NOT NULL,
                            owner_token TEXT NOT NULL,
                            message TEXT,
                            metadata_json TEXT NOT NULL,
                            created_at TEXT NOT NULL,
                            PRIMARY KEY (run_id, stage_name, event_sequence),
                            FOREIGN KEY (run_id, stage_name)
                                REFERENCES run_stages(run_id, stage_name) ON DELETE CASCADE
                        )
                        """
                    )
                    connection.execute(
                        "CREATE INDEX IF NOT EXISTS stage_attempt_events_attempt_idx "
                        "ON stage_attempt_events(run_id, stage_name, attempt, event_sequence)"
                    )
                    cache_columns = {
                        row["name"] for row in connection.execute("PRAGMA table_info(stage_cache)")
                    }
                    if "generation" not in cache_columns:
                        connection.execute(
                            "ALTER TABLE stage_cache ADD COLUMN "
                            "generation INTEGER NOT NULL DEFAULT 1 CHECK (generation > 0)"
                        )
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS cache_election_events (
                            stage_signature TEXT NOT NULL,
                            event_sequence INTEGER NOT NULL CHECK (event_sequence > 0),
                            generation INTEGER NOT NULL CHECK (generation > 0),
                            event_kind TEXT NOT NULL,
                            active INTEGER NOT NULL CHECK (active IN (0, 1)),
                            stage_name TEXT NOT NULL,
                            artifact_id TEXT NOT NULL,
                            metadata_json TEXT NOT NULL,
                            reason TEXT,
                            created_at TEXT NOT NULL,
                            PRIMARY KEY (stage_signature, event_sequence),
                            FOREIGN KEY (stage_signature)
                                REFERENCES stage_cache(stage_signature) ON DELETE CASCADE
                        )
                        """
                    )
                    connection.execute(
                        "CREATE INDEX IF NOT EXISTS cache_election_events_generation_idx "
                        "ON cache_election_events(stage_signature, generation, event_sequence)"
                    )
                    if current_version in {1, 2, 3}:
                        connection.execute(
                            """
                            INSERT INTO cache_election_events (
                                stage_signature, event_sequence, generation, event_kind,
                                active, stage_name, artifact_id, metadata_json, reason,
                                created_at
                            )
                            SELECT stage_signature, 1, generation, 'UNAVAILABLE_PRE_V4',
                                   active, stage_name, artifact_id, metadata_json,
                                   'cache election history predates schema v4', updated_at
                            FROM stage_cache
                            WHERE NOT EXISTS (
                                SELECT 1 FROM cache_election_events AS history
                                WHERE history.stage_signature = stage_cache.stage_signature
                            )
                            """
                        )
                    connection.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.timeout)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        return connection

    @staticmethod
    def _insert_attempt_event(
        connection: sqlite3.Connection,
        *,
        run_id: str,
        stage_name: str,
        attempt: int,
        event_kind: str,
        state: StageState,
        owner_token: str,
        message: str | None,
        metadata_json: str,
        created_at: str,
    ) -> None:
        sequence = int(
            connection.execute(
                "SELECT COALESCE(MAX(event_sequence), 0) + 1 "
                "FROM stage_attempt_events WHERE run_id = ? AND stage_name = ?",
                (run_id, stage_name),
            ).fetchone()[0]
        )
        connection.execute(
            """
            INSERT INTO stage_attempt_events (
                run_id, stage_name, event_sequence, attempt, event_kind, state,
                owner_token, message, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                stage_name,
                sequence,
                attempt,
                event_kind,
                state.value,
                owner_token,
                message,
                metadata_json,
                created_at,
            ),
        )

    @staticmethod
    def _insert_cache_election_event(
        connection: sqlite3.Connection,
        *,
        stage_signature: str,
        generation: int,
        event_kind: str,
        active: bool,
        stage_name: str,
        artifact_id: str,
        metadata_json: str,
        reason: str | None,
        created_at: str,
    ) -> None:
        if event_kind not in _CACHE_ELECTION_EVENT_KINDS:
            raise RunIndexValidationError(f"unknown cache election event: {event_kind!r}")
        sequence = int(
            connection.execute(
                "SELECT COALESCE(MAX(event_sequence), 0) + 1 "
                "FROM cache_election_events WHERE stage_signature = ?",
                (stage_signature,),
            ).fetchone()[0]
        )
        connection.execute(
            """
            INSERT INTO cache_election_events (
                stage_signature, event_sequence, generation, event_kind, active,
                stage_name, artifact_id, metadata_json, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stage_signature,
                sequence,
                generation,
                event_kind,
                int(active),
                stage_name,
                artifact_id,
                metadata_json,
                reason,
                created_at,
            ),
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            config_hash=row["config_hash"],
            provenance_artifact_id=row["provenance_artifact_id"],
            metadata=_metadata_from_text(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _stage_from_row(row: sqlite3.Row) -> StageRecord:
        return StageRecord(
            run_id=row["run_id"],
            stage_name=row["stage_name"],
            stage_signature=row["stage_signature"],
            state=StageState(row["state"]),
            owner_token=row["owner_token"],
            artifact_id=row["artifact_id"],
            attempts=row["attempts"],
            message=row["message"],
            metadata=_metadata_from_text(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _cache_from_row(row: sqlite3.Row) -> CacheRecord:
        return CacheRecord(
            stage_signature=row["stage_signature"],
            stage_name=row["stage_name"],
            artifact_id=row["artifact_id"],
            metadata=_metadata_from_text(row["metadata_json"]),
            generation=row["generation"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _cache_election_event_from_row(row: sqlite3.Row) -> CacheElectionEvent:
        event_kind = row["event_kind"]
        if event_kind not in _CACHE_ELECTION_EVENT_KINDS:
            raise RunIndexError(f"stored cache election event kind is invalid: {event_kind!r}")
        active = row["active"]
        if active not in {0, 1}:
            raise RunIndexError("stored cache election active flag is invalid")
        return CacheElectionEvent(
            stage_signature=row["stage_signature"],
            event_sequence=row["event_sequence"],
            generation=row["generation"],
            event_kind=event_kind,
            active=bool(active),
            stage_name=row["stage_name"],
            artifact_id=row["artifact_id"],
            metadata=_metadata_from_text(row["metadata_json"]),
            reason=row["reason"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _receipt_from_row(row: sqlite3.Row) -> ReceiptRecord:
        decoded = json.loads(row["requested_targets_json"])
        if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
            raise RunIndexError("stored receipt targets are not a string array")
        return ReceiptRecord(
            run_id=row["run_id"],
            sequence=row["receipt_sequence"],
            artifact_id=row["artifact_id"],
            requested_targets=tuple(decoded),
            created_at=row["created_at"],
        )

    @staticmethod
    def _attempt_event_from_row(row: sqlite3.Row) -> StageAttemptEvent:
        event_kind = row["event_kind"]
        if event_kind not in _ATTEMPT_EVENT_KINDS:
            raise RunIndexError(f"stored attempt event kind is invalid: {event_kind!r}")
        return StageAttemptEvent(
            run_id=row["run_id"],
            stage_name=row["stage_name"],
            event_sequence=row["event_sequence"],
            attempt=row["attempt"],
            event_kind=event_kind,
            state=StageState(row["state"]),
            owner_token=row["owner_token"],
            message=row["message"],
            metadata=_metadata_from_text(row["metadata_json"]),
            created_at=row["created_at"],
        )
