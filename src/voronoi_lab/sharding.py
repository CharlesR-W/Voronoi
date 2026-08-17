"""Deterministic, resumable execution for expensive stage shards.

This module deliberately keeps shard orchestration independent of models and
array formats.  A handler may publish any artifact kind, but the artifact must
carry the complete scientific identity supplied by :class:`ShardSpec`.
Mutable progress lives in :class:`~voronoi_lab.core.RunIndex`; shard payloads
and reducer manifests remain immutable content-addressed artifacts.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from operator import index as integer_index
from uuid import uuid4

from voronoi_lab.core import (
    ArtifactRef,
    ArtifactStore,
    CacheRecord,
    JSONLike,
    JSONValue,
    RunIndex,
    RunIndexConflictError,
    StageRecord,
    StageState,
    canonical_hash,
    freeze_json,
    thaw_json,
)

SHARD_EXECUTION_SCHEMA_VERSION = 1
REDUCER_MANIFEST_SCHEMA_VERSION = 1

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_COORDINATE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,63}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_KIND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_SHARD_DOMAIN = "voronoi-lab/shard-execution/v1"
_SHARD_KEY_DOMAIN = "voronoi-lab/shard-key/v1"
_SHARD_ROW_DOMAIN = "voronoi-lab/shard-row/v1"
_REDUCER_DOMAIN = "voronoi-lab/shard-reducer/v1"


class ShardError(RuntimeError):
    """Base class for shard planning, execution, and reduction failures."""


class ShardValidationError(ShardError):
    """Raised when shard identity or an immutable artifact is inconsistent."""


class ShardReductionError(ShardError):
    """Raised when a complete, exact shard set cannot be reduced safely."""


def _validate_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ShardValidationError(f"{label} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}}")
    return value


def _validate_digest(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise ShardValidationError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _freeze_mapping(
    value: Mapping[str, JSONLike], *, label: str, require_nonempty: bool = False
) -> Mapping[str, JSONLike]:
    if not isinstance(value, Mapping):
        raise ShardValidationError(f"{label} must be a JSON object")
    if require_nonempty and not value:
        raise ShardValidationError(f"{label} must not be empty")
    try:
        frozen = freeze_json(value)
    except (TypeError, ValueError) as error:
        raise ShardValidationError(f"{label} must contain only finite JSON values") from error
    if not isinstance(frozen, Mapping):  # Defensive: the input type already establishes this.
        raise ShardValidationError(f"{label} must be a JSON object")
    return frozen


def _plain_mapping(value: Mapping[str, JSONLike]) -> dict[str, JSONValue]:
    plain = thaw_json(value)
    assert isinstance(plain, dict)
    return plain


def _identity_mismatches(
    actual: Mapping[str, JSONLike], expected: Mapping[str, JSONLike]
) -> list[str]:
    return [
        name
        for name, value in expected.items()
        if name not in actual or canonical_hash(actual[name]) != canonical_hash(value)
    ]


@dataclass(frozen=True, slots=True)
class ShardKey:
    """Named semantic coordinates identifying one shard within a parent stage."""

    coordinates: Mapping[str, JSONLike]

    def __post_init__(self) -> None:
        coordinates = _freeze_mapping(
            self.coordinates, label="shard coordinates", require_nonempty=True
        )
        invalid = [name for name in coordinates if not _COORDINATE_RE.fullmatch(name)]
        if invalid:
            raise ShardValidationError(
                "shard coordinate names must match "
                f"[A-Za-z][A-Za-z0-9._-]{{0,63}}: {sorted(invalid)!r}"
            )
        object.__setattr__(self, "coordinates", coordinates)

    @property
    def signature(self) -> str:
        return canonical_hash({"domain": _SHARD_KEY_DOMAIN, "coordinates": self.coordinates})

    def to_dict(self) -> dict[str, JSONValue]:
        return _plain_mapping(self.coordinates)


@dataclass(frozen=True, slots=True)
class ShardSpec:
    """Complete cache identity and output contract for one independently runnable shard."""

    parent_stage: str
    parent_stage_signature: str
    key: ShardKey
    artifact_kind: str
    stage_config: Mapping[str, JSONLike]
    source_identity: Mapping[str, JSONLike]
    upstream_artifacts: Mapping[str, str]
    shard_version: int = 1

    def __post_init__(self) -> None:
        _validate_identifier(self.parent_stage, label="parent_stage")
        _validate_digest(self.parent_stage_signature, label="parent_stage_signature")
        if not isinstance(self.key, ShardKey):
            raise ShardValidationError("key must be a ShardKey")
        if not isinstance(self.artifact_kind, str) or not _KIND_RE.fullmatch(self.artifact_kind):
            raise ShardValidationError("artifact_kind must match [A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
        if (
            isinstance(self.shard_version, bool)
            or not isinstance(self.shard_version, int)
            or self.shard_version < 1
        ):
            raise ShardValidationError("shard_version must be a positive integer")
        stage_config = _freeze_mapping(self.stage_config, label="stage_config")
        source_identity = _freeze_mapping(self.source_identity, label="source_identity")
        if not isinstance(self.upstream_artifacts, Mapping):
            raise ShardValidationError("upstream_artifacts must be a JSON object")
        upstream: dict[str, JSONLike] = {}
        for name, artifact_id in self.upstream_artifacts.items():
            _validate_identifier(name, label="upstream stage name")
            upstream[name] = _validate_digest(artifact_id, label=f"upstream artifact {name}")
        upstream_frozen = _freeze_mapping(upstream, label="upstream_artifacts")
        object.__setattr__(self, "stage_config", stage_config)
        object.__setattr__(self, "source_identity", source_identity)
        object.__setattr__(self, "upstream_artifacts", upstream_frozen)

    @property
    def signature(self) -> str:
        return canonical_hash(self._signature_payload())

    @property
    def row_name(self) -> str:
        """Stable RunIndex row name for this parent-stage/key pair.

        The row deliberately excludes mutable execution inputs.  Reusing a run id
        with changed inputs therefore raises a RunIndex signature conflict instead
        of silently creating a second version of the same semantic shard.
        """

        row_digest = canonical_hash(
            {
                "domain": _SHARD_ROW_DOMAIN,
                "parent_stage": self.parent_stage,
                "shard_key": self.key.coordinates,
            }
        )
        return f"shard.{row_digest}"

    @property
    def artifact_metadata(self) -> dict[str, JSONLike]:
        """Required immutable metadata for artifacts produced by this shard."""

        return {
            "parent_stage": self.parent_stage,
            "parent_stage_signature": self.parent_stage_signature,
            "shard_execution_schema_version": SHARD_EXECUTION_SCHEMA_VERSION,
            "shard_key": self.key.to_dict(),
            "shard_key_signature": self.key.signature,
            "shard_signature": self.signature,
            "shard_version": self.shard_version,
            "source_identity": _plain_mapping(self.source_identity),
            "stage_config": _plain_mapping(self.stage_config),
            "upstream_artifacts": _plain_mapping(self.upstream_artifacts),
        }

    @property
    def reduction_identity(self) -> dict[str, JSONLike]:
        """Fields that every shard in one exact reduction must share."""

        return {
            "artifact_kind": self.artifact_kind,
            "parent_stage": self.parent_stage,
            "parent_stage_signature": self.parent_stage_signature,
            "shard_execution_schema_version": SHARD_EXECUTION_SCHEMA_VERSION,
            "shard_version": self.shard_version,
            "source_identity": _plain_mapping(self.source_identity),
            "stage_config": _plain_mapping(self.stage_config),
            "upstream_artifacts": _plain_mapping(self.upstream_artifacts),
        }

    def _signature_payload(self) -> dict[str, JSONLike]:
        return {
            "artifact_kind": self.artifact_kind,
            "domain": _SHARD_DOMAIN,
            "parent_stage": self.parent_stage,
            "parent_stage_signature": self.parent_stage_signature,
            "shard_key": self.key.coordinates,
            "shard_version": self.shard_version,
            "source_identity": self.source_identity,
            "stage_config": self.stage_config,
            "upstream_artifacts": self.upstream_artifacts,
        }


@dataclass(frozen=True, slots=True)
class ImageChunk:
    """One contiguous chunk in a deterministic image sequence."""

    ordinal: int
    start: int
    stop: int
    image_ids: tuple[int, ...]
    key: ShardKey


def plan_image_chunks(
    image_ids: Iterable[int],
    *,
    shard_images: int,
    coordinates: Mapping[str, JSONLike] | None = None,
) -> tuple[ImageChunk, ...]:
    """Partition an ordered image bank into stable chunks of at most ``shard_images``.

    Input order is scientific identity and is preserved.  Each returned key includes
    the exact image ids, not just an ordinal, so bank drift invalidates shard caches.
    """

    if isinstance(shard_images, bool) or not isinstance(shard_images, int) or shard_images < 1:
        raise ShardValidationError("shard_images must be a positive integer")
    normalized_ids: list[int] = []
    for image_id in image_ids:
        try:
            normalized = integer_index(image_id)
        except TypeError as error:
            raise ShardValidationError("image ids must be non-negative integers") from error
        if isinstance(image_id, bool) or normalized < 0:
            raise ShardValidationError("image ids must be non-negative integers")
        normalized_ids.append(normalized)
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ShardValidationError("image ids must be unique")
    base = (
        {}
        if coordinates is None
        else _plain_mapping(_freeze_mapping(coordinates, label="chunk coordinates"))
    )
    if "image_chunk" in base:
        raise ShardValidationError("image_chunk is reserved by the image chunk planner")

    chunks: list[ImageChunk] = []
    for ordinal, start in enumerate(range(0, len(normalized_ids), shard_images)):
        stop = min(start + shard_images, len(normalized_ids))
        chunk_ids = tuple(normalized_ids[start:stop])
        key = ShardKey(
            {
                **base,
                "image_chunk": {
                    "image_ids": chunk_ids,
                    "ordinal": ordinal,
                    "start": start,
                    "stop": stop,
                },
            }
        )
        chunks.append(
            ImageChunk(
                ordinal=ordinal,
                start=start,
                stop=stop,
                image_ids=chunk_ids,
                key=key,
            )
        )
    return tuple(chunks)


@dataclass(frozen=True, slots=True)
class ShardContext:
    """Handler context with a safe constructor for mandatory artifact metadata."""

    store: ArtifactStore
    index: RunIndex
    run_id: str
    spec: ShardSpec
    owner_token: str

    def artifact_metadata(self, extra: Mapping[str, JSONLike] | None = None) -> dict[str, JSONLike]:
        required = self.spec.artifact_metadata
        if extra is None:
            return required
        normalized = _plain_mapping(_freeze_mapping(extra, label="extra artifact metadata"))
        overlap = sorted(set(required).intersection(normalized))
        if overlap:
            raise ShardValidationError(
                f"extra artifact metadata overrides reserved identity fields: {overlap}"
            )
        return {**required, **normalized}


ShardHandler = Callable[[ShardContext], ArtifactRef]


def validate_shard_artifact(reference: ArtifactRef, spec: ShardSpec) -> None:
    """Validate kind and every required semantic identity field on a shard artifact."""

    if reference.manifest.kind != spec.artifact_kind:
        raise ShardValidationError(
            f"artifact kind mismatch for {spec.row_name}: "
            f"expected {spec.artifact_kind!r}, got {reference.manifest.kind!r}"
        )
    mismatches = _identity_mismatches(reference.manifest.metadata, spec.artifact_metadata)
    if mismatches:
        raise ShardValidationError(
            f"artifact has incompatible shard identity fields: {', '.join(mismatches)}"
        )


def _validate_shard_record(record: StageRecord, spec: ShardSpec) -> None:
    if record.stage_name != spec.row_name:
        raise ShardValidationError("RunIndex row name does not match the shard key")
    if record.stage_signature != spec.signature:
        raise ShardValidationError("RunIndex shard signature does not match the shard spec")
    mismatches = _identity_mismatches(record.metadata, spec.artifact_metadata)
    if mismatches:
        raise ShardValidationError(
            f"RunIndex row has incompatible shard identity fields: {', '.join(mismatches)}"
        )


def _run_provenance(index: RunIndex, run_id: str) -> str:
    run = index.get_run(run_id)
    if run is None:
        raise ShardValidationError(f"run is not registered: {run_id}")
    provenance_id = run.provenance_artifact_id
    if provenance_id is None:
        raise ShardValidationError(f"run {run_id} has no provenance artifact")
    return _validate_digest(provenance_id, label=f"run {run_id} provenance artifact")


def _cache_identity(spec: ShardSpec) -> dict[str, JSONLike]:
    return {
        "parent_stage": spec.parent_stage,
        "parent_stage_signature": spec.parent_stage_signature,
        "shard_execution_schema_version": SHARD_EXECUTION_SCHEMA_VERSION,
        "shard_key_signature": spec.key.signature,
    }


def _cache_metadata(
    spec: ShardSpec,
    *,
    producer_run_id: str,
    producer_provenance_artifact_id: str,
) -> dict[str, JSONLike]:
    return {
        **_cache_identity(spec),
        "producer_provenance_artifact_id": producer_provenance_artifact_id,
        "producer_run_id": producer_run_id,
    }


def _validate_cache_record(
    index: RunIndex, cached: CacheRecord, spec: ShardSpec
) -> tuple[str, str]:
    if cached.stage_name != spec.row_name:
        raise ShardValidationError("cache entry belongs to a different shard row")
    mismatches = _identity_mismatches(cached.metadata, _cache_identity(spec))
    if mismatches:
        raise ShardValidationError(
            f"cache entry has incompatible shard identity fields: {', '.join(mismatches)}"
        )
    producer_run_id = cached.metadata.get("producer_run_id")
    producer_provenance_id = cached.metadata.get("producer_provenance_artifact_id")
    if not isinstance(producer_run_id, str):
        raise ShardValidationError("cache producer_run_id must be a string identifier")
    if not isinstance(producer_provenance_id, str):
        raise ShardValidationError("cache producer provenance must be an artifact id")
    _validate_identifier(producer_run_id, label="cache producer_run_id")
    _validate_digest(producer_provenance_id, label="cache producer provenance artifact")
    producer = index.get_run(producer_run_id)
    if producer is None:
        raise ShardValidationError("cache producer run is not registered")
    if producer.provenance_artifact_id != producer_provenance_id:
        raise ShardValidationError("cache first-publisher provenance does not match its run")
    return producer_run_id, producer_provenance_id


def _consumer_metadata(
    spec: ShardSpec,
    *,
    cache_hit: bool,
    producer_run_id: str,
    producer_provenance_artifact_id: str,
    run_provenance_artifact_id: str,
) -> dict[str, JSONLike]:
    return {
        **spec.artifact_metadata,
        "cache_hit": cache_hit,
        "producer_provenance_artifact_id": producer_provenance_artifact_id,
        "producer_run_id": producer_run_id,
        "run_provenance_artifact_id": run_provenance_artifact_id,
    }


def _validate_completed_lineage(
    index: RunIndex, record: StageRecord, *, current_run_id: str
) -> None:
    cache_hit = record.metadata.get("cache_hit")
    if not isinstance(cache_hit, bool):
        raise ShardValidationError("completed shard has no boolean cache_hit lineage field")
    current_provenance_id = _run_provenance(index, current_run_id)
    if record.metadata.get("run_provenance_artifact_id") != current_provenance_id:
        raise ShardValidationError("completed shard current-run provenance does not match its run")
    producer_run_id = record.metadata.get("producer_run_id")
    producer_provenance_id = record.metadata.get("producer_provenance_artifact_id")
    if not isinstance(producer_run_id, str):
        raise ShardValidationError("completed shard producer_run_id is invalid")
    if not isinstance(producer_provenance_id, str):
        raise ShardValidationError("completed shard producer provenance is invalid")
    _validate_identifier(producer_run_id, label="shard producer_run_id")
    _validate_digest(producer_provenance_id, label="shard producer provenance artifact")
    producer = index.get_run(producer_run_id)
    if producer is None or producer.provenance_artifact_id != producer_provenance_id:
        raise ShardValidationError("completed shard first-publisher provenance is invalid")


class ShardExecutor:
    """Execute or resume independent shards with atomic per-shard ownership."""

    def __init__(
        self,
        store: ArtifactStore,
        index: RunIndex,
        run_id: str,
        *,
        owner_token: str | None = None,
    ) -> None:
        self.store = store
        self.index = index
        self.run_id = _validate_identifier(run_id, label="run_id")
        self.owner_token = owner_token or f"shard-worker-{uuid4().hex}"
        _validate_identifier(self.owner_token, label="owner_token")

    def execute(self, spec: ShardSpec, handler: ShardHandler) -> ArtifactRef:
        """Return a verified completed shard, or claim, run, and publish it once.

        Handler failures, including ``KeyboardInterrupt``, mark only this shard
        FAILED before being re-raised.  Uncatchable process death leaves a RUNNING
        claim, which another worker must take over via :meth:`reclaim` with a reason.
        """

        run_provenance_id = _run_provenance(self.index, self.run_id)
        existing = self.index.get_stage(self.run_id, spec.row_name)
        if existing is not None and existing.state is StageState.COMPLETED:
            _validate_shard_record(existing, spec)
            _validate_completed_lineage(self.index, existing, current_run_id=self.run_id)
            if existing.artifact_id is None:  # RunIndex prevents this; retain a hard boundary.
                raise ShardValidationError("completed shard has no artifact id")
            reference = self.store.get(existing.artifact_id, verify=True)
            validate_shard_artifact(reference, spec)
            return reference

        cached = self.index.cache_lookup(spec.signature)
        if cached is not None:
            try:
                producer_run_id, producer_provenance_id = _validate_cache_record(
                    self.index, cached, spec
                )
            except ShardValidationError:
                self.index.cache_forget(
                    spec.signature,
                    expected_generation=cached.generation,
                    reason="shard cache lineage validation failed",
                )
                raise
            try:
                reference = self.store.get(cached.artifact_id, verify=True)
                validate_shard_artifact(reference, spec)
            except Exception:
                self.index.cache_forget(
                    spec.signature,
                    expected_generation=cached.generation,
                    reason="cached shard artifact validation failed",
                )
                raise
            lineage = _consumer_metadata(
                spec,
                cache_hit=True,
                producer_run_id=producer_run_id,
                producer_provenance_artifact_id=producer_provenance_id,
                run_provenance_artifact_id=run_provenance_id,
            )
            self.index.claim_stage(
                self.run_id,
                spec.row_name,
                spec.signature,
                owner_token=self.owner_token,
                metadata=lineage,
            )
            self.index.finish_stage(
                self.run_id,
                spec.row_name,
                owner_token=self.owner_token,
                state=StageState.COMPLETED,
                artifact_id=reference.artifact_id,
                message="verified shard cache hit",
                metadata=lineage,
            )
            return reference

        running_metadata: dict[str, JSONLike] = {
            **spec.artifact_metadata,
            "cache_hit": False,
            "run_provenance_artifact_id": run_provenance_id,
        }
        self.index.claim_stage(
            self.run_id,
            spec.row_name,
            spec.signature,
            owner_token=self.owner_token,
            metadata=running_metadata,
        )
        context = ShardContext(
            store=self.store,
            index=self.index,
            run_id=self.run_id,
            spec=spec,
            owner_token=self.owner_token,
        )
        try:
            published = handler(context)
            reference = self.store.get(published.artifact_id, verify=True)
            validate_shard_artifact(reference, spec)
            try:
                cached_record = self.index.cache_store(
                    spec.signature,
                    stage_name=spec.row_name,
                    artifact_id=reference.artifact_id,
                    metadata=_cache_metadata(
                        spec,
                        producer_run_id=self.run_id,
                        producer_provenance_artifact_id=run_provenance_id,
                    ),
                )
            except RunIndexConflictError as conflict:
                # Another run may publish the same deterministic shard between our
                # cache lookup and store.  Its first-publisher lineage is canonical.
                cached_record = self.index.cache_lookup(spec.signature)
                if cached_record is None:
                    raise ShardValidationError(
                        "shard cache changed during concurrent publication"
                    ) from conflict
                if cached_record.artifact_id != reference.artifact_id:
                    raise ShardValidationError(
                        "concurrent shard publishers produced different artifacts"
                    ) from conflict
            try:
                producer_run_id, producer_provenance_id = _validate_cache_record(
                    self.index, cached_record, spec
                )
            except ShardValidationError:
                self.index.cache_forget(
                    spec.signature,
                    expected_generation=cached_record.generation,
                    reason="elected shard cache lineage validation failed",
                )
                raise
            lineage = _consumer_metadata(
                spec,
                cache_hit=False,
                producer_run_id=producer_run_id,
                producer_provenance_artifact_id=producer_provenance_id,
                run_provenance_artifact_id=run_provenance_id,
            )
            self.index.finish_stage(
                self.run_id,
                spec.row_name,
                owner_token=self.owner_token,
                state=StageState.COMPLETED,
                artifact_id=reference.artifact_id,
                metadata=lineage,
            )
            return reference
        except BaseException as error:
            try:
                self.index.finish_stage(
                    self.run_id,
                    spec.row_name,
                    owner_token=self.owner_token,
                    state=StageState.FAILED,
                    message=f"{type(error).__name__}: {error}",
                    metadata=running_metadata,
                )
            except Exception as finish_error:
                error.add_note(f"could not mark shard failed: {finish_error}")
            raise

    def reclaim(self, spec: ShardSpec, *, reason: str) -> StageRecord:
        """Explicitly take over a RUNNING shard after determining its worker crashed."""

        if not isinstance(reason, str) or not reason.strip():
            raise ShardValidationError("reclaim reason must be a non-empty string")
        existing = self.index.get_stage(self.run_id, spec.row_name)
        if existing is None:
            raise ShardValidationError("cannot reclaim a shard with no RunIndex row")
        _validate_shard_record(existing, spec)
        return self.index.reclaim_stage(
            self.run_id,
            spec.row_name,
            owner_token=self.owner_token,
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class ShardReduction:
    """An immutable reducer manifest plus its verified ordered shard inputs."""

    reference: ArtifactRef
    shard_artifacts: tuple[ArtifactRef, ...]
    reducer_signature: str


class ShardReducer:
    """Publish an exact ordered manifest only after every expected shard completes."""

    def __init__(self, store: ArtifactStore, index: RunIndex, run_id: str) -> None:
        self.store = store
        self.index = index
        self.run_id = _validate_identifier(run_id, label="run_id")

    def publish(self, specs: Sequence[ShardSpec]) -> ShardReduction:
        if not specs:
            raise ShardReductionError("a reducer requires at least one expected shard")
        expected = tuple(specs)
        base_identity = expected[0].reduction_identity
        incompatible = [
            index
            for index, spec in enumerate(expected[1:], start=1)
            if canonical_hash(spec.reduction_identity) != canonical_hash(base_identity)
        ]
        if incompatible:
            raise ShardReductionError(
                f"all reduced shards must share one parent/input identity: {incompatible}"
            )
        row_names = [spec.row_name for spec in expected]
        key_signatures = [spec.key.signature for spec in expected]
        signatures = [spec.signature for spec in expected]
        if (
            len(set(row_names)) != len(row_names)
            or len(set(key_signatures)) != len(key_signatures)
            or len(set(signatures)) != len(signatures)
        ):
            raise ShardReductionError("expected shard list contains duplicates")

        all_records = self.index.list_stages(self.run_id)
        parent_stage = expected[0].parent_stage
        observed = {
            record.stage_name: record
            for record in all_records
            if record.metadata.get("parent_stage") == parent_stage
            and (
                record.metadata.get("shard_execution_schema_version")
                == SHARD_EXECUTION_SCHEMA_VERSION
                or record.stage_name.startswith("shard.")
            )
        }
        unexpected = sorted(set(observed).difference(row_names))
        if unexpected:
            raise ShardReductionError(f"unexpected shards exist for {parent_stage}: {unexpected}")
        missing = [name for name in row_names if name not in observed]
        if missing:
            raise ShardReductionError(f"expected shards are missing: {missing}")
        incomplete = {
            name: observed[name].state.value
            for name in row_names
            if observed[name].state is not StageState.COMPLETED
        }
        if incomplete:
            raise ShardReductionError(f"expected shards are not complete: {incomplete}")

        shard_artifacts: list[ArtifactRef] = []
        entries: list[dict[str, JSONLike]] = []
        artifact_ids: list[str] = []
        for ordinal, spec in enumerate(expected):
            record = observed[spec.row_name]
            _validate_shard_record(record, spec)
            _validate_completed_lineage(self.index, record, current_run_id=self.run_id)
            if record.artifact_id is None:
                raise ShardReductionError(f"completed shard has no artifact id: {spec.row_name}")
            reference = self.store.get(record.artifact_id, verify=True)
            validate_shard_artifact(reference, spec)
            artifact_ids.append(reference.artifact_id)
            shard_artifacts.append(reference)
            entries.append(
                {
                    "artifact_id": reference.artifact_id,
                    "ordinal": ordinal,
                    "shard_key": spec.key.to_dict(),
                    "shard_key_signature": spec.key.signature,
                    "shard_signature": spec.signature,
                }
            )
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ShardReductionError("distinct expected shards resolved to duplicate artifacts")

        reducer_signature = canonical_hash(
            {
                "domain": _REDUCER_DOMAIN,
                "identity": base_identity,
                "ordered_shards": entries,
            }
        )
        payload: dict[str, JSONLike] = {
            "ordered_shards": entries,
            "reducer_manifest_schema_version": REDUCER_MANIFEST_SCHEMA_VERSION,
            "reducer_signature": reducer_signature,
        }
        metadata: dict[str, JSONLike] = {
            **base_identity,
            "reducer_manifest_schema_version": REDUCER_MANIFEST_SCHEMA_VERSION,
            "reducer_signature": reducer_signature,
            "shard_count": len(entries),
        }
        reference = self.store.put_json(
            payload,
            filename="shards.json",
            kind="shards/reducer-manifest",
            metadata=metadata,
        )
        return ShardReduction(
            reference=reference,
            shard_artifacts=tuple(shard_artifacts),
            reducer_signature=reducer_signature,
        )


__all__ = [
    "REDUCER_MANIFEST_SCHEMA_VERSION",
    "SHARD_EXECUTION_SCHEMA_VERSION",
    "ImageChunk",
    "ShardContext",
    "ShardError",
    "ShardExecutor",
    "ShardHandler",
    "ShardKey",
    "ShardReducer",
    "ShardReduction",
    "ShardReductionError",
    "ShardSpec",
    "ShardValidationError",
    "plan_image_chunks",
    "validate_shard_artifact",
]
