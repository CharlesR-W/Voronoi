"""Resumable execution of runnable DAG stages into immutable artifacts."""

from __future__ import annotations

import fcntl
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from voronoi_lab.config import LabConfig
from voronoi_lab.core import (
    ArtifactRef,
    ArtifactStore,
    GateResult,
    JSONLike,
    Provenance,
    ProvenanceError,
    RunIndex,
    RunIndexConflictError,
    RunRecord,
    StageRecord,
    StageState,
    canonical_hash,
    canonical_json_bytes,
    capture_provenance,
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
    stage_signature,
    validate_stage_output,
)
from voronoi_lab.receipts import ReceiptError, publish_run_receipt


class ExecutionError(RuntimeError):
    """Raised when a stage cannot safely run or resume."""


@dataclass(frozen=True, slots=True)
class StageContext:
    config: LabConfig
    store: ArtifactStore
    index: RunIndex
    run_id: str
    project_root: Path
    source_identity: dict[str, JSONLike]
    stage_spec: StageSpec


StageHandler = Callable[[StageContext, Mapping[str, ArtifactRef]], ArtifactRef]


def _fallback_source_inputs(project_root: Path) -> dict[str, Path]:
    """Inventory executable source independently of Git location metadata.

    Logical names deliberately omit absolute locations so identical unpacked
    source trees share cache identity while byte drift still invalidates it.
    """

    runtime_package = Path(__file__).resolve().parent
    local_package = project_root / "src" / "voronoi_lab"
    roots = [("runtime", runtime_package)]
    if local_package.is_dir() and local_package.resolve() != runtime_package:
        roots.append(("project", local_package.resolve()))
    files: dict[str, Path] = {}
    for role, package_root in roots:
        files.update(
            {
                f"source/{role}/voronoi_lab/{path.relative_to(package_root).as_posix()}": path
                for path in sorted(package_root.rglob("*"))
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            }
        )
    for name in ("README.md", "pyproject.toml"):
        path = project_root / name
        if path.is_file():
            files[f"project/{name}"] = path
    if not files:
        raise ExecutionError("cannot inventory executable source files")
    return files


def default_run_id(config: LabConfig, source_identity: Mapping[str, JSONLike]) -> str:
    config_hash = canonical_hash(config.model_dump(mode="json"))
    source_hash = canonical_hash(source_identity)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", config.protocol.name).strip("-.")
    if not safe_name:
        safe_name = "run"
    return f"{safe_name}-{config_hash[:10]}-{source_hash[:10]}"


@dataclass(frozen=True, slots=True)
class VerifiedRunIdentity:
    record: RunRecord
    config: LabConfig
    source_identity: dict[str, JSONLike]
    provenance_reference: ArtifactRef


def load_verified_run_identity(
    store: ArtifactStore,
    index: RunIndex,
    run_id: str,
) -> VerifiedRunIdentity:
    """Resolve a run only when its index row and immutable provenance agree."""

    record = index.get_run(run_id)
    if record is None or record.provenance_artifact_id is None:
        raise ExecutionError(f"run {run_id!r} has no registered provenance artifact")
    try:
        reference = store.get(record.provenance_artifact_id, verify=True)
    except Exception as error:
        raise ExecutionError(
            f"run {run_id!r} provenance artifact is unavailable or corrupt"
        ) from error
    if reference.manifest.kind != "run/provenance":
        raise ExecutionError(f"run {run_id!r} provenance has the wrong artifact kind")
    source_value = thaw_json(record.metadata.get("source_identity"))
    if not isinstance(source_value, dict):
        raise ExecutionError(f"run {run_id!r} has no valid source identity")
    source_identity: dict[str, JSONLike] = source_value  # type: ignore[assignment]
    expected_metadata: dict[str, JSONLike] = {
        "config_hash": record.config_hash,
        "run_id": run_id,
        "source_identity": source_identity,
    }
    mismatches = [
        key
        for key, value in expected_metadata.items()
        if canonical_hash(reference.manifest.metadata.get(key)) != canonical_hash(value)
    ]
    expected_metadata_keys = {"config_hash", "run_id", "source_identity"}
    if set(reference.manifest.metadata) != expected_metadata_keys:
        mismatches.append("provenance metadata keys")
    files = {entry.path: entry for entry in reference.manifest.files}
    expected_files = {"config.json", "provenance.json"}
    if set(files) != expected_files:
        mismatches.append("provenance payload inventory")
        saved_config: object = None
        saved_provenance: object = None
    else:
        if any(files[name].media_type != "application/json" for name in expected_files):
            mismatches.append("provenance payload media types")
        try:
            saved_config = store.read_json(reference.artifact_id, "config.json")
            saved_provenance = store.read_json(reference.artifact_id, "provenance.json")
        except Exception as error:
            raise ExecutionError(f"run {run_id!r} provenance payloads are unreadable") from error
        if canonical_hash(saved_config) != record.config_hash:
            mismatches.append("config.json")
    if mismatches:
        raise ExecutionError(
            f"run {run_id!r} has invalid provenance lineage: "
            + ", ".join(dict.fromkeys(mismatches))
        )
    try:
        config = LabConfig.model_validate(saved_config)
    except Exception as error:
        raise ExecutionError(f"run {run_id!r} saved configuration is invalid") from error
    if canonical_hash(config.model_dump(mode="json")) != record.config_hash:
        raise ExecutionError(f"run {run_id!r} saved configuration is not canonical")
    try:
        provenance = Provenance.from_dict(saved_provenance)
    except ProvenanceError as error:
        raise ExecutionError(f"run {run_id!r} saved provenance payload is invalid") from error
    if canonical_hash(provenance.source_identity) != canonical_hash(source_identity):
        raise ExecutionError(
            f"run {run_id!r} saved provenance does not match registered source identity"
        )
    if record.metadata.get("mode") != config.protocol.mode:
        raise ExecutionError(f"run {run_id!r} recorded mode does not match saved configuration")
    return VerifiedRunIdentity(record, config, source_identity, reference)


class ExperimentRunner:
    def __init__(
        self,
        config: LabConfig,
        *,
        project_root: str | Path,
        artifact_root: str | Path = "artifacts",
        run_index_path: str | Path = "runs/index.sqlite",
        registry: StageRegistry = DEFAULT_STAGES,
        handlers: Mapping[str, StageHandler] | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.project_root = Path(project_root).resolve()
        self.store = ArtifactStore(self.project_root / artifact_root)
        self.index = RunIndex(self.project_root / run_index_path)
        self.registry = registry
        self.handlers = dict(handlers or {})
        self.owner_token = f"runner-{uuid4().hex}"
        self._provenance_root = self.project_root if (self.project_root / ".git").exists() else None
        self._captured_provenance = self._capture_source_provenance()
        self.source_identity = self._captured_provenance.source_identity
        self.run_id = run_id or default_run_id(config, self.source_identity)
        self._last_receipt: ArtifactRef | None = None
        # Validate before interpolating a caller-controlled identifier into a lock path.
        # ``get_run`` is read-only and uses the RunIndex's canonical identifier grammar.
        self.index.get_run(self.run_id)
        self._register_run()

    def run(self, targets: Sequence[str]) -> dict[str, ArtifactRef]:
        requested_targets = tuple(targets)
        if len(requested_targets) != len(set(requested_targets)):
            raise ExecutionError("requested target stages must be unique")
        stages = self.registry.topological_order(requested_targets)
        planned = [
            stage.name
            for stage in stages
            if stage.implementation is not ImplementationStatus.RUNNABLE
        ]
        if planned:
            raise ExecutionError(
                "requested closure includes stages that are specified but not implemented: "
                + ", ".join(planned)
            )
        validation_context = StageValidationContext()
        results: dict[str, ArtifactRef] = {}
        for stage in stages:
            results[stage.name] = self._run_stage(
                stage,
                results,
                validation_context=validation_context,
            )
        self._assert_source_identity_current(boundary="before publishing run receipt")
        try:
            self._last_receipt = publish_run_receipt(
                self.store,
                self.index,
                run_id=self.run_id,
                config=self.config,
                source_identity=self.source_identity,
                requested_targets=requested_targets,
                ordered_stage_names=tuple(stage.name for stage in stages),
                registry=self.registry,
                validation_context=validation_context,
            )
        except ReceiptError as error:
            raise ExecutionError(f"failed to publish immutable run receipt: {error}") from error
        return results

    @property
    def receipt_reference(self) -> ArtifactRef | None:
        """Most recent immutable receipt published by this runner instance."""

        return self._last_receipt

    @property
    def receipt_artifact_id(self) -> str | None:
        return None if self._last_receipt is None else self._last_receipt.artifact_id

    def _register_run(self) -> None:
        lock_directory = self.index.path.parent / ".registration-locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        lock_path = lock_directory / f"{self.run_id}.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._register_run_locked()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _register_run_locked(self) -> None:
        config_hash = canonical_hash(self.config.model_dump(mode="json"))
        existing = self.index.get_run(self.run_id)
        if existing is not None:
            if (
                existing.config_hash != config_hash
                or existing.metadata.get("source_identity") != self.source_identity
            ):
                raise RunIndexConflictError(
                    f"run id {self.run_id!r} belongs to different config/source identity"
                )
            self._verified_run_provenance(
                self.run_id,
                expected_config_hash=config_hash,
                expected_source_identity=self.source_identity,
            )
            return
        provenance_bytes = canonical_json_bytes(self._captured_provenance.to_dict())
        config_bytes = canonical_json_bytes(self.config.model_dump(mode="json"))
        provenance_ref = self.store.put_files(
            {"config.json": config_bytes, "provenance.json": provenance_bytes},
            kind="run/provenance",
            metadata={
                "config_hash": config_hash,
                "run_id": self.run_id,
                "source_identity": self.source_identity,
            },
            media_types={
                "config.json": "application/json",
                "provenance.json": "application/json",
            },
        )
        self.index.register_run(
            self.run_id,
            config_hash=config_hash,
            provenance_artifact_id=provenance_ref.artifact_id,
            metadata={
                "mode": self.config.protocol.mode,
                "source_identity": self.source_identity,
            },
        )
        self._verified_run_provenance(
            self.run_id,
            expected_artifact_id=provenance_ref.artifact_id,
            expected_config_hash=config_hash,
            expected_source_identity=self.source_identity,
        )

    def _verified_run_provenance(
        self,
        run_id: str,
        *,
        expected_artifact_id: str | None = None,
        expected_config_hash: str | None = None,
        expected_source_identity: Mapping[str, JSONLike] | None = None,
    ) -> str:
        """Verify a run's mutable pointer and immutable provenance payload agree."""

        verified = load_verified_run_identity(self.store, self.index, run_id)
        record = verified.record
        provenance_id = verified.provenance_reference.artifact_id
        if expected_artifact_id is not None and provenance_id != expected_artifact_id:
            raise ExecutionError(f"run {run_id!r} provenance pointer changed unexpectedly")
        if expected_config_hash is not None and record.config_hash != expected_config_hash:
            raise ExecutionError(f"run {run_id!r} config hash does not match expected identity")
        if expected_source_identity is not None and canonical_hash(
            verified.source_identity
        ) != canonical_hash(expected_source_identity):
            raise ExecutionError(
                f"run {run_id!r} registered source identity does not match expected identity"
            )
        return provenance_id

    def _assert_source_identity_current(self, *, boundary: str) -> None:
        """Abort when runtime packages or Git source drift from run registration."""

        try:
            current = self._capture_source_provenance()
        except ProvenanceError as error:
            raise ExecutionError(f"unable to recapture source identity {boundary}") from error
        if canonical_hash(current.source_identity) != canonical_hash(self.source_identity):
            raise ExecutionError(f"source identity drift detected {boundary}")

    def _capture_source_provenance(self) -> Provenance:
        return capture_provenance(
            repo_root=self._provenance_root,
            input_files=_fallback_source_inputs(self.project_root),
        )

    def _validate_completed_stage_lineage(self, record: StageRecord) -> None:
        if not isinstance(record.metadata.get("cache_hit"), bool):
            raise ExecutionError("completed stage has no boolean cache_hit lineage")
        current_id = self._verified_run_provenance(self.run_id)
        if record.metadata.get("run_provenance_artifact_id") != current_id:
            raise ExecutionError("completed stage current-run provenance is invalid")
        producer_run_id = record.metadata.get("producer_run_id")
        producer_provenance_id = record.metadata.get("producer_provenance_artifact_id")
        if not isinstance(producer_run_id, str) or not isinstance(producer_provenance_id, str):
            raise ExecutionError("completed stage has no first-publisher provenance")
        self._verified_run_provenance(
            producer_run_id,
            expected_artifact_id=producer_provenance_id,
        )

    def _run_stage(
        self,
        stage: StageSpec,
        completed: Mapping[str, ArtifactRef],
        *,
        validation_context: StageValidationContext,
    ) -> ArtifactRef:
        if stage.name not in self.handlers:
            raise ExecutionError(f"no handler is registered for runnable stage {stage.name}")
        self._assert_source_identity_current(boundary=f"before resolving stage {stage.name}")
        upstream = {dependency: completed[dependency] for dependency in stage.dependencies}
        context = StageContext(
            config=self.config,
            store=self.store,
            index=self.index,
            run_id=self.run_id,
            project_root=self.project_root,
            source_identity=self.source_identity,
            stage_spec=stage,
        )
        preflight = getattr(self.handlers[stage.name], "__voronoi_stage_preflight__", None)
        if preflight is not None:
            if not callable(preflight):
                raise ExecutionError(f"stage {stage.name} declares a non-callable preflight")
            try:
                preflight(context)
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise ExecutionError(f"preflight failed for stage {stage.name}: {error}") from error
        upstream_ids = {name: reference.artifact_id for name, reference in upstream.items()}
        signature = stage_signature(
            stage,
            self.config,
            upstream_artifact_ids=upstream_ids,
            source_identity=self.source_identity,
        )
        gate_blockers = _gate_blockers(self.store, upstream)
        if gate_blockers:
            existing = self.index.get_stage(self.run_id, stage.name)
            if existing is not None and existing.state is StageState.COMPLETED:
                raise ExecutionError(
                    f"completed stage {stage.name} conflicts with blocking gate dependencies"
                )
            self.index.claim_stage(
                self.run_id,
                stage.name,
                signature,
                owner_token=self.owner_token,
                metadata={"gate_blockers": gate_blockers},
            )
            message = "gate dependencies block execution: " + ", ".join(gate_blockers)
            self.index.finish_stage(
                self.run_id,
                stage.name,
                owner_token=self.owner_token,
                state=StageState.BLOCKED,
                message=message,
                metadata={"gate_blockers": gate_blockers},
            )
            raise ExecutionError(message)
        existing = self.index.get_stage(self.run_id, stage.name)
        if existing is not None and existing.state is StageState.COMPLETED:
            if existing.stage_signature != signature or existing.artifact_id is None:
                raise ExecutionError(
                    f"completed stage {stage.name} does not match current stage identity"
                )
            reference = self.store.get(existing.artifact_id, verify=True)
            _validate_artifact_identity(
                reference,
                stage=stage,
                registry=self.registry,
                config=self.config,
                signature=signature,
                dependencies=upstream,
                store=self.store,
                source_identity=self.source_identity,
                validation_context=validation_context,
            )
            self._validate_completed_stage_lineage(existing)
            self._assert_source_identity_current(
                boundary=f"before reusing completed stage {stage.name}"
            )
            return reference

        cached = self.index.cache_lookup(signature)
        if cached is not None:
            if cached.stage_name != stage.name:
                self.index.cache_forget(
                    signature,
                    expected_generation=cached.generation,
                    reason="cache entry belongs to a different stage",
                )
                raise ExecutionError("cache entry belongs to a different stage")
            try:
                reference = self.store.get(cached.artifact_id, verify=True)
            except Exception:
                self.index.cache_forget(
                    signature,
                    expected_generation=cached.generation,
                    reason="cached artifact is unavailable or corrupt",
                )
            else:
                try:
                    _validate_artifact_identity(
                        reference,
                        stage=stage,
                        registry=self.registry,
                        config=self.config,
                        signature=signature,
                        dependencies=upstream,
                        store=self.store,
                        source_identity=self.source_identity,
                        validation_context=validation_context,
                    )
                except ExecutionError:
                    self.index.cache_forget(
                        signature,
                        expected_generation=cached.generation,
                        reason="cached artifact identity validation failed",
                    )
                    raise
                run_provenance_id = self._verified_run_provenance(self.run_id)
                producer_run_id = cached.metadata.get("producer_run_id")
                producer_provenance_id = cached.metadata.get("producer_provenance_artifact_id")
                if not isinstance(producer_run_id, str) or not isinstance(
                    producer_provenance_id, str
                ):
                    self.index.cache_forget(
                        signature,
                        expected_generation=cached.generation,
                        reason="cache entry has no first-publisher provenance",
                    )
                    raise ExecutionError("cache entry has no first-publisher provenance")
                try:
                    self._verified_run_provenance(
                        producer_run_id,
                        expected_artifact_id=producer_provenance_id,
                    )
                except ExecutionError:
                    self.index.cache_forget(
                        signature,
                        expected_generation=cached.generation,
                        reason="cache first-publisher provenance validation failed",
                    )
                    raise
                self._assert_source_identity_current(
                    boundary=f"before consuming cache for stage {stage.name}"
                )
                self.index.claim_stage(
                    self.run_id,
                    stage.name,
                    signature,
                    owner_token=self.owner_token,
                    metadata={"cache_hit": True},
                )
                self.index.finish_stage(
                    self.run_id,
                    stage.name,
                    owner_token=self.owner_token,
                    state=StageState.COMPLETED,
                    artifact_id=reference.artifact_id,
                    message="verified cache hit",
                    metadata={
                        "cache_hit": True,
                        "producer_provenance_artifact_id": producer_provenance_id,
                        "producer_run_id": producer_run_id,
                        "run_provenance_artifact_id": run_provenance_id,
                    },
                )
                return reference

        self.index.claim_stage(
            self.run_id,
            stage.name,
            signature,
            owner_token=self.owner_token,
            metadata={"cache_hit": False},
        )
        try:
            self._assert_source_identity_current(
                boundary=f"immediately before handler for stage {stage.name}"
            )
            reference = self.handlers[stage.name](context, upstream)
            self._assert_source_identity_current(
                boundary=f"immediately after handler for stage {stage.name}"
            )
            run_provenance_id = self._verified_run_provenance(self.run_id)
            # Never trust a handler-supplied filesystem path. Re-resolve the
            # content identity through the configured immutable store.
            reference = self.store.get(reference.artifact_id, verify=True)
            _validate_artifact_identity(
                reference,
                stage=stage,
                registry=self.registry,
                config=self.config,
                signature=signature,
                dependencies=upstream,
                store=self.store,
                source_identity=self.source_identity,
                validation_context=validation_context,
            )
            cached_record = self.index.cache_store(
                signature,
                stage_name=stage.name,
                artifact_id=reference.artifact_id,
                metadata={
                    "producer_provenance_artifact_id": run_provenance_id,
                    "producer_run_id": self.run_id,
                    "stage_version": stage.stage_version,
                },
            )
            producer_run_id = cached_record.metadata.get("producer_run_id")
            producer_provenance_id = cached_record.metadata.get("producer_provenance_artifact_id")
            if not isinstance(producer_run_id, str) or not isinstance(producer_provenance_id, str):
                self.index.cache_forget(
                    signature,
                    expected_generation=cached_record.generation,
                    reason="elected cache entry has no first-publisher provenance",
                )
                raise ExecutionError("cache entry did not retain first-publisher provenance")
            try:
                self._verified_run_provenance(
                    producer_run_id,
                    expected_artifact_id=producer_provenance_id,
                )
            except ExecutionError:
                self.index.cache_forget(
                    signature,
                    expected_generation=cached_record.generation,
                    reason="elected cache first-publisher provenance validation failed",
                )
                raise
            self.index.finish_stage(
                self.run_id,
                stage.name,
                owner_token=self.owner_token,
                state=StageState.COMPLETED,
                artifact_id=reference.artifact_id,
                metadata={
                    "cache_hit": False,
                    "producer_provenance_artifact_id": producer_provenance_id,
                    "producer_run_id": producer_run_id,
                    "run_provenance_artifact_id": run_provenance_id,
                },
            )
            return reference
        except BaseException as error:
            try:
                self.index.finish_stage(
                    self.run_id,
                    stage.name,
                    owner_token=self.owner_token,
                    state=StageState.FAILED,
                    message=f"{type(error).__name__}: {error}",
                    metadata={"cache_hit": False, "interrupted": not isinstance(error, Exception)},
                )
            except Exception as finish_error:
                error.add_note(
                    "failed to record terminal stage state: "
                    f"{type(finish_error).__name__}: {finish_error}"
                )
            raise


def verify_recorded_stage(
    store: ArtifactStore,
    index: RunIndex,
    run_id: str,
    stage_name: str,
    *,
    registry: StageRegistry = DEFAULT_STAGES,
    validation_context: StageValidationContext | None = None,
) -> ArtifactRef:
    """Recursively verify a completed run record from saved provenance alone."""

    run = load_verified_run_identity(store, index, run_id)
    context = StageValidationContext() if validation_context is None else validation_context
    resolved: dict[str, ArtifactRef] = {}

    def resolve(name: str) -> ArtifactRef:
        if name in resolved:
            return resolved[name]
        stage = registry.get(name)
        record = index.get_stage(run_id, name)
        if record is None or record.state is not StageState.COMPLETED or record.artifact_id is None:
            raise ExecutionError(f"recorded stage {run_id}/{name} is not complete")
        dependencies = {dependency: resolve(dependency) for dependency in stage.dependencies}
        signature = stage_signature(
            stage,
            run.config,
            upstream_artifact_ids={
                dependency: reference.artifact_id for dependency, reference in dependencies.items()
            },
            source_identity=run.source_identity,
        )
        if record.stage_signature != signature:
            raise ExecutionError(f"recorded stage {run_id}/{name} has an invalid signature")
        reference = store.get(record.artifact_id, verify=True)
        _validate_artifact_identity(
            reference,
            stage=stage,
            registry=registry,
            config=run.config,
            signature=signature,
            dependencies=dependencies,
            store=store,
            source_identity=run.source_identity,
            validation_context=context,
        )
        if not isinstance(record.metadata.get("cache_hit"), bool):
            raise ExecutionError(f"recorded stage {run_id}/{name} has no cache lineage")
        if (
            record.metadata.get("run_provenance_artifact_id")
            != run.provenance_reference.artifact_id
        ):
            raise ExecutionError(f"recorded stage {run_id}/{name} has invalid consumer provenance")
        producer_run_id = record.metadata.get("producer_run_id")
        producer_provenance_id = record.metadata.get("producer_provenance_artifact_id")
        if not isinstance(producer_run_id, str) or not isinstance(producer_provenance_id, str):
            raise ExecutionError(
                f"recorded stage {run_id}/{name} has no first-publisher provenance"
            )
        producer = load_verified_run_identity(store, index, producer_run_id)
        if producer.provenance_reference.artifact_id != producer_provenance_id:
            raise ExecutionError(f"recorded stage {run_id}/{name} has invalid producer provenance")
        resolved[name] = reference
        return reference

    return resolve(stage_name)


def artifact_metadata(
    context: StageContext,
    dependencies: Mapping[str, ArtifactRef],
) -> dict[str, JSONLike]:
    """Build the mandatory identity block shared by every stage artifact."""

    stage = context.stage_spec
    signature = stage_signature(
        stage,
        context.config,
        upstream_artifact_ids={name: ref.artifact_id for name, ref in dependencies.items()},
        source_identity=context.source_identity,
    )
    return {
        "inherited_gate_overrides": _inherited_gate_overrides(context.store, dependencies),
        "source_identity": context.source_identity,
        "stage": stage.name,
        "stage_config": stage_config(context.config, stage.config_paths),
        "stage_signature": signature,
        "stage_version": stage.stage_version,
        "upstream_artifacts": {
            name: reference.artifact_id for name, reference in sorted(dependencies.items())
        },
    }


def _validate_artifact_identity(
    reference: ArtifactRef,
    *,
    stage: StageSpec,
    registry: StageRegistry,
    config: LabConfig,
    signature: str,
    dependencies: Mapping[str, ArtifactRef],
    store: ArtifactStore,
    source_identity: Mapping[str, JSONLike],
    validation_context: StageValidationContext,
) -> None:
    """Reject a valid content object whose scientific cache identity is wrong."""

    expected: dict[str, JSONLike] = {
        "inherited_gate_overrides": _inherited_gate_overrides(store, dependencies),
        "source_identity": dict(source_identity),
        "stage": stage.name,
        "stage_config": stage_config(config, stage.config_paths),
        "stage_signature": signature,
        "stage_version": stage.stage_version,
        "upstream_artifacts": {
            name: reference.artifact_id for name, reference in sorted(dependencies.items())
        },
    }
    metadata = reference.manifest.metadata
    mismatches = [
        key
        for key, value in expected.items()
        if canonical_hash(metadata.get(key)) != canonical_hash(value)
    ]
    if mismatches:
        fields = ", ".join(mismatches)
        raise ExecutionError(
            f"artifact for {stage.name} has incompatible stage identity fields: {fields}"
        )
    try:
        gate_rule = None if stage.gate_payload_path is None else expected_gate_rule(stage, config)
        gate_authorization = (
            None
            if stage.gate_payload_path is None
            else expected_gate_override_authorization(stage, config)
        )
        validate_stage_output(
            reference,
            stage,
            store,
            gate_rule=gate_rule,
            gate_override_authorization=gate_authorization,
            config=config,
            registry=registry,
            source_identity=source_identity,
            validation_context=validation_context,
        )
    except PipelineError as error:
        raise ExecutionError(str(error)) from error


def _inherited_gate_overrides(
    store: ArtifactStore, dependencies: Mapping[str, ArtifactRef]
) -> list[JSONLike]:
    """Collect and de-duplicate direct and transitive gate-override lineage."""

    lineage: list[JSONLike] = []
    seen: set[str] = set()

    def add(entries: object, *, label: str) -> None:
        if not isinstance(entries, (list, tuple)):
            raise ExecutionError(f"{label} gate override lineage must be an array")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ExecutionError(f"{label} gate override lineage contains a non-object")
            reason = entry.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ExecutionError(f"{label} gate override lineage has no reason")
            digest = canonical_hash(entry)
            if digest not in seen:
                lineage.append(entry)  # type: ignore[arg-type]
                seen.add(digest)

    for name, reference in sorted(dependencies.items()):
        inherited = reference.manifest.metadata.get("inherited_gate_overrides", ())
        add(inherited, label=name)
        if (
            name.startswith("gate.")
            and reference.manifest.metadata.get("gate_status") == "OVERRIDDEN"
        ):
            payload = store.read_json(reference.artifact_id, "gate.json")
            if not isinstance(payload, Mapping):
                raise ExecutionError(f"{name} payload must be an object")
            add(payload.get("override_lineage"), label=name)
    return lineage


def _gate_blockers(store: ArtifactStore, dependencies: Mapping[str, ArtifactRef]) -> list[str]:
    """Return safe blockers for gate-stage dependencies, preserving explicit overrides."""

    blockers: list[str] = []
    for name, reference in sorted(dependencies.items()):
        if not name.startswith("gate."):
            continue
        metadata_status = reference.manifest.metadata.get("gate_status")
        try:
            payload = store.read_json(reference.artifact_id, "gate.json")
            result = GateResult.from_dict(payload)
        except Exception:
            blockers.append(f"{name}:INVALID_GATE_ARTIFACT")
            continue
        if result.status.value != metadata_status:
            blockers.append(f"{name}:INVALID_GATE_ARTIFACT")
            continue
        if result.status.value == "PASS":
            continue
        if result.status.value == "OVERRIDDEN":
            if result.override_lineage:
                continue
            blockers.append(f"{name}:INVALID_OVERRIDE")
            continue
        if result.status.value in {"FAIL", "NOT_EVALUABLE"}:
            blockers.append(f"{name}:{result.status.value}")
        else:
            blockers.append(f"{name}:INVALID_GATE_STATUS")
    return blockers
