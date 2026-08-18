"""Concrete handlers for the deliberately small runnable pipeline surface.

Handlers consume only configuration plus immutable dependency artifacts and
publish immutable artifacts of their own.  Scientific outcome stages that are
still marked ``PLANNED`` in :mod:`voronoi_lab.pipeline` intentionally have no
handler here.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from voronoi_lab.config import LabConfig
from voronoi_lab.core import (
    ArtifactRef,
    GateEvaluator,
    GateResult,
    GateRule,
    GateStatus,
    JSONLike,
    SeedDeriver,
    StageState,
    canonical_hash,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from voronoi_lab.execution import StageContext, StageHandler, artifact_metadata
from voronoi_lab.exp1.probe_artifact import ProbeArtifactError, build_probe_bank_files
from voronoi_lab.exp1.torch_mechanics import (
    summarize_resnet_mechanical_evidence,
    validate_jvp,
    zero_intervention_parity,
)
from voronoi_lab.exp1.tracking2 import (
    Tracking2Adapter,
    parse_tracking2_manifest_bytes,
    resolve_cut,
)
from voronoi_lab.exp1.tracking2_vgg import (
    Tracking2VGGAdapter,
    parse_tracking2_vgg_manifest_bytes,
)
from voronoi_lab.mechanical import replay_toy_geometry
from voronoi_lab.pipeline import expected_gate_rule, stage_config
from voronoi_lab.reporting.builder import render_report
from voronoi_lab.reporting.payload import make_mock_payload
from voronoi_lab.sharding import ShardContext, ShardExecutor, ShardKey, ShardReducer, ShardSpec

RESULT_SCHEMA_VERSION = 1
PLATEAU_RESULT_SCHEMA_VERSION = 2

_RAW_LOCAL_PLANE_JACOBIAN = "local_transition_plane_jacobian"
_RAW_ANCHOR_PLANE_JACOBIAN = "anchor_transition_plane_jacobian_by_context"
_RESIDUAL_LOCAL_PLANE_JACOBIAN = "local_residual_update_plane_jacobian"
_RESIDUAL_ANCHOR_PLANE_JACOBIAN = "anchor_residual_update_plane_jacobian_by_context"
_RAW_PLANE_ESTIMAND = "2D-plane-restricted ||DT||_F"
_RESIDUAL_PLANE_ESTIMAND = "2D-plane-restricted ||D(T-I)||_F"


class StageHandlerError(RuntimeError):
    """Raised when a runnable handler cannot honor its declared contract."""


def _resolve(project_root: Path, declared: str | Path) -> Path:
    path = Path(declared).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def tracking2_adapter_from_config(
    config: LabConfig, project_root: Path
) -> tuple[Path, Tracking2Adapter]:
    """Construct the one supported adapter, rejecting silent legacy overrides."""

    inputs = config.inputs.tracking2
    if not inputs.read_only:
        raise StageHandlerError("Tracking2 inputs must remain read-only")
    if inputs.checkpoint_files or inputs.dataset_root is not None or inputs.transplant_results:
        raise StageHandlerError(
            "per-file Tracking2 overrides are not implemented; use one hash-pinned manifest"
        )
    manifest_path = _resolve(project_root, inputs.manifest)
    observed_manifest_hash = sha256_file(manifest_path)
    if observed_manifest_hash != inputs.manifest_sha256:
        raise StageHandlerError(
            "Tracking2 manifest SHA-256 does not match the signed configuration: "
            f"expected {inputs.manifest_sha256}, observed {observed_manifest_hash}"
        )
    root = _resolve(project_root, inputs.root)
    adapter = Tracking2Adapter.from_yaml(manifest_path, root_override=root)
    expected = f"preactivation_resnet18_v2_width{adapter.manifest.architecture.width}"
    if inputs.expected_model != expected:
        raise StageHandlerError(
            f"expected_model={inputs.expected_model!r} does not match manifest model {expected!r}"
        )
    manifest = adapter.manifest
    configured_epochs = tuple(config.experiment1.checkpoints)
    manifest_epochs = tuple(item.epoch for item in manifest.checkpoints)
    if configured_epochs != manifest_epochs:
        raise StageHandlerError(
            "experiment1.checkpoints must exactly match the Tracking2 manifest epochs"
        )
    banks = config.experiment1.probe_banks
    if banks.fit_train_images + banks.independent_fit_train_images > manifest.training.train_size:
        raise StageHandlerError("disjoint codebook fit banks exceed the manifest training split")
    if banks.intervention_nested_in_geometry:
        if banks.intervention_test_images > banks.geometry_test_images:
            raise StageHandlerError("nested intervention bank cannot exceed the geometry bank")
        required_test = banks.geometry_test_images
    else:
        required_test = banks.geometry_test_images + banks.intervention_test_images
    if required_test > manifest.training.test_size:
        raise StageHandlerError("declared held-out probe banks exceed the manifest test split")
    return manifest_path, adapter


def _tracking2_adapter(context: StageContext) -> tuple[Path, Tracking2Adapter]:
    return tracking2_adapter_from_config(context.config, context.project_root)


def _preflight_inputs_tracking2(context: StageContext) -> None:
    """Verify every current external byte before completed/cache reuse."""

    _manifest_path, adapter = _tracking2_adapter(context)
    adapter.validate_all()


def tracking2_vgg_adapter_from_config(
    config: LabConfig, project_root: Path
) -> tuple[Path, Tracking2VGGAdapter]:
    """Construct the hash-pinned exploratory VGG control adapter."""

    inputs = config.inputs.tracking2_vgg
    if not inputs.read_only:
        raise StageHandlerError("Tracking2 VGG inputs must remain read-only")
    manifest_path = _resolve(project_root, inputs.manifest)
    observed_manifest_hash = sha256_file(manifest_path)
    if observed_manifest_hash != inputs.manifest_sha256:
        raise StageHandlerError(
            "Tracking2 VGG manifest SHA-256 does not match the signed configuration: "
            f"expected {inputs.manifest_sha256}, observed {observed_manifest_hash}"
        )
    root = _resolve(project_root, inputs.root)
    adapter = Tracking2VGGAdapter.from_yaml(manifest_path, root_override=root)
    manifest = adapter.manifest
    architecture = manifest.architecture
    expected = (
        f"vgg19_bn_classifier{architecture.classifier_width}_width{architecture.width_multiplier:g}"
    )
    if inputs.expected_model != expected:
        raise StageHandlerError(
            f"expected_model={inputs.expected_model!r} does not match manifest model {expected!r}"
        )
    if manifest.lineage_quality != "exploratory_legacy":
        raise StageHandlerError("Tracking2 VGG lineage must remain explicitly exploratory")
    configured_epochs = tuple(config.experiment1.checkpoints)
    manifest_epochs = tuple(item.epoch for item in manifest.checkpoints)
    if configured_epochs != manifest_epochs:
        raise StageHandlerError(
            "experiment1.checkpoints must exactly match the Tracking2 VGG manifest epochs"
        )
    banks = config.experiment1.probe_banks
    if banks.fit_train_images + banks.independent_fit_train_images > manifest.training.train_size:
        raise StageHandlerError("disjoint codebook fit banks exceed the VGG training split")
    if banks.intervention_nested_in_geometry:
        if banks.intervention_test_images > banks.geometry_test_images:
            raise StageHandlerError("nested intervention bank cannot exceed the geometry bank")
        required_test = banks.geometry_test_images
    else:
        required_test = banks.geometry_test_images + banks.intervention_test_images
    if required_test > manifest.training.test_size:
        raise StageHandlerError("declared held-out probe banks exceed the VGG test split")
    return manifest_path, adapter


def _tracking2_vgg_adapter(context: StageContext) -> tuple[Path, Tracking2VGGAdapter]:
    return tracking2_vgg_adapter_from_config(context.config, context.project_root)


def _preflight_inputs_tracking2_vgg(context: StageContext) -> None:
    """Verify the current VGG bytes and criticality metadata before reuse."""

    _manifest_path, adapter = _tracking2_vgg_adapter(context)
    adapter.validate_all()
    adapter.read_training_record()


def _stage_metadata(
    context: StageContext,
    dependencies: Mapping[str, ArtifactRef],
    **extra: JSONLike,
) -> dict[str, JSONLike]:
    return {
        **artifact_metadata(context, dependencies),
        "result_schema_version": RESULT_SCHEMA_VERSION,
        **extra,
    }


def _configure_plateau_torch_runtime(
    requested_device: str,
    *,
    torch_module: Any | None = None,
) -> tuple[str, dict[str, JSONLike]]:
    """Resolve the one accelerator-aware stage and record its backend contract."""

    if requested_device not in {"cpu", "cuda"}:
        raise StageHandlerError(f"unsupported CIFAR plateau device: {requested_device!r}")
    if requested_device == "cuda":
        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace_config is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif workspace_config not in {":16:8", ":4096:8"}:
            raise StageHandlerError(
                "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:16:8 or :4096:8"
            )
    torch = torch_module or importlib.import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    if requested_device == "cuda" and not cuda_available:
        raise StageHandlerError(
            "runtime.device='cuda' requires torch.cuda.is_available() to be true"
        )

    torch.use_deterministic_algorithms(True)
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False

    device_index: int | None = None
    device_name: str | None = None
    compute_capability: list[int] | None = None
    if requested_device == "cuda":
        device_index = int(torch.cuda.current_device())
        actual_device = f"cuda:{device_index}"
        device_name = str(torch.cuda.get_device_name(device_index))
        compute_capability = [
            int(value) for value in torch.cuda.get_device_capability(device_index)
        ]
    else:
        actual_device = "cpu"

    cudnn_version = None if cudnn is None else cudnn.version()
    provenance: dict[str, JSONLike] = {
        "requested_device": requested_device,
        "actual_device": actual_device,
        "torch_version": str(torch.__version__),
        "torch_cuda_build_version": (
            None
            if getattr(getattr(torch, "version", None), "cuda", None) is None
            else str(torch.version.cuda)
        ),
        "cuda_available": cuda_available,
        "cuda_device_index": device_index,
        "cuda_device_name": device_name,
        "cuda_compute_capability": compute_capability,
        "cudnn_version": None if cudnn_version is None else int(cudnn_version),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": None if cudnn is None else bool(cudnn.deterministic),
        "cudnn_benchmark": None if cudnn is None else bool(cudnn.benchmark),
        "cublas_workspace_config": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") if requested_device == "cuda" else None
        ),
        "persisted_arrays": "cpu_numpy_float32",
    }
    return actual_device, provenance


def _json_object(context: StageContext, reference: ArtifactRef, filename: str) -> dict[str, Any]:
    value = context.store.read_json(reference.artifact_id, filename)
    if not isinstance(value, dict):
        raise StageHandlerError(f"dependency payload {filename!r} must be a JSON object")
    return value


def _gate_with_optional_override(
    rule: GateRule,
    observations: Mapping[str, object],
    override_reason: str | None,
) -> GateResult:
    evaluator = GateEvaluator()
    natural = evaluator.evaluate(rule, observations)
    if override_reason is None or natural.status is GateStatus.PASS:
        return natural
    return evaluator.evaluate(rule, observations, override_reason=override_reason)


def handle_inputs_tracking2(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    """Validate all Tracking2 bytes and normalize the legacy transplant table."""

    if dependencies:
        raise StageHandlerError("inputs.tracking2 must not receive dependency artifacts")
    manifest_path, adapter = _tracking2_adapter(context)
    validated = adapter.validate_all()
    manifest = adapter.manifest
    references = {
        "model_source": manifest.architecture.source,
        **{f"checkpoint_epoch{item.epoch}": item for item in manifest.checkpoints},
        "dataset_train": manifest.datasets.train,
        "dataset_test": manifest.datasets.test,
        "transplant": manifest.transplant.file,
    }
    if set(validated) != set(references):
        raise StageHandlerError("Tracking2 adapter returned an unexpected validated-file inventory")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise StageHandlerError("could not snapshot the Tracking2 manifest") from error
    if sha256_bytes(manifest_bytes) != context.config.inputs.tracking2.manifest_sha256:
        raise StageHandlerError("Tracking2 manifest changed while the input stage was running")
    embedded_manifest = parse_tracking2_manifest_bytes(
        manifest_bytes, source=context.config.inputs.tracking2.manifest
    )
    if canonical_hash(embedded_manifest.model_dump(mode="json")) != canonical_hash(
        manifest.model_dump(mode="json")
    ):
        raise StageHandlerError("Tracking2 manifest changed while it was being parsed")
    transplant_bytes = adapter.read_validated_bytes(manifest.transplant.file)
    files = {
        name: {
            "declared_path": reference.path.as_posix(),
            "sha256": reference.sha256,
            "size_bytes": reference.size_bytes,
        }
        for name, reference in sorted(references.items())
    }
    rows = [
        row.model_dump(mode="json") for row in adapter.normalize_transplant_bytes(transplant_bytes)
    ]
    result: dict[str, JSONLike] = {
        "schema_version": RESULT_SCHEMA_VERSION,
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
        # Keep immutable scientific input identity independent of the machine's
        # checkout location. Resolved operational paths belong in run provenance,
        # while these declarations are already signed by the stage configuration.
        "external_root": context.config.inputs.tracking2.root.as_posix(),
        "lineage_note": manifest.lineage_note,
        "lineage_quality": manifest.lineage_quality,
        "manifest": {
            "path": context.config.inputs.tracking2.manifest.as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "observed_repository_revision": manifest.observed_repository_revision,
        "read_only": True,
        "training": manifest.training.model_dump(mode="json"),
        "validated_files": files,
        "transplant_rows": rows,
    }
    return context.store.put_files(
        {
            "inputs.json": canonical_json_bytes(result),
            "manifest.yaml": manifest_bytes,
            "transplant.json": transplant_bytes,
        },
        kind="stage/inputs-tracking2",
        metadata=_stage_metadata(
            context,
            dependencies,
            input_count=len(validated),
            lineage_quality=manifest.lineage_quality,
        ),
        media_types={
            "inputs.json": "application/json",
            "manifest.yaml": "application/yaml",
            "transplant.json": "application/json",
        },
    )


def handle_inputs_tracking2_vgg(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    """Snapshot the immutable legacy VGG control and its criticality record."""

    if dependencies:
        raise StageHandlerError("inputs.tracking2_vgg must not receive dependency artifacts")
    manifest_path, adapter = _tracking2_vgg_adapter(context)
    validated = adapter.validate_all()
    manifest = adapter.manifest
    if manifest.lineage_quality != "exploratory_legacy":
        raise StageHandlerError("Tracking2 VGG lineage must remain explicitly exploratory")
    references = {
        "model_source": manifest.architecture.source,
        **{f"checkpoint_epoch{item.epoch}": item for item in manifest.checkpoints},
        "dataset_train": manifest.datasets.train,
        "dataset_test": manifest.datasets.test,
        "training_record": manifest.training_record.file,
    }
    if set(validated) != set(references):
        raise StageHandlerError(
            "Tracking2 VGG adapter returned an unexpected validated-file inventory"
        )
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise StageHandlerError("could not snapshot the Tracking2 VGG manifest") from error
    if sha256_bytes(manifest_bytes) != context.config.inputs.tracking2_vgg.manifest_sha256:
        raise StageHandlerError("Tracking2 VGG manifest changed while the input stage was running")
    embedded_manifest = parse_tracking2_vgg_manifest_bytes(
        manifest_bytes, source=context.config.inputs.tracking2_vgg.manifest
    )
    if canonical_hash(embedded_manifest.model_dump(mode="json")) != canonical_hash(
        manifest.model_dump(mode="json")
    ):
        raise StageHandlerError("Tracking2 VGG manifest changed while it was being parsed")

    criticality_record = adapter.read_training_record()
    criticality_bytes = adapter.read_validated_bytes(manifest.training_record.file)
    try:
        embedded_criticality = json.loads(criticality_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageHandlerError("Tracking2 VGG criticality metadata is invalid JSON") from error
    if not isinstance(embedded_criticality, dict) or canonical_hash(
        embedded_criticality
    ) != canonical_hash(criticality_record):
        raise StageHandlerError("Tracking2 VGG criticality metadata changed while being read")

    files = {
        name: {
            "declared_path": reference.path.as_posix(),
            "sha256": reference.sha256,
            "size_bytes": reference.size_bytes,
        }
        for name, reference in sorted(references.items())
    }
    result: dict[str, JSONLike] = {
        "schema_version": RESULT_SCHEMA_VERSION,
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
        "external_root": context.config.inputs.tracking2_vgg.root.as_posix(),
        "lineage_note": manifest.lineage_note,
        "lineage_quality": "exploratory_legacy",
        "manifest": {
            "path": context.config.inputs.tracking2_vgg.manifest.as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "observed_repository_revision": manifest.observed_repository_revision,
        "read_only": True,
        "training": manifest.training.model_dump(mode="json"),
        "training_record": criticality_record,
        "validated_files": files,
    }
    return context.store.put_files(
        {
            "inputs.json": canonical_json_bytes(result),
            "manifest.yaml": manifest_bytes,
            "criticality.json": criticality_bytes,
        },
        kind="stage/inputs-tracking2-vgg",
        metadata=_stage_metadata(
            context,
            dependencies,
            criticality_experiment=manifest.training_record.experiment,
            input_count=len(validated),
            lineage_quality="exploratory_legacy",
        ),
        media_types={
            "inputs.json": "application/json",
            "manifest.yaml": "application/yaml",
            "criticality.json": "application/json",
        },
    )


def _build_probe_bank_files(
    context: StageContext, input_payload: Mapping[str, object]
) -> tuple[dict[str, bytes], dict[str, JSONLike]]:
    """Materialize the complete deterministic probe artifact before publication."""

    try:
        return build_probe_bank_files(context.config, input_payload)
    except (ProbeArtifactError, ValueError) as error:
        raise StageHandlerError(str(error)) from error


def handle_exp1_probe_banks(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    """Freeze image identities, per-cut sites, and held-out bootstrap draws."""

    try:
        input_ref = dependencies["inputs.tracking2"]
    except KeyError as error:
        raise StageHandlerError("exp1.probe_banks requires inputs.tracking2") from error
    input_payload = _json_object(context, input_ref, "inputs.json")
    files, plan_payload = _build_probe_bank_files(context, input_payload)
    return context.store.put_files(
        files,
        kind="stage/exp1-probe-banks",
        metadata=_stage_metadata(
            context,
            dependencies,
            role_count=len(plan_payload["roles"]),  # type: ignore[arg-type]
            cut_count=len(plan_payload["cuts"]),  # type: ignore[arg-type]
        ),
        media_types={"plan.json": "application/json"},
    )


def _plateau_settings(config: LabConfig) -> Any:
    """Build the torch-optional collector settings from strict config."""

    from voronoi_lab.exp1.plateau_collection import PlateauCollectionSettings

    protocol = config.experiment1.plateau_protocol
    return PlateauCollectionSettings(
        root_seed=config.protocol.root_seed,
        covariance_fit_images=protocol.covariance_fit_images,
        centers_per_kind=protocol.centers_per_kind,
        perturbation_directions_per_center=protocol.perturbation_directions_per_center,
        perturbation_steps=protocol.perturbation_steps,
        perturbation_max_scale=protocol.perturbation_max_scale,
        local_surface_grid_points=protocol.local_surface_grid_points,
        local_surface_extent=protocol.local_surface_extent,
        three_anchor_grid_points=protocol.three_anchor_grid_points,
        three_anchor_axis_min=protocol.three_anchor_axis_min,
        three_anchor_axis_max=protocol.three_anchor_axis_max,
        hutchinson_probes=protocol.hutchinson_probes,
        intervention_batch_size=protocol.intervention_batch_size,
    )


def handle_exp1_synthetic_task(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    """Train and publish the deterministic residual-MLP checkpoint trajectory."""

    if dependencies:
        raise StageHandlerError("exp1.synthetic_task must not receive dependency artifacts")
    from voronoi_lab.exp1.synthetic_task import (
        GaussianMixtureConfig,
        ResidualMLPConfig,
        SyntheticTrainingConfig,
        run_synthetic_residual_task,
        write_synthetic_artifact,
    )

    declared = context.config.experiment1.synthetic_plateau_task
    seeds = SeedDeriver(context.config.protocol.root_seed, ("exp1", "synthetic_task", "v1"))
    dataset_config = GaussianMixtureConfig(
        seed=seeds.derive("dataset", bits=32),
        train_samples_per_class=declared.train_samples_per_class,
        test_samples_per_class=declared.test_samples_per_class,
        radius=declared.class_radius,
        standard_deviation=declared.noise_standard_deviation,
    )
    model_config = ResidualMLPConfig(
        input_dim=declared.input_dimensions,
        width=declared.hidden_width,
        blocks=declared.residual_blocks,
        classes=declared.classes,
    )
    training_config = SyntheticTrainingConfig(
        seed=seeds.derive("training", bits=32),
        epochs=declared.epochs,
        checkpoint_epochs=context.config.experiment1.checkpoints,
        batch_size=declared.batch_size,
        learning_rate=declared.learning_rate,
        momentum=declared.momentum,
        weight_decay=declared.weight_decay,
    )
    run = run_synthetic_residual_task(
        dataset_config=dataset_config,
        model_config=model_config,
        training_config=training_config,
    )
    with tempfile.TemporaryDirectory(prefix="voronoi-synthetic-task-") as temporary:
        root = Path(temporary)
        write_synthetic_artifact(root, run)
        files = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
    if "inventory.json" not in files or len(run.checkpoints) != len(
        context.config.experiment1.checkpoints
    ):
        raise StageHandlerError("synthetic task serialization omitted its inventory or checkpoints")
    final = run.metrics[-1]
    media_types = {
        name: "application/json" if name.endswith(".json") else "application/x-npz"
        for name in files
    }
    return context.store.put_files(
        files,
        kind="stage/exp1-synthetic-task",
        metadata=_stage_metadata(
            context,
            dependencies,
            architecture="normalization_free_residual_mlp",
            checkpoint_count=len(run.checkpoints),
            evidence_class="synthetic_known_data_training_trajectory",
            final_test_accuracy=final.test_accuracy,
        ),
        media_types=media_types,
    )


def handle_exp1_plateau_synthetic(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    """Collect real/fake paths and activation planes on every synthetic checkpoint."""

    try:
        task_ref = dependencies["exp1.synthetic_task"]
    except KeyError as error:
        raise StageHandlerError("exp1.plateau.synthetic requires exp1.synthetic_task") from error
    if set(dependencies) != {"exp1.synthetic_task"}:
        raise StageHandlerError("exp1.plateau.synthetic received unexpected dependencies")
    from voronoi_lab.exp1.plateau_collection import (
        collect_plateau_checkpoint,
        make_synthetic_intervention_adapter,
        package_plateau_checkpoint,
    )
    from voronoi_lab.exp1.synthetic_task import load_synthetic_artifact

    loaded = load_synthetic_artifact(task_ref.path / "files")
    settings = _plateau_settings(context.config)
    cut = context.config.experiment1.synthetic_plateau_task.intervention_block
    files: dict[str, bytes] = {}
    rows: list[JSONLike] = []
    train_ids = np.arange(len(loaded.dataset.train_inputs), dtype=np.int64)
    test_ids = np.arange(len(loaded.dataset.test_inputs), dtype=np.int64)
    for epoch in context.config.experiment1.checkpoints:
        model = loaded.load_model(epoch)
        adapter = make_synthetic_intervention_adapter(model, cut)
        result = collect_plateau_checkpoint(
            adapter,
            epoch=epoch,
            train_images=loaded.dataset.train_inputs,
            train_image_ids=train_ids,
            test_images=loaded.dataset.test_inputs,
            test_image_ids=test_ids,
            test_labels=loaded.dataset.test_labels,
            settings=settings,
        )
        prefix = f"checkpoints/epoch_{epoch:05d}"
        packaged = package_plateau_checkpoint(result)
        for name, data in packaged.items():
            files[f"{prefix}/{name}"] = data
        rows.append(
            {
                "epoch": epoch,
                "arrays": f"{prefix}/arrays.npz",
                "metadata": f"{prefix}/metadata.json",
                "inventory": f"{prefix}/inventory.json",
            }
        )
    summary: dict[str, JSONLike] = {
        "schema_version": PLATEAU_RESULT_SCHEMA_VERSION,
        "task": "three_class_gaussian_mixture",
        "architecture": "normalization_free_residual_mlp",
        "checkpoint_rows": rows,
        "checkpoint_epochs": list(context.config.experiment1.checkpoints),
        "jacobian_display_contract": {
            "protocol_version": context.config.experiment1.plateau_protocol.protocol_version,
            "local_source_array": _RESIDUAL_LOCAL_PLANE_JACOBIAN,
            "anchor_source_array": _RESIDUAL_ANCHOR_PLANE_JACOBIAN,
            "estimand": _RESIDUAL_PLANE_ESTIMAND,
        },
        "source_separation": {
            "response_paths": "Heimersheim-Mendel analogue",
            "three_anchor_rgb": "Janiak-et-al analogue",
            "jacobian_surfaces": "new hybrid diagnostic",
        },
        "task_artifact_id": task_ref.artifact_id,
    }
    files["summary.json"] = canonical_json_bytes(summary)
    media_types = {
        name: "application/json" if name.endswith(".json") else "application/x-npz"
        for name in files
    }
    return context.store.put_files(
        files,
        kind="stage/exp1-plateau-synthetic",
        metadata=_stage_metadata(
            context,
            dependencies,
            result_schema_version=PLATEAU_RESULT_SCHEMA_VERSION,
            checkpoint_count=len(rows),
            evidence_class="synthetic_activation_plateau_diagnostic",
        ),
        media_types=media_types,
    )


def _read_npy_artifact(
    context: StageContext,
    reference: ArtifactRef,
    path: str,
) -> np.ndarray:
    try:
        value = np.load(
            io.BytesIO(context.store.read_bytes(reference.artifact_id, path)),
            allow_pickle=False,
        )
        # np.load returns an ndarray for .npy payloads, not an NpzFile.
        return np.asarray(value).copy()
    except (OSError, ValueError, EOFError, TypeError) as error:
        raise StageHandlerError(f"probe payload {path!r} is not a valid pickle-free NPY") from error


def _materialize_plateau_cifar_banks(
    context: StageContext,
    probe_ref: ArtifactRef,
    adapter: Tracking2Adapter,
) -> tuple[Any, Any]:
    from voronoi_lab.exp1.data import CifarParquetSource, InputRecipe

    protocol = context.config.experiment1.plateau_protocol
    fit_ids = _read_npy_artifact(context, probe_ref, "indices/codebook_fit.npy")
    intervention_ids = _read_npy_artifact(context, probe_ref, "indices/intervention.npy")
    if fit_ids.ndim != 1 or intervention_ids.ndim != 1:
        raise StageHandlerError("plateau probe image ids must be vectors")
    if len(fit_ids) < protocol.covariance_fit_images:
        raise StageHandlerError("codebook_fit bank is smaller than plateau covariance_fit_images")
    if len(intervention_ids) < protocol.centers_per_kind:
        raise StageHandlerError("intervention bank is smaller than plateau centers_per_kind")
    clean = next(
        (recipe for recipe in context.config.experiment1.input_recipes if recipe.kind == "clean"),
        None,
    )
    if clean is None:
        raise StageHandlerError("plateau CIFAR collection requires the declared clean recipe")
    recipe = InputRecipe(
        name=clean.name,
        kind=clean.kind,
        crop_padding=clean.crop_padding,
        flip_probability=clean.flip_probability,
        brightness_fraction=clean.brightness_fraction,
        recipe_version=clean.recipe_version,
    )
    manifest = adapter.manifest
    train_path = adapter.validate_file(manifest.datasets.train)
    test_path = adapter.validate_file(manifest.datasets.test)
    train_source = CifarParquetSource(
        train_path,
        split="train",
        expected_sha256=manifest.datasets.train.sha256,
        expected_size=manifest.datasets.train.size_bytes,
    )
    test_source = CifarParquetSource(
        test_path,
        split="test",
        expected_sha256=manifest.datasets.test.sha256,
        expected_size=manifest.datasets.test.size_bytes,
    )
    train_bank = train_source.materialize(
        fit_ids[: protocol.covariance_fit_images],
        recipe=recipe,
        root_seed=context.config.protocol.root_seed,
    )
    test_bank = test_source.materialize(
        intervention_ids,
        recipe=recipe,
        root_seed=context.config.protocol.root_seed,
    )
    return train_bank, test_bank


def handle_exp1_plateau_cifar(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    """Collect ten resumable ResNet/VGG checkpoint shards on fixed CIFAR images."""

    if context.config.runtime.dtype != "float32":
        raise StageHandlerError("CIFAR plateau collection requires deterministic fp32")
    actual_device, runtime_provenance = _configure_plateau_torch_runtime(
        context.config.runtime.device
    )
    expected_dependencies = {"inputs.tracking2", "inputs.tracking2_vgg", "exp1.probe_banks"}
    if set(dependencies) != expected_dependencies:
        raise StageHandlerError(
            "exp1.plateau.cifar requires ResNet, VGG, and fixed probe-bank artifacts"
        )
    from voronoi_lab.exp1.plateau_collection import (
        collect_plateau_checkpoint,
        make_resnet_intervention_adapter,
        make_vgg_intervention_adapter,
        package_plateau_checkpoint,
    )

    parent = context.index.get_stage(context.run_id, context.stage_spec.name)
    if parent is None or parent.state is not StageState.RUNNING:
        raise StageHandlerError("CIFAR plateau shards require a claimed RUNNING parent stage")
    _resnet_manifest, resnet = _tracking2_adapter(context)
    _vgg_manifest, vgg = _tracking2_vgg_adapter(context)
    if resnet.manifest.datasets.model_dump(mode="json") != vgg.manifest.datasets.model_dump(
        mode="json"
    ):
        raise StageHandlerError("ResNet and VGG manifests do not pin the same CIFAR bytes")
    train_bank, test_bank = _materialize_plateau_cifar_banks(
        context,
        dependencies["exp1.probe_banks"],
        resnet,
    )
    settings = _plateau_settings(context.config)
    selected_config = stage_config(context.config, context.stage_spec.config_paths)
    upstream_ids = {name: reference.artifact_id for name, reference in dependencies.items()}
    coordinates = tuple(
        (architecture, epoch)
        for architecture in ("resnet", "vgg")
        for epoch in context.config.experiment1.checkpoints
    )
    specs = tuple(
        ShardSpec(
            parent_stage=context.stage_spec.name,
            parent_stage_signature=parent.stage_signature,
            key=ShardKey({"architecture": architecture, "epoch": epoch}),
            artifact_kind="shard/exp1-plateau-cifar-checkpoint",
            stage_config=selected_config,
            source_identity=context.source_identity,
            upstream_artifacts=upstream_ids,
        )
        for architecture, epoch in coordinates
    )

    def execute_checkpoint(spec: ShardSpec) -> ArtifactRef:
        architecture = spec.key.coordinates["architecture"]
        epoch = spec.key.coordinates["epoch"]
        if architecture not in {"resnet", "vgg"} or type(epoch) is not int:
            raise StageHandlerError("CIFAR plateau shard coordinates are invalid")

        def publish(shard_context: ShardContext) -> ArtifactRef:
            if architecture == "resnet":
                model = resnet.load_model(epoch, device=actual_device)
                intervention = make_resnet_intervention_adapter(
                    model,
                    resnet,
                    context.config.experiment1.plateau_protocol.resnet_cut,
                    device=actual_device,
                )
            else:
                model = vgg.load_model(epoch, device=actual_device)
                intervention = make_vgg_intervention_adapter(
                    model,
                    vgg,
                    context.config.experiment1.plateau_protocol.vgg_cut,
                    device=actual_device,
                )
            result = collect_plateau_checkpoint(
                intervention,
                epoch=epoch,
                train_images=train_bank.tensors,
                train_image_ids=train_bank.image_ids,
                test_images=test_bank.tensors,
                test_image_ids=test_bank.image_ids,
                test_labels=test_bank.labels,
                settings=settings,
            )
            packaged = package_plateau_checkpoint(result)
            return shard_context.store.put_files(
                packaged,
                kind=spec.artifact_kind,
                metadata=shard_context.artifact_metadata(
                    {
                        "architecture": architecture,
                        "epoch": epoch,
                        "evidence_class": "cifar_single_seed_plateau_checkpoint",
                        "result_schema_version": PLATEAU_RESULT_SCHEMA_VERSION,
                        "test_bank_id": test_bank.bank_id,
                        "train_bank_id": train_bank.bank_id,
                        "runtime_provenance": runtime_provenance,
                    }
                ),
                media_types={
                    "arrays.npz": "application/x-npz",
                    "metadata.json": "application/json",
                    "inventory.json": "application/json",
                },
            )

        return ShardExecutor(context.store, context.index, context.run_id).execute(spec, publish)

    # Loading both large legacy models concurrently is less predictable on both
    # CPU and bounded-memory accelerators, so checkpoint shards execute in stable order.
    executed = tuple(execute_checkpoint(spec) for spec in specs)
    reduction = ShardReducer(context.store, context.index, context.run_id).publish(specs)
    if tuple(item.artifact_id for item in executed) != tuple(
        item.artifact_id for item in reduction.shard_artifacts
    ):
        raise StageHandlerError("CIFAR plateau reducer order differs from executed shards")
    rows: list[JSONLike] = []
    for (architecture, epoch), reference in zip(
        coordinates, reduction.shard_artifacts, strict=True
    ):
        metadata = _json_object(context, reference, "metadata.json")
        if metadata.get("architecture") is None or metadata.get("epoch") != epoch:
            raise StageHandlerError("CIFAR plateau checkpoint metadata is inconsistent")
        shard_runtime = reference.manifest.metadata.get("runtime_provenance")
        if not isinstance(shard_runtime, Mapping) or metadata.get(
            "compute_device"
        ) != shard_runtime.get("actual_device"):
            raise StageHandlerError("CIFAR plateau shard runtime provenance is inconsistent")
        rows.append(
            {
                "architecture": architecture,
                "epoch": epoch,
                "artifact_id": reference.artifact_id,
                "arrays": "arrays.npz",
                "metadata": "metadata.json",
                "inventory": "inventory.json",
                "runtime_provenance": dict(shard_runtime),
            }
        )
    summary: dict[str, JSONLike] = {
        "schema_version": PLATEAU_RESULT_SCHEMA_VERSION,
        "task": "cifar10",
        "architectures": ["resnet", "vgg"],
        "checkpoint_epochs": list(context.config.experiment1.checkpoints),
        "checkpoint_rows": rows,
        "reducer_artifact_id": reduction.reference.artifact_id,
        # This describes the process that assembled the parent. Each row above
        # preserves the producer runtime of its child even when a shard was reused.
        "orchestrator_runtime_provenance": runtime_provenance,
        "train_bank": {
            "bank_id": train_bank.bank_id,
            "image_ids": train_bank.image_ids.tolist(),
            "tensor_sha256": train_bank.tensor_sha256,
            "source_sha256": train_bank.source_sha256,
            "recipe": train_bank.recipe.to_dict(),
        },
        "test_bank": {
            "bank_id": test_bank.bank_id,
            "image_ids": test_bank.image_ids.tolist(),
            "labels": test_bank.labels.tolist(),
            "tensor_sha256": test_bank.tensor_sha256,
            "source_sha256": test_bank.source_sha256,
            "recipe": test_bank.recipe.to_dict(),
        },
        "lineage_scope": "exploratory_single_seed_descriptive_architecture_comparison",
        "jacobian_display_contract": {
            "protocol_version": context.config.experiment1.plateau_protocol.protocol_version,
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
            "comparison_scope": ("descriptive_confounded_side_by_side_with_row_specific_operators"),
        },
        "confounds": [
            "batch normalization differs",
            "parameter count differs",
            "weight decay differs",
            "training progress differs",
        ],
        "source_separation": {
            "response_paths": "Heimersheim-Mendel analogue",
            "three_anchor_rgb": "Janiak-et-al analogue",
            "jacobian_surfaces": "new hybrid diagnostic",
        },
    }
    return context.store.put_json(
        summary,
        filename="summary.json",
        kind="stage/exp1-plateau-cifar",
        metadata=_stage_metadata(
            context,
            dependencies,
            result_schema_version=PLATEAU_RESULT_SCHEMA_VERSION,
            evidence_class="cifar_single_seed_plateau_trajectory",
            reducer_artifact_id=reduction.reference.artifact_id,
            shard_count=len(rows),
        ),
    )


def _load_animation_arrays(
    context: StageContext,
    reference: ArtifactRef,
    path: str,
    required: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Load only declared numeric NPZ members used by the animation stage."""

    try:
        raw = context.store.read_bytes(reference.artifact_id, path)
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            missing = sorted(set(required) - set(archive.files))
            if missing:
                raise StageHandlerError(
                    f"animation source {path!r} is missing arrays: {', '.join(missing)}"
                )
            arrays = {name: np.asarray(archive[name]).copy() for name in required}
    except StageHandlerError:
        raise
    except (OSError, ValueError, EOFError, TypeError) as error:
        raise StageHandlerError(
            f"animation source {path!r} is not a valid pickle-free NPZ"
        ) from error
    for name, array in arrays.items():
        if array.dtype.kind not in "biuf" or array.size == 0 or not np.all(np.isfinite(array)):
            raise StageHandlerError(f"animation source array {name!r} must be finite and numeric")
    return arrays


def _real_fake_animation_frame(
    checkpoint: str,
    arrays: Mapping[str, np.ndarray],
    *,
    jacobian_field: str,
) -> Any:
    """Reduce saved checkpoint arrays into one source-bound display frame."""

    from voronoi_lab.exp1.animation import RealFakeFrame

    local_jacobian = arrays[jacobian_field]
    path_response = arrays["path_response_l2"]
    local_axis = arrays["local_axis"]
    path_axis = arrays["path_coefficients"]
    if local_jacobian.ndim != 4 or local_jacobian.shape[0] != 2:
        raise StageHandlerError(f"{jacobian_field} must have shape (2, centers, y, x)")
    if path_response.ndim != 4 or path_response.shape[0] != 2:
        raise StageHandlerError("path_response_l2 must have shape (2, centers, directions, steps)")
    if local_axis.ndim != 1 or local_jacobian.shape[2:] != (
        len(local_axis),
        len(local_axis),
    ):
        raise StageHandlerError("local animation axis does not align with its Jacobian fields")
    if path_axis.ndim != 1 or path_response.shape[-1] != len(path_axis):
        raise StageHandlerError("path animation axis does not align with its response fields")
    if np.any(local_jacobian < -1e-12) or np.any(path_response < -1e-12):
        raise StageHandlerError("animation Jacobian and response fields must be nonnegative")
    heatmaps = np.maximum(local_jacobian, 0.0).mean(axis=1)
    curves = np.median(np.maximum(path_response, 0.0), axis=(1, 2))
    return RealFakeFrame(
        checkpoint=checkpoint,
        real_heatmap=heatmaps[0],
        fake_heatmap=heatmaps[1],
        heatmap_x=local_axis,
        heatmap_y=local_axis,
        curve_x=path_axis,
        real_curve=curves[0],
        fake_curve=curves[1],
    )


def _architecture_animation_inputs(
    arrays_by_architecture: Mapping[str, list[Mapping[str, np.ndarray]]],
    checkpoints: tuple[int, ...],
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    """Prepare synchronized cell/Jacobian frames with fixed global RGB scales."""

    from voronoi_lab.exp1.animation import ArchitectureCellsFrame

    expected = {"resnet", "vgg"}
    if set(arrays_by_architecture) != expected or any(
        len(arrays_by_architecture[name]) != len(checkpoints) for name in expected
    ):
        raise StageHandlerError("architecture animation inputs do not align with checkpoints")

    maxima: dict[str, np.ndarray] = {}
    for architecture in ("resnet", "vgg"):
        distances = [
            item["anchor_output_distances"] for item in arrays_by_architecture[architecture]
        ]
        for field in distances:
            if field.ndim != 3 or field.shape[-1] != 3 or np.any(field < -1e-12):
                raise StageHandlerError(
                    "anchor_output_distances must be nonnegative fields with three channels"
                )
        maxima[architecture] = np.maximum.reduce(
            [np.maximum(field, 0.0).max(axis=(0, 1)) for field in distances]
        )
        if np.any(maxima[architecture] <= 0.0):
            raise StageHandlerError(
                "each architecture/anchor distance channel needs positive range"
            )

    frames: list[Any] = []
    for index, epoch in enumerate(checkpoints):
        prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for architecture in ("resnet", "vgg"):
            arrays = arrays_by_architecture[architecture][index]
            distances = np.maximum(arrays["anchor_output_distances"], 0.0)
            jacobian_field = (
                _RESIDUAL_ANCHOR_PLANE_JACOBIAN
                if architecture == "resnet"
                else _RAW_ANCHOR_PLANE_JACOBIAN
            )
            jacobian_by_context = arrays[jacobian_field]
            axis = arrays["anchor_axis"]
            anchors = arrays["anchor_coordinates"]
            if jacobian_by_context.ndim != 3 or jacobian_by_context.shape[0] != 3:
                raise StageHandlerError(f"{jacobian_field} must have three contexts")
            if np.any(jacobian_by_context < -1e-12):
                raise StageHandlerError("anchor transition Jacobian fields must be nonnegative")
            if axis.ndim != 1 or distances.shape[:2] != (len(axis), len(axis)):
                raise StageHandlerError("anchor animation axis does not align with distance fields")
            if jacobian_by_context.shape[1:] != distances.shape[:2]:
                raise StageHandlerError("anchor Jacobian and output-distance fields do not align")
            if anchors.shape != (3, 2):
                raise StageHandlerError("anchor coordinates must have shape (3, 2)")
            rgb = np.clip(
                1.0 - distances / maxima[architecture].reshape(1, 1, 3),
                0.0,
                1.0,
            )
            prepared[architecture] = (
                rgb,
                np.maximum(jacobian_by_context, 0.0).mean(axis=0),
                axis,
                anchors,
            )
        resnet = prepared["resnet"]
        vgg = prepared["vgg"]
        if not np.array_equal(resnet[2], vgg[2]):
            raise StageHandlerError("ResNet and VGG anchor axes must be identical")
        frames.append(
            ArchitectureCellsFrame(
                checkpoint=f"epoch {epoch}",
                resnet_rgb=resnet[0],
                resnet_jacobian=resnet[1],
                vgg_rgb=vgg[0],
                vgg_jacobian=vgg[1],
                alpha_coordinates=resnet[2],
                beta_coordinates=resnet[2],
                resnet_anchors=resnet[3],
                vgg_anchors=vgg[3],
            )
        )
    return frames, maxima["resnet"], maxima["vgg"]


def handle_exp1_plateau_animations(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    """Render four source-bound GIF bundles from immutable plateau arrays."""

    expected_dependencies = {"exp1.plateau.synthetic", "exp1.plateau.cifar"}
    if set(dependencies) != expected_dependencies:
        raise StageHandlerError(
            "exp1.plateau.animations requires synthetic and CIFAR plateau artifacts"
        )
    from voronoi_lab.exp1.animation import (
        AnimationTiming,
        render_architecture_cells_animation,
        render_real_fake_animation,
    )

    synthetic_ref = dependencies["exp1.plateau.synthetic"]
    cifar_ref = dependencies["exp1.plateau.cifar"]
    if synthetic_ref.manifest.kind != "stage/exp1-plateau-synthetic":
        raise StageHandlerError("synthetic animation dependency has the wrong artifact kind")
    if cifar_ref.manifest.kind != "stage/exp1-plateau-cifar":
        raise StageHandlerError("CIFAR animation dependency has the wrong artifact kind")
    synthetic_summary = _json_object(context, synthetic_ref, "summary.json")
    cifar_summary = _json_object(context, cifar_ref, "summary.json")
    if synthetic_summary.get("schema_version") != PLATEAU_RESULT_SCHEMA_VERSION:
        raise StageHandlerError("synthetic animation source must use plateau schema v2")
    if cifar_summary.get("schema_version") != PLATEAU_RESULT_SCHEMA_VERSION:
        raise StageHandlerError("CIFAR animation source must use plateau schema v2")
    checkpoints = tuple(context.config.experiment1.checkpoints)
    if synthetic_summary.get("checkpoint_epochs") != list(checkpoints):
        raise StageHandlerError("synthetic animation checkpoint axis differs from config")
    if cifar_summary.get("checkpoint_epochs") != list(checkpoints):
        raise StageHandlerError("CIFAR animation checkpoint axis differs from config")

    real_fake_base_required = (
        "local_axis",
        _RAW_LOCAL_PLANE_JACOBIAN,
        "path_coefficients",
        "path_response_l2",
    )
    synthetic_required = (*real_fake_base_required, _RESIDUAL_LOCAL_PLANE_JACOBIAN)
    synthetic_rows = synthetic_summary.get("checkpoint_rows")
    if not isinstance(synthetic_rows, list) or len(synthetic_rows) != len(checkpoints):
        raise StageHandlerError("synthetic animation checkpoint rows are invalid")
    synthetic_frames: list[Any] = []
    for epoch, row in zip(checkpoints, synthetic_rows, strict=True):
        if not isinstance(row, dict) or row.get("epoch") != epoch:
            raise StageHandlerError("synthetic animation checkpoint row is inconsistent")
        expected_path = f"checkpoints/epoch_{epoch:05d}/arrays.npz"
        if row.get("arrays") != expected_path:
            raise StageHandlerError("synthetic animation array path is inconsistent")
        arrays = _load_animation_arrays(
            context,
            synthetic_ref,
            expected_path,
            synthetic_required,
        )
        synthetic_frames.append(
            _real_fake_animation_frame(
                f"epoch {epoch}",
                arrays,
                jacobian_field=_RESIDUAL_LOCAL_PLANE_JACOBIAN,
            )
        )

    cifar_rows = cifar_summary.get("checkpoint_rows")
    expected_coordinates = tuple(
        (architecture, epoch) for architecture in ("resnet", "vgg") for epoch in checkpoints
    )
    if not isinstance(cifar_rows, list) or len(cifar_rows) != len(expected_coordinates):
        raise StageHandlerError("CIFAR animation checkpoint rows are invalid")
    architecture_base_required = (
        *real_fake_base_required,
        "anchor_axis",
        "anchor_coordinates",
        "anchor_output_distances",
        _RAW_ANCHOR_PLANE_JACOBIAN,
    )
    arrays_by_architecture: dict[str, list[dict[str, np.ndarray]]] = {
        "resnet": [],
        "vgg": [],
    }
    cifar_real_fake: dict[str, list[Any]] = {"resnet": [], "vgg": []}
    for (architecture, epoch), row in zip(expected_coordinates, cifar_rows, strict=True):
        if (
            not isinstance(row, dict)
            or row.get("architecture") != architecture
            or row.get("epoch") != epoch
            or row.get("arrays") != "arrays.npz"
        ):
            raise StageHandlerError("CIFAR animation checkpoint row is inconsistent")
        artifact_id = row.get("artifact_id")
        if not isinstance(artifact_id, str):
            raise StageHandlerError("CIFAR animation child artifact id is invalid")
        child = context.store.get(artifact_id, verify=True)
        if child.manifest.kind != "shard/exp1-plateau-cifar-checkpoint":
            raise StageHandlerError("CIFAR animation child has the wrong artifact kind")
        required = architecture_base_required
        if architecture == "resnet":
            required = (
                *required,
                _RESIDUAL_LOCAL_PLANE_JACOBIAN,
                _RESIDUAL_ANCHOR_PLANE_JACOBIAN,
            )
        arrays = _load_animation_arrays(context, child, "arrays.npz", required)
        arrays_by_architecture[architecture].append(arrays)
        cifar_real_fake[architecture].append(
            _real_fake_animation_frame(
                f"epoch {epoch}",
                arrays,
                jacobian_field=(
                    _RESIDUAL_LOCAL_PLANE_JACOBIAN
                    if architecture == "resnet"
                    else _RAW_LOCAL_PLANE_JACOBIAN
                ),
            )
        )

    architecture_frames, resnet_maxima, vgg_maxima = _architecture_animation_inputs(
        arrays_by_architecture,
        checkpoints,
    )
    protocol = context.config.experiment1.plateau_protocol
    timing = AnimationTiming(
        orientation_ms=protocol.orientation_frame_ms,
        checkpoint_ms=protocol.intermediate_frame_ms,
        conclusion_ms=protocol.final_frame_ms,
    )
    bundle_specs = (
        {
            "name": "synthetic_real_fake",
            "task": "three_class_gaussian_mixture",
            "architectures": ["normalization_free_residual_mlp"],
            "frames": synthetic_frames,
            "title": "Synthetic: real vs matched fake",
            "scalar_source_array": _RESIDUAL_LOCAL_PLANE_JACOBIAN,
            "scalar_estimand": _RESIDUAL_PLANE_ESTIMAND,
            "scalar_label": "2D ‖D(T-I)‖F",
        },
        {
            "name": "cifar_resnet_real_fake",
            "task": "cifar10",
            "architectures": ["tracking2_resnet18_v2_width64"],
            "frames": cifar_real_fake["resnet"],
            "title": "CIFAR ResNet: real vs fake",
            "scalar_source_array": _RESIDUAL_LOCAL_PLANE_JACOBIAN,
            "scalar_estimand": _RESIDUAL_PLANE_ESTIMAND,
            "scalar_label": "2D ‖D(T-I)‖F",
        },
        {
            "name": "cifar_vgg_real_fake",
            "task": "cifar10",
            "architectures": ["tracking2_vgg19_bn_width1_classifier512"],
            "frames": cifar_real_fake["vgg"],
            "title": "CIFAR VGG: real vs fake",
            "scalar_source_array": _RAW_LOCAL_PLANE_JACOBIAN,
            "scalar_estimand": _RAW_PLANE_ESTIMAND,
            "scalar_label": "2D ‖DT‖F",
        },
    )
    bundle_rows: list[JSONLike] = []
    media_types: dict[str, str] = {"summary.json": "application/json"}
    with tempfile.TemporaryDirectory(prefix="exp1-plateau-animations-") as raw_directory:
        directory = Path(raw_directory)
        generated: dict[str, Path | bytes] = {}
        for item in bundle_specs:
            name = str(item["name"])
            output = render_real_fake_animation(
                item["frames"],  # type: ignore[arg-type]
                directory / name,
                title=str(item["title"]),
                scalar_label=str(item["scalar_label"]),
                heatmap_classification_label="NEW HYBRID",
                scalar_estimand=str(item["scalar_estimand"]),
                curve_label="Logit L2",
                curve_estimand=("center-and-direction median downstream-logit L2 from path base"),
                final_callout=(
                    "Final checkpoint → compare real and matched-fake geometry. This is a "
                    "descriptive diagnostic, not a phenomenon gate or causal residual claim."
                ),
                timing=timing,
            )
            prefix = f"animations/{name}"
            paths = {
                "gif": f"{prefix}.gif",
                "final_png": f"{prefix}_final.png",
                "metadata": f"{prefix}_metadata.json",
            }
            generated[paths["gif"]] = output.gif_path
            generated[paths["final_png"]] = output.final_png_path
            generated[paths["metadata"]] = output.metadata_path
            media_types.update(
                {
                    paths["gif"]: "image/gif",
                    paths["final_png"]: "image/png",
                    paths["metadata"]: "application/json",
                }
            )
            bundle_rows.append(
                {
                    "name": name,
                    "task": item["task"],
                    "architectures": item["architectures"],
                    "animation_kind": "real_fake_scalar_fields",
                    **paths,
                    "scalar_source_array": item["scalar_source_array"],
                    "scalar_estimand": item["scalar_estimand"],
                    "scalar_selection": (
                        "residual_update_for_residual_transition"
                        if item["scalar_estimand"] == _RESIDUAL_PLANE_ESTIMAND
                        else "raw_transition_for_nonresidual_transition"
                    ),
                    "curve_estimand": (
                        "center-and-direction median downstream-logit L2 from path base"
                    ),
                    "method_classification": {
                        "curve": "architecture-adapted Heimersheim-Mendel analogue",
                        "heatmap": "new hybrid Jacobian diagnostic",
                    },
                }
            )

        architecture_output = render_architecture_cells_animation(
            architecture_frames,
            directory / "cifar_architecture_cells",
            resnet_rgb_channel_maxima=resnet_maxima,
            vgg_rgb_channel_maxima=vgg_maxima,
            title="CIFAR residual-adjusted / VGG raw fields",
            jacobian_label="2D ‖·‖F",
            resnet_jacobian_label="NEW HYBRID · D(T-I)",
            vgg_jacobian_label="NEW HYBRID · DT",
            resnet_jacobian_estimand=_RESIDUAL_PLANE_ESTIMAND,
            vgg_jacobian_estimand=_RAW_PLANE_ESTIMAND,
            comparison_note=(
                "The rows use different operators and legacy training recipes; this is a "
                "descriptive, confounded side-by-side view, not a causal architecture ablation."
            ),
            final_callout=(
                "Final checkpoint → inspect each row's sensitivity structure. ResNet shows "
                "2D D(T-I), VGG shows 2D DT; the view is descriptive and confounded."
            ),
            timing=timing,
        )
        architecture_prefix = "animations/cifar_architecture_cells"
        architecture_paths = {
            "gif": f"{architecture_prefix}.gif",
            "final_png": f"{architecture_prefix}_final.png",
            "metadata": f"{architecture_prefix}_metadata.json",
        }
        generated[architecture_paths["gif"]] = architecture_output.gif_path
        generated[architecture_paths["final_png"]] = architecture_output.final_png_path
        generated[architecture_paths["metadata"]] = architecture_output.metadata_path
        media_types.update(
            {
                architecture_paths["gif"]: "image/gif",
                architecture_paths["final_png"]: "image/png",
                architecture_paths["metadata"]: "application/json",
            }
        )
        bundle_rows.append(
            {
                "name": "cifar_architecture_cells",
                "task": "cifar10",
                "architectures": [
                    "tracking2_resnet18_v2_width64",
                    "tracking2_vgg19_bn_width1_classifier512",
                ],
                "animation_kind": "architecture_cells_and_jacobians",
                **architecture_paths,
                "cell_estimand": "three frozen-context downstream-logit L2 distances",
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
                "method_classification": {
                    "cells": "architecture-adapted Janiak-et-al Appendix-C analogue",
                    "jacobian": "new hybrid Jacobian diagnostic",
                },
            }
        )
        summary: dict[str, JSONLike] = {
            "schema_version": PLATEAU_RESULT_SCHEMA_VERSION,
            "task": "experiment1_activation_geometry_animations",
            "checkpoint_epochs": list(checkpoints),
            "source_artifacts": {
                "exp1.plateau.synthetic": synthetic_ref.artifact_id,
                "exp1.plateau.cifar": cifar_ref.artifact_id,
            },
            "timing_ms": {
                "orientation": timing.orientation_ms,
                "checkpoint": timing.checkpoint_ms,
                "conclusion": timing.conclusion_ms,
            },
            "bundles": bundle_rows,
            "normalization": {
                "real_fake_scalar_scales": "global within each bundle across all checkpoints",
                "real_fake_curve_scales": "global within each bundle across all checkpoints",
                "cell_rgb_raw_field": "anchor_output_distances",
                "cell_rgb_transform": "clip(1 - distance / channel_max, 0, 1)",
                "cell_rgb_scope": ("per architecture and anchor channel across every checkpoint"),
                "resnet_channel_maxima": resnet_maxima.tolist(),
                "vgg_channel_maxima": vgg_maxima.tolist(),
                "jacobian_scale": (
                    "one numerical display scale across row-specific estimands and all "
                    "checkpoints; not an equivalent-operator or causal comparison"
                ),
            },
            "jacobian_display_contract": {
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
            },
            "source_separation": {
                "response_paths": "Heimersheim-Mendel analogue",
                "three_anchor_rgb": "Janiak-et-al analogue",
                "jacobian_surfaces": "new hybrid diagnostic",
            },
            "nonclaims": [
                "No plateau or stable-region phenomenon is established by rendering.",
                "The VGG/ResNet comparison is single-seed, exploratory, and confounded.",
                "ResNet and VGG Jacobian rows use different operators and are not a direct "
                "causal architecture comparison.",
                "RGB output-distance cells are not Jacobian fields or Voronoi assignments.",
            ],
        }
        generated["summary.json"] = canonical_json_bytes(summary)
        return context.store.put_files(
            generated,
            kind="stage/exp1-plateau-animations",
            metadata=_stage_metadata(
                context,
                dependencies,
                result_schema_version=PLATEAU_RESULT_SCHEMA_VERSION,
                animation_count=len(bundle_rows),
                evidence_class="presentation_of_exploratory_plateau_diagnostics",
            ),
            media_types=media_types,
        )


def _probe_determinism_check(
    context: StageContext,
    input_payload: Mapping[str, object],
    saved_reference: ArtifactRef,
) -> bool:
    first, _ = _build_probe_bank_files(context, input_payload)
    second, _ = _build_probe_bank_files(context, input_payload)
    if first.keys() != second.keys() or any(first[name] != second[name] for name in first):
        return False
    saved_paths = {entry.path for entry in saved_reference.manifest.files}
    return saved_paths == set(first) and all(
        context.store.read_bytes(saved_reference.artifact_id, name) == expected
        for name, expected in first.items()
    )


def _empty_resnet_metrics() -> dict[str, JSONLike]:
    return {
        "actual_device": "cpu",
        "identity_exact": None,
        "identity_max_absolute_error": None,
        "identity_logits_by_cut": {},
        "identity_per_cut": {},
        "jvp_by_cut": {},
        "jvp_cuts_completed": 0,
        "jvp_failures": {},
        "jvp_median_relative_error": None,
        "jvp_p95_relative_error": None,
    }


def _run_resnet_mechanics(context: StageContext) -> tuple[dict[str, JSONLike], list[str]]:
    metrics = _empty_resnet_metrics()
    warnings: list[str] = []
    try:
        import torch

        _, adapter = _tracking2_adapter(context)
        target_epoch = adapter.manifest.training.target_epoch
        model = adapter.load_model(target_epoch, device="cpu")
        dtype = torch.float64 if context.config.runtime.dtype == "float64" else torch.float32
        model.to(dtype=dtype)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        protocol = context.config.experiment1.mechanical_protocol
        values_per_image = 3 * 32 * 32
        images = torch.linspace(
            protocol.input_min,
            protocol.input_max,
            protocol.input_batch_size * values_per_image,
            dtype=dtype,
        ).reshape(protocol.input_batch_size, 3, 32, 32)
        cuts = tuple(resolve_cut(name) for name in context.config.experiment1.cuts)
        parity = zero_intervention_parity(
            model,
            images,
            cuts,
            encode=adapter.encode,
            suffix=adapter.suffix,
        )
        identity_logits_by_cut: dict[str, dict[str, JSONLike]] = {
            name: {
                "full_logits": list(full),
                "split_logits": list(split),
            }
            for name, full, split in zip(
                parity.cuts,
                parity.full_logits,
                parity.split_logits,
                strict=True,
            )
        }

        direction_seeds = SeedDeriver(
            context.config.protocol.root_seed,
            ("exp1", "mechanical", "jvp_direction"),
        )
        jvp_outputs_by_cut: dict[str, dict[str, JSONLike]] = {}
        jvp_failures: dict[str, JSONLike] = {}
        epsilon = (
            protocol.jvp_epsilon_float64 if dtype == torch.float64 else protocol.jvp_epsilon_float32
        )
        for cut_name in context.config.experiment1.sentinel_cuts:
            cut = resolve_cut(cut_name)
            direction_generator = torch.Generator(device="cpu")
            direction_generator.manual_seed(direction_seeds.derive({"cut": cut.name}))
            with torch.no_grad():
                point = adapter.encode(model, images, cut).detach().clone()
            direction = torch.randn(
                point.shape,
                dtype=point.dtype,
                device=point.device,
                generator=direction_generator,
            )
            direction /= torch.sqrt(torch.mean(direction**2))
            try:
                check = validate_jvp(
                    lambda value, selected=cut: adapter.suffix(model, value, selected),
                    point,
                    direction,
                    epsilon=epsilon,
                    sample_axis=None,
                    denominator_floor=protocol.denominator_floor,
                )
            except (RuntimeError, TypeError, ValueError) as error:
                warnings.append(f"JVP unavailable at {cut.name}: {type(error).__name__}: {error}")
                jvp_failures[cut.name] = f"{type(error).__name__}: {error}"
                continue
            jvp_outputs_by_cut[cut.name] = {
                "epsilon": check.epsilon,
                "automatic_jvp": list(check.automatic_output),
                "finite_difference_jvp": list(check.finite_difference_output),
            }
        summary = summarize_resnet_mechanical_evidence(
            identity_logits_by_cut,
            jvp_outputs_by_cut,
            identity_cuts=context.config.experiment1.cuts,
            jvp_cuts=context.config.experiment1.sentinel_cuts,
            denominator_floor=protocol.denominator_floor,
        )
        metrics.update(
            {
                "identity_exact": summary.identity_exact,
                "identity_max_absolute_error": summary.identity_max_absolute_error,
                "identity_logits_by_cut": identity_logits_by_cut,
                "identity_per_cut": summary.identity_per_cut,
                "jvp_by_cut": {
                    cut_name: {
                        **raw,
                        "relative_error": summary.jvp_relative_error_by_cut[cut_name],
                    }
                    for cut_name, raw in jvp_outputs_by_cut.items()
                },
                "jvp_cuts_completed": summary.jvp_cuts_completed,
                "jvp_failures": jvp_failures,
                "jvp_median_relative_error": summary.jvp_median_relative_error,
                "jvp_p95_relative_error": summary.jvp_p95_relative_error,
            }
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        warnings.append(f"real ResNet mechanics unavailable: {type(error).__name__}: {error}")
    return metrics, warnings


def handle_exp1_mechanical(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    """Run deterministic toy invariants and bounded CPU checks on the pinned model."""

    try:
        input_ref = dependencies["inputs.tracking2"]
        probe_ref = dependencies["exp1.probe_banks"]
    except KeyError as error:
        raise StageHandlerError(
            "exp1.mechanical requires inputs.tracking2 and exp1.probe_banks"
        ) from error
    input_payload = _json_object(context, input_ref, "inputs.json")
    isolation = input_payload.get("dataset_isolation")
    distinct_train_test_sources = (
        isinstance(isolation, Mapping)
        and isolation.get("passed") is True
        and isolation.get("train_test_distinct_hashes") is True
        and isolation.get("train_test_distinct_paths") is True
    )
    geometry = replay_toy_geometry(
        context.config.protocol.root_seed,
        rms_epsilon=context.config.experiment1.state_metric.rms_epsilon,
    )
    resnet, warnings = _run_resnet_mechanics(context)
    result: dict[str, JSONLike] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol": context.config.experiment1.mechanical_protocol.model_dump(mode="json"),
        "geometry": geometry,
        "probe_banks": {
            "artifact_valid": True,
            "deterministic": _probe_determinism_check(context, input_payload, probe_ref),
            "distinct_train_test_sources": distinct_train_test_sources,
        },
        "resnet": resnet,
        "warnings": warnings,
    }
    return context.store.put_json(
        result,
        filename="mechanical.json",
        kind="stage/exp1-mechanical",
        metadata=_stage_metadata(
            context,
            dependencies,
            evidence_class="mechanical_validation",
            warning_count=len(warnings),
        ),
    )


def _mechanical_rule(context: StageContext) -> GateRule:
    return expected_gate_rule(context.stage_spec, context.config)


def handle_gate_mechanical(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    try:
        reference = dependencies["exp1.mechanical"]
    except KeyError as error:
        raise StageHandlerError("gate.mechanical requires exp1.mechanical") from error
    observations = _json_object(context, reference, "mechanical.json")
    authorization = context.config.gates.overrides.mechanical
    rule = _mechanical_rule(context)
    result = _gate_with_optional_override(
        rule,
        observations,
        None if authorization is None else authorization.reason,
    )
    return context.store.put_json(
        result.to_dict(),
        filename="gate.json",
        kind="gate/mechanical",
        metadata=_stage_metadata(
            context,
            dependencies,
            gate_id=result.gate_id,
            gate_status=result.status.value,
            gate_rule_signature=canonical_hash(rule.to_dict()),
            natural_status=result.natural_status.value,
            override_authorization=(
                None if authorization is None else authorization.model_dump(mode="json")
            ),
        ),
    )


def handle_report_build(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    """Publish the interpretation-first MOCKUP as an immutable report artifact."""

    if dependencies:
        raise StageHandlerError("report.build currently consumes no dependency artifacts")
    if not context.config.report.self_contained:
        raise StageHandlerError("report.build supports only self-contained reports")
    if not context.config.report.embed_spec:
        raise StageHandlerError("report.build requires the experiment specification to be embedded")
    readme_path = context.project_root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    payload = make_mock_payload()
    document = render_report(payload, readme)
    return context.store.put_files(
        {
            "report.html": document.encode("utf-8"),
            "report_payload.json": canonical_json_bytes(payload.model_dump(mode="json")),
            "spec.md": readme.encode("utf-8"),
        },
        kind="report/mockup",
        metadata=_stage_metadata(
            context,
            dependencies,
            evidence_mode="mockup",
            self_contained=True,
        ),
        media_types={
            "report.html": "text/html; charset=utf-8",
            "report_payload.json": "application/json",
            "spec.md": "text/markdown; charset=utf-8",
        },
    )


def default_handlers() -> dict[str, StageHandler]:
    """Return a fresh mapping for every stage declared runnable in the default DAG."""

    return {
        "inputs.tracking2": handle_inputs_tracking2,
        "inputs.tracking2_vgg": handle_inputs_tracking2_vgg,
        "exp1.synthetic_task": handle_exp1_synthetic_task,
        "exp1.plateau.synthetic": handle_exp1_plateau_synthetic,
        "exp1.plateau.cifar": handle_exp1_plateau_cifar,
        "exp1.plateau.animations": handle_exp1_plateau_animations,
        "exp1.probe_banks": handle_exp1_probe_banks,
        "exp1.mechanical": handle_exp1_mechanical,
        "gate.mechanical": handle_gate_mechanical,
        "report.build": handle_report_build,
    }


# ExperimentRunner recognizes this generic callable attribute and invokes it
# before considering completed-stage or cross-run cache reuse.
handle_inputs_tracking2.__dict__["__voronoi_stage_preflight__"] = _preflight_inputs_tracking2
handle_inputs_tracking2_vgg.__dict__["__voronoi_stage_preflight__"] = (
    _preflight_inputs_tracking2_vgg
)


__all__ = [
    "StageHandlerError",
    "default_handlers",
    "handle_exp1_mechanical",
    "handle_exp1_plateau_animations",
    "handle_exp1_plateau_cifar",
    "handle_exp1_plateau_synthetic",
    "handle_exp1_probe_banks",
    "handle_exp1_synthetic_task",
    "handle_gate_mechanical",
    "handle_inputs_tracking2",
    "handle_inputs_tracking2_vgg",
    "handle_report_build",
    "tracking2_adapter_from_config",
    "tracking2_vgg_adapter_from_config",
]
