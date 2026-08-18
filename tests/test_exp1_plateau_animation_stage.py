from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageSequence

from voronoi_lab import stage_handlers
from voronoi_lab.config import LabConfig
from voronoi_lab.core import (
    ArtifactRef,
    ArtifactStore,
    RunIndex,
    canonical_hash,
    canonical_json_bytes,
)
from voronoi_lab.execution import StageContext
from voronoi_lab.pipeline import DEFAULT_STAGES, PipelineError, validate_stage_output


def _config() -> LabConfig:
    payload = LabConfig().model_dump(mode="json")
    payload["experiment1"]["checkpoints"] = [0, 1, 3]
    payload["experiment1"]["synthetic_plateau_task"].update(
        {
            "train_samples_per_class": 8,
            "test_samples_per_class": 4,
            "epochs": 3,
        }
    )
    payload["experiment1"]["plateau_protocol"].update(
        {
            "covariance_fit_images": 6,
            "centers_per_kind": 3,
            "perturbation_directions_per_center": 1,
            "perturbation_steps": 3,
            "local_surface_grid_points": 3,
            "three_anchor_grid_points": 3,
            "hutchinson_probes": 1,
        }
    )
    return LabConfig.model_validate(payload)


def _context(tmp_path: Path, config: LabConfig) -> StageContext:
    store = ArtifactStore(tmp_path / "artifacts")
    index = RunIndex(tmp_path / "runs.sqlite")
    source_identity = {"git_commit": "fixture", "workspace_sha256": "fixture"}
    config_hash = canonical_hash(config.model_dump(mode="json"))
    provenance = store.put_json(
        {"schema_version": 1, "fixture": True},
        filename="provenance.json",
        kind="fixture/provenance",
    )
    index.register_run(
        "animation-fixture",
        config_hash=config_hash,
        provenance_artifact_id=provenance.artifact_id,
        metadata={"source_identity": source_identity},
    )
    return StageContext(
        config=config,
        store=store,
        index=index,
        run_id="animation-fixture",
        project_root=tmp_path,
        source_identity=source_identity,
        stage_spec=DEFAULT_STAGES.get("exp1.plateau.animations"),
    )


def _npz_bytes(
    epoch: int,
    *,
    architecture: str,
    include_residual_fields: bool = True,
) -> bytes:
    local_axis = np.linspace(-0.75, 0.75, 3, dtype=np.float32)
    path_axis = np.linspace(0.0, 1.0, 3, dtype=np.float32)
    displayed_local_jacobian = np.empty((2, 3, 3, 3), dtype=np.float32)
    path_response = np.empty((2, 3, 1, 3), dtype=np.float32)
    architecture_scale = 1.0 if architecture in {"synthetic", "resnet"} else 1.7
    for kind in range(2):
        displayed_local_jacobian[kind] = architecture_scale * (
            0.2 + 0.1 * epoch + 0.3 * kind + np.arange(9, dtype=np.float32).reshape(3, 3)
        )
        path_response[kind] = architecture_scale * (
            (0.1 + 0.2 * kind + 0.05 * epoch) * path_axis[None, None, :]
        )
    alpha, beta = np.meshgrid(
        np.linspace(-0.25, 1.25, 3, dtype=np.float32),
        np.linspace(-0.25, 1.25, 3, dtype=np.float32),
        indexing="xy",
    )
    channel_scale = np.array(
        [1.0, 2.0, 3.0] if architecture == "resnet" else [2.0, 3.0, 5.0],
        dtype=np.float32,
    )
    distances = (
        np.stack((alpha + 0.3, beta + 0.4, alpha + beta + 0.6), axis=-1)
        * channel_scale
        * (1.0 + 0.1 * epoch)
    )
    jacobian = np.stack(
        [
            architecture_scale * (1.0 + context + 0.1 * epoch + alpha**2 + beta**2)
            for context in range(3)
        ],
        axis=0,
    ).astype(np.float32)
    residual_architecture = architecture in {"synthetic", "resnet"}
    arrays = {
        "local_axis": local_axis,
        "local_transition_plane_jacobian": (
            displayed_local_jacobian + 100.0 if residual_architecture else displayed_local_jacobian
        ),
        "path_coefficients": path_axis,
        "path_response_l2": path_response,
        "anchor_axis": np.linspace(-0.25, 1.25, 3, dtype=np.float32),
        "anchor_coordinates": np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.5 if architecture != "vgg" else 0.6, 1.0]],
            dtype=np.float32,
        ),
        "anchor_output_distances": distances.astype(np.float32),
        "anchor_transition_plane_jacobian_by_context": (
            jacobian + 100.0 if residual_architecture else jacobian
        ),
    }
    if residual_architecture and include_residual_fields:
        arrays["local_residual_update_plane_jacobian"] = displayed_local_jacobian
        arrays["anchor_residual_update_plane_jacobian_by_context"] = jacobian
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _dependencies(
    context: StageContext,
    *,
    omit_synthetic_residual_fields: bool = False,
) -> dict[str, ArtifactRef]:
    checkpoints = tuple(context.config.experiment1.checkpoints)
    synthetic_rows = []
    synthetic_files: dict[str, bytes] = {}
    for epoch in checkpoints:
        prefix = f"checkpoints/epoch_{epoch:05d}"
        synthetic_files[f"{prefix}/arrays.npz"] = _npz_bytes(
            epoch,
            architecture="synthetic",
            include_residual_fields=not omit_synthetic_residual_fields,
        )
        synthetic_rows.append(
            {
                "epoch": epoch,
                "arrays": f"{prefix}/arrays.npz",
                "metadata": f"{prefix}/metadata.json",
                "inventory": f"{prefix}/inventory.json",
            }
        )
    synthetic_files["summary.json"] = canonical_json_bytes(
        {
            "schema_version": 2,
            "checkpoint_epochs": list(checkpoints),
            "checkpoint_rows": synthetic_rows,
            "jacobian_display_contract": {
                "protocol_version": 2,
                "local_source_array": "local_residual_update_plane_jacobian",
                "anchor_source_array": "anchor_residual_update_plane_jacobian_by_context",
                "estimand": "2D-plane-restricted ||D(T-I)||_F",
            },
        }
    )
    synthetic = context.store.put_files(
        synthetic_files,
        kind="stage/exp1-plateau-synthetic",
        media_types={
            path: "application/json" if path.endswith(".json") else "application/x-npz"
            for path in synthetic_files
        },
    )

    cifar_rows = []
    for architecture in ("resnet", "vgg"):
        for epoch in checkpoints:
            child = context.store.put_bytes(
                _npz_bytes(epoch, architecture=architecture),
                filename="arrays.npz",
                kind="shard/exp1-plateau-cifar-checkpoint",
                media_type="application/x-npz",
            )
            cifar_rows.append(
                {
                    "architecture": architecture,
                    "epoch": epoch,
                    "artifact_id": child.artifact_id,
                    "arrays": "arrays.npz",
                    "metadata": "metadata.json",
                    "inventory": "inventory.json",
                }
            )
    cifar = context.store.put_json(
        {
            "schema_version": 2,
            "checkpoint_epochs": list(checkpoints),
            "checkpoint_rows": cifar_rows,
            "jacobian_display_contract": {
                "protocol_version": 2,
                "resnet": {
                    "local_source_array": "local_residual_update_plane_jacobian",
                    "anchor_source_array": ("anchor_residual_update_plane_jacobian_by_context"),
                    "estimand": "2D-plane-restricted ||D(T-I)||_F",
                },
                "vgg": {
                    "local_source_array": "local_transition_plane_jacobian",
                    "anchor_source_array": "anchor_transition_plane_jacobian_by_context",
                    "estimand": "2D-plane-restricted ||DT||_F",
                },
                "comparison_scope": (
                    "descriptive_confounded_side_by_side_with_row_specific_operators"
                ),
            },
        },
        filename="summary.json",
        kind="stage/exp1-plateau-cifar",
    )
    return {
        "exp1.plateau.synthetic": synthetic,
        "exp1.plateau.cifar": cifar,
    }


def test_animation_handler_publishes_four_verified_source_bound_bundles(tmp_path: Path) -> None:
    context = _context(tmp_path, _config())
    dependencies = _dependencies(context)
    reference = stage_handlers.handle_exp1_plateau_animations(context, dependencies)

    validate_stage_output(
        reference,
        context.stage_spec,
        context.store,
        config=context.config,
    )
    summary = context.store.read_json(reference.artifact_id, "summary.json")
    assert [row["name"] for row in summary["bundles"]] == [
        "synthetic_real_fake",
        "cifar_resnet_real_fake",
        "cifar_vgg_real_fake",
        "cifar_architecture_cells",
    ]
    assert summary["source_artifacts"] == {
        name: dependency.artifact_id for name, dependency in dependencies.items()
    }
    assert len(reference.manifest.files) == 13
    assert summary["schema_version"] == 2
    assert summary["bundles"][0]["scalar_source_array"] == ("local_residual_update_plane_jacobian")
    assert summary["bundles"][1]["scalar_estimand"] == ("2D-plane-restricted ||D(T-I)||_F")
    assert summary["bundles"][2]["scalar_source_array"] == ("local_transition_plane_jacobian")
    assert summary["bundles"][0]["architectures"] == ["normalization_free_residual_mlp"]
    assert summary["bundles"][1]["architectures"] == ["tracking2_resnet18_v2_width64"]
    assert summary["bundles"][2]["architectures"] == ["tracking2_vgg19_bn_width1_classifier512"]
    assert summary["bundles"][3]["jacobian_source_arrays"] == {
        "resnet": "anchor_residual_update_plane_jacobian_by_context",
        "vgg": "anchor_transition_plane_jacobian_by_context",
    }
    synthetic_metadata = context.store.read_json(
        reference.artifact_id,
        "animations/synthetic_real_fake_metadata.json",
    )
    assert synthetic_metadata["labels"]["scalar"] == "2D ‖D(T-I)‖F"
    assert synthetic_metadata["labels"]["heatmap_classification"] == "NEW HYBRID"
    assert synthetic_metadata["estimands"]["heatmap"] == ("2D-plane-restricted ||D(T-I)||_F")
    assert synthetic_metadata["scales"]["heatmap"]["data_max"] < 100.0
    architecture_metadata = context.store.read_json(
        reference.artifact_id,
        "animations/cifar_architecture_cells_metadata.json",
    )
    assert architecture_metadata["labels"]["jacobian"] == "2D ‖·‖F"
    assert architecture_metadata["labels"]["rgb"] == ("SOURCE ANALOGUE · THREE-ANCHOR RGB")
    assert architecture_metadata["labels"]["resnet_jacobian"] == "NEW HYBRID · D(T-I)"
    assert architecture_metadata["labels"]["vgg_jacobian"] == "NEW HYBRID · DT"
    assert architecture_metadata["scales"]["jacobian"]["data_max"] < 100.0
    with Image.open(
        io.BytesIO(
            context.store.read_bytes(
                reference.artifact_id,
                "animations/cifar_architecture_cells.gif",
            )
        )
    ) as gif:
        assert gif.n_frames == 4
        assert [int(frame.info["duration"]) for frame in ImageSequence.Iterator(gif)] == [
            3000,
            1000,
            1000,
            6000,
        ]

    repeated = stage_handlers.handle_exp1_plateau_animations(context, dependencies)
    assert repeated.artifact_id == reference.artifact_id


def test_animation_validator_recomputes_rgb_maxima_from_raw_child_arrays(tmp_path: Path) -> None:
    context = _context(tmp_path, _config())
    reference = stage_handlers.handle_exp1_plateau_animations(context, _dependencies(context))
    files = {
        entry.path: context.store.read_bytes(reference.artifact_id, entry.path)
        for entry in reference.manifest.files
    }
    summary = json.loads(files["summary.json"])
    summary["normalization"]["resnet_channel_maxima"][0] += 1.0
    files["summary.json"] = canonical_json_bytes(summary)
    media_types = {entry.path: entry.media_type for entry in reference.manifest.files}
    corrupted = context.store.put_files(
        files,
        kind=reference.manifest.kind,
        metadata=reference.manifest.metadata,
        media_types={name: value for name, value in media_types.items() if value is not None},
    )

    with pytest.raises(PipelineError, match="normalization is not derived"):
        validate_stage_output(
            corrupted,
            context.stage_spec,
            context.store,
            config=context.config,
        )


def test_animation_handler_rejects_residual_source_without_adjusted_plane_fields(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, _config())
    dependencies = _dependencies(context, omit_synthetic_residual_fields=True)

    with pytest.raises(
        stage_handlers.StageHandlerError,
        match="local_residual_update_plane_jacobian",
    ):
        stage_handlers.handle_exp1_plateau_animations(context, dependencies)


def test_animation_validator_rejects_wrong_declared_jacobian_source_array(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, _config())
    reference = stage_handlers.handle_exp1_plateau_animations(context, _dependencies(context))
    files = {
        entry.path: context.store.read_bytes(reference.artifact_id, entry.path)
        for entry in reference.manifest.files
    }
    summary = json.loads(files["summary.json"])
    summary["bundles"][0]["scalar_source_array"] = "local_transition_plane_jacobian"
    files["summary.json"] = canonical_json_bytes(summary)
    media_types = {entry.path: entry.media_type for entry in reference.manifest.files}
    corrupted = context.store.put_files(
        files,
        kind=reference.manifest.kind,
        metadata=reference.manifest.metadata,
        media_types={name: value for name, value in media_types.items() if value is not None},
    )

    with pytest.raises(PipelineError, match="source-field contract"):
        validate_stage_output(
            corrupted,
            context.stage_spec,
            context.store,
            config=context.config,
        )


def test_animation_validator_recomputes_curve_scale_extrema_from_source_arrays(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, _config())
    reference = stage_handlers.handle_exp1_plateau_animations(context, _dependencies(context))
    files = {
        entry.path: context.store.read_bytes(reference.artifact_id, entry.path)
        for entry in reference.manifest.files
    }
    metadata_path = "animations/synthetic_real_fake_metadata.json"
    metadata = json.loads(files[metadata_path])
    metadata["scales"]["curve_y"]["data_max"] += 1.0
    files[metadata_path] = canonical_json_bytes(metadata)
    media_types = {entry.path: entry.media_type for entry in reference.manifest.files}
    corrupted = context.store.put_files(
        files,
        kind=reference.manifest.kind,
        metadata=reference.manifest.metadata,
        media_types={name: value for name, value in media_types.items() if value is not None},
    )

    with pytest.raises(PipelineError, match="scales or heatmap coordinates"):
        validate_stage_output(
            corrupted,
            context.stage_spec,
            context.store,
            config=context.config,
        )


def test_animation_validator_replays_display_ranges_and_coordinate_metadata(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, _config())
    reference = stage_handlers.handle_exp1_plateau_animations(context, _dependencies(context))
    original_files = {
        entry.path: context.store.read_bytes(reference.artifact_id, entry.path)
        for entry in reference.manifest.files
    }
    media_types = {entry.path: entry.media_type for entry in reference.manifest.files}
    mutations = (
        (
            "animations/synthetic_real_fake_metadata.json",
            ("scales", "heatmap", "display_min"),
            "scales or heatmap coordinates",
        ),
        (
            "animations/synthetic_real_fake_metadata.json",
            ("scales", "heatmap_extent", 0),
            "scales or heatmap coordinates",
        ),
        (
            "animations/cifar_architecture_cells_metadata.json",
            ("scales", "jacobian", "display_max"),
            "scales or plane coordinates",
        ),
        (
            "animations/cifar_architecture_cells_metadata.json",
            ("scales", "alpha_coordinates", 0),
            "scales or plane coordinates",
        ),
    )
    for metadata_path, key_path, error_match in mutations:
        files = dict(original_files)
        metadata = json.loads(files[metadata_path])
        target = metadata
        for key in key_path[:-1]:
            target = target[key]
        final_key = key_path[-1]
        target[final_key] += 1.0
        files[metadata_path] = canonical_json_bytes(metadata)
        corrupted = context.store.put_files(
            files,
            kind=reference.manifest.kind,
            metadata=reference.manifest.metadata,
            media_types={name: value for name, value in media_types.items() if value is not None},
        )
        with pytest.raises(PipelineError, match=error_match):
            validate_stage_output(
                corrupted,
                context.stage_spec,
                context.store,
                config=context.config,
            )


def test_animation_validator_rejects_tampered_bundle_architecture_identity(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, _config())
    reference = stage_handlers.handle_exp1_plateau_animations(context, _dependencies(context))
    files = {
        entry.path: context.store.read_bytes(reference.artifact_id, entry.path)
        for entry in reference.manifest.files
    }
    summary = json.loads(files["summary.json"])
    summary["bundles"][1]["architectures"] = ["vgg19_bn"]
    files["summary.json"] = canonical_json_bytes(summary)
    media_types = {entry.path: entry.media_type for entry in reference.manifest.files}
    corrupted = context.store.put_files(
        files,
        kind=reference.manifest.kind,
        metadata=reference.manifest.metadata,
        media_types={name: value for name, value in media_types.items() if value is not None},
    )

    with pytest.raises(PipelineError, match="task or architecture identity"):
        validate_stage_output(
            corrupted,
            context.stage_spec,
            context.store,
            config=context.config,
        )
