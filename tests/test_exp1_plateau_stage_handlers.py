from __future__ import annotations

from pathlib import Path

from voronoi_lab import stage_handlers
from voronoi_lab.config import LabConfig
from voronoi_lab.core import ArtifactStore, RunIndex, canonical_hash
from voronoi_lab.execution import StageContext
from voronoi_lab.pipeline import DEFAULT_STAGES, validate_stage_output


def _small_plateau_config() -> LabConfig:
    payload = LabConfig().model_dump(mode="json")
    payload["experiment1"]["checkpoints"] = [0, 1, 3]
    payload["experiment1"]["synthetic_plateau_task"].update(
        {
            "train_samples_per_class": 16,
            "test_samples_per_class": 8,
            "hidden_width": 8,
            "residual_blocks": 3,
            "intervention_block": 1,
            "epochs": 3,
            "batch_size": 16,
        }
    )
    payload["experiment1"]["plateau_protocol"].update(
        {
            "covariance_fit_images": 12,
            "centers_per_kind": 3,
            "perturbation_directions_per_center": 1,
            "perturbation_steps": 3,
            "local_surface_grid_points": 3,
            "three_anchor_grid_points": 3,
            "hutchinson_probes": 2,
            "intervention_batch_size": 16,
        }
    )
    return LabConfig.model_validate(payload)


def _context(
    tmp_path: Path,
    config: LabConfig,
    stage_name: str,
) -> StageContext:
    store = ArtifactStore(tmp_path / "artifacts")
    index = RunIndex(tmp_path / "runs.sqlite")
    source_identity = {"git_commit": "fixture", "workspace_sha256": "fixture"}
    config_hash = canonical_hash(config.model_dump(mode="json"))
    run_id = f"plateau-fixture-{config_hash[:12]}"
    provenance = store.put_json(
        {"schema_version": 1, "fixture": True},
        filename="provenance.json",
        kind="fixture/provenance",
    )
    index.register_run(
        run_id,
        config_hash=config_hash,
        provenance_artifact_id=provenance.artifact_id,
        metadata={"source_identity": source_identity},
    )
    return StageContext(
        config=config,
        store=store,
        index=index,
        run_id=run_id,
        project_root=tmp_path,
        source_identity=source_identity,
        stage_spec=DEFAULT_STAGES.get(stage_name),
    )


def test_synthetic_task_and_plateau_handlers_publish_valid_replayable_artifacts(
    tmp_path: Path,
) -> None:
    config = _small_plateau_config()
    task_context = _context(tmp_path, config, "exp1.synthetic_task")
    task = stage_handlers.handle_exp1_synthetic_task(task_context, {})
    validate_stage_output(
        task,
        DEFAULT_STAGES.get("exp1.synthetic_task"),
        task_context.store,
        config=config,
    )
    assert task.manifest.metadata["final_test_accuracy"] > 0.9

    plateau_context = _context(tmp_path, config, "exp1.plateau.synthetic")
    plateau = stage_handlers.handle_exp1_plateau_synthetic(
        plateau_context,
        {"exp1.synthetic_task": task},
    )
    validate_stage_output(
        plateau,
        DEFAULT_STAGES.get("exp1.plateau.synthetic"),
        plateau_context.store,
        config=config,
    )
    summary = plateau_context.store.read_json(plateau.artifact_id, "summary.json")
    assert summary["schema_version"] == 2
    assert summary["checkpoint_epochs"] == [0, 1, 3]
    assert len(summary["checkpoint_rows"]) == 3
    assert summary["jacobian_display_contract"] == {
        "protocol_version": 2,
        "local_source_array": "local_residual_update_plane_jacobian",
        "anchor_source_array": "anchor_residual_update_plane_jacobian_by_context",
        "estimand": "2D-plane-restricted ||D(T-I)||_F",
    }

    repeated = stage_handlers.handle_exp1_plateau_synthetic(
        plateau_context,
        {"exp1.synthetic_task": task},
    )
    assert repeated.artifact_id == plateau.artifact_id
