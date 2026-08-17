from __future__ import annotations

import json
from typing import Literal

import numpy as np
import pytest

from voronoi_lab.exp1.analysis import (
    BootstrapSpec,
    BoundaryPathSpec,
    GeometryStatisticRecord,
    InterventionMetricObservation,
    StaticAggregationSpec,
    deterministic_image_bootstrap_interval,
    distortion_from_image_features,
    join_geometry_with_legacy_transplants,
    make_boundary_shift_plan,
    static_distortion_features,
    summarize_boundary_paths,
    summarize_snapping_recovery_controls,
    summarize_static_geometry,
    weighted_mean,
)
from voronoi_lab.exp1.geometry import Codebook, CoordinateBatch
from voronoi_lab.exp1.tracking2 import TransplantRow


def _static_spec() -> StaticAggregationSpec:
    return StaticAggregationSpec(
        method_version="static_geometry_v1",
        evaluation_bank_id="geometry-sites-fixture",
        image_weighting="equal_image_weight_v1",
        margin_scale=2.0,
        margin_scale_id="fixture_scale",
        margin_reducer_id="weighted_mean_v1",
        assignment_stability_id="aligned_exact_agreement_v1",
    )


def test_static_summary_uses_equal_image_mass_and_preserves_image_rows() -> None:
    batch = CoordinateBatch(np.asarray([[0.0], [0.0], [8.0]]), "raw")
    codebook = Codebook(np.asarray([[0.0], [10.0]]), "raw", "fit-bank")

    summary = summarize_static_geometry(
        batch,
        codebook,
        [10, 10, 20],
        site_weights=[1.0, 1.0, 7.0],
        assignment_stability_scores=[1.0, 0.0, 0.5],
        spec=_static_spec(),
        margin_reducer=weighted_mean,
    )

    assert summary.normalized_distortion == pytest.approx(0.125)
    assert summary.normalized_margin == pytest.approx(2.0)
    assert summary.effective_occupied_cells == pytest.approx(2.0)
    assert summary.assignment_stability == pytest.approx(0.5)
    assert summary.images[0].occupancy_mass == pytest.approx((1.0, 0.0))
    assert summary.images[1].occupancy_mass == pytest.approx((0.0, 1.0))
    assert summary.images[0].site_count == 2
    assert summary.images[1].site_count == 1
    features = static_distortion_features(summary)
    assert distortion_from_image_features(features) == pytest.approx(summary.normalized_distortion)
    json.dumps(summary.to_artifact(), allow_nan=False)


@pytest.mark.parametrize(
    ("stability", "weights", "match"),
    [
        ([1.0, np.nan, 0.0], [1.0, 1.0, 1.0], "finite"),
        ([1.0, 0.0], [1.0, 1.0, 1.0], "length"),
        ([1.0, 0.0, 0.0], [1.0, 0.0, 1.0], "positive"),
    ],
)
def test_static_summary_rejects_invalid_inputs(stability, weights, match: str) -> None:
    batch = CoordinateBatch(np.asarray([[0.0], [0.0], [8.0]]), "raw")
    codebook = Codebook(np.asarray([[0.0], [10.0]]), "raw", "fit-bank")
    with pytest.raises(ValueError, match=match):
        summarize_static_geometry(
            batch,
            codebook,
            [10, 10, 20],
            site_weights=weights,
            assignment_stability_scores=stability,
            spec=_static_spec(),
            margin_reducer=weighted_mean,
        )


def test_image_bootstrap_is_deterministic_and_accepts_ratio_statistics() -> None:
    spec = BootstrapSpec(
        method_version="percentile_image_bootstrap_v1",
        root_seed=17,
        namespace="static/distortion",
        input_artifact_id="static-summary-fixture",
        resamples=128,
        confidence_level=0.95,
        quantile_method="linear",
        statistic_id="ratio_of_feature_means_v1",
    )
    features = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])

    def statistic(rows: np.ndarray) -> float:
        return float(rows[:, 1].mean() / rows[:, 0].mean())

    first = deterministic_image_bootstrap_interval(
        [10, 20, 30], features, statistic=statistic, spec=spec
    )
    second = deterministic_image_bootstrap_interval(
        [10, 20, 30], features, statistic=statistic, spec=spec
    )

    assert first == second
    assert first.point_estimate == pytest.approx(7 / 3)
    assert first.lower <= first.point_estimate <= first.upper
    assert len(first.replicate_values) == 128
    json.dumps(first.to_artifact(), allow_nan=False)


def test_image_bootstrap_rejects_duplicate_images_and_nonfinite_statistics() -> None:
    spec = BootstrapSpec(
        method_version="v1",
        root_seed=0,
        namespace="x",
        input_artifact_id="fixture",
        resamples=4,
        confidence_level=0.9,
        quantile_method="linear",
        statistic_id="mean_v1",
    )
    with pytest.raises(ValueError, match="one row per image"):
        deterministic_image_bootstrap_interval([1, 1], [[1.0], [2.0]], statistic=np.mean, spec=spec)
    with pytest.raises(ValueError, match="nonfinite"):
        deterministic_image_bootstrap_interval(
            [1, 2], [[1.0], [2.0]], statistic=lambda _rows: np.nan, spec=spec
        )


def _boundary_spec(
    energy_weighting: Literal[
        "discrete_grid_mass", "continuous_path_trapezoid"
    ] = "discrete_grid_mass",
) -> BoundaryPathSpec:
    return BoundaryPathSpec(
        method_version="boundary_paths_v1",
        path_bank_id="paths-fixture",
        shift_plan_id="shift-plan-fixture",
        direction_family="empirical_chord",
        boundary_window=(0.9, 1.1),
        energy_weighting=energy_weighting,
        path_weighting="weighted_within_image_equal_images_v1",
        null_method="within_path_circular_energy_shift_v1",
        comparison_statistic="fraction_near_boundary",
    )


def test_boundary_summary_compares_image_weighted_response_to_shift_null() -> None:
    coordinates = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0])
    energy = np.tile([1.0, 1.0, 10.0, 1.0, 1.0], (3, 1))
    shift_plan = make_boundary_shift_plan(
        path_count=3,
        path_length=5,
        draws=12,
        allowed_shifts=[1],
        root_seed=4,
        namespace="fixture",
    )

    summary = summarize_boundary_paths(
        coordinates,
        energy,
        [10, 10, 20],
        path_weights=[1.0, 3.0, 2.0],
        shift_plan=shift_plan,
        spec=_boundary_spec(),
    )

    assert summary.image_count == 2
    assert summary.path_count == 3
    assert summary.shifted_null.observed == pytest.approx(10 / 14)
    assert summary.shifted_null.null_mean == pytest.approx(1 / 14)
    assert summary.shifted_null.observed_minus_null_mean == pytest.approx(9 / 14)
    assert summary.shifted_null.empirical_percentile == pytest.approx(1.0)
    assert summary.images[0].path_count == 2
    json.dumps(summary.to_artifact(), allow_nan=False)


def test_boundary_summary_records_continuous_path_weighting() -> None:
    summary = summarize_boundary_paths(
        [0.0, 0.5, 1.0, 1.5, 2.0],
        [[1.0, 1.0, 10.0, 1.0, 1.0]],
        [10],
        path_weights=[1.0],
        shift_plan=[[1]],
        spec=_boundary_spec("continuous_path_trapezoid"),
    )

    assert summary.spec.energy_weighting == "continuous_path_trapezoid"
    assert summary.shifted_null.observed == pytest.approx(0.28)


def test_boundary_shift_plan_and_summary_reject_identity_or_nonfinite_paths() -> None:
    with pytest.raises(ValueError, match="identity"):
        make_boundary_shift_plan(
            path_count=1,
            path_length=5,
            draws=2,
            allowed_shifts=[0],
            root_seed=0,
            namespace="x",
        )
    with pytest.raises(ValueError, match="energy"):
        summarize_boundary_paths(
            [0.0, 1.0],
            [[1.0, np.nan]],
            [1],
            path_weights=[1.0],
            shift_plan=[[1]],
            spec=_boundary_spec(),
        )


def _intervention_rows() -> list[InterventionMetricObservation]:
    values = {
        "predictive_kl": {
            "snap": [0.1, 0.2],
            "random": [0.5, 0.6],
            "away": [0.4, 0.5],
        },
        "clean_cell_recovery": {
            "snap": [0.9, 0.8],
            "random": [0.6, 0.5],
            "away": [0.5, 0.4],
        },
    }
    rows = []
    for metric, arms in values.items():
        for arm, observations in arms.items():
            for image_id, value in zip((10, 20), observations, strict=True):
                rows.append(InterventionMetricObservation(image_id, arm, metric, value))
    return rows


def test_snapping_recovery_summary_keeps_raw_and_directional_paired_contrasts() -> None:
    summary = summarize_snapping_recovery_controls(
        _intervention_rows(),
        target_arm="snap",
        control_arms=("random", "away"),
        metric_directions={"predictive_kl": "lower", "clean_cell_recovery": "higher"},
        protocol_id="hard_snap",
        stratum_id="cut1/alpha1/all_tokens",
        method_version="paired_equal_image_v1",
    )
    by_key = {(row.metric, row.control_arm): row for row in summary.contrasts}

    kl = by_key[("predictive_kl", "random")]
    assert kl.target_mean == pytest.approx(0.15)
    assert kl.control_mean == pytest.approx(0.55)
    assert kl.target_minus_control == pytest.approx(-0.4)
    assert kl.favorable_advantage == pytest.approx(0.4)

    recovery = by_key[("clean_cell_recovery", "random")]
    assert recovery.target_minus_control == pytest.approx(0.3)
    assert recovery.favorable_advantage == pytest.approx(0.3)
    json.dumps(summary.to_artifact(), allow_nan=False)


def test_snapping_recovery_summary_requires_complete_pairing_and_finite_values() -> None:
    rows = _intervention_rows()
    with pytest.raises(ValueError, match="complete paired"):
        summarize_snapping_recovery_controls(
            rows[:-1],
            target_arm="snap",
            control_arms=("random", "away"),
            metric_directions={"predictive_kl": "lower", "clean_cell_recovery": "higher"},
            protocol_id="hard_snap",
            stratum_id="cut1",
            method_version="v1",
        )
    with pytest.raises(ValueError, match="finite"):
        InterventionMetricObservation(1, "snap", "kappa", np.inf)


def _transplant(source_epoch: int, delta_loss: float) -> TransplantRow:
    return TransplantRow(
        seed=0,
        target_epoch=100,
        cut_index=0,
        cut_name="stage1.block1",
        tracking2_module_name="stage1.resblk1",
        transplant_module_index=1,
        source_kind="checkpoint",
        source_epoch=source_epoch,
        loss=1.0 + delta_loss,
        accuracy=0.8,
        error=0.2,
        delta_loss=delta_loss,
        delta_error=0.0,
    )


def _geometry(epoch: int, value: float) -> GeometryStatisticRecord:
    return GeometryStatisticRecord(
        seed=0,
        checkpoint_epoch=epoch,
        cut_index=0,
        cut_name="stage1.block1",
        statistic="normalized_distortion",
        value=value,
        coordinate_metric="standardized",
        k=32,
        analysis_version="static_v1",
        artifact_id=f"geometry-epoch{epoch}",
    )


def test_geometry_transplant_join_matches_source_epochs_and_labels_lineage() -> None:
    joined = join_geometry_with_legacy_transplants(
        [_geometry(1, 0.8), _geometry(5, 0.4)],
        [_transplant(1, 0.2), _transplant(5, 0.7)],
        transplant_statistic="delta_loss",
        source_policy="checkpoint_only",
        unmatched_policy="error",
        join_version="source_epoch_cut_seed_v1",
    )

    assert [(row.source_epoch, row.geometry_value, row.transplant_value) for row in joined] == [
        (1, 0.8, 0.2),
        (5, 0.4, 0.7),
    ]
    assert all(row.interpretation == "descriptive_only" for row in joined)
    assert all(row.legacy_lineage == "exploratory_legacy" for row in joined)
    json.dumps([row.to_artifact() for row in joined], allow_nan=False)


def test_geometry_transplant_join_has_explicit_unmatched_policy() -> None:
    geometry = [_geometry(1, 0.8), _geometry(5, 0.4)]
    transplants = [_transplant(1, 0.2)]
    with pytest.raises(ValueError, match="unmatched"):
        join_geometry_with_legacy_transplants(
            geometry,
            transplants,
            transplant_statistic="delta_loss",
            source_policy="checkpoint_only",
            unmatched_policy="error",
            join_version="v1",
        )
    dropped = join_geometry_with_legacy_transplants(
        geometry,
        transplants,
        transplant_statistic="delta_loss",
        source_policy="checkpoint_only",
        unmatched_policy="drop",
        join_version="v1",
    )
    assert len(dropped) == 1
