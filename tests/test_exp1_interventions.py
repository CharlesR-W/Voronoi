from __future__ import annotations

import numpy as np
import pytest

from voronoi_lab.exp1.directions import off_cloud_direction
from voronoi_lab.exp1.geometry import CoordinateBatch
from voronoi_lab.exp1.interventions import (
    centered_directional_derivative,
    circular_shift_null,
    finite_recovery_gain,
    insert_site_displacements,
    partial_snap,
    predictive_derivative_energy,
    predictive_kl,
    relative_vector_error,
    summarize_boundary_energy,
)


def test_snapping_holds_clean_assignment_fixed() -> None:
    clean = np.asarray([[0.0], [10.0]])
    centers = np.asarray([[1.0], [9.0]])
    fixed = np.asarray([1, 0])
    snapped = partial_snap(clean, centers, fixed, 1.0)
    assert np.array_equal(snapped[:, 0], [9.0, 1.0])


def test_tensor_perturbation_and_gain_use_complete_support() -> None:
    clean = np.ones((2, 3, 2, 2))
    perturbed = insert_site_displacements(
        clean,
        batch_indices=[0, 1],
        rows=[0, 1],
        columns=[1, 0],
        native_displacements=[[1, 2, 3], [-1, -2, -3]],
    )
    assert np.array_equal(perturbed[0, :, 0, 1], [2, 3, 4])
    assert np.array_equal(perturbed[1, :, 1, 0], [0, -1, -2])
    output_clean = 2 * clean
    output_perturbed = 2 * perturbed
    gain = finite_recovery_gain(clean, perturbed, output_clean, output_perturbed)
    assert gain == pytest.approx(1.0)


def test_tensor_perturbation_accumulates_repeated_sites() -> None:
    result = insert_site_displacements(
        np.zeros((1, 2, 1, 1)),
        batch_indices=[0, 0],
        rows=[0, 0],
        columns=[0, 0],
        native_displacements=[[1.0, 2.0], [3.0, 4.0]],
    )
    assert np.array_equal(result[0, :, 0, 0], [4.0, 6.0])


@pytest.mark.parametrize(
    ("index_name", "indices"),
    [
        ("batch_indices", [-1]),
        ("batch_indices", [2]),
        ("rows", [-1]),
        ("rows", [2]),
        ("columns", [-1]),
        ("columns", [2]),
        ("columns", [0.5]),
    ],
)
def test_tensor_perturbation_rejects_invalid_site_indices(
    index_name: str, indices: list[float]
) -> None:
    kwargs = {"batch_indices": [0], "rows": [0], "columns": [0]}
    kwargs[index_name] = indices
    with pytest.raises(ValueError):
        insert_site_displacements(
            np.zeros((2, 1, 2, 2)),
            **kwargs,
            native_displacements=[[1.0]],
        )


@pytest.mark.parametrize(
    ("activation", "displacement"),
    [
        (np.asarray([[[[np.nan]]]]), [[0.0]]),
        (np.zeros((1, 1, 1, 1)), [[np.inf]]),
    ],
)
def test_tensor_perturbation_rejects_nonfinite_arrays(
    activation: np.ndarray, displacement: list[list[float]]
) -> None:
    with pytest.raises(ValueError, match="finite"):
        insert_site_displacements(
            activation,
            batch_indices=[0],
            rows=[0],
            columns=[0],
            native_displacements=displacement,
        )


def test_centered_difference_validates_linear_jvp() -> None:
    matrix = np.asarray([[1.0, 2.0], [-3.0, 0.5]])
    point = np.asarray([0.4, -0.2])
    direction = np.asarray([0.6, 0.8])
    estimate = centered_directional_derivative(
        lambda x: np.tanh(matrix @ x), point, direction, epsilon=1e-5
    )
    y = matrix @ point
    exact = (1.0 - np.tanh(y) ** 2) * (matrix @ direction)
    assert relative_vector_error(exact, estimate) < 1e-9


def test_boundary_energy_summary_and_shift_null() -> None:
    r = np.asarray([0.5, 0.9, 1.0, 1.1, 1.5])
    energy = np.asarray([1.0, 2.0, 10.0, 2.0, 1.0])
    summary = summarize_boundary_energy(
        r, energy, window=(0.9, 1.1), weighting="discrete_grid_mass"
    )
    assert summary.fraction_near_boundary == pytest.approx(14 / 16)
    assert summary.peak_offset == pytest.approx(0.0)
    assert np.array_equal(circular_shift_null(energy, 2), np.roll(energy, 2))


def test_continuous_boundary_summary_integrates_nonuniform_grid() -> None:
    summary = summarize_boundary_energy(
        [0.0, 1.0, 3.0],
        [0.0, 2.0, 0.0],
        window=(0.5, 2.0),
        weighting="continuous_path_trapezoid",
    )
    assert summary.fraction_near_boundary == pytest.approx(0.75)
    assert summary.energy_80_interval == pytest.approx((np.sqrt(0.3), 3.0 - np.sqrt(0.6)))
    assert summary.peak_offset == pytest.approx(0.0)


def test_continuous_peak_uses_density_not_grid_cell_mass() -> None:
    summary = summarize_boundary_energy(
        [0.0, 0.01, 10.0],
        [0.0, 10.0, 9.0],
        window=(20.0, 30.0),
        weighting="continuous_path_trapezoid",
    )
    assert summary.fraction_near_boundary == 0.0
    assert summary.peak_offset == pytest.approx(-0.99)


def test_boundary_summary_requires_explicit_valid_weighting() -> None:
    with pytest.raises(TypeError):
        summarize_boundary_energy([0.0], [1.0], window=(0.0, 1.0))
    with pytest.raises(ValueError, match="unsupported"):
        summarize_boundary_energy(
            [0.0],
            [1.0],
            window=(0.0, 1.0),
            weighting="implicit",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="at least two"):
        summarize_boundary_energy(
            [0.0], [1.0], window=(0.0, 1.0), weighting="continuous_path_trapezoid"
        )


def test_predictive_derivative_energy_is_zero_for_constant_path() -> None:
    probabilities = np.tile([0.2, 0.8], (4, 1))
    assert np.allclose(predictive_derivative_energy(probabilities, delta_r=0.1), 0.0)


def test_predictive_kl_normalizes_positive_rows_and_stays_finite() -> None:
    result = predictive_kl([[2.0, 0.0]], [[0.0, 3.0]], epsilon=1e-12)
    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(-np.log(1e-12))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([[np.nan, 1.0]], [[0.5, 0.5]]),
        ([[0.5, 0.5]], [[np.inf, 1.0]]),
        ([[0.0, 0.0]], [[0.5, 0.5]]),
        ([[0.5, 0.5]], [[0.0, 0.0]]),
    ],
)
def test_predictive_kl_rejects_nonfinite_or_zero_mass_rows(left, right) -> None:
    with pytest.raises(ValueError):
        predictive_kl(left, right)


@pytest.mark.parametrize("epsilon", [0.0, -1.0, np.nan, np.inf, 1.0, 2.0, 1j, "bad"])
def test_predictive_kl_rejects_invalid_epsilon(epsilon) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        predictive_kl([0.5, 0.5], [0.5, 0.5], epsilon=epsilon)


def test_off_cloud_direction_is_orthogonal_to_local_span() -> None:
    x = np.linspace(-2, 2, 40)
    observed = CoordinateBatch(np.column_stack([x, 2 * x, np.zeros_like(x)]), "standardized")
    direction = off_cloud_direction(
        [0.0, 0.0, 0.0],
        observed,
        neighbors=20,
        pca_rank=1,
        rng=np.random.default_rng(4),
    )
    line = np.asarray([1.0, 2.0, 0.0])
    assert abs(float(direction.vector @ line)) < 1e-10
