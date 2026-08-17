from __future__ import annotations

import numpy as np

from voronoi_lab.synthetic import (
    FactorSpace,
    Labeling,
    coordinate_to_flat,
    flat_to_coordinate,
    lift_local_operator,
    permutation_matrix,
    to_grid,
    to_observed,
)


def test_c_order_product_indexing_round_trip() -> None:
    space = FactorSpace((2, 3, 2))
    assert coordinate_to_flat((0, 0, 0), space) == 0
    assert coordinate_to_flat((0, 1, 0), space) == 2
    assert coordinate_to_flat((1, 0, 0), space) == 6
    for index in range(space.n_states):
        assert coordinate_to_flat(flat_to_coordinate(index, space), space) == index


def test_observed_grid_conjugation_matches_permutation_matrix() -> None:
    space = FactorSpace((2, 3))
    labeling = Labeling(space, np.array([4, 0, 5, 2, 1, 3]))
    grid = np.arange(36, dtype=np.float64).reshape(6, 6)
    observed = to_observed(grid, labeling)
    permutation = permutation_matrix(labeling)
    np.testing.assert_array_equal(observed, permutation.T @ grid @ permutation)
    np.testing.assert_array_equal(to_grid(observed, labeling), grid)


def test_local_lift_matches_kronecker_convention() -> None:
    space = FactorSpace((2, 3))
    left = np.array([[1.0, 2.0], [3.0, 4.0]])
    right = np.arange(9, dtype=np.float64).reshape(3, 3)
    np.testing.assert_array_equal(lift_local_operator(left, space, (0,)), np.kron(left, np.eye(3)))
    np.testing.assert_array_equal(
        lift_local_operator(right, space, (1,)), np.kron(np.eye(2), right)
    )
    np.testing.assert_array_equal(lift_local_operator(np.eye(6), space, (0, 1)), np.eye(6))


def test_invalid_labeling_and_support_are_rejected() -> None:
    space = FactorSpace((2, 3))
    with np.testing.assert_raises(ValueError):
        Labeling(space, np.array([0, 1, 2, 3, 4, 4]))
    with np.testing.assert_raises(ValueError):
        lift_local_operator(np.eye(2), space, (1, 0))
