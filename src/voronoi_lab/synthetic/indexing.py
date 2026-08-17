"""Product indexing, lifting, and observed/latent conjugation utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .schema import FactorSpace, FloatArray, IntArray, Labeling, Support


def coordinate_to_flat(coordinate: Sequence[int], space: FactorSpace) -> int:
    """Flatten one product coordinate in C order (last factor fastest)."""

    values = tuple(int(value) for value in coordinate)
    if len(values) != space.n_factors:
        raise ValueError("coordinate length does not match the factor space")
    if any(value < 0 or value >= size for value, size in zip(values, space.sizes, strict=True)):
        raise ValueError(f"coordinate {values} is outside factor sizes {space.sizes}")
    return int(np.ravel_multi_index(values, space.sizes, order="C"))


def flat_to_coordinate(index: int, space: FactorSpace) -> tuple[int, ...]:
    """Invert :func:`coordinate_to_flat`."""

    if index < 0 or index >= space.n_states:
        raise ValueError(f"flat index {index} is outside [0, {space.n_states})")
    return tuple(int(value) for value in np.unravel_index(index, space.sizes, order="C"))


def labeling_coordinates(labeling: Labeling) -> IntArray:
    """Return an ``(observed_state, factor)`` coordinate table."""

    unraveled = np.unravel_index(labeling.obs_to_grid, labeling.space.sizes, order="C")
    return np.stack(unraveled, axis=-1).astype(np.int64, copy=False)


def permutation_matrix(labeling: Labeling) -> FloatArray:
    """Return Pi with ``Pi[grid, observed] = 1``.

    Thus ``p_grid = Pi @ p_observed`` and
    ``L_observed = Pi.T @ L_grid @ Pi``.
    """

    matrix = np.zeros((labeling.space.n_states, labeling.space.n_states), dtype=np.float64)
    matrix[labeling.obs_to_grid, np.arange(labeling.space.n_states)] = 1.0
    return matrix


def _take_square_axes(operator: ArrayLike, indices: IntArray) -> FloatArray:
    values = np.asarray(operator, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2] != values.shape[-1]:
        raise ValueError("operator arrays must end in equal destination/source axes")
    if values.shape[-1] != indices.size:
        raise ValueError("operator size and permutation size do not match")
    return np.take(np.take(values, indices, axis=-2), indices, axis=-1)


def to_grid(operator_observed: ArrayLike, labeling: Labeling) -> FloatArray:
    """Conjugate observed-coordinate operators into the proposed product grid."""

    return _take_square_axes(operator_observed, labeling.grid_to_obs)


def to_observed(operator_grid: ArrayLike, labeling: Labeling) -> FloatArray:
    """Conjugate product-grid operators into observed coordinates."""

    return _take_square_axes(operator_grid, labeling.obs_to_grid)


def lift_local_operator(
    local_operator: ArrayLike,
    space: FactorSpace,
    support: Support,
) -> FloatArray:
    """Lift a local operator by identities on all complementary factors.

    The explicit enumeration is intentionally simple and convention-safe. The
    benchmark's initial largest state space has only twenty states.
    """

    support = space.validate_support(support)
    local_sizes = tuple(space.sizes[axis] for axis in support)
    local_states = int(np.prod(local_sizes, dtype=np.int64)) if local_sizes else 1
    local = np.asarray(local_operator, dtype=np.float64)
    if local.shape != (local_states, local_states):
        raise ValueError(
            f"local operator for support {support} must have shape "
            f"{(local_states, local_states)}, got {local.shape}"
        )

    lifted = np.zeros((space.n_states, space.n_states), dtype=np.float64)
    for source in range(space.n_states):
        source_coordinate = list(flat_to_coordinate(source, space))
        local_source_coordinate = tuple(source_coordinate[axis] for axis in support)
        local_source = (
            int(np.ravel_multi_index(local_source_coordinate, local_sizes, order="C"))
            if support
            else 0
        )
        for local_destination in range(local_states):
            destination_coordinate = source_coordinate.copy()
            if support:
                replacement = np.unravel_index(local_destination, local_sizes, order="C")
                for axis, value in zip(support, replacement, strict=True):
                    destination_coordinate[axis] = int(value)
            destination = coordinate_to_flat(destination_coordinate, space)
            lifted[destination, source] = local[local_destination, local_source]
    return lifted
