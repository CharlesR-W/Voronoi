"""Finite permutation groups and generator-cone-preserving twirls."""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import ArrayLike

from .indexing import coordinate_to_flat, flat_to_coordinate
from .schema import FactorSpace, FloatArray


def cyclic_permutation(size: int, shift: int) -> FloatArray:
    """Permutation matrix for ``i -> (i + shift) mod size``."""

    if size < 1:
        raise ValueError("cyclic action size must be positive")
    matrix = np.zeros((size, size), dtype=np.float64)
    source = np.arange(size)
    matrix[(source + int(shift)) % size, source] = 1.0
    return matrix


def cyclic_group(size: int) -> FloatArray:
    """Enumerate the regular cyclic permutation representation."""

    return np.stack([cyclic_permutation(size, shift) for shift in range(size)])


def cyclic_product_group(space: FactorSpace) -> FloatArray:
    """Enumerate the compatible product of cyclic factor actions."""

    matrices: list[FloatArray] = []
    for shifts in product(*(range(size) for size in space.sizes)):
        destination_of_source = np.empty(space.n_states, dtype=np.int64)
        for source in range(space.n_states):
            coordinate = flat_to_coordinate(source, space)
            shifted = tuple(
                (value + shift) % size
                for value, shift, size in zip(coordinate, shifts, space.sizes, strict=True)
            )
            destination_of_source[source] = coordinate_to_flat(shifted, space)
        matrix = np.zeros((space.n_states, space.n_states), dtype=np.float64)
        matrix[destination_of_source, np.arange(space.n_states)] = 1.0
        matrices.append(matrix)
    return np.stack(matrices)


def twirl_operator(operator: ArrayLike, group: ArrayLike) -> FloatArray:
    """Orthogonally average an operator under permutation conjugation."""

    values = np.asarray(operator, dtype=np.float64)
    matrices = np.asarray(group, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("operator must be square")
    if matrices.ndim != 3 or matrices.shape[1:] != values.shape:
        raise ValueError("group must have shape (element, state, state)")
    if matrices.shape[0] < 1:
        raise ValueError("group must contain at least one element")
    return np.mean(matrices @ values @ np.swapaxes(matrices, -1, -2), axis=0)


def relative_twirl_defect(operator: ArrayLike, group: ArrayLike) -> float:
    values = np.asarray(operator, dtype=np.float64)
    denominator = np.linalg.norm(values)
    residual = np.linalg.norm(values - twirl_operator(values, group))
    return float(residual / denominator) if denominator else float(residual)
