"""Orthogonal identity/traceless support decomposition of product operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike

from .schema import FactorSpace, FloatArray, Support


def all_supports(space: FactorSpace) -> tuple[Support, ...]:
    return tuple(
        support
        for order in range(space.n_factors + 1)
        for support in combinations(range(space.n_factors), order)
    )


def _to_paired_tensor(operator: FloatArray, space: FactorSpace) -> FloatArray:
    tensor = operator.reshape((*space.sizes, *space.sizes), order="C")
    interleave = tuple(
        axis for factor in range(space.n_factors) for axis in (factor, factor + space.n_factors)
    )
    tensor = np.transpose(tensor, interleave)
    return tensor.reshape(tuple(size * size for size in space.sizes), order="C")


def _from_paired_tensor(tensor: FloatArray, space: FactorSpace) -> FloatArray:
    interleaved_shape = tuple(value for size in space.sizes for value in (size, size))
    interleaved = tensor.reshape(interleaved_shape, order="C")
    destination_axes = tuple(2 * factor for factor in range(space.n_factors))
    source_axes = tuple(2 * factor + 1 for factor in range(space.n_factors))
    conventional = np.transpose(interleaved, (*destination_axes, *source_axes))
    return conventional.reshape((space.n_states, space.n_states), order="C")


def _identity_projection(tensor: FloatArray, axis: int, size: int) -> FloatArray:
    identity_direction = np.eye(size, dtype=np.float64).reshape(-1) / np.sqrt(size)
    moved = np.moveaxis(tensor, axis, 0)
    coefficient = np.tensordot(identity_direction, moved, axes=(0, 0))
    projected = identity_direction.reshape((-1,) + (1,) * coefficient.ndim) * coefficient
    return np.moveaxis(projected, 0, axis)


def support_decomposition(operator: ArrayLike, space: FactorSpace) -> dict[Support, FloatArray]:
    """Return the unique Frobenius-orthogonal exact-support components."""

    values = np.asarray(operator, dtype=np.float64)
    if values.shape != (space.n_states, space.n_states):
        raise ValueError(
            f"operator must have shape {(space.n_states, space.n_states)}, got {values.shape}"
        )
    branches: dict[Support, FloatArray] = {(): _to_paired_tensor(values, space)}
    for axis, size in enumerate(space.sizes):
        next_branches: dict[Support, FloatArray] = {}
        for support, tensor in branches.items():
            identity_part = _identity_projection(tensor, axis, size)
            next_branches[support] = identity_part
            next_branches[(*support, axis)] = tensor - identity_part
        branches = next_branches
    return {support: _from_paired_tensor(tensor, space) for support, tensor in branches.items()}


def reconstruct_support(components: Mapping[Support, ArrayLike], space: FactorSpace) -> FloatArray:
    result = np.zeros((space.n_states, space.n_states), dtype=np.float64)
    expected = set(all_supports(space))
    if set(components) != expected:
        raise ValueError("components must contain every support exactly once")
    for support, component in components.items():
        space.validate_support(support)
        values = np.asarray(component, dtype=np.float64)
        if values.shape != result.shape:
            raise ValueError("support component has the wrong operator shape")
        result += values
    return result


def support_order_energy(
    operator: ArrayLike,
    space: FactorSpace,
    *,
    normalize: bool = False,
) -> FloatArray:
    components = support_decomposition(operator, space)
    energy = np.zeros(space.n_factors + 1, dtype=np.float64)
    for support, component in components.items():
        energy[len(support)] += np.vdot(component, component).real
    if normalize:
        total = float(energy.sum())
        if total:
            energy /= total
    return energy


def weighted_high_order_objective(
    generators: ArrayLike,
    space: FactorSpace,
    penalties: Sequence[float],
) -> float:
    """Evaluate the README's exact fixed-label support-energy objective."""

    values = np.asarray(generators, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1:] != (space.n_states, space.n_states):
        raise ValueError("generators have the wrong shape")
    if len(penalties) < space.n_factors + 1:
        raise ValueError("penalties must cover every interaction order")
    total = 0.0
    for operator in values:
        for support, component in support_decomposition(operator, space).items():
            total += float(penalties[len(support)]) * float(np.vdot(component, component).real)
    return total
