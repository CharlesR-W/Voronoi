"""Commutator spectra and planted symmetry-subspace comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .schema import FloatArray, SymmetrySpectrum


def vectorize_operator(operator: ArrayLike) -> FloatArray:
    return np.asarray(operator, dtype=np.float64).reshape(-1, order="F")


def unvectorize_operator(vector: ArrayLike, size: int) -> FloatArray:
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (size * size,):
        raise ValueError("operator vector has the wrong size")
    return values.reshape((size, size), order="F")


def identity_complement(size: int) -> FloatArray:
    """Deterministic orthonormal basis for matrices orthogonal to identity."""

    if size < 2:
        raise ValueError("identity quotient needs a state space of size at least two")
    dimension = size * size
    direction = vectorize_operator(np.eye(size)) / np.sqrt(size)
    first_basis = np.zeros(dimension, dtype=np.float64)
    first_basis[0] = 1.0
    difference = first_basis - direction
    householder = np.eye(dimension) - 2.0 * np.outer(difference, difference) / np.dot(
        difference, difference
    )
    complement = householder[:, 1:]
    if not np.allclose(complement.T @ complement, np.eye(dimension - 1), atol=1e-12):
        raise RuntimeError("failed to construct an orthonormal identity complement")
    return complement


def commutator_matrix(
    generators: ArrayLike,
    *,
    weights: Sequence[float] | None = None,
) -> FloatArray:
    """Matrix of ``X -> ([X,L_1],...)`` using Fortran vectorization."""

    values = np.asarray(generators, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-2] != values.shape[-1]:
        raise ValueError("generators must be square matrices or a family")
    count, size, _ = values.shape
    selected_weights = np.ones(count, dtype=np.float64)
    if weights is not None:
        selected_weights = np.asarray(tuple(weights), dtype=np.float64)
        if selected_weights.shape != (count,) or np.any(selected_weights < 0):
            raise ValueError("weights must be one nonnegative value per primitive")
    identity = np.eye(size, dtype=np.float64)
    blocks = [
        np.sqrt(selected_weights[index])
        * (np.kron(operator.T, identity) - np.kron(identity, operator))
        for index, operator in enumerate(values)
    ]
    return np.vstack(blocks)


def commutator_spectrum(
    generators: ArrayLike,
    *,
    weights: Sequence[float] | None = None,
) -> SymmetrySpectrum:
    """Compute ascending nontrivial commutator singular modes."""

    values = np.asarray(generators, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, ...]
    size = values.shape[-1]
    commutator = commutator_matrix(values, weights=weights)
    quotient = identity_complement(size)
    reduced = commutator @ quotient
    _, singular_values, right_adjoint = np.linalg.svd(reduced, full_matrices=False)
    order = np.argsort(singular_values)
    singular_values = singular_values[order]
    vectors = quotient @ right_adjoint.T[:, order]
    modes = np.stack(
        [unvectorize_operator(vectors[:, index], size) for index in range(vectors.shape[1])]
    )
    residuals = np.linalg.norm(commutator @ vectors, axis=0)
    return SymmetrySpectrum(
        singular_values=singular_values,
        modes=modes,
        commutator_residuals=residuals,
    )


def operator_span_without_identity(operators: ArrayLike, *, tolerance: float = 1e-12) -> FloatArray:
    """Orthonormalize an operator span after removing the scalar identity."""

    values = np.asarray(operators, dtype=np.float64)
    if values.ndim != 3 or values.shape[-2] != values.shape[-1]:
        raise ValueError("operators must have shape (element, state, state)")
    size = values.shape[-1]
    quotient = identity_complement(size)
    coordinates = quotient.T @ np.stack([vectorize_operator(value) for value in values], axis=1)
    left, singular_values, _ = np.linalg.svd(coordinates, full_matrices=False)
    if singular_values.size == 0:
        return np.empty((size * size, 0), dtype=np.float64)
    keep = singular_values > tolerance * singular_values[0]
    return quotient @ left[:, keep]


def principal_angles(left_basis: ArrayLike, right_basis: ArrayLike) -> FloatArray:
    """Principal angles between column spans, in ascending order."""

    left = np.asarray(left_basis, dtype=np.float64)
    right = np.asarray(right_basis, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("subspace bases must be two-dimensional with equal ambient size")
    left_q, _ = np.linalg.qr(left)
    right_q, _ = np.linalg.qr(right)
    singular_values = np.linalg.svd(left_q.T @ right_q, compute_uv=False)
    return np.arccos(np.clip(singular_values, -1.0, 1.0))
