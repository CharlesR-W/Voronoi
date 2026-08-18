"""Pure geometry and output-field calculations for activation-slice probes.

The functions in this module deliberately know nothing about PyTorch or model
architectures.  A collector supplies activation tensors and downstream outputs;
these helpers construct the two declared planes and turn saved outputs into the
raw fields used by plots and animations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _finite_array(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def _strict_grid(values: ArrayLike, *, name: str) -> FloatArray:
    grid = _finite_array(values, name=name)
    if grid.ndim != 1 or len(grid) < 3:
        raise ValueError(f"{name} must contain at least three coordinates")
    if np.any(np.diff(grid) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return grid


@dataclass(frozen=True, slots=True)
class ThreeAnchorSlice:
    """The paper's plane ``A + alpha(B-A) + beta(C-P)``.

    ``P`` is the orthogonal projection of ``C`` onto the line through ``A`` and
    ``B``.  Thus the two stored axes are orthogonal, while retaining the paper's
    unnormalized alpha/beta coordinates and the true plotted location of C.
    """

    origin: FloatArray
    alpha_axis: FloatArray
    beta_axis: FloatArray
    c_alpha: float

    def __post_init__(self) -> None:
        origin = _finite_array(self.origin, name="origin")
        alpha_axis = _finite_array(self.alpha_axis, name="alpha_axis")
        beta_axis = _finite_array(self.beta_axis, name="beta_axis")
        if origin.shape != alpha_axis.shape or origin.shape != beta_axis.shape:
            raise ValueError("slice origin and axes must have matching activation shapes")
        alpha_flat = alpha_axis.reshape(-1)
        beta_flat = beta_axis.reshape(-1)
        alpha_norm = float(np.linalg.norm(alpha_flat))
        beta_norm = float(np.linalg.norm(beta_flat))
        if alpha_norm <= 0 or beta_norm <= 0:
            raise ValueError("three-anchor slice axes must be nonzero")
        cosine = float(alpha_flat @ beta_flat / (alpha_norm * beta_norm))
        if abs(cosine) > 1e-8:
            raise ValueError("three-anchor slice axes must be orthogonal")
        if not np.isfinite(self.c_alpha):
            raise ValueError("C alpha coordinate must be finite")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "alpha_axis", alpha_axis)
        object.__setattr__(self, "beta_axis", beta_axis)

    @classmethod
    def from_anchors(
        cls,
        anchor_a: ArrayLike,
        anchor_b: ArrayLike,
        anchor_c: ArrayLike,
        *,
        tolerance: float = 1e-12,
    ) -> ThreeAnchorSlice:
        a = _finite_array(anchor_a, name="anchor_a")
        b = _finite_array(anchor_b, name="anchor_b")
        c = _finite_array(anchor_c, name="anchor_c")
        if a.shape != b.shape or a.shape != c.shape:
            raise ValueError("three anchors must have matching activation shapes")
        alpha_axis = b - a
        alpha_flat = alpha_axis.reshape(-1)
        squared_norm = float(alpha_flat @ alpha_flat)
        if squared_norm <= tolerance:
            raise ValueError("anchors A and B must be distinct")
        c_alpha = float((c - a).reshape(-1) @ alpha_flat / squared_norm)
        projection = a + c_alpha * alpha_axis
        beta_axis = c - projection
        if float(np.linalg.norm(beta_axis.reshape(-1))) <= tolerance:
            raise ValueError("the three anchors must not be collinear")
        return cls(a, alpha_axis, beta_axis, c_alpha)

    @property
    def alpha_scale(self) -> float:
        return float(np.linalg.norm(self.alpha_axis.reshape(-1)))

    @property
    def beta_scale(self) -> float:
        return float(np.linalg.norm(self.beta_axis.reshape(-1)))

    @property
    def anchor_coordinates(self) -> FloatArray:
        return np.asarray(((0.0, 0.0), (1.0, 0.0), (self.c_alpha, 1.0)), dtype=np.float64)

    def grid(self, alpha_values: ArrayLike, beta_values: ArrayLike) -> FloatArray:
        alpha = _strict_grid(alpha_values, name="alpha_values")
        beta = _strict_grid(beta_values, name="beta_values")
        alpha_mesh, beta_mesh = np.meshgrid(alpha, beta, indexing="xy")
        coefficient_shape = alpha_mesh.shape + (1,) * self.origin.ndim
        return (
            self.origin
            + alpha_mesh.reshape(coefficient_shape) * self.alpha_axis
            + beta_mesh.reshape(coefficient_shape) * self.beta_axis
        )


@dataclass(frozen=True, slots=True)
class OrthonormalPerturbationPlane:
    """A two-direction plane around one real or covariance-matched fake base."""

    base: FloatArray
    first: FloatArray
    second: FloatArray
    scale: float

    def __post_init__(self) -> None:
        base = _finite_array(self.base, name="base")
        first = _finite_array(self.first, name="first")
        second = _finite_array(self.second, name="second")
        if base.shape != first.shape or base.shape != second.shape:
            raise ValueError("perturbation base and directions must have matching shapes")
        first_flat = first.reshape(-1)
        second_flat = second.reshape(-1)
        if not np.isclose(np.linalg.norm(first_flat), 1.0, rtol=0, atol=1e-8):
            raise ValueError("first perturbation direction must be unit norm")
        if not np.isclose(np.linalg.norm(second_flat), 1.0, rtol=0, atol=1e-8):
            raise ValueError("second perturbation direction must be unit norm")
        if abs(float(first_flat @ second_flat)) > 1e-8:
            raise ValueError("perturbation directions must be orthogonal")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("perturbation scale must be positive and finite")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)

    @classmethod
    def from_targets(
        cls,
        base: ArrayLike,
        first_target: ArrayLike,
        second_target: ArrayLike,
        *,
        scale: float,
        tolerance: float = 1e-12,
    ) -> OrthonormalPerturbationPlane:
        center = _finite_array(base, name="base")
        first_delta = _finite_array(first_target, name="first_target") - center
        second_delta = _finite_array(second_target, name="second_target") - center
        if first_delta.shape != center.shape or second_delta.shape != center.shape:
            raise ValueError("perturbation targets must match the base activation shape")
        first_flat = first_delta.reshape(-1)
        first_norm = float(np.linalg.norm(first_flat))
        if first_norm <= tolerance:
            raise ValueError("first perturbation target equals the base")
        first_flat = first_flat / first_norm
        second_flat = second_delta.reshape(-1)
        second_flat = second_flat - float(second_flat @ first_flat) * first_flat
        second_norm = float(np.linalg.norm(second_flat))
        if second_norm <= tolerance:
            raise ValueError("perturbation targets do not span a plane")
        second_flat = second_flat / second_norm
        return cls(
            center,
            first_flat.reshape(center.shape),
            second_flat.reshape(center.shape),
            scale,
        )

    def grid(self, first_values: ArrayLike, second_values: ArrayLike) -> FloatArray:
        first = _strict_grid(first_values, name="first_values")
        second = _strict_grid(second_values, name="second_values")
        first_mesh, second_mesh = np.meshgrid(first, second, indexing="xy")
        coefficient_shape = first_mesh.shape + (1,) * self.base.ndim
        return (
            self.base
            + self.scale * first_mesh.reshape(coefficient_shape) * self.first
            + self.scale * second_mesh.reshape(coefficient_shape) * self.second
        )


def empirical_covariance_gaussian(
    bank: ArrayLike,
    coefficients: ArrayLike,
) -> FloatArray:
    """Sample the exact empirical Gaussian without materializing its covariance.

    If ``bank`` has ``n`` rows and ``g ~ N(0, I_n)``, then
    ``mean + g @ centered / sqrt(n - 1)`` has the bank's unbiased empirical
    covariance in the full flattened activation space.
    """

    values = _finite_array(bank, name="bank")
    draws = _finite_array(coefficients, name="coefficients")
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError("activation bank must contain at least two rows")
    if draws.ndim != 2 or draws.shape[1] != values.shape[0]:
        raise ValueError("coefficient rows must match the activation-bank size")
    centered = values - values.mean(axis=0, keepdims=True)
    return values.mean(axis=0) + np.tensordot(
        draws,
        centered,
        axes=((1,), (0,)),
    ) / np.sqrt(values.shape[0] - 1)


def median_pair_distance(bank: ArrayLike) -> float:
    """Deterministic robust L2 scale from adjacent rows of a fixed bank."""

    values = _finite_array(bank, name="bank")
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError("activation bank must contain at least two rows")
    flattened = values.reshape(values.shape[0], -1)
    distances = np.linalg.norm(flattened[1:] - flattened[:-1], axis=1)
    scale = float(np.median(distances))
    if scale <= 0:
        raise ValueError("activation bank has zero median adjacent-pair distance")
    return scale


def output_distance_fields(
    outputs: ArrayLike,
    reference_outputs: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return raw anchor distances and the paper's per-channel RGB field."""

    grid = _finite_array(outputs, name="outputs")
    references = _finite_array(reference_outputs, name="reference_outputs")
    if grid.ndim < 3 or references.ndim != grid.ndim - 1:
        raise ValueError("outputs must be a 2D grid followed by output dimensions")
    if references.shape[0] != 3 or references.shape[1:] != grid.shape[2:]:
        raise ValueError("reference_outputs must contain three matching output tensors")
    flat_grid = grid.reshape(*grid.shape[:2], -1)
    flat_references = references.reshape(3, -1)
    distances = np.linalg.norm(
        flat_grid[:, :, None, :] - flat_references[None, None, :, :],
        axis=-1,
    )
    maxima = distances.max(axis=(0, 1))
    if np.any(maxima <= 0):
        raise ValueError("RGB normalization is undefined for a constant anchor distance")
    rgb = np.clip(1.0 - distances / maxima[None, None, :], 0.0, 1.0)
    return distances, rgb


def contextual_output_distance_fields(
    outputs_by_context: ArrayLike,
    reference_outputs: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return three-anchor distances while preserving each anchor's host context."""

    values = _finite_array(outputs_by_context, name="outputs_by_context")
    references = _finite_array(reference_outputs, name="reference_outputs")
    if values.ndim < 4 or values.shape[2] != 3:
        raise ValueError("outputs_by_context must be a 2D grid with three contexts")
    if references.shape != values.shape[2:]:
        raise ValueError("reference_outputs must match the three contextual outputs")
    flattened = values.reshape(*values.shape[:3], -1)
    flat_references = references.reshape(3, -1)
    distances = np.linalg.norm(
        flattened - flat_references[None, None, :, :],
        axis=-1,
    )
    maxima = distances.max(axis=(0, 1))
    if np.any(maxima <= 0):
        raise ValueError("RGB normalization is undefined for a constant contextual distance")
    rgb = np.clip(1.0 - distances / maxima[None, None, :], 0.0, 1.0)
    return distances, rgb


def path_response_fields(
    logits: ArrayLike,
    base_logits: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return L2 and ``KL(base || perturbed)`` along batched 1D paths."""

    values = _finite_array(logits, name="logits")
    bases = _finite_array(base_logits, name="base_logits")
    if values.ndim < 3 or bases.shape != values.shape[:-2] + values.shape[-1:]:
        raise ValueError("base_logits must match every path and the final logit axis")
    l2 = np.linalg.norm(values - bases[..., None, :], axis=-1)
    base_shifted = bases - np.max(bases, axis=-1, keepdims=True)
    base_log_prob = base_shifted - np.log(np.exp(base_shifted).sum(axis=-1, keepdims=True))
    shifted = values - np.max(values, axis=-1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    base_prob = np.exp(base_log_prob)
    kl = np.sum(
        base_prob[..., None, :] * (base_log_prob[..., None, :] - log_prob),
        axis=-1,
    )
    return l2, np.maximum(kl, 0.0)


def path_directional_jacobian_norm(
    outputs: ArrayLike,
    coefficients: ArrayLike,
    native_direction_norms: ArrayLike,
) -> FloatArray:
    """Finite-difference ``||J d_hat||`` along one or more saved paths."""

    values = _finite_array(outputs, name="outputs")
    steps = _strict_grid(coefficients, name="coefficients")
    scales = _finite_array(native_direction_norms, name="native_direction_norms")
    if values.ndim < 3 or values.shape[-2] != len(steps):
        raise ValueError("outputs must have a penultimate path-step axis")
    if scales.shape != values.shape[:-2] or np.any(scales <= 0):
        raise ValueError("native_direction_norms must be positive and match the path batch")
    flattened = values.reshape(*values.shape[:-1], -1)
    derivatives = np.gradient(flattened, steps, axis=-2, edge_order=2)
    coefficient_norm = np.linalg.norm(derivatives, axis=-1)
    return coefficient_norm / scales[..., None]


def base_response_fields(
    logits: ArrayLike,
    base_logits: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return logit L2 response and ``KL(base || perturbed)`` on a 2D grid."""

    values = _finite_array(logits, name="logits")
    base = _finite_array(base_logits, name="base_logits")
    if values.ndim != 3 or base.shape != values.shape[2:]:
        raise ValueError("logits must have shape (rows, columns, classes)")
    l2 = np.linalg.norm(values - base[None, None, :], axis=-1)
    base_shifted = base - np.max(base)
    base_log_prob = base_shifted - np.log(np.exp(base_shifted).sum())
    shifted = values - np.max(values, axis=-1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    base_prob = np.exp(base_log_prob)
    kl = np.sum(base_prob[None, None, :] * (base_log_prob - log_prob), axis=-1)
    return l2, np.maximum(kl, 0.0)


def plane_pullback_jacobian_frobenius(
    outputs: ArrayLike,
    first_values: ArrayLike,
    second_values: ArrayLike,
    *,
    first_scale: float,
    second_scale: float,
) -> FloatArray:
    """Finite-difference Frobenius norm restricted to an orthogonal 2D plane.

    Axis zero of ``outputs`` corresponds to ``second_values`` and axis one to
    ``first_values``, matching ``numpy.meshgrid(..., indexing='xy')``.  Dividing
    by the native axis lengths makes the result a derivative per unit native L2
    displacement, rather than per arbitrary interpolation coefficient.
    """

    values = _finite_array(outputs, name="outputs")
    first = _strict_grid(first_values, name="first_values")
    second = _strict_grid(second_values, name="second_values")
    if values.ndim < 3 or values.shape[:2] != (len(second), len(first)):
        raise ValueError("outputs grid does not match the declared plane coordinates")
    if first_scale <= 0 or second_scale <= 0:
        raise ValueError("plane axis scales must be positive")
    flattened = values.reshape(len(second), len(first), -1)
    derivative_second, derivative_first = np.gradient(
        flattened,
        second * second_scale,
        first * first_scale,
        axis=(0, 1),
        edge_order=2,
    )
    return np.sqrt(np.sum(derivative_first**2, axis=-1) + np.sum(derivative_second**2, axis=-1))


__all__ = [
    "OrthonormalPerturbationPlane",
    "ThreeAnchorSlice",
    "base_response_fields",
    "contextual_output_distance_fields",
    "empirical_covariance_gaussian",
    "median_pair_distance",
    "output_distance_fields",
    "path_directional_jacobian_norm",
    "path_response_fields",
    "plane_pullback_jacobian_frobenius",
]
