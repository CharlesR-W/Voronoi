"""Euler transition construction and uniform per-state multinomial sampling."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .generators import assert_generator
from .schema import FactorSpace, FloatArray, TransitionCounts


def safe_tau(generators: ArrayLike, *, fraction: float = 0.5) -> float:
    values = np.asarray(generators, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, ...]
    if not 0 < fraction < 1:
        raise ValueError("fraction must lie strictly between zero and one")
    max_exit = float(np.max(-np.diagonal(values, axis1=-2, axis2=-1)))
    if max_exit <= 0:
        raise ValueError("generator family has no positive exit rate")
    return fraction / max_exit


def euler_transition(generators: ArrayLike, tau: float) -> FloatArray:
    """Form ``P = I + tau L`` and validate column stochasticity."""

    values = np.asarray(generators, dtype=np.float64)
    was_matrix = values.ndim == 2
    if was_matrix:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-2] != values.shape[-1]:
        raise ValueError("generators must be square matrices or a family of them")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    for operator in values:
        assert_generator(operator)
    identity = np.eye(values.shape[-1], dtype=np.float64)
    transition = identity + float(tau) * values
    tolerance = 1e-12
    if float(transition.min(initial=0.0)) < -tolerance:
        raise ValueError("tau is too large: I + tau L has a negative probability")
    if not np.allclose(transition.sum(axis=-2), 1.0, atol=tolerance, rtol=0):
        raise ValueError("Euler transition columns are not stochastic")
    transition = np.maximum(transition, 0.0)
    transition /= transition.sum(axis=-2, keepdims=True)
    return transition[0] if was_matrix else transition


def sample_transition_counts(
    transitions: ArrayLike,
    transitions_per_state: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample each primitive/source column independently."""

    probabilities = np.asarray(transitions, dtype=np.float64)
    if probabilities.ndim == 2:
        probabilities = probabilities[None, ...]
    if probabilities.ndim != 3 or probabilities.shape[-2] != probabilities.shape[-1]:
        raise ValueError("transitions must be square matrices or a family of them")
    if transitions_per_state < 1:
        raise ValueError("transitions_per_state must be positive")
    if np.any(probabilities < -1e-12) or not np.allclose(
        probabilities.sum(axis=-2), 1.0, atol=1e-12, rtol=0
    ):
        raise ValueError("transition columns must be nonnegative and sum to one")
    counts = np.empty(probabilities.shape, dtype=np.int64)
    for primitive in range(probabilities.shape[0]):
        for source in range(probabilities.shape[-1]):
            counts[primitive, :, source] = rng.multinomial(
                transitions_per_state, probabilities[primitive, :, source]
            )
    return counts


def sample_generator_family(
    generators: ArrayLike,
    space: FactorSpace,
    transitions_per_state: int,
    rng: np.random.Generator,
    *,
    tau: float | None = None,
    tau_fraction: float = 0.5,
) -> TransitionCounts:
    values = np.asarray(generators, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, ...]
    selected_tau = safe_tau(values, fraction=tau_fraction) if tau is None else float(tau)
    probabilities = euler_transition(values, selected_tau)
    counts = sample_transition_counts(probabilities, transitions_per_state, rng)
    return TransitionCounts(space=space, counts=counts, tau=selected_tau)


def estimate_generator(samples: TransitionCounts) -> FloatArray:
    """Return the direct known-tau empirical generator estimate."""

    totals = samples.counts.sum(axis=-2, keepdims=True)
    probabilities = samples.counts / totals
    identity = np.eye(samples.space.n_states, dtype=np.float64)
    estimates = (probabilities - identity) / samples.tau
    for estimate in estimates:
        assert_generator(estimate, atol=1e-10)
    return estimates
