"""Synthetic continuous-time Markov generator construction."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike

from .groups import cyclic_product_group, relative_twirl_defect, twirl_operator
from .indexing import lift_local_operator, to_observed
from .schema import (
    FactorSpace,
    FloatArray,
    Labeling,
    ObservedGeneratorFamily,
    Support,
    SyntheticInstance,
    SyntheticTruth,
)
from .support import support_decomposition, support_order_energy


def generator_tolerance(generator: ArrayLike) -> float:
    values = np.asarray(generator)
    scale = max(1.0, float(np.linalg.norm(values, ord=np.inf)))
    dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float64
    return 100.0 * np.finfo(dtype).eps * values.shape[-1] * scale


def assert_generator(generator: ArrayLike, *, atol: float | None = None) -> None:
    """Validate nonnegative off-diagonals and zero column sums."""

    values = np.asarray(generator, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("generator must be a square matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("generator must be finite")
    tolerance = generator_tolerance(values) if atol is None else float(atol)
    off_diagonal = values.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if float(off_diagonal.min(initial=0.0)) < -tolerance:
        raise ValueError("generator has a negative off-diagonal entry")
    if float(np.diag(values).max(initial=0.0)) > tolerance:
        raise ValueError("generator has a positive diagonal entry")
    if float(np.max(np.abs(values.sum(axis=0)), initial=0.0)) > tolerance:
        raise ValueError("generator columns do not sum to zero")


def normalize_mean_exit(generator: ArrayLike) -> FloatArray:
    values = np.asarray(generator, dtype=np.float64)
    assert_generator(values)
    mean_exit = float(np.mean(-np.diag(values)))
    if mean_exit <= 0:
        raise ValueError("cannot normalize a generator with zero mean exit rate")
    return values / mean_exit


def draw_sparse_generator(
    size: int,
    rng: np.random.Generator,
    *,
    density: float = 0.5,
    rate_shape: float = 2.0,
) -> FloatArray:
    """Draw a sparse irreducible generator using a mandatory directed cycle."""

    if size < 2:
        raise ValueError("random generator size must be at least two")
    if not 0 < density <= 1:
        raise ValueError("density must lie in (0, 1]")
    if not np.isfinite(rate_shape) or rate_shape <= 0:
        raise ValueError("rate_shape must be finite and positive")
    mask = rng.random((size, size)) < density
    np.fill_diagonal(mask, False)
    sources = np.arange(size)
    mask[(sources + 1) % size, sources] = True
    rates = np.zeros((size, size), dtype=np.float64)
    edge_count = int(mask.sum())
    rates[mask] = rng.gamma(shape=rate_shape, scale=1.0 / rate_shape, size=edge_count)
    rates[np.diag_indices(size)] = -rates.sum(axis=0)
    result = normalize_mean_exit(rates)
    assert_generator(result)
    return result


def _normalized_weights(count: int, rng: np.random.Generator) -> FloatArray:
    if count == 0:
        return np.empty(0, dtype=np.float64)
    weights = rng.exponential(size=count)
    return weights / weights.sum()


def generate_synthetic_instance(
    space: FactorSpace,
    n_primitives: int,
    rng: np.random.Generator,
    *,
    density: float = 0.5,
    rate_shape: float = 2.0,
    unary_weight: float = 1.0,
    rho: float = 0.0,
    delta: float = 1.0,
    support_policy: str = "mixed",
    random_relabel: bool = True,
) -> SyntheticInstance:
    """Generate and arbitrarily relabel a family of planted product generators."""

    if n_primitives < 1:
        raise ValueError("n_primitives must be positive")
    if not np.isfinite(unary_weight) or unary_weight <= 0:
        raise ValueError("unary_weight must be finite and positive")
    if not np.isfinite(density) or not 0 < density <= 1:
        raise ValueError("density must lie in (0, 1]")
    if not np.isfinite(rate_shape) or rate_shape <= 0:
        raise ValueError("rate_shape must be finite and positive")
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("rho must be nonnegative")
    if not np.isfinite(delta) or not 0 <= delta <= 1:
        raise ValueError("delta must lie in [0, 1]")
    if support_policy not in {"mixed", "anchored"}:
        raise ValueError("support_policy must be 'mixed' or 'anchored'")
    if any(size < 2 for size in space.sizes):
        raise ValueError("synthetic generator factors must each have at least two states")

    unary_supports = tuple((axis,) for axis in range(space.n_factors))
    pair_supports = tuple(combinations(range(space.n_factors), 2))
    latent_generators: list[FloatArray] = []
    all_raw_terms: list[dict[Support, FloatArray]] = []

    for primitive in range(n_primitives):
        if support_policy == "anchored":
            unary_weights = np.zeros(len(unary_supports), dtype=np.float64)
            unary_weights[primitive % len(unary_supports)] = 1.0
        else:
            unary_weights = _normalized_weights(len(unary_supports), rng)
        pair_weights = _normalized_weights(len(pair_supports), rng)

        terms: dict[Support, FloatArray] = {}
        unscaled = np.zeros((space.n_states, space.n_states), dtype=np.float64)
        for order_scale, supports, weights in (
            (unary_weight, unary_supports, unary_weights),
            (rho, pair_supports, pair_weights),
        ):
            if order_scale == 0:
                continue
            for support, weight in zip(supports, weights, strict=True):
                if weight == 0:
                    continue
                local_space = FactorSpace(tuple(space.sizes[axis] for axis in support))
                local = draw_sparse_generator(
                    local_space.n_states,
                    rng,
                    density=density,
                    rate_shape=rate_shape,
                )
                local_group = cyclic_product_group(local_space)
                symmetric = twirl_operator(local, local_group)
                broken = (1.0 - delta) * symmetric + delta * local
                assert_generator(broken)
                lifted = lift_local_operator(broken, space, support)
                term = float(order_scale * weight) * lifted
                terms[support] = term
                unscaled += term
        scaled = normalize_mean_exit(unscaled)
        scale = float(np.mean(-np.diag(unscaled)))
        terms = {support: term / scale for support, term in terms.items()}
        assert_generator(scaled)
        latent_generators.append(scaled)
        all_raw_terms.append(terms)

    latent = np.stack(latent_generators)
    mapping = (
        rng.permutation(space.n_states).astype(np.int64)
        if random_relabel
        else np.arange(space.n_states, dtype=np.int64)
    )
    labeling = Labeling(space=space, obs_to_grid=mapping)
    observed_values = to_observed(latent, labeling)
    for operator in observed_values:
        assert_generator(operator)

    support_components = tuple(support_decomposition(operator, space) for operator in latent)
    realized_order_energy = np.stack(
        [support_order_energy(operator, space, normalize=True) for operator in latent]
    )
    global_group_latent = cyclic_product_group(space)
    global_group_observed = to_observed(global_group_latent, labeling)
    if delta == 0:
        for operator in latent:
            if relative_twirl_defect(operator, global_group_latent) > 1e-12:
                raise RuntimeError("delta=0 construction failed the global twirl invariant")

    observed = ObservedGeneratorFamily(space=space, generators=observed_values)
    truth = SyntheticTruth(
        labeling=labeling,
        latent_generators=latent,
        support_components=support_components,
        raw_terms=tuple(all_raw_terms),
        observed_group=global_group_observed,
        realized_order_energy=realized_order_energy,
        rho=float(rho),
        delta=float(delta),
    )
    return SyntheticInstance(observed=observed, truth=truth)
