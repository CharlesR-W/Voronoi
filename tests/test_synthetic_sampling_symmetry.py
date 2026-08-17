from __future__ import annotations

import numpy as np

from voronoi_lab.synthetic import (
    FactorSpace,
    commutator_matrix,
    commutator_spectrum,
    estimate_generator,
    generate_synthetic_instance,
    operator_span_without_identity,
    principal_angles,
    safe_tau,
    sample_generator_family,
)
from voronoi_lab.synthetic.generators import assert_generator
from voronoi_lab.synthetic.symmetry import vectorize_operator


def test_euler_multinomial_counts_and_empirical_generator() -> None:
    instance = generate_synthetic_instance(
        FactorSpace((2, 3)),
        3,
        np.random.default_rng(21),
        rho=0.2,
        delta=0.4,
    )
    tau = safe_tau(instance.observed.generators, fraction=0.4)
    samples = sample_generator_family(
        instance.observed.generators,
        instance.observed.space,
        500,
        np.random.default_rng(22),
        tau=tau,
    )
    np.testing.assert_array_equal(samples.counts.sum(axis=-2), 500)
    estimates = estimate_generator(samples)
    for estimate in estimates:
        assert_generator(estimate, atol=1e-10)


def test_commutator_matrix_matches_direct_matrix_commutators() -> None:
    rng = np.random.default_rng(23)
    generators = rng.normal(size=(3, 4, 4))
    candidate = rng.normal(size=(4, 4))
    explicit = commutator_matrix(generators) @ vectorize_operator(candidate)
    direct = np.concatenate(
        [vectorize_operator(candidate @ operator - operator @ candidate) for operator in generators]
    )
    np.testing.assert_allclose(explicit, direct, atol=2e-14)


def test_identity_quotient_recovers_planted_cyclic_symmetry_subspace() -> None:
    instance = generate_synthetic_instance(
        FactorSpace((2, 3)),
        8,
        np.random.default_rng(24),
        density=0.8,
        rho=0.35,
        delta=0.0,
        support_policy="mixed",
    )
    planted = operator_span_without_identity(instance.truth.observed_group)
    spectrum = commutator_spectrum(instance.observed.generators)
    planted_dimension = planted.shape[1]
    recovered = np.stack(
        [vectorize_operator(mode) for mode in spectrum.modes[:planted_dimension]], axis=1
    )

    assert planted_dimension == instance.observed.space.n_states - 1
    assert spectrum.singular_values[planted_dimension - 1] < 2e-12
    assert spectrum.singular_values[planted_dimension] > 1e-4
    assert max(principal_angles(planted, recovered), default=0.0) < 3e-8
    assert all(abs(np.trace(mode)) < 2e-13 for mode in spectrum.modes)
    np.testing.assert_allclose(spectrum.singular_values, spectrum.commutator_residuals, atol=2e-13)
