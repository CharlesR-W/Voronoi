from __future__ import annotations

import numpy as np
import pytest

from voronoi_lab.synthetic import (
    FactorSpace,
    assert_generator,
    cyclic_product_group,
    draw_sparse_generator,
    generate_synthetic_instance,
    reconstruct_support,
    support_decomposition,
    support_order_energy,
    twirl_operator,
)
from voronoi_lab.synthetic.groups import relative_twirl_defect


def test_sparse_generator_and_twirl_preserve_generator_cone() -> None:
    rng = np.random.default_rng(11)
    generator = draw_sparse_generator(6, rng, density=0.25)
    assert_generator(generator)
    np.testing.assert_allclose(np.mean(-np.diag(generator)), 1.0, atol=1e-14)

    group = cyclic_product_group(FactorSpace((2, 3)))
    twirled = twirl_operator(generator, group)
    assert_generator(twirled)
    np.testing.assert_allclose(twirl_operator(twirled, group), twirled, atol=1e-14)
    assert relative_twirl_defect(twirled, group) < 1e-14


def test_synthetic_instance_uses_the_configured_generator_rate_shape() -> None:
    space = FactorSpace((2, 3))
    baseline = generate_synthetic_instance(
        space,
        3,
        np.random.default_rng(111),
        rate_shape=2.0,
    )
    changed = generate_synthetic_instance(
        space,
        3,
        np.random.default_rng(111),
        rate_shape=5.0,
    )
    assert not np.allclose(
        baseline.truth.latent_generators,
        changed.truth.latent_generators,
    )

    with pytest.raises(ValueError, match="rate_shape"):
        generate_synthetic_instance(space, 1, np.random.default_rng(112), rate_shape=0.0)


def test_support_decomposition_reconstructs_and_is_orthogonal() -> None:
    rng = np.random.default_rng(12)
    space = FactorSpace((2, 3, 2))
    operator = rng.normal(size=(space.n_states, space.n_states))
    components = support_decomposition(operator, space)
    np.testing.assert_allclose(reconstruct_support(components, space), operator, atol=2e-14)

    supports = tuple(components)
    for left_index, left_support in enumerate(supports):
        for right_support in supports[left_index + 1 :]:
            assert abs(np.vdot(components[left_support], components[right_support])) < 2e-12
        redecomposed = support_decomposition(components[left_support], space)
        for projected_support, projected in redecomposed.items():
            if projected_support == left_support:
                np.testing.assert_allclose(projected, components[left_support], atol=2e-14)
            else:
                assert np.linalg.norm(projected) < 2e-14


def test_rho_and_delta_control_distinct_structures() -> None:
    space = FactorSpace((2, 3))
    symmetric_interacting = generate_synthetic_instance(
        space,
        4,
        np.random.default_rng(13),
        density=0.8,
        rho=0.6,
        delta=0.0,
        support_policy="mixed",
        random_relabel=False,
    )
    assert np.mean(symmetric_interacting.truth.realized_order_energy[:, 2]) > 1e-5
    latent_group = cyclic_product_group(space)
    assert all(
        relative_twirl_defect(operator, latent_group) < 1e-12
        for operator in symmetric_interacting.truth.latent_generators
    )

    broken_noninteracting = generate_synthetic_instance(
        space,
        4,
        np.random.default_rng(14),
        density=0.8,
        rho=0.0,
        delta=1.0,
        support_policy="mixed",
        random_relabel=False,
    )
    assert all(
        support_order_energy(operator, space)[2] < 1e-24
        for operator in broken_noninteracting.truth.latent_generators
    )
    assert any(
        relative_twirl_defect(operator, latent_group) > 1e-3
        for operator in broken_noninteracting.truth.latent_generators
    )


def test_raw_lifted_terms_sum_to_the_valid_full_generator() -> None:
    instance = generate_synthetic_instance(
        FactorSpace((2, 3)),
        3,
        np.random.default_rng(15),
        rho=0.4,
        delta=0.3,
    )
    for generator, terms in zip(
        instance.truth.latent_generators, instance.truth.raw_terms, strict=True
    ):
        assert_generator(generator)
        np.testing.assert_allclose(sum(terms.values()), generator, atol=2e-14)
