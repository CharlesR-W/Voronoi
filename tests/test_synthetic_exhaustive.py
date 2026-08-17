from __future__ import annotations

import numpy as np
import pytest

from voronoi_lab.core import canonical_hash
from voronoi_lab.synthetic import (
    FactorSpace,
    Labeling,
    align_labeling,
    aligned_support_error,
    exhaustive_label_search,
    generate_synthetic_instance,
    labeling_objective,
    run_oracle_exhaustive_instance,
    run_oracle_exhaustive_smoke,
)
from voronoi_lab.synthetic.indexing import labeling_coordinates


def _gauge_transform(labeling: Labeling) -> Labeling:
    coordinates = labeling_coordinates(labeling)
    transformed = np.column_stack((1 - coordinates[:, 0], (coordinates[:, 1] + 1) % 3))
    mapping = np.ravel_multi_index(transformed.T, labeling.space.sizes, order="C")
    return Labeling(labeling.space, mapping)


def test_alignment_and_objective_are_gauge_invariant() -> None:
    instance = generate_synthetic_instance(
        FactorSpace((2, 3)),
        6,
        np.random.default_rng(31),
        density=0.75,
        rho=0.0,
        delta=1.0,
        support_policy="anchored",
    )
    transformed = _gauge_transform(instance.truth.labeling)
    alignment = align_labeling(transformed, instance.truth.labeling)
    assert alignment.exact
    assert alignment.coordinate_ami == 1.0
    assert (
        aligned_support_error(
            instance.observed.generators, transformed, instance.truth.labeling, alignment
        )
        < 2e-14
    )
    np.testing.assert_allclose(
        labeling_objective(instance.observed.generators, transformed),
        labeling_objective(instance.observed.generators, instance.truth.labeling),
        atol=2e-14,
    )


def test_tiny_exhaustive_search_recovers_only_gauge_equivalent_labels() -> None:
    instance = generate_synthetic_instance(
        FactorSpace((2, 3)),
        6,
        np.random.default_rng(32),
        density=0.75,
        rho=0.0,
        delta=1.0,
        support_policy="anchored",
    )
    result = exhaustive_label_search(instance.observed.generators, instance.observed.space)
    assert result.evaluated == 720
    assert len(result.best_labelings) == 12  # 2! * 3! local gauge transformations.
    assert result.best_objective < 1e-24
    assert all(
        align_labeling(labeling, instance.truth.labeling).exact
        for labeling in result.best_labelings
    )


def test_deterministic_oracle_exhaustive_smoke_runner() -> None:
    result = run_oracle_exhaustive_smoke(seed=20260816, instances=2)
    assert result.instances == 2
    assert result.exact_instances == 2
    assert result.evaluated_labelings == 1440
    assert result.worst_support_error < 1e-12
    assert result.max_best_objective < 1e-24
    assert result.heldout_primitives == 3
    assert result.random_relabel is True
    assert result.unary_weight == 1.0
    assert result.seed_namespace == ("exp2", "exact", "oracle_exhaustive", "v1")
    assert [row.instance_index for row in result.instance_results] == [0, 1]
    assert len({row.seed for row in result.instance_results}) == 2
    assert all(row.exact and row.heldout_support_error < 1e-12 for row in result.instance_results)
    for row in result.instance_results:
        assert row.observed_generator_family_hash == canonical_hash(row.observed_generator_family)
        assert row.realized_normalized_support_order_spectrum_hash == canonical_hash(
            row.realized_normalized_support_order_spectrum
        )
        assert row.truth_labeling_hash == canonical_hash(row.truth_labeling)
        assert row.selected_labeling_hash == canonical_hash(row.selected_labeling)
        assert row.train_primitive_indices == tuple(range(result.train_primitives))
        assert row.heldout_primitive_indices == tuple(
            range(result.train_primitives, result.train_primitives + result.heldout_primitives)
        )
        np.testing.assert_allclose(
            np.sum(row.realized_normalized_support_order_spectrum, axis=1),
            1.0,
        )


def test_independent_instance_keeps_global_semantic_index() -> None:
    complete = run_oracle_exhaustive_smoke(seed=81, instances=3)
    isolated = run_oracle_exhaustive_instance(seed=81, instance_index=2)
    assert isolated == complete.instance_results[2]
    assert isolated.seed != complete.instance_results[0].seed


def test_exact_generator_and_search_protocol_is_consumed_and_recorded() -> None:
    result = run_oracle_exhaustive_smoke(
        seed=19,
        instances=1,
        factor_sizes=(2, 2),
        generator_rate_shape=3.5,
        generator_connectivity_policy="mandatory_directed_cycle",
        generator_normalization="unit_mean_exit_rate",
        exhaustive_tie_atol=2e-9,
        exhaustive_tie_rtol=3e-9,
    )
    assert result.generator_rate_shape == 3.5
    assert result.generator_connectivity_policy == "mandatory_directed_cycle"
    assert result.generator_normalization == "unit_mean_exit_rate"
    assert result.exhaustive_tie_atol == 2e-9
    assert result.exhaustive_tie_rtol == 3e-9


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("generator_rate_shape", 0.0),
        ("generator_connectivity_policy", "optional_cycle"),
        ("generator_normalization", "none"),
        ("exhaustive_tie_atol", -1.0),
        ("exhaustive_tie_rtol", float("nan")),
    ],
)
def test_exact_runner_rejects_unsupported_generator_and_search_choices(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=name):
        run_oracle_exhaustive_instance(instance_index=0, **{name: value})  # type: ignore[arg-type]
