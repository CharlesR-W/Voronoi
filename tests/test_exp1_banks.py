from __future__ import annotations

import numpy as np
import pytest

from voronoi_lab.core import CanonicalJSONError
from voronoi_lab.exp1.banks import (
    image_bootstrap_means,
    make_image_bootstrap_plan,
    make_probe_index_plan,
    make_site_bank,
    semantic_seed,
)


def test_semantic_seed_is_deterministic_and_preserves_coordinate_boundaries() -> None:
    expected = semantic_seed(17, "probe", "a\x1fb", "c")

    assert semantic_seed(17, "probe", "a\x1fb", "c") == expected
    assert semantic_seed(17, "probe", "a", "b", "c") != expected


def test_semantic_seed_preserves_canonical_value_types() -> None:
    assert semantic_seed(17, "coordinate", 1) != semantic_seed(17, "coordinate", "1")
    assert semantic_seed(17, "coordinate", True) != semantic_seed(17, "coordinate", 1)
    assert semantic_seed(17, "coordinate", None) != semantic_seed(17, "coordinate", "None")
    assert semantic_seed(17, "coordinate", ["a", "b"]) != semantic_seed(
        17, "coordinate", "['a', 'b']"
    )


def test_semantic_seed_rejects_noncanonical_coordinates() -> None:
    class StringAlias:
        def __str__(self) -> str:
            return "probe"

    with pytest.raises(CanonicalJSONError, match="unsupported value type"):
        semantic_seed(17, StringAlias())  # type: ignore[arg-type]


def test_probe_banks_are_deterministic_and_split_isolated() -> None:
    kwargs = dict(
        train_size=50,
        test_size=30,
        fit_train_images=10,
        independent_fit_train_images=8,
        geometry_test_images=12,
        intervention_test_images=6,
        intervention_nested_in_geometry=False,
        root_seed=9,
    )
    left = make_probe_index_plan(**kwargs)
    right = make_probe_index_plan(**kwargs)
    assert all(np.array_equal(left.roles[key], right.roles[key]) for key in left.roles)
    left.assert_disjoint("codebook_fit", "independent_codebook_fit")
    left.assert_disjoint("geometry", "intervention")


def test_site_bank_caps_sites_and_weights_images_equally() -> None:
    bank = make_site_bank(
        [7, 11, 20], height=8, width=8, max_sites_per_image=10, root_seed=2, namespace="x"
    )
    assert len(bank.image_ids) == 30
    assert max(np.bincount(np.searchsorted(np.unique(bank.image_ids), bank.image_ids))) == 10
    bank.assert_equal_image_weight()


def test_bootstrap_averages_within_image_before_resampling() -> None:
    image_ids = np.asarray([10, 11])
    plan = make_image_bootstrap_plan(image_ids, resamples=5, root_seed=3, namespace="metric")
    token_ids = np.asarray([10, 10, 11, 11, 11])
    values = np.asarray([0.0, 2.0, 8.0, 10.0, 12.0])
    observed = image_bootstrap_means(values, token_ids, plan)
    expected = np.asarray([[1.0 if i == 10 else 10.0 for i in row] for row in plan]).mean(axis=1)
    assert np.allclose(observed, expected)


def test_nested_intervention_bank_must_fit_inside_geometry() -> None:
    with pytest.raises(ValueError, match="test split"):
        make_probe_index_plan(
            train_size=20,
            test_size=20,
            fit_train_images=5,
            independent_fit_train_images=5,
            geometry_test_images=4,
            intervention_test_images=5,
            intervention_nested_in_geometry=True,
            root_seed=0,
        )
