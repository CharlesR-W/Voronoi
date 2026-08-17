from __future__ import annotations

import numpy as np

from voronoi_lab.exp1.nulls import fit_conditional_gaussian, fit_gaussian


def test_global_gaussian_preserves_first_two_moments_in_large_sample() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(size=(1000, 3)) @ np.asarray(
        [[1.0, 0.2, 0.0], [0.0, 2.0, 0.3], [0.0, 0.0, 0.5]]
    ) + [1.0, -2.0, 0.5]
    model = fit_gaussian(values, shrinkage=0.0)
    sampled = model.sample(100_000, np.random.default_rng(6))
    assert np.allclose(sampled.mean(axis=0), model.mean, atol=0.02)
    assert np.allclose(np.cov(sampled, rowvar=False, bias=True), model.covariance, atol=0.04)


def test_conditional_gaussian_keeps_group_mixture_separate() -> None:
    rng = np.random.default_rng(7)
    labels = np.repeat([0, 1], [300, 700])
    values = rng.normal(size=(1000, 2)) + labels[:, None] * 5
    model = fit_conditional_gaussian(values, labels)
    sampled, sampled_labels = model.sample(20_000, np.random.default_rng(8))
    assert abs((sampled_labels == 1).mean() - 0.7) < 0.02
    assert sampled[sampled_labels == 1].mean() > sampled[sampled_labels == 0].mean() + 4
