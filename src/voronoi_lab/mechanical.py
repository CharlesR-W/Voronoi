"""Deterministic, independently replayable mechanical-smoke fixtures."""

from __future__ import annotations

import math

import numpy as np

from voronoi_lab.core import JSONLike, SeedDeriver
from voronoi_lab.exp1.geometry import (
    ChannelStandardizer,
    Codebook,
    CoordinateBatch,
    first_positive_boundary_crossing,
    fit_codebook,
    normalized_distortion,
)
from voronoi_lab.exp1.nulls import fit_gaussian


def replay_toy_geometry(root_seed: int, *, rms_epsilon: float) -> dict[str, JSONLike]:
    """Recompute the fixed metric/null smoke from its signed seed and tolerance."""

    seeds = SeedDeriver(root_seed, ("exp1", "mechanical", "toy_geometry"))
    rng = np.random.default_rng(seeds.derive("roundtrip"))
    scales = np.geomspace(0.1, 10.0, 9)
    offsets = np.linspace(-3.0, 3.0, 9)
    native = (rng.normal(size=(512, 9)) * scales + offsets).astype(np.float32)
    standardizer = ChannelStandardizer.fit(native, epsilon=rms_epsilon)
    standardized = standardizer.transform(native).values.astype(np.float32)
    reconstructed = standardizer.inverse(standardized).astype(np.float32)
    numerator = float(np.sqrt(np.mean((reconstructed - native) ** 2, dtype=np.float64)))
    denominator = float(np.sqrt(np.mean(native**2, dtype=np.float64)))
    roundtrip = numerator / denominator

    standardized_centroid = rng.normal(size=(1, native.shape[1])).astype(np.float32)
    native_centroid = standardizer.inverse(standardized_centroid).astype(np.float32)
    reconstructed_centroid = standardizer.mean.astype(
        np.float32
    ) + standardized_centroid * standardizer.rms.astype(np.float32)
    centroid_denominator = max(
        float(np.sqrt(np.mean(native_centroid**2, dtype=np.float64))),
        np.finfo(np.float32).tiny,
    )
    centroid_error = float(
        np.sqrt(np.mean((native_centroid - reconstructed_centroid) ** 2, dtype=np.float64))
        / centroid_denominator
    )

    boundary_codebook = Codebook(
        np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]]), "raw", "mechanical"
    )
    boundary_point = CoordinateBatch(np.asarray([[0.25, 0.5]]), "raw")
    boundary_crossing = float(
        first_positive_boundary_crossing(
            boundary_point,
            np.asarray([[1.0, 0.0]]),
            boundary_codebook,
        )[0]
    )

    mixture_rng = np.random.default_rng(seeds.derive("mixture_gaussian"))
    mixture_centers = np.asarray([[-4.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

    def mixture_sample(count: int) -> np.ndarray:
        labels = mixture_rng.integers(0, 2, size=count)
        return mixture_centers[labels] + mixture_rng.normal(scale=0.35, size=(count, 3))

    mixture_fit = mixture_sample(768)
    mixture_eval = mixture_sample(768)
    moments = fit_gaussian(mixture_fit, shrinkage=0.0)
    gaussian_fit = moments.sample(768, mixture_rng)
    gaussian_eval = moments.sample(768, mixture_rng)
    mixture_codebook = fit_codebook(
        CoordinateBatch(mixture_fit, "raw"),
        k=2,
        seed=seeds.derive("mixture_codebook", bits=32),
        fit_bank_id="mechanical-mixture-fit",
        algorithm="kmeans",
        n_init=5,
        max_iter=100,
    )
    gaussian_codebook = fit_codebook(
        CoordinateBatch(gaussian_fit, "raw"),
        k=2,
        seed=seeds.derive("gaussian_codebook", bits=32),
        fit_bank_id="mechanical-gaussian-fit",
        algorithm="kmeans",
        n_init=5,
        max_iter=100,
    )
    mixture_distortion = normalized_distortion(
        CoordinateBatch(mixture_eval, "raw"), mixture_codebook
    )
    gaussian_distortion = normalized_distortion(
        CoordinateBatch(gaussian_eval, "raw"), gaussian_codebook
    )
    return {
        "boundary": {
            "absolute_error": abs(boundary_crossing - 0.75),
            "absolute_tolerance": 1e-12,
            "expected_crossing": 0.75,
            "observed_crossing": boundary_crossing,
            "passed": math.isclose(boundary_crossing, 0.75, abs_tol=1e-12),
        },
        "centroid_reconstruction": {"relative_rms_error": centroid_error},
        "mixture_gaussian": {
            "gaussian_normalized_distortion": gaussian_distortion,
            "mixture_normalized_distortion": mixture_distortion,
            "passed": mixture_distortion < gaussian_distortion,
        },
        "roundtrip": {"relative_rms_error": roundtrip},
    }


__all__ = ["replay_toy_geometry"]
