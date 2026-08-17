from __future__ import annotations

import numpy as np
import pytest

from voronoi_lab.exp1.geometry import (
    ChannelStandardizer,
    Codebook,
    CoordinateBatch,
    effective_occupied_cells,
    first_positive_boundary_crossing,
    nearest_boundary_margin,
    normalized_distortion,
)


def test_standardized_native_round_trip_and_displacement() -> None:
    rng = np.random.default_rng(1)
    native = rng.normal(size=(128, 7)) * np.arange(1, 8) + np.arange(7)
    standardizer = ChannelStandardizer.fit(native)
    standardized = standardizer.transform(native)
    reconstructed = standardizer.inverse(standardized.values)
    relative_rms = np.sqrt(np.mean((reconstructed - native) ** 2)) / np.sqrt(np.mean(native**2))
    assert relative_rms < 1e-12
    displacement = rng.normal(size=(3, 7))
    assert np.allclose(
        standardizer.inverse(standardized.values[:3] + displacement)
        - standardizer.inverse(standardized.values[:3]),
        standardizer.displacement_to_native(displacement),
    )


def test_nearest_boundary_is_not_assumed_to_be_second_point_nearest() -> None:
    # The far centroid's x=5 bisector is closer even though the near vertical
    # centroid is second-nearest by point distance from the anchor.
    codebook = Codebook(np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 2.0]]), "raw", "fixture")
    point = CoordinateBatch(np.asarray([[4.9, 0.8]]), "raw")
    distances = codebook.squared_distances(point)[0]
    assert np.argsort(distances)[1] == 2
    margin = nearest_boundary_margin(point, codebook)[0]
    brute = min(
        (distances[j] - distances[0])
        / (2 * np.linalg.norm(codebook.centers[j] - codebook.centers[0]))
        for j in (1, 2)
    )
    assert margin == pytest.approx(brute)


def test_first_crossing_hits_known_boundary() -> None:
    codebook = Codebook(np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]]), "raw", "x")
    batch = CoordinateBatch(np.asarray([[0.25, 0.5]]), "raw")
    directions = np.asarray([[1.0, 0.0]])
    crossing = first_positive_boundary_crossing(batch, directions, codebook)
    assert crossing[0] == pytest.approx(0.75)


def test_metric_mixing_is_rejected() -> None:
    codebook = Codebook(np.asarray([[0.0], [1.0]]), "standardized", "fit")
    with pytest.raises(ValueError, match="metric mismatch"):
        codebook.assign(CoordinateBatch(np.asarray([[0.2]]), "raw"))


def test_distortion_and_effective_occupancy_have_expected_limits() -> None:
    batch = CoordinateBatch(np.asarray([[0.0], [0.0], [2.0], [2.0]]), "raw")
    codebook = Codebook(np.asarray([[0.0], [2.0]]), "raw", "fit")
    assignments = codebook.assign(batch)
    assert normalized_distortion(batch, codebook) == pytest.approx(0.0)
    assert effective_occupied_cells(assignments, k=2) == pytest.approx(2.0)
