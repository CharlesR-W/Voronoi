"""Metric-aware codebooks and exact Voronoi geometry for Experiment 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans, MiniBatchKMeans

FloatArray = NDArray[np.floating]
MetricName = Literal["standardized", "raw"]


def _matrix(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (samples, channels)")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class CoordinateBatch:
    """A matrix tagged with the metric in which its distances are meaningful."""

    values: FloatArray
    metric: MetricName

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _matrix(self.values, name="values"))


@dataclass(frozen=True)
class ChannelStandardizer:
    """Channel mean and population RMS fitted on the codebook bank only."""

    mean: FloatArray
    rms: FloatArray
    epsilon: float

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        rms = np.asarray(self.rms, dtype=np.float64)
        if mean.ndim != 1 or rms.shape != mean.shape:
            raise ValueError("mean and rms must be matching channel vectors")
        if self.epsilon <= 0 or np.any(rms < self.epsilon) or not np.all(np.isfinite(rms)):
            raise ValueError("rms must be finite and at least epsilon")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "rms", rms)

    @classmethod
    def fit(
        cls,
        native: ArrayLike,
        *,
        epsilon: float = 1e-6,
        sample_weight: ArrayLike | None = None,
    ) -> ChannelStandardizer:
        values = _matrix(native, name="native")
        weights = _weights(sample_weight, len(values))
        mean = np.average(values, axis=0, weights=weights)
        variance = np.average((values - mean) ** 2, axis=0, weights=weights)
        rms = np.maximum(np.sqrt(variance), epsilon)
        return cls(mean=mean, rms=rms, epsilon=epsilon)

    def transform(self, native: ArrayLike) -> CoordinateBatch:
        values = _matrix(native, name="native")
        self._check_channels(values)
        return CoordinateBatch((values - self.mean) / self.rms, "standardized")

    def inverse(self, standardized: ArrayLike) -> FloatArray:
        values = _matrix(standardized, name="standardized")
        self._check_channels(values)
        return self.mean + values * self.rms

    def displacement_to_native(self, standardized_displacement: ArrayLike) -> FloatArray:
        values = np.asarray(standardized_displacement, dtype=np.float64)
        if values.shape[-1] != len(self.mean):
            raise ValueError("displacement channel dimension does not match standardizer")
        return values * self.rms

    def standardized_normal_to_native(self, normal: ArrayLike) -> FloatArray:
        values = np.asarray(normal, dtype=np.float64)
        if values.shape[-1] != len(self.mean):
            raise ValueError("normal channel dimension does not match standardizer")
        return values / self.rms

    def _check_channels(self, values: FloatArray) -> None:
        if values.shape[1] != len(self.mean):
            raise ValueError("channel dimension does not match standardizer")


@dataclass(frozen=True)
class Codebook:
    centers: FloatArray
    metric: MetricName
    fit_bank_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "centers", _matrix(self.centers, name="centers"))
        if not self.fit_bank_id:
            raise ValueError("fit_bank_id is required for provenance")

    @property
    def k(self) -> int:
        return int(self.centers.shape[0])

    @property
    def channels(self) -> int:
        return int(self.centers.shape[1])

    def require_compatible(self, batch: CoordinateBatch) -> FloatArray:
        if batch.metric != self.metric:
            raise ValueError(
                "metric mismatch: "
                f"codebook={self.metric}, coordinates={batch.metric}; refit required"
            )
        if batch.values.shape[1] != self.channels:
            raise ValueError("channel dimension does not match codebook")
        return batch.values

    def squared_distances(self, batch: CoordinateBatch) -> FloatArray:
        values = self.require_compatible(batch)
        return np.sum((values[:, None, :] - self.centers[None, :, :]) ** 2, axis=2)

    def assign(self, batch: CoordinateBatch) -> NDArray[np.int64]:
        return np.argmin(self.squared_distances(batch), axis=1).astype(np.int64)


def fit_codebook(
    batch: CoordinateBatch,
    *,
    k: int,
    seed: int,
    fit_bank_id: str,
    algorithm: Literal["minibatch_kmeans", "kmeans"] = "minibatch_kmeans",
    n_init: int = 10,
    max_iter: int = 300,
    batch_size: int = 4096,
    initialization: Literal["k-means++", "random"] = "k-means++",
    sample_weight: ArrayLike | None = None,
) -> Codebook:
    """Fit a deterministic codebook in exactly one declared coordinate metric."""

    if not 1 < k <= len(batch.values):
        raise ValueError("k must be between 2 and the number of samples")
    weights = _weights(sample_weight, len(batch.values))
    common = dict(
        n_clusters=k,
        init=initialization,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    if algorithm == "minibatch_kmeans":
        estimator = MiniBatchKMeans(**common, batch_size=batch_size, reassignment_ratio=0.0)
    elif algorithm == "kmeans":
        estimator = KMeans(**common, algorithm="lloyd")
    else:  # pragma: no cover - protected by the type/config schema
        raise ValueError(f"unsupported algorithm: {algorithm}")
    estimator.fit(batch.values, sample_weight=weights)
    return Codebook(estimator.cluster_centers_, batch.metric, fit_bank_id)


def normalized_distortion(
    batch: CoordinateBatch,
    codebook: Codebook,
    *,
    sample_weight: ArrayLike | None = None,
) -> float:
    values = codebook.require_compatible(batch)
    weights = _weights(sample_weight, len(values))
    distances = codebook.squared_distances(batch)
    assigned = np.min(distances, axis=1)
    mean = np.average(values, axis=0, weights=weights)
    denominator = float(np.average(np.sum((values - mean) ** 2, axis=1), weights=weights))
    if denominator <= 0:
        raise ValueError("normalized distortion is undefined for a constant evaluation bank")
    return float(np.average(assigned, weights=weights) / denominator)


def effective_occupied_cells(
    assignments: ArrayLike,
    *,
    k: int | None = None,
    sample_weight: ArrayLike | None = None,
) -> float:
    labels = np.asarray(assignments, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0 or np.any(labels < 0):
        raise ValueError("assignments must be a nonempty vector of nonnegative labels")
    cell_count = int(labels.max()) + 1 if k is None else k
    if np.any(labels >= cell_count):
        raise ValueError("assignment is outside declared cell range")
    weights = _weights(sample_weight, len(labels))
    mass = np.bincount(labels, weights=weights, minlength=cell_count)
    probability = mass[mass > 0] / mass.sum()
    entropy = -np.sum(probability * np.log(probability))
    return float(np.exp(entropy))


def nearest_boundary_margin(batch: CoordinateBatch, codebook: Codebook) -> FloatArray:
    """Exact distance to the complement of each assigned Euclidean Voronoi cell."""

    values = codebook.require_compatible(batch)
    distances = codebook.squared_distances(batch)
    assignments = np.argmin(distances, axis=1)
    assigned_centers = codebook.centers[assignments]
    center_deltas = codebook.centers[None, :, :] - assigned_centers[:, None, :]
    denominators = 2.0 * np.linalg.norm(center_deltas, axis=2)
    numerators = distances - distances[np.arange(len(values)), assignments, None]
    margins = np.full_like(numerators, np.inf)
    valid = denominators > 0
    np.divide(numerators, denominators, out=margins, where=valid)
    margins[np.arange(len(values)), assignments] = np.inf
    return np.min(margins, axis=1)


def first_positive_boundary_crossing(
    batch: CoordinateBatch,
    directions: ArrayLike,
    codebook: Codebook,
    *,
    tolerance: float = 1e-12,
) -> FloatArray:
    """Return the first fitted-cell crossing distance along each unit direction."""

    values = codebook.require_compatible(batch)
    vectors = _matrix(directions, name="directions")
    if vectors.shape != values.shape:
        raise ValueError("directions must match the coordinate batch shape")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-8):
        raise ValueError("directions must have unit Euclidean norm")
    distances = codebook.squared_distances(batch)
    assignments = np.argmin(distances, axis=1)
    assigned_centers = codebook.centers[assignments]
    deltas = codebook.centers[None, :, :] - assigned_centers[:, None, :]
    projections = np.einsum("nd,nkd->nk", vectors, deltas)
    numerators = distances - distances[np.arange(len(values)), assignments, None]
    crossings = np.full_like(numerators, np.inf)
    valid = projections > tolerance
    np.divide(numerators, 2.0 * projections, out=crossings, where=valid)
    crossings[crossings < -tolerance] = np.inf
    crossings[np.arange(len(values)), assignments] = np.inf
    return np.min(crossings, axis=1)


def codebook_scale(
    codebook: Codebook,
    strategy: Literal["median_nearest_centroid_distance", "rms_centroid_radius"],
) -> float:
    centers = codebook.centers
    if strategy == "median_nearest_centroid_distance":
        distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        scale = float(np.median(np.min(distances, axis=1)))
    elif strategy == "rms_centroid_radius":
        centered = centers - centers.mean(axis=0)
        scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    else:  # pragma: no cover - protected by config
        raise ValueError(f"unknown codebook scale: {strategy}")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("codebook scale is undefined for coincident centroids")
    return scale


def _weights(sample_weight: ArrayLike | None, size: int) -> FloatArray | None:
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (size,) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("sample_weight must be a nonnegative vector with positive mass")
    return weights
