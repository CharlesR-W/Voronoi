"""Moment-matched Gaussian controls fitted without labels leaking into the codebook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GaussianMoments:
    mean: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if mean.ndim != 1 or covariance.shape != (len(mean), len(mean)):
            raise ValueError("invalid Gaussian moment shapes")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("covariance must be symmetric")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    def sample(self, count: int, rng: np.random.Generator) -> FloatArray:
        if count <= 0:
            raise ValueError("sample count must be positive")
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        transform = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
        return self.mean + rng.normal(size=(count, len(self.mean))) @ transform.T


def fit_gaussian(values: ArrayLike, *, shrinkage: float = 1e-3) -> GaussianMoments:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 2 or len(samples) < 2 or not 0 <= shrinkage <= 1:
        raise ValueError("values must be a matrix with at least two rows and valid shrinkage")
    mean = samples.mean(axis=0)
    centered = samples - mean
    covariance = centered.T @ centered / len(samples)
    isotropic = np.trace(covariance) / len(mean)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * isotropic * np.eye(len(mean))
    return GaussianMoments(mean, covariance)


@dataclass(frozen=True)
class ConditionalGaussianMoments:
    by_group: dict[int, GaussianMoments]
    group_probability: dict[int, float]

    def sample(self, count: int, rng: np.random.Generator) -> tuple[FloatArray, NDArray[np.int64]]:
        groups = np.asarray(sorted(self.by_group), dtype=np.int64)
        probability = np.asarray([self.group_probability[int(group)] for group in groups])
        labels = rng.choice(groups, size=count, p=probability)
        width = len(next(iter(self.by_group.values())).mean)
        values = np.empty((count, width), dtype=np.float64)
        for group in groups:
            mask = labels == group
            values[mask] = self.by_group[int(group)].sample(int(mask.sum()), rng)
        return values, labels


def fit_conditional_gaussian(
    values: ArrayLike, groups: ArrayLike, *, shrinkage: float = 1e-3
) -> ConditionalGaussianMoments:
    samples = np.asarray(values, dtype=np.float64)
    labels = np.asarray(groups, dtype=np.int64)
    if samples.ndim != 2 or labels.shape != (len(samples),):
        raise ValueError("values/groups have incompatible shapes")
    unique, counts = np.unique(labels, return_counts=True)
    if np.any(counts < 2):
        raise ValueError("each conditional Gaussian group needs at least two samples")
    moments = {
        int(group): fit_gaussian(samples[labels == group], shrinkage=shrinkage) for group in unique
    }
    probability = {
        int(group): float(count / len(samples)) for group, count in zip(unique, counts, strict=True)
    }
    return ConditionalGaussianMoments(moments, probability)
