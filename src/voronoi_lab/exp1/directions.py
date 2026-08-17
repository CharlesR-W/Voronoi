"""Declared perturbation-direction families in standardized coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import Codebook, CoordinateBatch

FloatArray = NDArray[np.float64]


def unit(vector: ArrayLike, *, tolerance: float = 1e-12) -> FloatArray:
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("direction must be one finite vector")
    norm = float(np.linalg.norm(values))
    if norm <= tolerance:
        raise ValueError("direction has zero norm")
    return values / norm


@dataclass(frozen=True)
class Direction:
    vector: FloatArray
    family: str
    metadata: dict[str, int | float | str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", unit(self.vector))


def empirical_chord(
    anchor: ArrayLike,
    anchor_assignment: int,
    observed: CoordinateBatch,
    observed_assignments: ArrayLike,
) -> Direction:
    z = np.asarray(anchor, dtype=np.float64)
    labels = np.asarray(observed_assignments, dtype=np.int64)
    if z.shape != (observed.values.shape[1],) or labels.shape != (len(observed.values),):
        raise ValueError("anchor/assignment shape mismatch")
    candidates = np.flatnonzero(labels != anchor_assignment)
    if not len(candidates):
        raise ValueError("empirical chord requires an observed point in another cell")
    distances = np.linalg.norm(observed.values[candidates] - z, axis=1)
    selected = int(candidates[np.argmin(distances)])
    return Direction(
        observed.values[selected] - z,
        "empirical_chord",
        {"target_index": selected, "target_assignment": int(labels[selected])},
    )


def boundary_normal(anchor: ArrayLike, assignment: int, codebook: Codebook) -> Direction:
    z = np.asarray(anchor, dtype=np.float64)
    if z.shape != (codebook.channels,) or not 0 <= assignment < codebook.k:
        raise ValueError("anchor or assignment does not match codebook")
    distances = np.sum((codebook.centers - z) ** 2, axis=1)
    delta = codebook.centers - codebook.centers[assignment]
    denominator = 2.0 * np.linalg.norm(delta, axis=1)
    margins = np.full(codebook.k, np.inf)
    valid = denominator > 0
    margins[valid] = (distances[valid] - distances[assignment]) / denominator[valid]
    margins[assignment] = np.inf
    target = int(np.argmin(margins))
    return Direction(
        codebook.centers[target] - codebook.centers[assignment],
        "boundary_normal",
        {"target_assignment": target, "margin": float(margins[target])},
    )


def local_neighbors(anchor: ArrayLike, observed: CoordinateBatch, count: int) -> FloatArray:
    z = np.asarray(anchor, dtype=np.float64)
    if z.shape != (observed.values.shape[1],) or count <= 1:
        raise ValueError("invalid anchor or neighbor count")
    count = min(count, len(observed.values))
    order = np.argsort(np.linalg.norm(observed.values - z, axis=1), kind="stable")
    return observed.values[order[:count]]


def off_cloud_direction(
    anchor: ArrayLike,
    observed: CoordinateBatch,
    *,
    neighbors: int,
    pca_rank: int,
    rng: np.random.Generator,
) -> Direction:
    z = np.asarray(anchor, dtype=np.float64)
    local = local_neighbors(z, observed, neighbors)
    centered = local - local.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rank = min(pca_rank, len(vh), observed.values.shape[1] - 1)
    if rank < 0:
        raise ValueError("off-cloud direction requires at least two channels")
    basis = vh[:rank]
    for _ in range(16):
        candidate = rng.normal(size=observed.values.shape[1])
        if rank:
            candidate = candidate - basis.T @ (basis @ candidate)
        norm = np.linalg.norm(candidate)
        if norm > 1e-10:
            return Direction(
                candidate / norm,
                "off_cloud",
                {"neighbors": len(local), "pca_rank": rank},
            )
    raise ValueError("could not sample a direction outside the local PCA span")


def local_covariance_direction(
    anchor: ArrayLike,
    observed: CoordinateBatch,
    *,
    neighbors: int,
    rng: np.random.Generator,
) -> Direction:
    z = np.asarray(anchor, dtype=np.float64)
    local = local_neighbors(z, observed, neighbors)
    centered = local - local.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(local) - 1)
    values, vectors = np.linalg.eigh(covariance)
    values = np.clip(values, 0.0, None)
    candidate = vectors @ (np.sqrt(values) * rng.normal(size=len(values)))
    return Direction(candidate, "local_covariance", {"neighbors": len(local)})
