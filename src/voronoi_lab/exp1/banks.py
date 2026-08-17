"""Deterministic, image-weighted probe and bootstrap banks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from voronoi_lab.core import JSONLike, SeedDeriver

IntArray = NDArray[np.int64]

_BANK_SEED_NAMESPACE = {"component": "voronoi_lab.exp1.banks", "schema_version": 1}


def semantic_seed(root_seed: int, *parts: JSONLike) -> int:
    """Derive a typed, stable NumPy seed from canonical semantic coordinates."""

    return SeedDeriver(root_seed).child(_BANK_SEED_NAMESPACE).derive(*parts, bits=64)


@dataclass(frozen=True)
class ProbeIndexPlan:
    roles: Mapping[str, IntArray]

    def __post_init__(self) -> None:
        normalized: dict[str, IntArray] = {}
        for role, values in self.roles.items():
            indices = np.asarray(values, dtype=np.int64)
            if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0):
                raise ValueError(f"role {role!r} must contain nonnegative dataset indices")
            if len(np.unique(indices)) != len(indices):
                raise ValueError(f"role {role!r} contains duplicate images")
            normalized[role] = indices
        object.__setattr__(self, "roles", normalized)

    def assert_disjoint(self, *roles: str) -> None:
        for i, left in enumerate(roles):
            for right in roles[i + 1 :]:
                overlap = np.intersect1d(self.roles[left], self.roles[right])
                if len(overlap):
                    raise ValueError(f"probe roles {left!r} and {right!r} overlap")


def make_probe_index_plan(
    *,
    train_size: int,
    test_size: int,
    fit_train_images: int,
    independent_fit_train_images: int,
    geometry_test_images: int,
    intervention_test_images: int,
    intervention_nested_in_geometry: bool,
    root_seed: int,
) -> ProbeIndexPlan:
    """Select fixed dataset indices while making split semantics explicit."""

    if min(train_size, test_size) <= 0:
        raise ValueError("dataset sizes must be positive")
    if fit_train_images + independent_fit_train_images > train_size:
        raise ValueError("training split is too small for disjoint codebook banks")
    required_test = geometry_test_images + (
        0 if intervention_nested_in_geometry else intervention_test_images
    )
    if required_test > test_size or (
        intervention_nested_in_geometry and intervention_test_images > geometry_test_images
    ):
        raise ValueError("test split is too small for requested probe banks")

    train_rng = np.random.default_rng(semantic_seed(root_seed, "probe", "train"))
    test_rng = np.random.default_rng(semantic_seed(root_seed, "probe", "test"))
    train = train_rng.permutation(train_size)
    test = test_rng.permutation(test_size)
    fit_end = fit_train_images
    geometry_end = geometry_test_images
    roles: dict[str, IntArray] = {
        "codebook_fit": train[:fit_end],
        "independent_codebook_fit": train[fit_end : fit_end + independent_fit_train_images],
        "geometry": test[:geometry_end],
    }
    if intervention_nested_in_geometry:
        roles["intervention"] = roles["geometry"][:intervention_test_images]
    else:
        roles["intervention"] = test[geometry_end : geometry_end + intervention_test_images]
    plan = ProbeIndexPlan(roles)
    plan.assert_disjoint("codebook_fit", "independent_codebook_fit")
    if not intervention_nested_in_geometry:
        plan.assert_disjoint("geometry", "intervention")
    return plan


@dataclass(frozen=True)
class SiteBank:
    image_ids: IntArray
    rows: IntArray
    columns: IntArray
    weights: NDArray[np.float64]
    height: int
    width: int

    def __post_init__(self) -> None:
        image_ids = np.asarray(self.image_ids, dtype=np.int64)
        rows = np.asarray(self.rows, dtype=np.int64)
        columns = np.asarray(self.columns, dtype=np.int64)
        weights = np.asarray(self.weights, dtype=np.float64)
        if not (image_ids.shape == rows.shape == columns.shape == weights.shape):
            raise ValueError("site-bank vectors must have matching shapes")
        if image_ids.ndim != 1 or len(image_ids) == 0:
            raise ValueError("site bank cannot be empty")
        if np.any(rows < 0) or np.any(rows >= self.height):
            raise ValueError("row is outside activation grid")
        if np.any(columns < 0) or np.any(columns >= self.width):
            raise ValueError("column is outside activation grid")
        if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("site weights must be positive and sum to one")
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "weights", weights)

    def assert_equal_image_weight(self, *, tolerance: float = 1e-12) -> None:
        unique, inverse = np.unique(self.image_ids, return_inverse=True)
        mass = np.bincount(inverse, weights=self.weights, minlength=len(unique))
        if np.max(np.abs(mass - 1.0 / len(unique))) > tolerance:
            raise ValueError("site weights do not give equal total mass to every image")


def make_site_bank(
    image_ids: ArrayLike,
    *,
    height: int,
    width: int,
    max_sites_per_image: int,
    root_seed: int,
    namespace: str,
) -> SiteBank:
    images = np.asarray(image_ids, dtype=np.int64)
    if images.ndim != 1 or len(images) == 0 or len(np.unique(images)) != len(images):
        raise ValueError("image_ids must be a nonempty unique vector")
    if min(height, width, max_sites_per_image) <= 0:
        raise ValueError("grid dimensions and site cap must be positive")
    count = min(height * width, max_sites_per_image)
    all_images: list[int] = []
    all_rows: list[int] = []
    all_columns: list[int] = []
    all_weights: list[float] = []
    for image_id in images:
        rng = np.random.default_rng(
            semantic_seed(root_seed, "sites", namespace, height, width, int(image_id))
        )
        flat = rng.choice(height * width, size=count, replace=False)
        all_images.extend([int(image_id)] * count)
        all_rows.extend((flat // width).tolist())
        all_columns.extend((flat % width).tolist())
        all_weights.extend([1.0 / (len(images) * count)] * count)
    bank = SiteBank(
        np.asarray(all_images),
        np.asarray(all_rows),
        np.asarray(all_columns),
        np.asarray(all_weights),
        height,
        width,
    )
    bank.assert_equal_image_weight()
    return bank


def make_image_bootstrap_plan(
    image_ids: ArrayLike, *, resamples: int, root_seed: int, namespace: str
) -> IntArray:
    images = np.asarray(image_ids, dtype=np.int64)
    if images.ndim != 1 or len(images) == 0 or len(np.unique(images)) != len(images):
        raise ValueError("bootstrap image_ids must be unique")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(semantic_seed(root_seed, "bootstrap", namespace))
    return rng.choice(images, size=(resamples, len(images)), replace=True)


def image_bootstrap_means(
    values: ArrayLike,
    token_image_ids: ArrayLike,
    bootstrap_images: ArrayLike,
) -> NDArray[np.float64]:
    """Average tokens within image before applying an image-level bootstrap."""

    observations = np.asarray(values, dtype=np.float64)
    token_ids = np.asarray(token_image_ids, dtype=np.int64)
    plan = np.asarray(bootstrap_images, dtype=np.int64)
    if observations.ndim != 1 or token_ids.shape != observations.shape or plan.ndim != 2:
        raise ValueError("invalid bootstrap array shapes")
    unique = np.unique(token_ids)
    per_image = {int(i): float(observations[token_ids == i].mean()) for i in unique}
    if any(int(i) not in per_image for i in plan.ravel()):
        raise ValueError("bootstrap plan references an image without observations")
    return np.asarray([[per_image[int(i)] for i in row] for row in plan]).mean(axis=1)
