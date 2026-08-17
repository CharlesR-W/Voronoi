"""Artifact-oriented, model-independent summaries for Experiment 1.

This module deliberately starts after model execution.  It combines the exact
geometry and intervention primitives into image-weighted records that can be
stored as JSON, bootstrapped without treating spatial sites as independent,
and joined to the legacy Tracking2 transplant table.  Every choice that is not
fixed by the README is represented by an argument or a method/version tag.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .banks import make_image_bootstrap_plan, semantic_seed
from .geometry import (
    Codebook,
    CoordinateBatch,
    effective_occupied_cells,
    nearest_boundary_margin,
    normalized_distortion,
)
from .interventions import BoundaryEnergySummary, summarize_boundary_energy
from .tracking2 import TransplantRow, resolve_cut

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
WeightedReducer: TypeAlias = Callable[[FloatArray, FloatArray], float]
BootstrapStatistic: TypeAlias = Callable[[FloatArray], float]


class ArtifactRecord:
    """Mixin for frozen records whose fields contain only JSON-compatible values."""

    def to_artifact(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]


def weighted_mean(values: FloatArray, weights: FloatArray) -> float:
    """A named reducer callers may select explicitly in their analysis spec."""

    return float(np.average(values, weights=weights))


@dataclass(frozen=True, slots=True)
class StaticAggregationSpec(ArtifactRecord):
    """Declared analysis choices for one static-geometry summary."""

    method_version: str
    evaluation_bank_id: str
    image_weighting: Literal["equal_image_weight_v1"]
    margin_scale: float
    margin_scale_id: str
    margin_reducer_id: str
    assignment_stability_id: str

    def __post_init__(self) -> None:
        _nonblank(self.method_version, name="method_version")
        _nonblank(self.evaluation_bank_id, name="evaluation_bank_id")
        if self.image_weighting != "equal_image_weight_v1":
            raise ValueError("unsupported image_weighting")
        _positive_finite(self.margin_scale, name="margin_scale")
        _nonblank(self.margin_scale_id, name="margin_scale_id")
        _nonblank(self.margin_reducer_id, name="margin_reducer_id")
        _nonblank(self.assignment_stability_id, name="assignment_stability_id")


@dataclass(frozen=True, slots=True)
class StaticImageSummary(ArtifactRecord):
    """Sufficient image-level statistics for equal-image aggregation/resampling."""

    image_id: int
    site_count: int
    distortion_numerator: float
    distortion_denominator_at_evaluation_mean: float
    coordinate_mean: tuple[float, ...]
    coordinate_variance_about_image_mean: float
    normalized_margin: float
    occupancy_mass: tuple[float, ...]
    assignment_stability: float

    def __post_init__(self) -> None:
        _nonnegative_int(self.image_id, name="image_id")
        _positive_int(self.site_count, name="site_count")
        _nonnegative_finite(self.distortion_numerator, name="distortion_numerator")
        _nonnegative_finite(
            self.distortion_denominator_at_evaluation_mean,
            name="distortion_denominator_at_evaluation_mean",
        )
        coordinate_mean = _finite_vector(self.coordinate_mean, name="coordinate_mean")
        _nonnegative_finite(
            self.coordinate_variance_about_image_mean,
            name="coordinate_variance_about_image_mean",
        )
        _nonnegative_finite(self.normalized_margin, name="normalized_margin")
        _probability(self.assignment_stability, name="assignment_stability")
        mass = _finite_vector(self.occupancy_mass, name="occupancy_mass")
        if np.any(mass < 0) or not np.isclose(mass.sum(), 1.0, atol=1e-12):
            raise ValueError("occupancy_mass must be nonnegative and sum to one")
        object.__setattr__(self, "occupancy_mass", tuple(float(value) for value in mass))
        object.__setattr__(
            self,
            "coordinate_mean",
            tuple(float(value) for value in coordinate_mean),
        )


@dataclass(frozen=True, slots=True)
class StaticGeometrySummary(ArtifactRecord):
    """Image-weighted static-geometry result plus its resampling units."""

    spec: StaticAggregationSpec
    metric: Literal["standardized", "raw"]
    fit_bank_id: str
    k: int
    image_count: int
    normalized_distortion: float
    normalized_margin: float
    effective_occupied_cells: float
    assignment_stability: float
    images: tuple[StaticImageSummary, ...]

    def __post_init__(self) -> None:
        _nonblank(self.fit_bank_id, name="fit_bank_id")
        if self.metric not in {"standardized", "raw"}:
            raise ValueError("metric must be 'standardized' or 'raw'")
        _positive_int(self.k, name="k")
        _positive_int(self.image_count, name="image_count")
        if self.k <= 1 or self.image_count != len(self.images):
            raise ValueError("invalid codebook or image count")
        if len({row.image_id for row in self.images}) != len(self.images):
            raise ValueError("static image summaries must have unique image IDs")
        if any(len(row.occupancy_mass) != self.k for row in self.images):
            raise ValueError("static image occupancy does not match k")
        channel_counts = {len(row.coordinate_mean) for row in self.images}
        if len(channel_counts) != 1:
            raise ValueError("static image coordinate means have inconsistent widths")
        _nonnegative_finite(self.normalized_distortion, name="normalized_distortion")
        _finite(self.normalized_margin, name="normalized_margin")
        _positive_finite(self.effective_occupied_cells, name="effective_occupied_cells")
        _probability(self.assignment_stability, name="assignment_stability")


def summarize_static_geometry(
    batch: CoordinateBatch,
    codebook: Codebook,
    image_ids: ArrayLike,
    *,
    site_weights: ArrayLike,
    assignment_stability_scores: ArrayLike,
    spec: StaticAggregationSpec,
    margin_reducer: WeightedReducer,
) -> StaticGeometrySummary:
    """Summarize static geometry while giving every image equal total mass.

    ``assignment_stability_scores`` must already encode the caller's declared
    label-alignment/bootstrap convention.  This layer only aggregates those
    per-site scores; it never silently assumes raw cluster labels are aligned.
    ``site_weights`` determine relative weight *within* each image and are then
    renormalized so all images contribute equally.
    """

    values = codebook.require_compatible(batch)
    count = len(values)
    images = _int_vector(image_ids, name="image_ids", size=count)
    raw_weights = _positive_vector(site_weights, name="site_weights", size=count)
    stability = _finite_vector(
        assignment_stability_scores,
        name="assignment_stability_scores",
        size=count,
    )
    if np.any((stability < 0) | (stability > 1)):
        raise ValueError("assignment_stability_scores must lie in [0, 1]")

    unique_images = np.unique(images)
    equal_weights = _equal_image_weights(images, raw_weights)
    assignments = codebook.assign(batch)
    squared_distances = codebook.squared_distances(batch)
    assigned_distance = squared_distances[np.arange(count), assignments]
    global_center = np.average(values, axis=0, weights=equal_weights)
    total_variance = np.sum((values - global_center) ** 2, axis=1)
    margins = nearest_boundary_margin(batch, codebook) / spec.margin_scale
    if not np.all(np.isfinite(margins)):
        raise ValueError("normalized margins must be finite; check for coincident centroids")

    image_rows: list[StaticImageSummary] = []
    for image_id in unique_images:
        mask = images == image_id
        local_weights = equal_weights[mask]
        local_weights = local_weights / local_weights.sum()
        local_center = np.average(values[mask], axis=0, weights=local_weights)
        occupancy = np.bincount(
            assignments[mask], weights=local_weights, minlength=codebook.k
        ).astype(np.float64)
        reduced_margin = float(margin_reducer(margins[mask], local_weights))
        if not np.isfinite(reduced_margin):
            raise ValueError("margin_reducer returned a nonfinite value")
        image_rows.append(
            StaticImageSummary(
                image_id=int(image_id),
                site_count=int(mask.sum()),
                distortion_numerator=float(
                    np.average(assigned_distance[mask], weights=local_weights)
                ),
                distortion_denominator_at_evaluation_mean=float(
                    np.average(total_variance[mask], weights=local_weights)
                ),
                coordinate_mean=tuple(float(value) for value in local_center),
                coordinate_variance_about_image_mean=float(
                    np.average(
                        np.sum((values[mask] - local_center) ** 2, axis=1),
                        weights=local_weights,
                    )
                ),
                normalized_margin=reduced_margin,
                occupancy_mass=tuple(float(value) for value in occupancy),
                assignment_stability=float(np.average(stability[mask], weights=local_weights)),
            )
        )

    # Evaluate the existing primitives using the exact equal-image weights as a
    # cross-module invariant, then retain per-image sufficient statistics above.
    distortion = normalized_distortion(batch, codebook, sample_weight=equal_weights)
    occupancy = effective_occupied_cells(
        assignments,
        k=codebook.k,
        sample_weight=equal_weights,
    )
    image_margin = float(np.mean([row.normalized_margin for row in image_rows]))
    image_stability = float(np.mean([row.assignment_stability for row in image_rows]))

    row_distortion = float(
        np.mean([row.distortion_numerator for row in image_rows])
        / np.mean([row.distortion_denominator_at_evaluation_mean for row in image_rows])
    )
    if not np.isclose(distortion, row_distortion, rtol=1e-12, atol=1e-12):
        raise RuntimeError("static image rows do not reproduce equal-image distortion")

    return StaticGeometrySummary(
        spec=spec,
        metric=codebook.metric,
        fit_bank_id=codebook.fit_bank_id,
        k=codebook.k,
        image_count=len(image_rows),
        normalized_distortion=distortion,
        normalized_margin=image_margin,
        effective_occupied_cells=occupancy,
        assignment_stability=image_stability,
        images=tuple(image_rows),
    )


def static_distortion_features(summary: StaticGeometrySummary) -> FloatArray:
    """Return per-image sufficient statistics for a recomputed-mean bootstrap.

    Columns are assigned squared distance, within-image coordinate variance,
    then the coordinate mean vector.  A resample can therefore recompute
    ``E||z - E[z]||^2`` stably instead of freezing the original bank mean.
    """

    return np.asarray(
        [
            (
                row.distortion_numerator,
                row.coordinate_variance_about_image_mean,
                *row.coordinate_mean,
            )
            for row in summary.images
        ],
        dtype=np.float64,
    )


def distortion_from_image_features(rows: FloatArray) -> float:
    """Compute distortion from rows returned by :func:`static_distortion_features`."""

    features = np.asarray(rows, dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape[0] == 0
        or features.shape[1] < 3
        or not np.all(np.isfinite(features))
    ):
        raise ValueError("distortion features must be a finite matrix with at least 3 columns")
    numerator = float(features[:, 0].mean())
    image_means = features[:, 2:]
    coordinate_mean = image_means.mean(axis=0)
    denominator = float(
        features[:, 1].mean() + np.mean(np.sum((image_means - coordinate_mean) ** 2, axis=1))
    )
    if denominator <= 0:
        raise ValueError("distortion is undefined for a constant resampled bank")
    return numerator / denominator


@dataclass(frozen=True, slots=True)
class BootstrapSpec(ArtifactRecord):
    """Fully declared deterministic percentile-bootstrap choices."""

    method_version: str
    root_seed: int
    namespace: str
    input_artifact_id: str
    resamples: int
    confidence_level: float
    quantile_method: str
    statistic_id: str

    def __post_init__(self) -> None:
        _nonblank(self.method_version, name="method_version")
        _nonblank(self.namespace, name="namespace")
        _nonblank(self.input_artifact_id, name="input_artifact_id")
        _nonblank(self.quantile_method, name="quantile_method")
        _nonblank(self.statistic_id, name="statistic_id")
        _nonnegative_int(self.root_seed, name="root_seed")
        _positive_int(self.resamples, name="resamples")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between zero and one")


@dataclass(frozen=True, slots=True)
class BootstrapInterval(ArtifactRecord):
    """Point estimate, percentile interval, and reproducible replicate values."""

    spec: BootstrapSpec
    image_count: int
    point_estimate: float
    lower: float
    upper: float
    replicate_values: tuple[float, ...]

    def __post_init__(self) -> None:
        _positive_int(self.image_count, name="image_count")
        if len(self.replicate_values) != self.spec.resamples:
            raise ValueError("bootstrap result count does not match its specification")
        for name in ("point_estimate", "lower", "upper"):
            _finite(getattr(self, name), name=name)
        replicates = _finite_vector(self.replicate_values, name="replicate_values")
        if self.lower > self.upper:
            raise ValueError("bootstrap lower endpoint exceeds upper endpoint")
        object.__setattr__(
            self,
            "replicate_values",
            tuple(float(value) for value in replicates),
        )


def deterministic_image_bootstrap_interval(
    image_ids: ArrayLike,
    image_features: ArrayLike,
    *,
    statistic: BootstrapStatistic,
    spec: BootstrapSpec,
) -> BootstrapInterval:
    """Bootstrap rows that have already been reduced to exactly one row/image.

    Keeping within-image reduction outside this function makes the resampling
    unit auditable and lets one callable handle scalar means, ratios of paired
    sufficient statistics, or target-control contrasts.
    """

    features = np.asarray(image_features, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    if (
        features.ndim != 2
        or not len(features)
        or features.shape[1] == 0
        or not np.all(np.isfinite(features))
    ):
        raise ValueError("image_features must be a nonempty finite matrix")
    images = _int_vector(image_ids, name="image_ids", size=len(features))
    if len(np.unique(images)) != len(images):
        raise ValueError("bootstrap input must contain exactly one row per image")

    point = _evaluate_statistic(statistic, features)
    plan = make_image_bootstrap_plan(
        images,
        resamples=spec.resamples,
        root_seed=spec.root_seed,
        namespace=spec.namespace,
    )
    row_by_image = {int(image_id): index for index, image_id in enumerate(images)}
    replicates = np.asarray(
        [
            _evaluate_statistic(
                statistic,
                features[[row_by_image[int(image_id)] for image_id in bootstrap_row]],
            )
            for bootstrap_row in plan
        ],
        dtype=np.float64,
    )
    tail = (1.0 - spec.confidence_level) / 2.0
    try:
        lower, upper = np.quantile(
            replicates,
            [tail, 1.0 - tail],
            method=spec.quantile_method,
        )
    except ValueError as exc:
        raise ValueError(
            f"unsupported bootstrap quantile_method: {spec.quantile_method!r}"
        ) from exc
    return BootstrapInterval(
        spec=spec,
        image_count=len(images),
        point_estimate=point,
        lower=float(lower),
        upper=float(upper),
        replicate_values=tuple(float(value) for value in replicates),
    )


@dataclass(frozen=True, slots=True)
class BoundaryPathSpec(ArtifactRecord):
    method_version: str
    path_bank_id: str
    shift_plan_id: str
    direction_family: str
    boundary_window: tuple[float, float]
    energy_weighting: Literal["discrete_grid_mass", "continuous_path_trapezoid"]
    path_weighting: Literal["weighted_within_image_equal_images_v1"]
    null_method: Literal["within_path_circular_energy_shift_v1"]
    comparison_statistic: Literal["fraction_near_boundary"]

    def __post_init__(self) -> None:
        _nonblank(self.method_version, name="method_version")
        _nonblank(self.path_bank_id, name="path_bank_id")
        _nonblank(self.shift_plan_id, name="shift_plan_id")
        _nonblank(self.direction_family, name="direction_family")
        if self.path_weighting != "weighted_within_image_equal_images_v1":
            raise ValueError("unsupported path_weighting")
        if self.energy_weighting not in {
            "discrete_grid_mass",
            "continuous_path_trapezoid",
        }:
            raise ValueError("unsupported energy_weighting")
        if self.null_method != "within_path_circular_energy_shift_v1":
            raise ValueError("unsupported null_method")
        if self.comparison_statistic != "fraction_near_boundary":
            raise ValueError("unsupported comparison_statistic")
        if (
            len(self.boundary_window) != 2
            or not np.all(np.isfinite(self.boundary_window))
            or self.boundary_window[0] > self.boundary_window[1]
        ):
            raise ValueError("boundary_window must be a finite ordered pair")


@dataclass(frozen=True, slots=True)
class BoundaryImageSummary(ArtifactRecord):
    image_id: int
    path_count: int
    fraction_near_boundary: float
    energy_80_lower: float
    energy_80_upper: float
    peak_offset: float

    def __post_init__(self) -> None:
        _nonnegative_int(self.image_id, name="image_id")
        _positive_int(self.path_count, name="path_count")
        _probability(self.fraction_near_boundary, name="fraction_near_boundary")
        for name in ("energy_80_lower", "energy_80_upper", "peak_offset"):
            _finite(getattr(self, name), name=name)
        if self.energy_80_lower > self.energy_80_upper:
            raise ValueError("energy interval endpoints are reversed")


@dataclass(frozen=True, slots=True)
class ShiftNullComparison(ArtifactRecord):
    method_version: str
    observed: float
    null_mean: float
    observed_minus_null_mean: float
    empirical_percentile: float
    null_values: tuple[float, ...]

    def __post_init__(self) -> None:
        _nonblank(self.method_version, name="method_version")
        for name in ("observed", "null_mean", "observed_minus_null_mean"):
            _finite(getattr(self, name), name=name)
        _probability(self.empirical_percentile, name="empirical_percentile")
        values = _finite_vector(self.null_values, name="null_values")
        object.__setattr__(self, "null_values", tuple(float(value) for value in values))


@dataclass(frozen=True, slots=True)
class BoundaryPathResponseSummary(ArtifactRecord):
    spec: BoundaryPathSpec
    image_count: int
    path_count: int
    images: tuple[BoundaryImageSummary, ...]
    shifted_null: ShiftNullComparison

    def __post_init__(self) -> None:
        _positive_int(self.image_count, name="image_count")
        _positive_int(self.path_count, name="path_count")
        if self.image_count != len(self.images) or self.path_count != sum(
            row.path_count for row in self.images
        ):
            raise ValueError("boundary path counts are inconsistent")
        if len({row.image_id for row in self.images}) != len(self.images):
            raise ValueError("boundary image summaries must have unique IDs")


def make_boundary_shift_plan(
    *,
    path_count: int,
    path_length: int,
    draws: int,
    allowed_shifts: Sequence[int],
    root_seed: int,
    namespace: str,
) -> IntArray:
    """Create a deterministic plan from an explicitly declared shift set."""

    _positive_int(path_count, name="path_count")
    _positive_int(path_length, name="path_length")
    _positive_int(draws, name="draws")
    _nonnegative_int(root_seed, name="root_seed")
    if path_length < 2:
        raise ValueError("path_count/draws must be positive and path_length at least two")
    _nonblank(namespace, name="namespace")
    shifts = _int_vector(allowed_shifts, name="allowed_shifts")
    if np.any(np.mod(shifts, path_length) == 0):
        raise ValueError("allowed_shifts cannot contain an identity circular shift")
    rng = np.random.default_rng(
        semantic_seed(root_seed, "boundary_shift", namespace, path_count, path_length)
    )
    return rng.choice(shifts, size=(draws, path_count), replace=True).astype(np.int64)


def summarize_boundary_paths(
    coordinates: ArrayLike,
    energy: ArrayLike,
    path_image_ids: ArrayLike,
    *,
    path_weights: ArrayLike,
    shift_plan: ArrayLike,
    spec: BoundaryPathSpec,
) -> BoundaryPathResponseSummary:
    """Aggregate path responses by image and compare to circular-shift nulls."""

    r = _finite_vector(coordinates, name="coordinates")
    if len(r) < 2 or np.any(np.diff(r) <= 0):
        raise ValueError("coordinates must be strictly increasing with at least two entries")
    paths = np.asarray(energy, dtype=np.float64)
    if (
        paths.ndim != 2
        or paths.shape[1] != len(r)
        or not len(paths)
        or not np.all(np.isfinite(paths))
        or np.any(paths < 0)
        or np.any(paths.sum(axis=1) <= 0)
    ):
        raise ValueError("energy must be a finite nonnegative path matrix with positive row mass")
    images = _int_vector(path_image_ids, name="path_image_ids", size=len(paths))
    raw_weights = _positive_vector(path_weights, name="path_weights", size=len(paths))
    equal_weights = _equal_image_weights(images, raw_weights)
    shifts = _integer_matrix(shift_plan, name="shift_plan")
    if shifts.ndim != 2 or shifts.shape[1] != len(paths) or not len(shifts):
        raise ValueError("shift_plan must have shape (draws, paths)")
    if np.any(np.mod(shifts, len(r)) == 0):
        raise ValueError("shift_plan must break boundary alignment on every path")

    path_summaries = [
        summarize_boundary_energy(
            r,
            row,
            window=spec.boundary_window,
            weighting=spec.energy_weighting,
        )
        for row in paths
    ]
    image_rows = _aggregate_boundary_images(images, equal_weights, path_summaries)
    observed = float(np.mean([row.fraction_near_boundary for row in image_rows]))

    null_values: list[float] = []
    for draw in shifts:
        shifted_summaries = [
            summarize_boundary_energy(
                r,
                np.roll(paths[path_index], int(draw[path_index])),
                window=spec.boundary_window,
                weighting=spec.energy_weighting,
            )
            for path_index in range(len(paths))
        ]
        shifted_images = _aggregate_boundary_images(images, equal_weights, shifted_summaries)
        null_values.append(float(np.mean([row.fraction_near_boundary for row in shifted_images])))
    null_array = np.asarray(null_values, dtype=np.float64)
    null_mean = float(null_array.mean())
    comparison = ShiftNullComparison(
        method_version="empirical_cdf_less_equal_v1",
        observed=observed,
        null_mean=null_mean,
        observed_minus_null_mean=observed - null_mean,
        empirical_percentile=float(np.mean(null_array <= observed)),
        null_values=tuple(null_values),
    )
    return BoundaryPathResponseSummary(
        spec=spec,
        image_count=len(image_rows),
        path_count=len(paths),
        images=tuple(image_rows),
        shifted_null=comparison,
    )


@dataclass(frozen=True, slots=True)
class InterventionMetricObservation(ArtifactRecord):
    """One already image-reduced snapping/recovery metric observation."""

    image_id: int
    arm: str
    metric: str
    value: float

    def __post_init__(self) -> None:
        _nonnegative_int(self.image_id, name="image_id")
        _nonblank(self.arm, name="arm")
        _nonblank(self.metric, name="metric")
        _finite(self.value, name="value")


@dataclass(frozen=True, slots=True)
class PairedControlContrast(ArtifactRecord):
    metric: str
    target_arm: str
    control_arm: str
    favorable_direction: Literal["lower", "higher"]
    image_ids: tuple[int, ...]
    target_mean: float
    control_mean: float
    target_minus_control: float
    favorable_advantage: float
    per_image_target_minus_control: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("metric", "target_arm", "control_arm"):
            _nonblank(getattr(self, name), name=name)
        if not self.image_ids or len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("contrast image IDs must be nonempty and unique")
        if any(not _is_nonnegative_int(image_id) for image_id in self.image_ids):
            raise ValueError("contrast image IDs must be nonnegative integers")
        if self.favorable_direction not in {"lower", "higher"}:
            raise ValueError("favorable_direction must be 'lower' or 'higher'")
        differences = _finite_vector(
            self.per_image_target_minus_control,
            name="per_image_target_minus_control",
            size=len(self.image_ids),
        )
        for name in (
            "target_mean",
            "control_mean",
            "target_minus_control",
            "favorable_advantage",
        ):
            _finite(getattr(self, name), name=name)
        object.__setattr__(
            self,
            "per_image_target_minus_control",
            tuple(float(value) for value in differences),
        )


@dataclass(frozen=True, slots=True)
class SnappingRecoverySummary(ArtifactRecord):
    """Paired image-level comparisons for one explicitly named protocol stratum."""

    method_version: str
    protocol_id: str
    stratum_id: str
    target_arm: str
    control_arms: tuple[str, ...]
    metric_directions: tuple[tuple[str, Literal["lower", "higher"]], ...]
    contrasts: tuple[PairedControlContrast, ...]

    def __post_init__(self) -> None:
        for name in ("method_version", "protocol_id", "stratum_id", "target_arm"):
            _nonblank(getattr(self, name), name=name)
        if not self.control_arms or len(set(self.control_arms)) != len(self.control_arms):
            raise ValueError("control_arms must be nonempty and unique")
        if self.target_arm in self.control_arms:
            raise ValueError("target_arm cannot also be a control")
        if not self.metric_directions or not self.contrasts:
            raise ValueError("summary must contain metrics and contrasts")


def summarize_snapping_recovery_controls(
    observations: Iterable[InterventionMetricObservation],
    *,
    target_arm: str,
    control_arms: Sequence[str],
    metric_directions: Mapping[str, Literal["lower", "higher"]],
    protocol_id: str,
    stratum_id: str,
    method_version: str,
) -> SnappingRecoverySummary:
    """Build paired target-control summaries with image as the comparison unit.

    Each metric must form a complete ``image x arm`` grid.  ``lower`` metrics
    (for example predictive KL or κ) report ``control - target`` as favorable
    advantage; ``higher`` metrics (for example clean-cell recovery) report
    ``target - control``.  The raw target-minus-control difference is retained.
    """

    _nonblank(target_arm, name="target_arm")
    controls = tuple(control_arms)
    if not controls or len(set(controls)) != len(controls) or target_arm in controls:
        raise ValueError("control_arms must be unique, nonempty, and exclude target_arm")
    directions = tuple(sorted(metric_directions.items()))
    if not directions:
        raise ValueError("metric_directions cannot be empty")
    if any(direction not in {"lower", "higher"} for _, direction in directions):
        raise ValueError("metric directions must be 'lower' or 'higher'")
    for metric, _ in directions:
        _nonblank(metric, name="metric name")

    rows = tuple(observations)
    if not rows:
        raise ValueError("observations cannot be empty")
    expected_metrics = {metric for metric, _ in directions}
    if {row.metric for row in rows} != expected_metrics:
        raise ValueError("observed metrics do not exactly match metric_directions")
    allowed_arms = {target_arm, *controls}
    if {row.arm for row in rows} != allowed_arms:
        raise ValueError("observed arms do not exactly match target and control arms")

    lookup: dict[tuple[str, str, int], float] = {}
    image_sets: dict[str, set[int]] = {}
    for row in rows:
        key = (row.metric, row.arm, row.image_id)
        if key in lookup:
            raise ValueError(f"duplicate intervention observation: {key!r}")
        lookup[key] = row.value
        image_sets.setdefault(row.metric, set()).add(row.image_id)

    distinct_image_sets = {tuple(sorted(image_ids)) for image_ids in image_sets.values()}
    if len(distinct_image_sets) != 1:
        raise ValueError("all metrics in a protocol stratum must use the same paired images")

    contrasts: list[PairedControlContrast] = []
    for metric, direction in directions:
        image_ids = tuple(sorted(image_sets[metric]))
        expected_keys = {(metric, arm, image_id) for arm in allowed_arms for image_id in image_ids}
        metric_keys = {key for key in lookup if key[0] == metric}
        if metric_keys != expected_keys:
            raise ValueError(f"metric {metric!r} does not form a complete paired image/arm grid")
        target_values = np.asarray(
            [lookup[(metric, target_arm, image_id)] for image_id in image_ids],
            dtype=np.float64,
        )
        for control in controls:
            control_values = np.asarray(
                [lookup[(metric, control, image_id)] for image_id in image_ids],
                dtype=np.float64,
            )
            difference = target_values - control_values
            target_mean = float(target_values.mean())
            control_mean = float(control_values.mean())
            target_minus_control = float(difference.mean())
            advantage = -target_minus_control if direction == "lower" else target_minus_control
            contrasts.append(
                PairedControlContrast(
                    metric=metric,
                    target_arm=target_arm,
                    control_arm=control,
                    favorable_direction=direction,
                    image_ids=image_ids,
                    target_mean=target_mean,
                    control_mean=control_mean,
                    target_minus_control=target_minus_control,
                    favorable_advantage=advantage,
                    per_image_target_minus_control=tuple(float(value) for value in difference),
                )
            )

    return SnappingRecoverySummary(
        method_version=method_version,
        protocol_id=protocol_id,
        stratum_id=stratum_id,
        target_arm=target_arm,
        control_arms=controls,
        metric_directions=directions,
        contrasts=tuple(contrasts),
    )


@dataclass(frozen=True, slots=True)
class GeometryStatisticRecord(ArtifactRecord):
    """Normalized long-form geometry statistic ready for joins/reports."""

    seed: int
    checkpoint_epoch: int
    cut_index: int
    cut_name: str
    statistic: str
    value: float
    coordinate_metric: Literal["standardized", "raw"]
    k: int
    analysis_version: str
    artifact_id: str

    def __post_init__(self) -> None:
        for name in ("seed", "checkpoint_epoch", "cut_index"):
            _nonnegative_int(getattr(self, name), name=name)
        _positive_int(self.k, name="k")
        if self.k <= 1:
            raise ValueError("k must exceed one")
        if self.coordinate_metric not in {"standardized", "raw"}:
            raise ValueError("coordinate_metric must be 'standardized' or 'raw'")
        cut = resolve_cut(self.cut_index)
        if self.cut_name != cut.name:
            raise ValueError("geometry cut_index and cut_name are inconsistent")
        for name in ("statistic", "analysis_version", "artifact_id"):
            _nonblank(getattr(self, name), name=name)
        _finite(self.value, name="value")


@dataclass(frozen=True, slots=True)
class GeometryTransplantJoinRecord(ArtifactRecord):
    """Descriptive-only paired row; it is not an inferential association claim."""

    join_version: str
    interpretation: Literal["descriptive_only"]
    legacy_lineage: Literal["exploratory_legacy"]
    seed: int
    target_epoch: int
    source_epoch: int
    cut_index: int
    cut_name: str
    geometry_statistic: str
    geometry_value: float
    coordinate_metric: Literal["standardized", "raw"]
    k: int
    geometry_analysis_version: str
    geometry_artifact_id: str
    transplant_statistic: Literal["delta_loss", "delta_error"]
    transplant_value: float

    def __post_init__(self) -> None:
        _nonblank(self.join_version, name="join_version")
        if self.interpretation != "descriptive_only":
            raise ValueError("joined legacy rows must remain descriptive_only")
        if self.legacy_lineage != "exploratory_legacy":
            raise ValueError("joined legacy rows must retain exploratory_legacy lineage")
        if self.coordinate_metric not in {"standardized", "raw"}:
            raise ValueError("coordinate_metric must be 'standardized' or 'raw'")
        if self.transplant_statistic not in {"delta_loss", "delta_error"}:
            raise ValueError("unsupported transplant_statistic")
        for name in ("seed", "target_epoch", "source_epoch", "cut_index"):
            _nonnegative_int(getattr(self, name), name=name)
        _positive_int(self.k, name="k")
        for name in (
            "geometry_statistic",
            "geometry_analysis_version",
            "geometry_artifact_id",
        ):
            _nonblank(getattr(self, name), name=name)
        for name in ("geometry_value", "transplant_value"):
            _finite(getattr(self, name), name=name)
        cut = resolve_cut(self.cut_index)
        if self.cut_name != cut.name:
            raise ValueError("joined cut_index and cut_name are inconsistent")


def join_geometry_with_legacy_transplants(
    geometry_rows: Iterable[GeometryStatisticRecord],
    transplant_rows: Iterable[TransplantRow],
    *,
    transplant_statistic: Literal["delta_loss", "delta_error"],
    source_policy: Literal["checkpoint_only"],
    unmatched_policy: Literal["error", "drop"],
    join_version: str,
) -> tuple[GeometryTransplantJoinRecord, ...]:
    """Join geometry at epoch ``s`` to the matching checkpoint transplant ``s→target``.

    Random transplant rows have no matching geometry epoch and are explicitly
    excluded by ``source_policy='checkpoint_only'``.  The returned lineage and
    interpretation labels prevent this legacy, one-seed comparison from being
    mistaken for confirmatory evidence.
    """

    _nonblank(join_version, name="join_version")
    if source_policy != "checkpoint_only":  # pragma: no cover - protected by typing
        raise ValueError("only checkpoint_only source_policy is supported")
    if unmatched_policy not in {"error", "drop"}:
        raise ValueError("unmatched_policy must be 'error' or 'drop'")
    if transplant_statistic not in {"delta_loss", "delta_error"}:
        raise ValueError("transplant_statistic must be 'delta_loss' or 'delta_error'")
    geometry = tuple(geometry_rows)
    transplants = tuple(row for row in transplant_rows if row.source_kind == "checkpoint")
    if not geometry or not transplants:
        raise ValueError("geometry and checkpoint transplant rows must both be nonempty")

    transplant_by_key: dict[tuple[int, int, int], TransplantRow] = {}
    for row in transplants:
        if row.source_epoch is None:  # protected by TransplantRow validation
            raise ValueError("checkpoint transplant row lacks source_epoch")
        key = (row.seed, row.source_epoch, row.cut_index)
        if key in transplant_by_key:
            raise ValueError(f"duplicate checkpoint transplant join key: {key!r}")
        transplant_by_key[key] = row

    geometry_keys: set[tuple[int, int, int]] = set()
    exact_geometry_keys: set[tuple[int, int, int, str, str, int, str]] = set()
    joined: list[GeometryTransplantJoinRecord] = []
    missing_geometry_keys: list[tuple[int, int, int]] = []
    for row in geometry:
        key = (row.seed, row.checkpoint_epoch, row.cut_index)
        exact_key = (
            *key,
            row.statistic,
            row.coordinate_metric,
            row.k,
            row.analysis_version,
        )
        if exact_key in exact_geometry_keys:
            raise ValueError(f"duplicate geometry join row: {exact_key!r}")
        exact_geometry_keys.add(exact_key)
        geometry_keys.add(key)
        transplant = transplant_by_key.get(key)
        if transplant is None:
            missing_geometry_keys.append(key)
            continue
        joined.append(
            GeometryTransplantJoinRecord(
                join_version=join_version,
                interpretation="descriptive_only",
                legacy_lineage="exploratory_legacy",
                seed=row.seed,
                target_epoch=transplant.target_epoch,
                source_epoch=row.checkpoint_epoch,
                cut_index=row.cut_index,
                cut_name=row.cut_name,
                geometry_statistic=row.statistic,
                geometry_value=row.value,
                coordinate_metric=row.coordinate_metric,
                k=row.k,
                geometry_analysis_version=row.analysis_version,
                geometry_artifact_id=row.artifact_id,
                transplant_statistic=transplant_statistic,
                transplant_value=float(getattr(transplant, transplant_statistic)),
            )
        )

    missing_transplant_keys = sorted(set(transplant_by_key) - geometry_keys)
    if unmatched_policy == "error" and (missing_geometry_keys or missing_transplant_keys):
        raise ValueError(
            "unmatched geometry/transplant join keys: "
            f"geometry_without_transplant={sorted(set(missing_geometry_keys))}, "
            f"transplant_without_geometry={missing_transplant_keys}"
        )
    return tuple(
        sorted(
            joined,
            key=lambda row: (
                row.seed,
                row.source_epoch,
                row.cut_index,
                row.geometry_statistic,
                row.coordinate_metric,
                row.k,
            ),
        )
    )


def _aggregate_boundary_images(
    images: IntArray,
    equal_weights: FloatArray,
    summaries: Sequence[BoundaryEnergySummary],
) -> list[BoundaryImageSummary]:
    rows: list[BoundaryImageSummary] = []
    for image_id in np.unique(images):
        indices = np.flatnonzero(images == image_id)
        weights = equal_weights[indices]
        weights = weights / weights.sum()
        rows.append(
            BoundaryImageSummary(
                image_id=int(image_id),
                path_count=len(indices),
                fraction_near_boundary=float(
                    np.average(
                        [summaries[index].fraction_near_boundary for index in indices],
                        weights=weights,
                    )
                ),
                energy_80_lower=float(
                    np.average(
                        [summaries[index].energy_80_interval[0] for index in indices],
                        weights=weights,
                    )
                ),
                energy_80_upper=float(
                    np.average(
                        [summaries[index].energy_80_interval[1] for index in indices],
                        weights=weights,
                    )
                ),
                peak_offset=float(
                    np.average(
                        [summaries[index].peak_offset for index in indices],
                        weights=weights,
                    )
                ),
            )
        )
    return rows


def _evaluate_statistic(statistic: BootstrapStatistic, rows: FloatArray) -> float:
    value = float(statistic(rows))
    if not np.isfinite(value):
        raise ValueError("bootstrap statistic returned a nonfinite value")
    return value


def _equal_image_weights(images: IntArray, weights: FloatArray) -> FloatArray:
    unique = np.unique(images)
    result = np.empty_like(weights, dtype=np.float64)
    for image_id in unique:
        mask = images == image_id
        total = float(weights[mask].sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"image {int(image_id)} has no positive weight")
        result[mask] = weights[mask] / (len(unique) * total)
    if not np.isclose(result.sum(), 1.0, atol=1e-12):  # pragma: no cover - arithmetic guard
        raise RuntimeError("equal-image weights do not sum to one")
    return result


def _integer_matrix(values: ArrayLike, *, name: str) -> IntArray:
    raw = np.asarray(values)
    try:
        finite = bool(np.all(np.isfinite(raw)))
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty finite integer matrix") from exc
    if raw.ndim != 2 or raw.size == 0 or not finite:
        raise ValueError(f"{name} must be a nonempty finite integer matrix")
    converted = raw.astype(np.int64)
    if not np.array_equal(raw, converted):
        raise ValueError(f"{name} must contain integers")
    return converted


def _int_vector(values: ArrayLike, *, name: str, size: int | None = None) -> IntArray:
    raw = np.asarray(values)
    try:
        finite = bool(np.all(np.isfinite(raw)))
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty finite integer vector") from exc
    if raw.ndim != 1 or len(raw) == 0 or not finite:
        raise ValueError(f"{name} must be a nonempty finite integer vector")
    converted = raw.astype(np.int64)
    if not np.array_equal(raw, converted):
        raise ValueError(f"{name} must contain integers")
    if size is not None and len(converted) != size:
        raise ValueError(f"{name} has length {len(converted)}, expected {size}")
    return converted


def _finite_vector(
    values: ArrayLike,
    *,
    name: str,
    size: int | None = None,
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    if size is not None and len(array) != size:
        raise ValueError(f"{name} has length {len(array)}, expected {size}")
    return array


def _positive_vector(values: ArrayLike, *, name: str, size: int) -> FloatArray:
    array = _finite_vector(values, name=name, size=size)
    if np.any(array <= 0):
        raise ValueError(f"{name} must contain only positive values")
    return array


def _nonblank(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be blank")


def _finite(value: float, *, name: str) -> None:
    try:
        finite = bool(np.isfinite(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not finite:
        raise ValueError(f"{name} must be finite")


def _is_nonnegative_int(value: object) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and value >= 0
    )


def _nonnegative_int(value: object, *, name: str) -> None:
    if not _is_nonnegative_int(value):
        raise ValueError(f"{name} must be a nonnegative integer")


def _positive_int(value: object, *, name: str) -> None:
    if not _is_nonnegative_int(value) or value <= 0:  # type: ignore[operator]
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_finite(value: float, *, name: str) -> None:
    _finite(value, name=name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _positive_finite(value: float, *, name: str) -> None:
    _finite(value, name=name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _probability(value: float, *, name: str) -> None:
    _finite(value, name=name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")


__all__ = [
    "BootstrapInterval",
    "BootstrapSpec",
    "BoundaryImageSummary",
    "BoundaryPathResponseSummary",
    "BoundaryPathSpec",
    "GeometryStatisticRecord",
    "GeometryTransplantJoinRecord",
    "InterventionMetricObservation",
    "PairedControlContrast",
    "ShiftNullComparison",
    "SnappingRecoverySummary",
    "StaticAggregationSpec",
    "StaticGeometrySummary",
    "StaticImageSummary",
    "deterministic_image_bootstrap_interval",
    "distortion_from_image_features",
    "join_geometry_with_legacy_transplants",
    "make_boundary_shift_plan",
    "static_distortion_features",
    "summarize_boundary_paths",
    "summarize_snapping_recovery_controls",
    "summarize_static_geometry",
    "weighted_mean",
]
