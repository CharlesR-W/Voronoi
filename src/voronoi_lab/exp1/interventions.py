"""Pure intervention mechanics and path summaries, independent of any model backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoundaryEnergyWeighting = Literal["discrete_grid_mass", "continuous_path_trapezoid"]


def partial_snap(
    clean: ArrayLike, centers: ArrayLike, clean_assignments: ArrayLike, alpha: float
) -> FloatArray:
    """Move toward the clean assigned centroid without ever recomputing assignment."""

    values = np.asarray(clean, dtype=np.float64)
    centroids = np.asarray(centers, dtype=np.float64)
    assignments = np.asarray(clean_assignments, dtype=np.int64)
    if values.ndim != 2 or centroids.ndim != 2 or values.shape[1] != centroids.shape[1]:
        raise ValueError("clean states and centroids must be compatible matrices")
    if (
        assignments.shape != (len(values),)
        or np.any(assignments < 0)
        or np.any(assignments >= len(centroids))
    ):
        raise ValueError("clean_assignments do not match states/centroids")
    return values + alpha * (centroids[assignments] - values)


def insert_site_displacements(
    activation: ArrayLike,
    *,
    batch_indices: ArrayLike,
    rows: ArrayLike,
    columns: ArrayLike,
    native_displacements: ArrayLike,
) -> FloatArray:
    """Insert complete native channel displacements into a BCHW activation clone.

    Repeated site indices intentionally accumulate in input order.
    """

    result = np.asarray(activation, dtype=np.float64).copy()
    displacement = np.asarray(native_displacements, dtype=np.float64)
    if result.ndim != 4:
        raise ValueError("activation must have BCHW shape")
    if not np.all(np.isfinite(result)):
        raise ValueError("activation must be finite")
    if displacement.ndim != 2 or displacement.shape[1] != result.shape[1]:
        raise ValueError("displacements must have one native vector per site")
    if not np.all(np.isfinite(displacement)):
        raise ValueError("native_displacements must be finite")

    batch = _bounded_site_indices(batch_indices, name="batch_indices", upper=result.shape[0])
    row = _bounded_site_indices(rows, name="rows", upper=result.shape[2])
    column = _bounded_site_indices(columns, name="columns", upper=result.shape[3])
    if not (batch.shape == row.shape == column.shape == (displacement.shape[0],)):
        raise ValueError("site indices and displacements have incompatible shapes")

    for i in range(len(batch)):
        with np.errstate(over="ignore", invalid="ignore"):
            result[batch[i], :, row[i], column[i]] += displacement[i]
    if not np.all(np.isfinite(result)):
        raise ValueError("inserted activation must remain finite")
    return result


def _bounded_site_indices(values: ArrayLike, *, name: str, upper: int) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a vector of integer-valued indices")
    numeric = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ValueError(f"{name} must be a vector of integer-valued indices")
    if np.any(numeric < 0) or np.any(numeric >= upper):
        raise ValueError(f"{name} contains an out-of-bounds index")
    return numeric.astype(np.int64)


def rms(values: ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array**2)))


def finite_recovery_gain(
    clean_input: ArrayLike,
    perturbed_input: ArrayLike,
    clean_output: ArrayLike,
    perturbed_output: ArrayLike,
) -> float:
    """Compute the README's normalized next-block RMS gain κ."""

    clean_in = np.asarray(clean_input, dtype=np.float64)
    perturbed_in = np.asarray(perturbed_input, dtype=np.float64)
    clean_out = np.asarray(clean_output, dtype=np.float64)
    perturbed_out = np.asarray(perturbed_output, dtype=np.float64)
    if clean_in.shape != perturbed_in.shape or clean_out.shape != perturbed_out.shape:
        raise ValueError("clean and perturbed tensors must have matching shapes")
    input_scale = rms(clean_in)
    output_scale = rms(clean_out)
    input_change = rms(perturbed_in - clean_in)
    output_change = rms(perturbed_out - clean_out)
    if min(input_scale, output_scale, input_change) <= 0:
        raise ValueError("finite-recovery gain is undefined at zero RMS scale/change")
    return (output_change / output_scale) / (input_change / input_scale)


def predictive_kl(p: ArrayLike, q: ArrayLike, *, epsilon: float = 1e-12) -> FloatArray:
    left = np.asarray(p, dtype=np.float64)
    right = np.asarray(q, dtype=np.float64)
    if left.shape != right.shape or left.ndim < 1:
        raise ValueError("probability arrays must have matching shapes")
    if left.shape[-1] == 0:
        raise ValueError("probability arrays must have a nonempty class axis")
    if not np.isscalar(epsilon) or isinstance(epsilon, (bool, np.bool_)):
        raise ValueError("epsilon must be a finite scalar strictly between zero and one")
    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("epsilon must be a finite scalar strictly between zero and one") from error
    if not np.isfinite(epsilon_value) or not 0 < epsilon_value < 1:
        raise ValueError("epsilon must be a finite scalar strictly between zero and one")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("probabilities must be finite")
    if np.any(left < 0) or np.any(right < 0):
        raise ValueError("probabilities cannot be negative")
    left_mass = left.sum(axis=-1, keepdims=True)
    right_mass = right.sum(axis=-1, keepdims=True)
    if (
        not np.all(np.isfinite(left_mass))
        or not np.all(np.isfinite(right_mass))
        or np.any(left_mass <= 0)
        or np.any(right_mass <= 0)
    ):
        raise ValueError("every probability row must have finite positive mass")
    left = left / left_mass
    right = right / right_mass
    if (
        not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or not np.allclose(left.sum(axis=-1), 1.0)
        or not np.allclose(right.sum(axis=-1), 1.0)
    ):
        raise ValueError("probability normalization must produce finite unit-mass rows")
    log_ratio = np.log(np.clip(left, epsilon_value, None)) - np.log(
        np.clip(right, epsilon_value, None)
    )
    result = np.sum(left * log_ratio, axis=-1)
    if not np.all(np.isfinite(result)):
        raise ValueError("predictive KL must be finite")
    return result


def predictive_derivative_energy(probabilities: ArrayLike, *, delta_r: float) -> FloatArray:
    distributions = np.asarray(probabilities, dtype=np.float64)
    if distributions.ndim != 2 or len(distributions) < 2 or delta_r <= 0:
        raise ValueError("probabilities must be a path-by-class matrix and delta_r positive")
    return 2.0 * predictive_kl(distributions[:-1], distributions[1:]) / delta_r**2


@dataclass(frozen=True)
class BoundaryEnergySummary:
    fraction_near_boundary: float
    energy_80_interval: tuple[float, float]
    peak_offset: float


def summarize_boundary_energy(
    r: ArrayLike,
    energy: ArrayLike,
    *,
    window: tuple[float, float],
    weighting: BoundaryEnergyWeighting,
) -> BoundaryEnergySummary:
    """Summarize nonnegative path energy under an explicit measure.

    ``discrete_grid_mass`` treats each energy value as an atom at its grid point.
    ``continuous_path_trapezoid`` treats values as samples of a piecewise-linear
    density with respect to ``r``. Its window mass and central 80% interval use
    exact segment integration. In both modes the peak is the leftmost grid point
    attaining the largest value (mass in discrete mode, density in continuous mode).
    """

    coordinates = np.asarray(r, dtype=np.float64)
    values = np.asarray(energy, dtype=np.float64)
    if coordinates.shape != values.shape or coordinates.ndim != 1 or len(values) == 0:
        raise ValueError("r and energy must be matching nonempty vectors")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("r must be finite")
    if np.any(np.diff(coordinates) <= 0):
        raise ValueError("r must be strictly increasing")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("energy must be finite and nonnegative")
    window_array = np.asarray(window, dtype=np.float64)
    if (
        window_array.shape != (2,)
        or not np.all(np.isfinite(window_array))
        or window_array[0] > window_array[1]
    ):
        raise ValueError("window must be a finite ordered pair")
    lo, hi = (float(value) for value in window_array)

    if weighting == "discrete_grid_mass":
        total = float(values.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("energy must have finite positive mass")
        weights = values / total
        fraction = float(weights[(coordinates >= lo) & (coordinates <= hi)].sum())
        cdf = np.cumsum(weights)
        lower = float(coordinates[np.searchsorted(cdf, 0.1, side="left")])
        upper = float(coordinates[np.searchsorted(cdf, 0.9, side="left")])
    elif weighting == "continuous_path_trapezoid":
        if len(coordinates) < 2:
            raise ValueError("continuous path weighting requires at least two grid points")
        total = _piecewise_linear_integral(
            coordinates, values, float(coordinates[0]), float(coordinates[-1])
        )
        if not np.isfinite(total) or total <= 0:
            raise ValueError("energy must have finite positive path integral")
        window_mass = _piecewise_linear_integral(coordinates, values, lo, hi)
        fraction = float(np.clip(window_mass / total, 0.0, 1.0))
        lower = _piecewise_linear_quantile(coordinates, values, total=total, fraction=0.1)
        upper = _piecewise_linear_quantile(coordinates, values, total=total, fraction=0.9)
    else:
        raise ValueError(f"unsupported boundary-energy weighting: {weighting!r}")

    peak_offset = float(coordinates[np.argmax(values)] - 1.0)
    return BoundaryEnergySummary(fraction, (lower, upper), peak_offset)


def _piecewise_linear_integral(
    coordinates: FloatArray, values: FloatArray, lower: float, upper: float
) -> float:
    left = max(lower, float(coordinates[0]))
    right = min(upper, float(coordinates[-1]))
    if left >= right:
        return 0.0

    total = 0.0
    for index in range(len(coordinates) - 1):
        x0 = float(coordinates[index])
        x1 = float(coordinates[index + 1])
        segment_left = max(left, x0)
        segment_right = min(right, x1)
        if segment_left >= segment_right:
            continue
        y0 = float(values[index])
        y1 = float(values[index + 1])
        slope = (y1 - y0) / (x1 - x0)
        value_left = y0 + slope * (segment_left - x0)
        value_right = y0 + slope * (segment_right - x0)
        total += 0.5 * (value_left + value_right) * (segment_right - segment_left)
    return float(total)


def _piecewise_linear_quantile(
    coordinates: FloatArray,
    values: FloatArray,
    *,
    total: float,
    fraction: float,
) -> float:
    target = fraction * total
    accumulated = 0.0
    for index in range(len(coordinates) - 1):
        x0 = float(coordinates[index])
        x1 = float(coordinates[index + 1])
        y0 = float(values[index])
        y1 = float(values[index + 1])
        width = x1 - x0
        slope = (y1 - y0) / width
        segment_mass = 0.5 * (y0 + y1) * width
        if target <= accumulated + segment_mass:
            remaining = max(0.0, target - accumulated)
            low = 0.0
            high = width
            for _ in range(64):
                offset = 0.5 * (low + high)
                mass = y0 * offset + 0.5 * slope * offset**2
                if mass < remaining:
                    low = offset
                else:
                    high = offset
            return x0 + 0.5 * (low + high)
        accumulated += segment_mass
    return float(coordinates[-1])


def circular_shift_null(values: ArrayLike, shift: int) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("path values must be a vector with at least two entries")
    normalized_shift = shift % len(array)
    if normalized_shift == 0:
        raise ValueError("null shift must break boundary alignment")
    return np.roll(array, normalized_shift)


def centered_directional_derivative(
    function,
    point: ArrayLike,
    direction: ArrayLike,
    *,
    epsilon: float,
) -> FloatArray:
    x = np.asarray(point, dtype=np.float64)
    v = np.asarray(direction, dtype=np.float64)
    if x.shape != v.shape or epsilon <= 0:
        raise ValueError("point/direction mismatch or nonpositive epsilon")
    return (
        np.asarray(function(x + epsilon * v), dtype=np.float64)
        - np.asarray(function(x - epsilon * v), dtype=np.float64)
    ) / (2.0 * epsilon)


def relative_vector_error(
    reference: ArrayLike, estimate: ArrayLike, *, floor: float = 1e-12
) -> float:
    truth = np.asarray(reference, dtype=np.float64)
    approximation = np.asarray(estimate, dtype=np.float64)
    if truth.shape != approximation.shape:
        raise ValueError("vectors must have matching shapes")
    return float(np.linalg.norm(approximation - truth) / max(np.linalg.norm(truth), floor))
