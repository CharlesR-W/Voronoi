"""Optional-PyTorch mechanical checks shared by real and toy residual models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError("PyTorch mechanics require the 'resnet' optional dependency") from error
    return torch


@dataclass(frozen=True, slots=True)
class ParityResult:
    cuts: tuple[str, ...]
    max_absolute_errors: tuple[float, ...]
    full_logits: tuple[tuple[float, ...], ...]
    split_logits: tuple[tuple[float, ...], ...]

    @property
    def exact(self) -> bool:
        return all(error == 0.0 for error in self.max_absolute_errors)


@dataclass(frozen=True, slots=True)
class JVPValidationResult:
    relative_errors: tuple[float, ...]
    median_relative_error: float
    p95_relative_error: float
    epsilon: float
    automatic_output: tuple[float, ...]
    finite_difference_output: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ResNetMechanicalEvidenceSummary:
    """Metrics deterministically reconstructed from saved numeric evidence."""

    identity_per_cut: dict[str, float]
    identity_exact: bool | None
    identity_max_absolute_error: float | None
    jvp_relative_error_by_cut: dict[str, float]
    jvp_cuts_completed: int
    jvp_median_relative_error: float | None
    jvp_p95_relative_error: float | None


def _axis(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    axis = tuple(values)
    if not axis or any(not isinstance(value, str) or not value for value in axis):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(set(axis)) != len(axis):
        raise ValueError(f"{label} must be unique")
    return axis


def _flat_finite_vector(value: object, *, label: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        raw = value
        if raw.dtype.kind not in "iuf":
            raise ValueError(f"{label} must contain real numbers, not booleans")
    else:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError(f"{label} must be a flat numeric sequence")
        if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
            raise ValueError(f"{label} must contain real numbers, not booleans")
        raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{label} must be a non-empty flattened vector")
    try:
        array = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must contain finite real numbers") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite real numbers")
    return array


def _paired_vectors(
    row: Mapping[str, object],
    *,
    left_key: str,
    right_key: str,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(row, Mapping):
        raise ValueError(f"{label} must be an object")
    left = _flat_finite_vector(row.get(left_key), label=f"{label}.{left_key}")
    right = _flat_finite_vector(row.get(right_key), label=f"{label}.{right_key}")
    if left.shape != right.shape:
        raise ValueError(f"{label} saved vectors must have matching shapes")
    return left, right


def summarize_resnet_mechanical_evidence(
    identity_logits_by_cut: Mapping[str, Mapping[str, object]],
    jvp_outputs_by_cut: Mapping[str, Mapping[str, object]],
    *,
    identity_cuts: Sequence[str],
    jvp_cuts: Sequence[str],
    denominator_floor: float,
) -> ResNetMechanicalEvidenceSummary:
    """Recompute every ResNet mechanical metric from flattened raw arrays.

    This routine deliberately has no PyTorch dependency.  Both the producer and
    artifact validator call it, so claimed scalar errors are never trusted as
    primary evidence.
    """

    declared_identity = _axis(identity_cuts, label="identity_cuts")
    declared_jvp = _axis(jvp_cuts, label="jvp_cuts")
    if (
        isinstance(denominator_floor, bool)
        or not isinstance(denominator_floor, Real)
        or not np.isfinite(float(denominator_floor))
        or float(denominator_floor) <= 0
    ):
        raise ValueError("denominator_floor must be a finite positive number")
    identity_names = set(identity_logits_by_cut)
    if identity_names and identity_names != set(declared_identity):
        raise ValueError("identity evidence must cover the complete declared cut axis")
    unknown_jvp = set(jvp_outputs_by_cut) - set(declared_jvp)
    if unknown_jvp:
        raise ValueError(f"JVP evidence contains undeclared cuts: {sorted(unknown_jvp)}")

    identity_errors: dict[str, float] = {}
    for cut_name in declared_identity:
        if cut_name not in identity_logits_by_cut:
            continue
        full, split = _paired_vectors(
            identity_logits_by_cut[cut_name],
            left_key="full_logits",
            right_key="split_logits",
            label=f"identity evidence {cut_name!r}",
        )
        error = float(np.max(np.abs(split - full)))
        if not np.isfinite(error):
            raise ValueError(f"identity evidence {cut_name!r} produces a non-finite error")
        identity_errors[cut_name] = error

    jvp_errors: dict[str, float] = {}
    for cut_name in declared_jvp:
        if cut_name not in jvp_outputs_by_cut:
            continue
        automatic, finite = _paired_vectors(
            jvp_outputs_by_cut[cut_name],
            left_key="automatic_jvp",
            right_key="finite_difference_jvp",
            label=f"JVP evidence {cut_name!r}",
        )
        numerator = float(np.linalg.norm(automatic - finite))
        denominator = max(float(np.linalg.norm(automatic)), float(denominator_floor))
        error = numerator / denominator
        if not np.isfinite(error):
            raise ValueError(f"JVP evidence {cut_name!r} produces a non-finite error")
        jvp_errors[cut_name] = error

    identity_available = bool(identity_errors)
    identity_max = max(identity_errors.values()) if identity_available else None
    identity_exact = (
        all(error == 0.0 for error in identity_errors.values()) if identity_available else None
    )
    jvp_complete = set(jvp_errors) == set(declared_jvp)
    if jvp_complete:
        ordered_errors = [jvp_errors[cut_name] for cut_name in declared_jvp]
        jvp_median = float(np.median(ordered_errors))
        jvp_p95 = float(np.quantile(ordered_errors, 0.95))
    else:
        jvp_median = None
        jvp_p95 = None
    return ResNetMechanicalEvidenceSummary(
        identity_per_cut=identity_errors,
        identity_exact=identity_exact,
        identity_max_absolute_error=identity_max,
        jvp_relative_error_by_cut=jvp_errors,
        jvp_cuts_completed=len(jvp_errors),
        jvp_median_relative_error=jvp_median,
        jvp_p95_relative_error=jvp_p95,
    )


def zero_intervention_parity(
    model: Any,
    images: Any,
    cuts: Sequence[Any],
    *,
    encode: Callable[[Any, Any, Any], Any],
    suffix: Callable[[Any, Any, Any], Any],
) -> ParityResult:
    """Compare full inference to encode/identity/suffix at every declared cut."""

    torch = _torch()
    model.eval()
    names: list[str] = []
    full_logits: list[tuple[float, ...]] = []
    split_logits: list[tuple[float, ...]] = []
    with torch.inference_mode():
        expected = model(images)
        for cut in cuts:
            representation = encode(model, images, cut)
            observed = suffix(model, representation.clone(), cut)
            if observed.shape != expected.shape:
                raise ValueError("split suffix output shape does not match full model output")
            names.append(str(getattr(cut, "name", cut)))
            full_logits.append(
                tuple(float(value) for value in expected.detach().cpu().reshape(-1).tolist())
            )
            split_logits.append(
                tuple(float(value) for value in observed.detach().cpu().reshape(-1).tolist())
            )
    evidence = {
        name: {
            "full_logits": full,
            "split_logits": split,
        }
        for name, full, split in zip(names, full_logits, split_logits, strict=True)
    }
    summary = summarize_resnet_mechanical_evidence(
        evidence,
        {},
        identity_cuts=names,
        jvp_cuts=names,
        denominator_floor=1.0,
    )
    return ParityResult(
        tuple(names),
        tuple(summary.identity_per_cut[name] for name in names),
        tuple(full_logits),
        tuple(split_logits),
    )


def edit_activation_sites(
    activation: Any,
    *,
    batch_indices: Any,
    rows: Any,
    columns: Any,
    native_displacements: Any,
) -> Any:
    """Clone a BCHW tensor and add one complete channel vector at each site."""

    torch = _torch()
    if activation.ndim != 4:
        raise ValueError("activation must have BCHW shape")
    batch = torch.as_tensor(batch_indices, dtype=torch.long, device=activation.device)
    row = torch.as_tensor(rows, dtype=torch.long, device=activation.device)
    column = torch.as_tensor(columns, dtype=torch.long, device=activation.device)
    displacement = torch.as_tensor(
        native_displacements, dtype=activation.dtype, device=activation.device
    )
    if not (batch.ndim == row.ndim == column.ndim == 1):
        raise ValueError("site indices must be vectors")
    if not (len(batch) == len(row) == len(column) == len(displacement)):
        raise ValueError("site indices and displacements must have equal length")
    if displacement.ndim != 2 or displacement.shape[1] != activation.shape[1]:
        raise ValueError("displacements must have one complete channel vector per site")
    if (
        torch.any(batch < 0)
        or torch.any(batch >= activation.shape[0])
        or torch.any(row < 0)
        or torch.any(row >= activation.shape[2])
        or torch.any(column < 0)
        or torch.any(column >= activation.shape[3])
    ):
        raise ValueError("site index lies outside the activation tensor")
    result = activation.clone()
    # Repeated sites intentionally accumulate rather than silently overwrite.
    for index in range(len(batch)):
        result[batch[index], :, row[index], column[index]] += displacement[index]
    return result


def validate_jvp(
    function: Callable[[Any], Any],
    point: Any,
    direction: Any,
    *,
    epsilon: float = 1e-5,
    sample_axis: int | None = 0,
    denominator_floor: float = 1e-12,
) -> JVPValidationResult:
    """Compare automatic JVPs with centered finite differences sample by sample."""

    torch = _torch()
    if epsilon <= 0 or denominator_floor <= 0:
        raise ValueError("epsilon and denominator_floor must be positive")
    if point.shape != direction.shape:
        raise ValueError("point and direction must have matching shapes")
    _, automatic = torch.func.jvp(function, (point,), (direction,))
    with torch.no_grad():
        finite = (function(point + epsilon * direction) - function(point - epsilon * direction)) / (
            2.0 * epsilon
        )
    automatic_array = automatic.detach().cpu().numpy().astype(np.float64)
    finite_array = finite.detach().cpu().numpy().astype(np.float64)
    difference = automatic_array - finite_array
    if sample_axis is None:
        numerator = np.asarray([np.linalg.norm(difference.reshape(-1))], dtype=np.float64)
        denominator = np.asarray([np.linalg.norm(automatic_array.reshape(-1))], dtype=np.float64)
    else:
        if not 0 <= sample_axis < automatic_array.ndim:
            raise ValueError("sample_axis is outside the function output")
        moved_difference = np.moveaxis(difference, sample_axis, 0)
        moved_automatic = np.moveaxis(automatic_array, sample_axis, 0)
        flattened_difference = moved_difference.reshape(len(moved_difference), -1)
        numerator = np.linalg.norm(flattened_difference, axis=1)
        denominator = np.linalg.norm(moved_automatic.reshape(len(moved_automatic), -1), axis=1)
    values = numerator / np.maximum(denominator, denominator_floor)
    return JVPValidationResult(
        relative_errors=tuple(float(value) for value in values),
        median_relative_error=float(np.median(values)),
        p95_relative_error=float(np.quantile(values, 0.95)),
        epsilon=float(epsilon),
        automatic_output=tuple(float(value) for value in automatic_array.reshape(-1)),
        finite_difference_output=tuple(float(value) for value in finite_array.reshape(-1)),
    )
