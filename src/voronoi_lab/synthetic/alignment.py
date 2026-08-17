"""Gauge-aware coordinate and support-component recovery metrics."""

from __future__ import annotations

from itertools import permutations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score

from .indexing import labeling_coordinates, to_grid, to_observed
from .schema import AlignmentResult, Labeling
from .support import all_supports, support_decomposition


def _legal_axis_permutations(labeling: Labeling) -> tuple[tuple[int, ...], ...]:
    sizes = labeling.space.sizes
    return tuple(
        candidate_axes
        for candidate_axes in permutations(range(labeling.space.n_factors))
        if all(
            sizes[candidate_axes[truth_axis]] == sizes[truth_axis]
            for truth_axis in range(len(sizes))
        )
    )


def align_labeling(candidate: Labeling, truth: Labeling) -> AlignmentResult:
    """Optimally align all legal local-label and equal-factor gauges."""

    if candidate.space != truth.space:
        raise ValueError("candidate and truth must use the same factor sizes")
    candidate_coordinates = labeling_coordinates(candidate)
    truth_coordinates = labeling_coordinates(truth)
    best: AlignmentResult | None = None
    for candidate_axis_for_truth in _legal_axis_permutations(candidate):
        aligned = np.empty_like(truth_coordinates)
        maps: list[np.ndarray] = []
        amis: list[float] = []
        for truth_axis, candidate_axis in enumerate(candidate_axis_for_truth):
            size = truth.space.sizes[truth_axis]
            contingency = np.zeros((size, size), dtype=np.int64)
            np.add.at(
                contingency,
                (candidate_coordinates[:, candidate_axis], truth_coordinates[:, truth_axis]),
                1,
            )
            candidate_values, truth_values = linear_sum_assignment(-contingency)
            mapping = np.empty(size, dtype=np.int64)
            mapping[candidate_values] = truth_values
            aligned[:, truth_axis] = mapping[candidate_coordinates[:, candidate_axis]]
            maps.append(mapping)
            amis.append(
                float(
                    adjusted_mutual_info_score(
                        truth_coordinates[:, truth_axis], candidate_coordinates[:, candidate_axis]
                    )
                )
            )
        tuple_accuracy = float(np.mean(np.all(aligned == truth_coordinates, axis=1)))
        result = AlignmentResult(
            candidate_axis_for_truth=tuple(candidate_axis_for_truth),
            candidate_value_to_truth=tuple(maps),
            aligned_coordinates=aligned,
            coordinate_ami=float(np.mean(amis)),
            tuple_accuracy=tuple_accuracy,
        )
        if best is None or (result.tuple_accuracy, result.coordinate_ami) > (
            best.tuple_accuracy,
            best.coordinate_ami,
        ):
            best = result
    if best is None:  # Every factor space has at least the identity axis permutation.
        raise RuntimeError("no legal factor-axis alignment exists")
    return best


def aligned_support_error(
    generators_observed: ArrayLike,
    candidate: Labeling,
    truth: Labeling,
    alignment: AlignmentResult | None = None,
) -> float:
    """Aggregate relative component error after legal support-axis alignment."""

    values = np.asarray(generators_observed, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1:] != (truth.space.n_states, truth.space.n_states):
        raise ValueError("generator family has the wrong shape")
    selected_alignment = alignment or align_labeling(candidate, truth)
    numerator = 0.0
    denominator = 0.0
    for operator in values:
        truth_components = support_decomposition(to_grid(operator, truth), truth.space)
        candidate_components = support_decomposition(to_grid(operator, candidate), candidate.space)
        for truth_support in all_supports(truth.space):
            candidate_support = tuple(
                sorted(selected_alignment.candidate_axis_for_truth[axis] for axis in truth_support)
            )
            truth_observed = to_observed(truth_components[truth_support], truth)
            candidate_observed = to_observed(candidate_components[candidate_support], candidate)
            difference = candidate_observed - truth_observed
            numerator += float(np.vdot(difference, difference).real)
            denominator += float(np.vdot(truth_observed, truth_observed).real)
    return float(np.sqrt(numerator / denominator)) if denominator else float(np.sqrt(numerator))
