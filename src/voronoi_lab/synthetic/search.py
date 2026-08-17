"""Exact product-label objectives and tiny exhaustive search."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations
from math import factorial

import numpy as np
from numpy.typing import ArrayLike

from .indexing import to_grid
from .schema import ExhaustiveSearchResult, FactorSpace, Labeling
from .support import weighted_high_order_objective


def labeling_objective(
    generators_observed: ArrayLike,
    labeling: Labeling,
    penalties: Sequence[float] = (0.0, 0.0, 1.0),
) -> float:
    grid_generators = to_grid(generators_observed, labeling)
    return weighted_high_order_objective(grid_generators, labeling.space, penalties)


def exhaustive_label_search(
    generators_observed: ArrayLike,
    space: FactorSpace,
    *,
    penalties: Sequence[float] = (0.0, 0.0, 1.0),
    tie_atol: float = 1e-10,
    tie_rtol: float = 1e-10,
    max_states: int = 8,
) -> ExhaustiveSearchResult:
    """Enumerate every observed-to-grid bijection for a tiny state space."""

    if space.n_states > max_states:
        raise ValueError(
            f"exhaustive search is limited to {max_states} states, got {space.n_states}"
        )
    if not np.isfinite(tie_atol) or tie_atol < 0:
        raise ValueError("tie_atol must be finite and nonnegative")
    if not np.isfinite(tie_rtol) or tie_rtol < 0:
        raise ValueError("tie_rtol must be finite and nonnegative")
    mappings: list[np.ndarray] = []
    objectives = np.empty(factorial(space.n_states), dtype=np.float64)
    for index, mapping_tuple in enumerate(permutations(range(space.n_states))):
        mapping = np.fromiter(mapping_tuple, dtype=np.int64, count=space.n_states)
        labeling = Labeling(space=space, obs_to_grid=mapping)
        objectives[index] = labeling_objective(generators_observed, labeling, penalties)
        mappings.append(mapping)
    best_objective = float(objectives.min())
    tolerance = tie_atol + tie_rtol * max(1.0, abs(best_objective))
    best_indices = np.flatnonzero(objectives <= best_objective + tolerance)
    best_labelings = tuple(
        Labeling(space=space, obs_to_grid=mappings[int(index)]) for index in best_indices
    )
    return ExhaustiveSearchResult(
        best_objective=best_objective,
        best_labelings=best_labelings,
        evaluated=objectives.size,
        objective_values=objectives,
    )
