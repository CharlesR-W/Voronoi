"""Validated array schemas for the synthetic factor-recovery benchmark.

The package uses a single convention throughout: matrices are indexed as
``[..., destination, source]`` and a proposed labeling maps observed state
indices to C-order flattened product-grid indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Support = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FactorSpace:
    """A finite product state space with C-order tuple enumeration."""

    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        sizes = tuple(int(size) for size in self.sizes)
        if not sizes:
            raise ValueError("a factor space must contain at least one factor")
        if any(size < 1 for size in sizes):
            raise ValueError("factor sizes must be positive")
        object.__setattr__(self, "sizes", sizes)

    @property
    def n_factors(self) -> int:
        return len(self.sizes)

    @property
    def n_states(self) -> int:
        return prod(self.sizes)

    @property
    def full_support(self) -> Support:
        return tuple(range(self.n_factors))

    def validate_support(self, support: Support) -> Support:
        normalized = tuple(int(axis) for axis in support)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("support axes must be unique and sorted")
        if any(axis < 0 or axis >= self.n_factors for axis in normalized):
            raise ValueError(f"support {normalized} is outside {self.n_factors} factors")
        return normalized


@dataclass(frozen=True, slots=True)
class Labeling:
    """A bijection from observed states to flattened product-grid labels."""

    space: FactorSpace
    obs_to_grid: IntArray

    def __post_init__(self) -> None:
        mapping = np.asarray(self.obs_to_grid, dtype=np.int64)
        if mapping.shape != (self.space.n_states,):
            raise ValueError(
                f"obs_to_grid must have shape {(self.space.n_states,)}, got {mapping.shape}"
            )
        if not np.array_equal(np.sort(mapping), np.arange(self.space.n_states)):
            raise ValueError("obs_to_grid must be a permutation of all state indices")
        object.__setattr__(self, "obs_to_grid", mapping.copy())

    @classmethod
    def identity(cls, space: FactorSpace) -> Labeling:
        return cls(space=space, obs_to_grid=np.arange(space.n_states, dtype=np.int64))

    @property
    def grid_to_obs(self) -> IntArray:
        return np.argsort(self.obs_to_grid).astype(np.int64, copy=False)


@dataclass(frozen=True, slots=True)
class ObservedGeneratorFamily:
    """Generators exposed to a recovery algorithm, without latent truth."""

    space: FactorSpace
    generators: FloatArray
    primitive_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray(self.generators, dtype=np.float64)
        expected_tail = (self.space.n_states, self.space.n_states)
        if values.ndim != 3 or values.shape[1:] != expected_tail:
            raise ValueError(
                f"generators must have shape (primitive, {expected_tail[0]}, {expected_tail[1]})"
            )
        if values.shape[0] < 1:
            raise ValueError("a generator family must contain at least one primitive")
        if not np.all(np.isfinite(values)):
            raise ValueError("generators must be finite")
        names = self.primitive_names or tuple(f"primitive_{i}" for i in range(values.shape[0]))
        if len(names) != values.shape[0] or len(set(names)) != len(names):
            raise ValueError("primitive names must be unique and match the family size")
        object.__setattr__(self, "generators", values.copy())
        object.__setattr__(self, "primitive_names", tuple(names))

    @property
    def n_primitives(self) -> int:
        return self.generators.shape[0]


@dataclass(frozen=True, slots=True)
class TransitionCounts:
    """Multinomial transition counts under the destination/source convention."""

    space: FactorSpace
    counts: IntArray
    tau: float

    def __post_init__(self) -> None:
        values = np.asarray(self.counts, dtype=np.int64)
        expected_tail = (self.space.n_states, self.space.n_states)
        if values.ndim != 3 or values.shape[1:] != expected_tail:
            raise ValueError(
                f"counts must have shape (primitive, {expected_tail[0]}, {expected_tail[1]})"
            )
        if np.any(values < 0):
            raise ValueError("transition counts cannot be negative")
        if not np.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("tau must be finite and positive")
        column_totals = values.sum(axis=-2)
        if np.any(column_totals <= 0):
            raise ValueError("every primitive/source column needs at least one transition")
        object.__setattr__(self, "counts", values.copy())
        object.__setattr__(self, "tau", float(self.tau))


@dataclass(frozen=True, slots=True)
class SyntheticTruth:
    """Latent data kept out of the recovery-facing observed-family schema."""

    labeling: Labeling
    latent_generators: FloatArray
    support_components: tuple[dict[Support, FloatArray], ...]
    raw_terms: tuple[dict[Support, FloatArray], ...]
    observed_group: FloatArray
    realized_order_energy: FloatArray
    rho: float
    delta: float


@dataclass(frozen=True, slots=True)
class SyntheticInstance:
    observed: ObservedGeneratorFamily
    truth: SyntheticTruth


@dataclass(frozen=True, slots=True)
class SymmetrySpectrum:
    """Commutator singular modes after quotienting the scalar identity."""

    singular_values: FloatArray
    modes: FloatArray
    commutator_residuals: FloatArray


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Best legal gauge alignment of a candidate labeling to truth."""

    candidate_axis_for_truth: tuple[int, ...]
    candidate_value_to_truth: tuple[IntArray, ...]
    aligned_coordinates: IntArray
    coordinate_ami: float
    tuple_accuracy: float

    @property
    def exact(self) -> bool:
        return self.tuple_accuracy == 1.0


@dataclass(frozen=True, slots=True)
class ExhaustiveSearchResult:
    best_objective: float
    best_labelings: tuple[Labeling, ...]
    evaluated: int
    objective_values: FloatArray
