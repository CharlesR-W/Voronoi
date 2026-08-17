"""Small deterministic smoke runners for the initial synthetic gate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from voronoi_lab.core import SeedDeriver, canonical_hash

from .alignment import align_labeling, aligned_support_error
from .generators import generate_synthetic_instance
from .schema import FactorSpace
from .search import exhaustive_label_search, labeling_objective

_GENERATOR_CONNECTIVITY_POLICY = "mandatory_directed_cycle"
_GENERATOR_NORMALIZATION = "unit_mean_exit_rate"


@dataclass(frozen=True, slots=True)
class OracleExhaustiveInstanceResult:
    instance_index: int
    seed: int
    exact: bool
    best_tie_count: int
    train_support_error: float
    heldout_support_error: float
    train_objective: float
    heldout_objective: float
    heldout_oracle_objective: float
    heldout_excess_objective: float
    evaluated_labelings: int
    observed_primitive_names: tuple[str, ...]
    observed_generator_family: tuple[tuple[tuple[float, ...], ...], ...]
    observed_generator_family_hash: str
    realized_normalized_support_order_spectrum: tuple[tuple[float, ...], ...]
    realized_normalized_support_order_spectrum_hash: str
    train_primitive_indices: tuple[int, ...]
    heldout_primitive_indices: tuple[int, ...]
    truth_labeling: tuple[int, ...]
    truth_labeling_hash: str
    selected_labeling: tuple[int, ...]
    selected_labeling_hash: str


@dataclass(frozen=True, slots=True)
class OracleExhaustiveSmokeResult:
    instances: int
    exact_instances: int
    worst_support_error: float
    worst_train_support_error: float
    max_best_objective: float
    max_heldout_objective: float
    max_heldout_oracle_objective: float
    max_heldout_excess_objective: float
    evaluated_labelings: int
    factor_sizes: tuple[int, ...]
    train_primitives: int
    heldout_primitives: int
    density: float
    unary_weight: float
    rho: float
    delta: float
    support_policy: str
    random_relabel: bool
    penalties: tuple[float, ...]
    max_states: int
    seed_namespace: tuple[str, ...]
    protocol_version: int
    generator_rate_shape: float
    generator_connectivity_policy: str
    generator_normalization: str
    exhaustive_tie_atol: float
    exhaustive_tie_rtol: float
    instance_results: tuple[OracleExhaustiveInstanceResult, ...]


def _validated_protocol(
    *,
    seed: int,
    factor_sizes: Sequence[int],
    train_primitives: int,
    heldout_primitives: int,
    penalties: Sequence[float],
    max_states: int,
) -> tuple[FactorSpace, tuple[float, ...], tuple[str, ...]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        isinstance(train_primitives, bool)
        or not isinstance(train_primitives, int)
        or isinstance(heldout_primitives, bool)
        or not isinstance(heldout_primitives, int)
        or train_primitives < 1
        or heldout_primitives < 1
    ):
        raise ValueError("train_primitives and heldout_primitives must be positive")
    normalized_penalties = tuple(float(value) for value in penalties)
    if any(not np.isfinite(value) or value < 0 for value in normalized_penalties):
        raise ValueError("penalties must be finite and nonnegative")
    if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    space = FactorSpace(tuple(factor_sizes))
    if space.n_states > max_states:
        raise ValueError(
            f"exact factor space has {space.n_states} states but max_states={max_states}"
        )
    if len(normalized_penalties) < space.n_factors + 1:
        raise ValueError("penalties must cover every interaction order")
    return (
        space,
        normalized_penalties,
        ("exp2", "exact", "oracle_exhaustive", "v1"),
    )


def _validated_exact_choices(
    *,
    generator_rate_shape: float,
    generator_connectivity_policy: str,
    generator_normalization: str,
    exhaustive_tie_atol: float,
    exhaustive_tie_rtol: float,
) -> tuple[float, str, str, float, float]:
    numeric_choices: list[tuple[str, float, bool]] = [
        ("generator_rate_shape", generator_rate_shape, True),
        ("exhaustive_tie_atol", exhaustive_tie_atol, False),
        ("exhaustive_tie_rtol", exhaustive_tie_rtol, False),
    ]
    normalized: dict[str, float] = {}
    for name, value, strictly_positive in numeric_choices:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number")
        try:
            converted = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite number") from error
        if not np.isfinite(converted) or (converted <= 0 if strictly_positive else converted < 0):
            qualifier = "positive" if strictly_positive else "nonnegative"
            raise ValueError(f"{name} must be finite and {qualifier}")
        normalized[name] = converted
    if generator_connectivity_policy != _GENERATOR_CONNECTIVITY_POLICY:
        raise ValueError("generator_connectivity_policy must be 'mandatory_directed_cycle'")
    if generator_normalization != _GENERATOR_NORMALIZATION:
        raise ValueError("generator_normalization must be 'unit_mean_exit_rate'")
    return (
        normalized["generator_rate_shape"],
        generator_connectivity_policy,
        generator_normalization,
        normalized["exhaustive_tie_atol"],
        normalized["exhaustive_tie_rtol"],
    )


def run_oracle_exhaustive_instance(
    *,
    seed: int = 20260816,
    instance_index: int,
    factor_sizes: Sequence[int] = (2, 3),
    train_primitives: int = 6,
    heldout_primitives: int = 3,
    density: float = 0.75,
    unary_weight: float = 1.0,
    rho: float = 0.0,
    delta: float = 1.0,
    support_policy: str = "anchored",
    random_relabel: bool = True,
    penalties: Sequence[float] = (0.0, 0.0, 1.0),
    max_states: int = 8,
    generator_rate_shape: float = 2.0,
    generator_connectivity_policy: str = _GENERATOR_CONNECTIVITY_POLICY,
    generator_normalization: str = _GENERATOR_NORMALIZATION,
    exhaustive_tie_atol: float = 1e-10,
    exhaustive_tie_rtol: float = 1e-10,
) -> OracleExhaustiveInstanceResult:
    """Generate and solve one globally indexed exact-protocol instance.

    ``instance_index`` is part of the semantic seed path.  Splitting a run into
    shards or retrying only selected shards therefore cannot renumber or alter
    any generated instance.
    """

    if (
        isinstance(instance_index, bool)
        or not isinstance(instance_index, int)
        or instance_index < 0
    ):
        raise ValueError("instance_index must be a nonnegative integer")
    space, normalized_penalties, seed_namespace = _validated_protocol(
        seed=seed,
        factor_sizes=factor_sizes,
        train_primitives=train_primitives,
        heldout_primitives=heldout_primitives,
        penalties=penalties,
        max_states=max_states,
    )
    (
        normalized_rate_shape,
        _,
        _,
        normalized_tie_atol,
        normalized_tie_rtol,
    ) = _validated_exact_choices(
        generator_rate_shape=generator_rate_shape,
        generator_connectivity_policy=generator_connectivity_policy,
        generator_normalization=generator_normalization,
        exhaustive_tie_atol=exhaustive_tie_atol,
        exhaustive_tie_rtol=exhaustive_tie_rtol,
    )
    instance_seed = SeedDeriver(seed, seed_namespace).derive("instance", instance_index)
    instance = generate_synthetic_instance(
        space,
        n_primitives=train_primitives + heldout_primitives,
        rng=np.random.default_rng(instance_seed),
        density=density,
        rate_shape=normalized_rate_shape,
        unary_weight=unary_weight,
        rho=rho,
        delta=delta,
        support_policy=support_policy,
        random_relabel=random_relabel,
    )
    train = instance.observed.generators[:train_primitives]
    heldout = instance.observed.generators[train_primitives:]
    search = exhaustive_label_search(
        train,
        space,
        penalties=normalized_penalties,
        tie_atol=normalized_tie_atol,
        tie_rtol=normalized_tie_rtol,
        max_states=max_states,
    )
    alignments = [
        align_labeling(labeling, instance.truth.labeling) for labeling in search.best_labelings
    ]
    selected = search.best_labelings[0]
    selected_alignment = alignments[0]
    train_support_error = aligned_support_error(
        train,
        selected,
        instance.truth.labeling,
        selected_alignment,
    )
    heldout_support_error = aligned_support_error(
        heldout,
        selected,
        instance.truth.labeling,
        selected_alignment,
    )
    heldout_objective = labeling_objective(heldout, selected, normalized_penalties)
    heldout_oracle = labeling_objective(heldout, instance.truth.labeling, normalized_penalties)
    observed = tuple(
        tuple(tuple(float(value) for value in row) for row in generator)
        for generator in instance.observed.generators
    )
    spectrum = tuple(
        tuple(float(value) for value in row) for row in instance.truth.realized_order_energy
    )
    truth_labeling = tuple(int(value) for value in instance.truth.labeling.obs_to_grid)
    selected_labeling = tuple(int(value) for value in selected.obs_to_grid)
    return OracleExhaustiveInstanceResult(
        instance_index=instance_index,
        seed=instance_seed,
        exact=all(alignment.exact for alignment in alignments),
        best_tie_count=len(search.best_labelings),
        train_support_error=train_support_error,
        heldout_support_error=heldout_support_error,
        train_objective=search.best_objective,
        heldout_objective=heldout_objective,
        heldout_oracle_objective=heldout_oracle,
        heldout_excess_objective=heldout_objective - heldout_oracle,
        evaluated_labelings=search.evaluated,
        observed_primitive_names=instance.observed.primitive_names,
        observed_generator_family=observed,
        observed_generator_family_hash=canonical_hash(observed),
        realized_normalized_support_order_spectrum=spectrum,
        realized_normalized_support_order_spectrum_hash=canonical_hash(spectrum),
        train_primitive_indices=tuple(range(train_primitives)),
        heldout_primitive_indices=tuple(
            range(train_primitives, train_primitives + heldout_primitives)
        ),
        truth_labeling=truth_labeling,
        truth_labeling_hash=canonical_hash(truth_labeling),
        selected_labeling=selected_labeling,
        selected_labeling_hash=canonical_hash(selected_labeling),
    )


def summarize_oracle_exhaustive_instances(
    instance_results: Sequence[OracleExhaustiveInstanceResult],
    *,
    seed: int = 20260816,
    factor_sizes: Sequence[int] = (2, 3),
    train_primitives: int = 6,
    heldout_primitives: int = 3,
    density: float = 0.75,
    unary_weight: float = 1.0,
    rho: float = 0.0,
    delta: float = 1.0,
    support_policy: str = "anchored",
    random_relabel: bool = True,
    penalties: Sequence[float] = (0.0, 0.0, 1.0),
    max_states: int = 8,
    generator_rate_shape: float = 2.0,
    generator_connectivity_policy: str = _GENERATOR_CONNECTIVITY_POLICY,
    generator_normalization: str = _GENERATOR_NORMALIZATION,
    exhaustive_tie_atol: float = 1e-10,
    exhaustive_tie_rtol: float = 1e-10,
) -> OracleExhaustiveSmokeResult:
    """Aggregate a complete ordered set of independently computed instances."""

    space, normalized_penalties, seed_namespace = _validated_protocol(
        seed=seed,
        factor_sizes=factor_sizes,
        train_primitives=train_primitives,
        heldout_primitives=heldout_primitives,
        penalties=penalties,
        max_states=max_states,
    )
    (
        normalized_rate_shape,
        normalized_connectivity_policy,
        normalized_normalization,
        normalized_tie_atol,
        normalized_tie_rtol,
    ) = _validated_exact_choices(
        generator_rate_shape=generator_rate_shape,
        generator_connectivity_policy=generator_connectivity_policy,
        generator_normalization=generator_normalization,
        exhaustive_tie_atol=exhaustive_tie_atol,
        exhaustive_tie_rtol=exhaustive_tie_rtol,
    )
    results = tuple(instance_results)
    if not results:
        raise ValueError("instance_results must not be empty")
    expected_indices = tuple(range(len(results)))
    if tuple(item.instance_index for item in results) != expected_indices:
        raise ValueError("instance_results must be ordered and cover indices 0..N-1")
    expected_seeds = SeedDeriver(seed, seed_namespace)
    if any(item.seed != expected_seeds.derive("instance", item.instance_index) for item in results):
        raise ValueError("instance result seed does not match its global semantic index")
    return OracleExhaustiveSmokeResult(
        instances=len(results),
        exact_instances=sum(item.exact for item in results),
        worst_support_error=max(item.heldout_support_error for item in results),
        worst_train_support_error=max(item.train_support_error for item in results),
        max_best_objective=max(item.train_objective for item in results),
        max_heldout_objective=max(item.heldout_objective for item in results),
        max_heldout_oracle_objective=max(item.heldout_oracle_objective for item in results),
        max_heldout_excess_objective=max(item.heldout_excess_objective for item in results),
        evaluated_labelings=sum(item.evaluated_labelings for item in results),
        factor_sizes=space.sizes,
        train_primitives=train_primitives,
        heldout_primitives=heldout_primitives,
        density=float(density),
        unary_weight=float(unary_weight),
        rho=float(rho),
        delta=float(delta),
        support_policy=support_policy,
        random_relabel=bool(random_relabel),
        penalties=normalized_penalties,
        max_states=max_states,
        seed_namespace=seed_namespace,
        protocol_version=1,
        generator_rate_shape=normalized_rate_shape,
        generator_connectivity_policy=normalized_connectivity_policy,
        generator_normalization=normalized_normalization,
        exhaustive_tie_atol=normalized_tie_atol,
        exhaustive_tie_rtol=normalized_tie_rtol,
        instance_results=results,
    )


def run_oracle_exhaustive_smoke(
    *,
    seed: int = 20260816,
    instances: int = 2,
    factor_sizes: Sequence[int] = (2, 3),
    train_primitives: int = 6,
    heldout_primitives: int = 3,
    density: float = 0.75,
    unary_weight: float = 1.0,
    rho: float = 0.0,
    delta: float = 1.0,
    support_policy: str = "anchored",
    random_relabel: bool = True,
    penalties: Sequence[float] = (0.0, 0.0, 1.0),
    max_states: int = 8,
    generator_rate_shape: float = 2.0,
    generator_connectivity_policy: str = _GENERATOR_CONNECTIVITY_POLICY,
    generator_normalization: str = _GENERATOR_NORMALIZATION,
    exhaustive_tie_atol: float = 1e-10,
    exhaustive_tie_rtol: float = 1e-10,
) -> OracleExhaustiveSmokeResult:
    """Search on noiseless train primitives and validate the labeling on heldout ones."""

    if isinstance(instances, bool) or not isinstance(instances, int) or instances < 1:
        raise ValueError("instances must be positive")
    instance_results = tuple(
        run_oracle_exhaustive_instance(
            seed=seed,
            instance_index=instance_index,
            factor_sizes=factor_sizes,
            train_primitives=train_primitives,
            heldout_primitives=heldout_primitives,
            density=density,
            unary_weight=unary_weight,
            rho=rho,
            delta=delta,
            support_policy=support_policy,
            random_relabel=random_relabel,
            penalties=penalties,
            max_states=max_states,
            generator_rate_shape=generator_rate_shape,
            generator_connectivity_policy=generator_connectivity_policy,
            generator_normalization=generator_normalization,
            exhaustive_tie_atol=exhaustive_tie_atol,
            exhaustive_tie_rtol=exhaustive_tie_rtol,
        )
        for instance_index in range(instances)
    )
    return summarize_oracle_exhaustive_instances(
        instance_results,
        seed=seed,
        factor_sizes=factor_sizes,
        train_primitives=train_primitives,
        heldout_primitives=heldout_primitives,
        density=density,
        unary_weight=unary_weight,
        rho=rho,
        delta=delta,
        support_policy=support_policy,
        random_relabel=random_relabel,
        penalties=penalties,
        max_states=max_states,
        generator_rate_shape=generator_rate_shape,
        generator_connectivity_policy=generator_connectivity_policy,
        generator_normalization=generator_normalization,
        exhaustive_tie_atol=exhaustive_tie_atol,
        exhaustive_tie_rtol=exhaustive_tie_rtol,
    )
