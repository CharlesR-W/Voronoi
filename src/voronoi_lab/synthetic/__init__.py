"""Synthetic known-factor and approximate-symmetry benchmark."""

from .alignment import align_labeling, aligned_support_error
from .generators import assert_generator, draw_sparse_generator, generate_synthetic_instance
from .groups import cyclic_group, cyclic_product_group, twirl_operator
from .indexing import (
    coordinate_to_flat,
    flat_to_coordinate,
    labeling_coordinates,
    lift_local_operator,
    permutation_matrix,
    to_grid,
    to_observed,
)
from .runner import (
    OracleExhaustiveInstanceResult,
    OracleExhaustiveSmokeResult,
    run_oracle_exhaustive_instance,
    run_oracle_exhaustive_smoke,
    summarize_oracle_exhaustive_instances,
)
from .sampling import (
    estimate_generator,
    euler_transition,
    safe_tau,
    sample_generator_family,
    sample_transition_counts,
)
from .schema import FactorSpace, Labeling, ObservedGeneratorFamily, SyntheticInstance
from .search import exhaustive_label_search, labeling_objective
from .support import (
    all_supports,
    reconstruct_support,
    support_decomposition,
    support_order_energy,
    weighted_high_order_objective,
)
from .symmetry import (
    commutator_matrix,
    commutator_spectrum,
    operator_span_without_identity,
    principal_angles,
)

__all__ = [
    "FactorSpace",
    "Labeling",
    "ObservedGeneratorFamily",
    "OracleExhaustiveInstanceResult",
    "OracleExhaustiveSmokeResult",
    "SyntheticInstance",
    "align_labeling",
    "aligned_support_error",
    "all_supports",
    "assert_generator",
    "commutator_matrix",
    "commutator_spectrum",
    "coordinate_to_flat",
    "cyclic_group",
    "cyclic_product_group",
    "draw_sparse_generator",
    "estimate_generator",
    "euler_transition",
    "exhaustive_label_search",
    "flat_to_coordinate",
    "generate_synthetic_instance",
    "labeling_coordinates",
    "labeling_objective",
    "lift_local_operator",
    "operator_span_without_identity",
    "permutation_matrix",
    "principal_angles",
    "reconstruct_support",
    "run_oracle_exhaustive_instance",
    "run_oracle_exhaustive_smoke",
    "safe_tau",
    "sample_generator_family",
    "sample_transition_counts",
    "summarize_oracle_exhaustive_instances",
    "support_decomposition",
    "support_order_energy",
    "to_grid",
    "to_observed",
    "twirl_operator",
    "weighted_high_order_objective",
]
