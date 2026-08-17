"""Strict, versioned configuration for every scientifically meaningful choice."""

from __future__ import annotations

import json
from datetime import datetime
from math import prod
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

PositiveInt = Annotated[int, Field(gt=0, strict=True)]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
StrictFloat = Annotated[float, Field(strict=True)]
NonNegativeFloat = Annotated[float, Field(ge=0, strict=True)]
StrictBoolean = Annotated[bool, Field(strict=True)]
VersionOne = Annotated[int, Field(ge=1, le=1, strict=True)]


def _require_true(value: bool) -> bool:
    if value is not True:
        raise ValueError("value must be true")
    return value


StrictTrue = Annotated[bool, Field(strict=True), AfterValidator(_require_true)]

Experiment1Cut = Literal[
    "stage1.block1",
    "stage1.block2",
    "stage2.block1",
    "stage2.block2",
    "stage3.block1",
    "stage3.block2",
    "stage4.block1",
    "stage4.block2",
]


class StrictModel(BaseModel):
    """Base model that rejects silent typos and is safe to hash."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ProtocolConfig(StrictModel):
    name: str = "coarse-seed0"
    mode: Literal["exploratory", "confirmatory"] = "exploratory"
    root_seed: NonNegativeInt = 20260816


class RuntimeConfig(StrictModel):
    # Every currently runnable numerical stage is deliberately CPU-only. Future
    # accelerator stages must add a resolved backend contract before widening this.
    device: Literal["cpu"] = "cpu"
    dtype: Literal["float32", "float64"] = "float32"
    workers: Annotated[int, Field(ge=0, le=64, strict=True)] = 0
    deterministic: StrictTrue = True
    shard_images: PositiveInt = 64


class Tracking2Inputs(StrictModel):
    manifest: Path = Path("configs/inputs/tracking2_seed0.yaml")
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)] = (
        "15a822883e269d9accfbfeba35a5fa33ac04caebb30b970a8e0d48f92a565d84"
    )
    root: Path = Path("../Experiments/Tracking2")
    dataset_root: Path | None = None
    checkpoint_files: dict[int, Path] = Field(default_factory=dict)
    transplant_results: Path | None = None
    expected_model: str = "preactivation_resnet18_v2_width64"
    read_only: StrictTrue = True


class InputsConfig(StrictModel):
    tracking2: Tracking2Inputs = Field(default_factory=Tracking2Inputs)


class ProbeBanksConfig(StrictModel):
    fit_train_images: PositiveInt = 2000
    geometry_test_images: PositiveInt = 2000
    intervention_test_images: PositiveInt = 256
    independent_fit_train_images: PositiveInt = 2000
    max_sites_per_image: PositiveInt = 32
    equal_weight_per_image: StrictTrue = True
    intervention_nested_in_geometry: StrictBoolean = False
    require_disjoint_fit_banks: StrictTrue = True


class StateMetricConfig(StrictModel):
    primary: Literal["standardized", "raw"] = "standardized"
    sensitivity: tuple[Literal["standardized", "raw"], ...] = ("standardized", "raw")
    rms_epsilon: Annotated[float, Field(gt=0, strict=True)] = 1e-6
    codebook_scale: Literal["median_nearest_centroid_distance", "rms_centroid_radius"] = (
        "median_nearest_centroid_distance"
    )

    @model_validator(mode="after")
    def include_primary(self) -> StateMetricConfig:
        if self.primary not in self.sensitivity:
            raise ValueError("primary state metric must be included in sensitivity")
        if not self.sensitivity or len(set(self.sensitivity)) != len(self.sensitivity):
            raise ValueError("state-metric sensitivity values must be nonempty and unique")
        return self


class CodebookConfig(StrictModel):
    primary_k: PositiveInt = 32
    k_values: tuple[PositiveInt, ...] = (16, 32, 64)
    algorithm: Literal["minibatch_kmeans", "kmeans"] = "minibatch_kmeans"
    n_init: PositiveInt = 10
    max_iter: PositiveInt = 300
    batch_size: PositiveInt = 4096
    initialization: Literal["k-means++", "random"] = "k-means++"
    stability_protocol: Literal["refit_hungarian", "fixed_codebook_resample", "disabled"] = (
        "refit_hungarian"
    )

    @model_validator(mode="after")
    def primary_is_in_sweep(self) -> CodebookConfig:
        if self.primary_k not in self.k_values:
            raise ValueError("primary_k must be included in k_values")
        if len(set(self.k_values)) != len(self.k_values):
            raise ValueError("k_values must not contain duplicates")
        if tuple(sorted(self.k_values)) != self.k_values:
            raise ValueError("k_values must be strictly increasing")
        return self


class BoundaryPathConfig(StrictModel):
    directions: tuple[
        Literal["empirical_chord", "off_cloud", "local_covariance", "boundary_normal"],
        ...,
    ] = ("empirical_chord", "off_cloud")
    r_grid: tuple[StrictFloat, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        0.9,
        0.95,
        1.0,
        1.05,
        1.1,
        1.25,
        1.5,
    )
    finite_difference_delta_r: Annotated[float, Field(gt=0, strict=True)] = 1e-2
    local_neighbors: PositiveInt = 64
    local_pca_rank: PositiveInt = 8
    boundary_energy_window: tuple[StrictFloat, StrictFloat] = (0.9, 1.1)
    boundary_energy_weighting: Literal["discrete_grid_mass", "continuous_path_trapezoid"] = (
        "continuous_path_trapezoid"
    )
    shifted_null: Literal["circular"] = "circular"

    @model_validator(mode="after")
    def validate_grid(self) -> BoundaryPathConfig:
        if not self.directions or len(set(self.directions)) != len(self.directions):
            raise ValueError("boundary directions must be nonempty and unique")
        if tuple(sorted(self.r_grid)) != self.r_grid or len(set(self.r_grid)) != len(self.r_grid):
            raise ValueError("r_grid must be strictly increasing")
        if 1.0 not in self.r_grid:
            raise ValueError("r_grid must include the fitted boundary r=1")
        if self.local_pca_rank > self.local_neighbors:
            raise ValueError("local_pca_rank cannot exceed local_neighbors")
        lo, hi = self.boundary_energy_window
        if not lo < 1.0 < hi:
            raise ValueError("boundary_energy_window must straddle r=1")
        if lo < self.r_grid[0] or hi > self.r_grid[-1]:
            raise ValueError("boundary_energy_window must lie within the r_grid extent")
        return self


class SnappingConfig(StrictModel):
    alphas: tuple[Annotated[float, Field(ge=0, le=1, strict=True)], ...] = (
        0.0,
        0.25,
        0.5,
        1.0,
    )
    boundary_fractions: tuple[Annotated[float, Field(gt=0, strict=True)], ...] = (
        0.5,
        0.9,
        1.1,
    )
    spatial_supports: tuple[Literal["single_token", "sparse_tokens", "all_tokens"], ...] = (
        "single_token",
        "sparse_tokens",
        "all_tokens",
    )
    controls: tuple[
        Literal[
            "identity",
            "same_norm_random",
            "away_from_centroid",
            "toward_other_centroid",
            "independent_codebook",
        ],
        ...,
    ] = (
        "identity",
        "same_norm_random",
        "away_from_centroid",
        "toward_other_centroid",
        "independent_codebook",
    )

    @model_validator(mode="after")
    def validate_intervention_axes(self) -> SnappingConfig:
        for name, values in (
            ("alphas", self.alphas),
            ("boundary_fractions", self.boundary_fractions),
            ("spatial_supports", self.spatial_supports),
            ("controls", self.controls),
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
        if tuple(sorted(self.alphas)) != self.alphas:
            raise ValueError("alphas must be strictly increasing")
        if tuple(sorted(self.boundary_fractions)) != self.boundary_fractions:
            raise ValueError("boundary_fractions must be strictly increasing")
        required_controls = {
            "identity",
            "same_norm_random",
            "away_from_centroid",
            "toward_other_centroid",
            "independent_codebook",
        }
        if set(self.controls) != required_controls:
            raise ValueError("controls must contain every required matched control exactly once")
        return self


class NullConfig(StrictModel):
    families: tuple[
        Literal[
            "epoch0",
            "global_gaussian",
            "class_conditional_gaussian",
            "position_conditional_gaussian",
        ],
        ...,
    ] = ("epoch0", "global_gaussian", "class_conditional_gaussian")
    covariance_shrinkage: Annotated[float, Field(ge=0, le=1, strict=True)] = 1e-3
    include_position_null_if_mutual_information_exceeds: NonNegativeFloat = 0.1

    @model_validator(mode="after")
    def validate_families(self) -> NullConfig:
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("null families must be nonempty and unique")
        return self


class BootstrapConfig(StrictModel):
    resamples: PositiveInt = 1000
    unit: Literal["image"] = "image"
    interval: Literal["percentile"] = "percentile"
    confidence: Annotated[float, Field(gt=0, lt=1, strict=True)] = 0.95


class InputRecipeConfig(StrictModel):
    """One versioned, materialized input-derived state recipe."""

    recipe_version: VersionOne = 1
    name: str
    kind: Literal["clean", "crop_flip", "mild_color"]
    crop_padding: NonNegativeInt
    flip_probability: Annotated[float, Field(ge=0, le=1, strict=True)]
    brightness_fraction: Annotated[float, Field(ge=0, lt=1, strict=True)]

    @model_validator(mode="after")
    def reject_ignored_parameters(self) -> InputRecipeConfig:
        if not self.name.strip():
            raise ValueError("input recipe name cannot be blank")
        if self.kind == "clean" and (
            self.crop_padding != 0 or self.flip_probability != 0 or self.brightness_fraction != 0
        ):
            raise ValueError("clean recipe parameters must all be zero")
        if self.kind == "crop_flip" and self.brightness_fraction != 0:
            raise ValueError("crop_flip brightness_fraction must be zero")
        if self.kind == "mild_color" and (self.crop_padding != 0 or self.flip_probability != 0):
            raise ValueError("mild_color crop/flip parameters must be zero")
        return self


def _default_input_recipes() -> tuple[InputRecipeConfig, ...]:
    return (
        InputRecipeConfig(
            name="clean",
            kind="clean",
            crop_padding=0,
            flip_probability=0.0,
            brightness_fraction=0.0,
        ),
        InputRecipeConfig(
            name="crop_flip",
            kind="crop_flip",
            crop_padding=4,
            flip_probability=0.5,
            brightness_fraction=0.0,
        ),
        InputRecipeConfig(
            name="mild_color",
            kind="mild_color",
            crop_padding=0,
            flip_probability=0.0,
            brightness_fraction=0.1,
        ),
    )


class ConfirmationTrainingConfig(StrictModel):
    """Explicit, currently unfrozen recipe for future independent model seeds."""

    protocol_version: VersionOne = 1
    status: Literal["unfrozen"] = "unfrozen"
    architecture: Literal["preactivation_resnet18_v2_width64"] = "preactivation_resnet18_v2_width64"
    model_source: Literal["pinned_tracking2_models_py"] = "pinned_tracking2_models_py"
    dataset: Literal["cifar10_pinned_parquet"] = "cifar10_pinned_parquet"
    training_seeds: tuple[NonNegativeInt, ...] = (0, 1, 2)
    epochs: PositiveInt = 100
    batch_size: PositiveInt = 128
    optimizer: Literal["sgd"] = "sgd"
    learning_rate: Annotated[float, Field(gt=0, strict=True)] = 0.05
    momentum: Annotated[float, Field(ge=0, lt=1, strict=True)] = 0.9
    weight_decay: NonNegativeFloat = 0.0
    scheduler: Literal["multistep"] = "multistep"
    milestones: tuple[PositiveInt, ...] = (30, 60, 90)
    gamma: Annotated[float, Field(gt=0, lt=1, strict=True)] = 0.2
    checkpoint_every_epochs: PositiveInt = 1
    early_batch_checkpoints: tuple[NonNegativeInt, ...] = (0, 1, 5, 20, 100)
    mixed_precision: StrictBoolean = False

    @model_validator(mode="after")
    def validate_training_schedule(self) -> ConfirmationTrainingConfig:
        if len(self.training_seeds) != 3 or len(set(self.training_seeds)) != 3:
            raise ValueError("confirmation training requires exactly three unique seeds")
        for name, values in (
            ("milestones", self.milestones),
            ("early_batch_checkpoints", self.early_batch_checkpoints),
        ):
            if not values or tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be nonempty, unique, and increasing")
        if self.milestones[-1] >= self.epochs:
            raise ValueError("scheduler milestones must occur before the final epoch")
        return self


class MechanicalProtocolConfig(StrictModel):
    """Versioned choices for the bounded model/JVP implementation smoke."""

    protocol_version: VersionOne = 1
    input_recipe: Literal["deterministic_linspace"] = "deterministic_linspace"
    input_min: StrictFloat = -2.0
    input_max: StrictFloat = 2.0
    input_batch_size: VersionOne = 1
    directions_per_cut: VersionOne = 1
    jvp_epsilon_float32: Annotated[float, Field(gt=0, strict=True)] = 1e-3
    jvp_epsilon_float64: Annotated[float, Field(gt=0, strict=True)] = 1e-5
    denominator_floor: Annotated[float, Field(gt=0, strict=True)] = 1e-12
    sample_axis: Literal["whole_tensor"] = "whole_tensor"
    aggregation: Literal["median_linear_p95"] = "median_linear_p95"

    @model_validator(mode="after")
    def validate_input_range(self) -> MechanicalProtocolConfig:
        if self.input_min >= self.input_max:
            raise ValueError("mechanical input_min must be less than input_max")
        return self


class Experiment1Config(StrictModel):
    checkpoints: tuple[NonNegativeInt, ...] = (0, 1, 5, 20, 100)
    cuts: tuple[Experiment1Cut, ...] = (
        "stage1.block1",
        "stage1.block2",
        "stage2.block1",
        "stage2.block2",
        "stage3.block1",
        "stage3.block2",
        "stage4.block1",
        "stage4.block2",
    )
    sentinel_cuts: tuple[Experiment1Cut, ...] = (
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    )
    input_recipes: tuple[InputRecipeConfig, ...] = Field(default_factory=_default_input_recipes)
    confirmation_training: ConfirmationTrainingConfig = Field(
        default_factory=ConfirmationTrainingConfig
    )
    mechanical_protocol: MechanicalProtocolConfig = Field(default_factory=MechanicalProtocolConfig)
    probe_banks: ProbeBanksConfig = Field(default_factory=ProbeBanksConfig)
    state_metric: StateMetricConfig = Field(default_factory=StateMetricConfig)
    codebooks: CodebookConfig = Field(default_factory=CodebookConfig)
    boundary_paths: BoundaryPathConfig = Field(default_factory=BoundaryPathConfig)
    snapping: SnappingConfig = Field(default_factory=SnappingConfig)
    nulls: NullConfig = Field(default_factory=NullConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)

    @model_validator(mode="after")
    def validate_axes(self) -> Experiment1Config:
        if not self.checkpoints or tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must be nonempty, unique, and strictly increasing")
        if not self.cuts or len(set(self.cuts)) != len(self.cuts):
            raise ValueError("cuts must be nonempty and unique")
        if not self.sentinel_cuts or len(set(self.sentinel_cuts)) != len(self.sentinel_cuts):
            raise ValueError("sentinel_cuts must be nonempty and unique")
        unknown = set(self.sentinel_cuts) - set(self.cuts)
        if unknown:
            raise ValueError(f"sentinel_cuts are not declared cuts: {sorted(unknown)}")
        if not self.input_recipes:
            raise ValueError("input_recipes must be nonempty")
        names = [recipe.name for recipe in self.input_recipes]
        kinds = [recipe.kind for recipe in self.input_recipes]
        if len(set(names)) != len(names):
            raise ValueError("input recipe names must be unique")
        if set(kinds) != {"clean", "crop_flip", "mild_color"} or len(kinds) != 3:
            raise ValueError("input_recipes must declare clean, crop_flip, and mild_color once")
        return self


class SyntheticSamplingConfig(StrictModel):
    tau: Annotated[float, Field(gt=0, strict=True)] = 0.1
    transitions_per_state: PositiveInt = 5000
    easy_rho: NonNegativeFloat = 0.05
    easy_delta: Annotated[float, Field(ge=0, le=1, strict=True)] = 0.1


class SyntheticExactProtocolConfig(StrictModel):
    """Frozen choices specific to the tiny exhaustive optimization subgate."""

    rho: NonNegativeFloat = 0.0
    delta: Annotated[float, Field(ge=0, le=1, strict=True)] = 1.0
    support_policy: Literal["anchored", "mixed"] = "anchored"
    random_relabel: StrictTrue = True
    max_states: Annotated[int, Field(gt=0, le=8, strict=True)] = 8
    generator_rate_shape: Annotated[float, Field(gt=0, strict=True)] = 2.0
    generator_connectivity_policy: Literal["mandatory_directed_cycle"] = "mandatory_directed_cycle"
    generator_normalization: Literal["unit_mean_exit_rate"] = "unit_mean_exit_rate"
    exhaustive_tie_atol: NonNegativeFloat = 1e-10
    exhaustive_tie_rtol: NonNegativeFloat = 1e-10


class Experiment2Config(StrictModel):
    oracle_factor_sizes: tuple[PositiveInt, ...] = (2, 3)
    refinement_factor_sizes: tuple[PositiveInt, ...] = (4, 5)
    train_primitives: PositiveInt = 6
    heldout_primitives: PositiveInt = 3
    generator_density: Annotated[float, Field(gt=0, le=1, strict=True)] = 0.5
    unary_weight: Annotated[float, Field(gt=0, strict=True)] = 1.0
    interaction_rho_values: tuple[NonNegativeFloat, ...] = (0.0, 0.05, 0.2, 0.5)
    symmetry_delta_values: tuple[Annotated[float, Field(ge=0, le=1, strict=True)], ...] = (
        0.0,
        0.1,
        0.5,
        1.0,
    )
    group: Literal["cyclic"] = "cyclic"
    exact_instances: PositiveInt = 20
    null_instances: PositiveInt = 100
    support_penalties: tuple[NonNegativeFloat, ...] = (0.0, 0.0, 1.0, 2.0)
    exact_protocol: SyntheticExactProtocolConfig = Field(
        default_factory=SyntheticExactProtocolConfig
    )
    sampling: SyntheticSamplingConfig = Field(default_factory=SyntheticSamplingConfig)

    @model_validator(mode="after")
    def require_nontrivial_products(self) -> Experiment2Config:
        if len(self.oracle_factor_sizes) < 2 or len(self.refinement_factor_sizes) < 2:
            raise ValueError("synthetic benchmarks require at least two factors")
        if any(size < 2 for size in (*self.oracle_factor_sizes, *self.refinement_factor_sizes)):
            raise ValueError("every synthetic factor must contain at least two states")
        for name, values in (
            ("interaction_rho_values", self.interaction_rho_values),
            ("symmetry_delta_values", self.symmetry_delta_values),
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
            if tuple(sorted(values)) != values:
                raise ValueError(f"{name} must be strictly increasing")
        if any(v > 1 for v in self.symmetry_delta_values):
            raise ValueError("symmetry_delta_values must be in [0, 1]")
        if prod(self.oracle_factor_sizes) > self.exact_protocol.max_states:
            raise ValueError("oracle_factor_sizes exceed exact_protocol.max_states")
        if len(self.support_penalties) < len(self.oracle_factor_sizes) + 1:
            raise ValueError("support_penalties must cover every oracle interaction order")
        if self.support_penalties[:2] != (0.0, 0.0) or any(
            right < left
            for left, right in zip(self.support_penalties, self.support_penalties[1:], strict=False)
        ):
            raise ValueError("support_penalties must start (0, 0) and be nondecreasing")
        return self


class MechanicalGateConfig(StrictModel):
    roundtrip_relative_rms_max: Annotated[float, Field(gt=0, strict=True)] = 1e-6
    jvp_median_relative_error_max: Annotated[float, Field(gt=0, strict=True)] = 1e-2
    jvp_p95_relative_error_max: Annotated[float, Field(gt=0, strict=True)] = 5e-2


class CoarseGateConfig(StrictModel):
    required_passing_sentinel_cuts: PositiveInt = 2
    require_nonfinal_cut: StrictBoolean = True
    confidence: Annotated[float, Field(gt=0, lt=1, strict=True)] = 0.95
    shifted_null_percentile: Annotated[float, Field(gt=0, lt=1, strict=True)] = 0.95


class SyntheticGateConfig(StrictModel):
    noiseless_instances: PositiveInt = 20
    exact_tuple_recovery_fraction_min: Annotated[float, Field(ge=0, le=1, strict=True)] = 1.0
    relative_support_error_max: Annotated[float, Field(gt=0, strict=True)] = 1e-8
    easy_sampled_median_ami_min: Annotated[float, Field(ge=0, le=1, strict=True)] = 0.9
    null_false_positives_max: NonNegativeInt = 5
    null_instances: PositiveInt = 100

    @model_validator(mode="after")
    def validate_null_allowance(self) -> SyntheticGateConfig:
        if self.null_false_positives_max > self.null_instances:
            raise ValueError("null_false_positives_max cannot exceed null_instances")
        return self


class FunctionalGateConfig(StrictModel):
    required_passing_cuts: PositiveInt = 2
    confidence: Annotated[float, Field(gt=0, lt=1, strict=True)] = 0.95
    require_lower_damage_than_every_control: StrictTrue = True
    kappa_max: Annotated[float, Field(gt=0, strict=True)] = 1.0
    require_higher_clean_cell_recovery: StrictTrue = True
    primary_direction: Literal["empirical_chord"] = "empirical_chord"


class ConfirmationGateConfig(StrictModel):
    training_seeds: tuple[NonNegativeInt, ...] = (0, 1, 2)
    required_passing_cuts: PositiveInt = 2
    confidence: Annotated[float, Field(gt=0, lt=1, strict=True)] = 0.95
    require_same_cuts: StrictTrue = True
    require_same_effect_direction: StrictTrue = True
    replication_unit: Literal["training_seed"] = "training_seed"

    @model_validator(mode="after")
    def validate_replication_seeds(self) -> ConfirmationGateConfig:
        if len(self.training_seeds) != 3 or len(set(self.training_seeds)) != 3:
            raise ValueError("confirmation requires exactly three unique training seeds")
        return self


class RealAlgebraGateConfig(StrictModel):
    null_false_positive_rate_max: Annotated[float, Field(gt=0, le=0.05, strict=True)] = 0.05
    confidence: Annotated[float, Field(gt=0, lt=1, strict=True)] = 0.95
    require_positive_heldout_compression: StrictTrue = True
    calibration_suite: Literal["synthetic_and_unfactored_nulls"] = "synthetic_and_unfactored_nulls"


class GateOverrideAuthorization(StrictModel):
    """Auditable authorization for one named gate, never a global bypass."""

    target_gate: Literal[
        "mechanical",
        "coarse",
        "functional",
        "synthetic_exact",
        "synthetic",
        "confirmation",
        "real_algebra",
    ]
    scope: Literal["gate"] = "gate"
    mode: Literal["diagnostic_only"] = "diagnostic_only"
    reason: str
    authorized_by: str
    recorded_at: str

    @model_validator(mode="after")
    def validate_authorization(self) -> GateOverrideAuthorization:
        if not self.reason.strip():
            raise ValueError("gate override reason cannot be blank")
        if not self.authorized_by.strip():
            raise ValueError("gate override authorized_by cannot be blank")
        try:
            recorded = datetime.fromisoformat(self.recorded_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("gate override recorded_at must be ISO-8601") from error
        if recorded.tzinfo is None or recorded.utcoffset() is None:
            raise ValueError("gate override recorded_at must include a timezone")
        return self


class GateOverridesConfig(StrictModel):
    mechanical: GateOverrideAuthorization | None = None
    coarse: GateOverrideAuthorization | None = None
    functional: GateOverrideAuthorization | None = None
    synthetic_exact: GateOverrideAuthorization | None = None
    synthetic: GateOverrideAuthorization | None = None
    confirmation: GateOverrideAuthorization | None = None
    real_algebra: GateOverrideAuthorization | None = None

    @model_validator(mode="after")
    def validate_targets(self) -> GateOverridesConfig:
        for target, authorization in (
            ("mechanical", self.mechanical),
            ("coarse", self.coarse),
            ("functional", self.functional),
            ("synthetic_exact", self.synthetic_exact),
            ("synthetic", self.synthetic),
            ("confirmation", self.confirmation),
            ("real_algebra", self.real_algebra),
        ):
            if authorization is not None and authorization.target_gate != target:
                raise ValueError(f"{target} override must target {target!r}")
        return self


class GatesConfig(StrictModel):
    mechanical: MechanicalGateConfig = Field(default_factory=MechanicalGateConfig)
    coarse: CoarseGateConfig = Field(default_factory=CoarseGateConfig)
    functional: FunctionalGateConfig = Field(default_factory=FunctionalGateConfig)
    synthetic: SyntheticGateConfig = Field(default_factory=SyntheticGateConfig)
    confirmation: ConfirmationGateConfig = Field(default_factory=ConfirmationGateConfig)
    real_algebra: RealAlgebraGateConfig = Field(default_factory=RealAlgebraGateConfig)
    overrides: GateOverridesConfig = Field(default_factory=GateOverridesConfig)


class ReportConfig(StrictModel):
    output: Path = Path("reports/voronoi_lab.html")
    mockup_output: Path = Path("reports/MOCKUP/voronoi_lab_MOCKUP.html")
    self_contained: StrictTrue = True
    embed_spec: StrictTrue = True


class LabConfig(StrictModel):
    schema_version: VersionOne = 1
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    inputs: InputsConfig = Field(default_factory=InputsConfig)
    experiment1: Experiment1Config = Field(default_factory=Experiment1Config)
    experiment2: Experiment2Config = Field(default_factory=Experiment2Config)
    gates: GatesConfig = Field(default_factory=GatesConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @model_validator(mode="after")
    def cross_validate_gate_sample_counts(self) -> LabConfig:
        if self.protocol.mode == "confirmatory":
            raise ValueError(
                "confirmatory mode is not yet supported: freeze and register the full "
                "protocol hash before enabling confirmatory claims"
            )
        if self.experiment2.exact_instances != self.gates.synthetic.noiseless_instances:
            raise ValueError(
                "experiment2.exact_instances must equal gates.synthetic.noiseless_instances"
            )
        if self.experiment2.null_instances != self.gates.synthetic.null_instances:
            raise ValueError("experiment2.null_instances must equal gates.synthetic.null_instances")
        available_sentinel_cuts = len(self.experiment1.sentinel_cuts)
        for gate_name, required in (
            ("coarse", self.gates.coarse.required_passing_sentinel_cuts),
            ("functional", self.gates.functional.required_passing_cuts),
            ("confirmation", self.gates.confirmation.required_passing_cuts),
        ):
            if required > available_sentinel_cuts:
                raise ValueError(
                    f"gates.{gate_name} required passing cuts cannot exceed the "
                    f"{available_sentinel_cuts} available sentinel cuts"
                )
        if self.experiment2.sampling.easy_rho not in self.experiment2.interaction_rho_values:
            raise ValueError("sampling.easy_rho must be included in interaction_rho_values")
        if self.experiment2.sampling.easy_delta not in self.experiment2.symmetry_delta_values:
            raise ValueError("sampling.easy_delta must be included in symmetry_delta_values")
        if (
            self.experiment1.confirmation_training.training_seeds
            != self.gates.confirmation.training_seeds
        ):
            raise ValueError(
                "confirmation training seeds must equal gates.confirmation.training_seeds"
            )
        if self.gates.coarse.require_nonfinal_cut and not any(
            cut != "stage4.block2" for cut in self.experiment1.sentinel_cuts
        ):
            raise ValueError("gates.coarse.require_nonfinal_cut requires a non-final sentinel cut")
        if not self.report.self_contained:
            raise ValueError("only self-contained reports are supported")
        if not self.report.embed_spec:
            raise ValueError("reports must embed the experiment specification")
        return self

    def canonical_json(self) -> str:
        """Return stable JSON used for configuration identity and provenance."""

        payload = self.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate scientific choices."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_config(path: str | Path) -> LabConfig:
    """Load a strict YAML configuration, resolving paths from the project cwd.

    Paths intentionally remain as declared rather than being silently made absolute;
    manifests record both the declaration and the resolved external input path.
    """

    config_path = Path(path)
    raw = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a YAML mapping: {config_path}")
    return LabConfig.model_validate(raw)


def write_resolved_config(config: LabConfig, path: str | Path) -> Path:
    """Write all defaults explicitly so a run never depends on future defaults."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output
