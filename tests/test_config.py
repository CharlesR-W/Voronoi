from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from voronoi_lab.config import LabConfig, load_config


def test_checked_in_pilot_config_is_complete_and_stable() -> None:
    config = load_config("configs/pilot.yaml")
    assert config.schema_version == 1
    assert config.experiment1.checkpoints == (0, 1, 5, 20, 100)
    assert len(config.experiment1.cuts) == 8
    assert config.experiment1.bootstrap.unit == "image"
    assert config.inputs.tracking2_vgg.expected_model == "vgg19_bn_classifier512_width1"
    assert config.experiment1.plateau_protocol.resnet_cut == "stage2.block1"
    assert config.experiment1.plateau_protocol.vgg_cut == "stage2.conv1"
    assert config.experiment1.plateau_protocol.protocol_version == 2
    assert (
        config.experiment1.plateau_protocol.animation_jacobian_selection
        == "residual_update_for_residual_raw_transition_for_nonresidual"
    )
    assert config.experiment1.synthetic_plateau_task.residual_blocks == 4
    assert config.experiment2.oracle_factor_sizes == (2, 3)
    assert config.experiment2.exact_protocol.generator_rate_shape == 2.0
    assert (
        config.experiment2.exact_protocol.generator_connectivity_policy
        == "mandatory_directed_cycle"
    )
    assert config.experiment2.exact_protocol.generator_normalization == "unit_mean_exit_rate"
    assert config.experiment2.exact_protocol.exhaustive_tie_atol == 1e-10
    assert config.experiment2.exact_protocol.exhaustive_tie_rtol == 1e-10
    digest = hashlib.sha256(config.canonical_json().encode()).hexdigest()
    assert len(digest) == 64


def test_unknown_configuration_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        LabConfig.model_validate({"schema_version": 1, "typo": True})


def test_primary_k_must_be_in_sweep() -> None:
    with pytest.raises(ValidationError, match="primary_k"):
        LabConfig.model_validate(
            {
                "schema_version": 1,
                "experiment1": {"codebooks": {"primary_k": 7, "k_values": [8, 16]}},
            }
        )


def test_sentinel_cut_must_be_declared() -> None:
    with pytest.raises(ValidationError, match="sentinel_cuts"):
        LabConfig.model_validate(
            {
                "schema_version": 1,
                "experiment1": {"sentinel_cuts": ["missing.cut"]},
            }
        )


def test_experiment1_cut_names_are_a_closed_supported_set() -> None:
    with pytest.raises(ValidationError, match="cuts"):
        LabConfig.model_validate(
            {
                "schema_version": 1,
                "experiment1": {
                    "cuts": ["stage1.block1", "stage1.typo"],
                    "sentinel_cuts": ["stage1.block1"],
                },
            }
        )


@pytest.mark.parametrize(
    ("gate_name", "count_name"),
    [
        ("coarse", "required_passing_sentinel_cuts"),
        ("functional", "required_passing_cuts"),
        ("confirmation", "required_passing_cuts"),
    ],
)
def test_gate_cut_requirements_cannot_exceed_available_sentinel_cuts(
    gate_name: str,
    count_name: str,
) -> None:
    gates = {
        "coarse": {"required_passing_sentinel_cuts": 1},
        "functional": {"required_passing_cuts": 1},
        "confirmation": {"required_passing_cuts": 1},
    }
    gates[gate_name][count_name] = 2
    with pytest.raises(ValidationError, match=f"gates.{gate_name}"):
        LabConfig.model_validate(
            {
                "schema_version": 1,
                "experiment1": {
                    "cuts": ["stage1.block1"],
                    "sentinel_cuts": ["stage1.block1"],
                },
                "gates": gates,
            }
        )


@pytest.mark.parametrize(
    "boundary_paths",
    [
        {"local_neighbors": 4, "local_pca_rank": 5},
        {
            "r_grid": [0.95, 1.0, 1.05],
            "boundary_energy_window": [0.9, 1.1],
        },
    ],
)
def test_boundary_protocol_is_feasible_on_its_declared_grid(boundary_paths) -> None:
    with pytest.raises(ValidationError):
        LabConfig.model_validate(
            {
                "schema_version": 1,
                "experiment1": {"boundary_paths": boundary_paths},
            }
        )


def test_synthetic_null_false_positive_allowance_cannot_exceed_trials() -> None:
    with pytest.raises(ValidationError, match="null_false_positives_max"):
        LabConfig.model_validate(
            {
                "schema_version": 1,
                "experiment2": {"null_instances": 3},
                "gates": {
                    "synthetic": {
                        "null_instances": 3,
                        "null_false_positives_max": 4,
                    }
                },
            }
        )


@pytest.mark.parametrize(
    "exact_protocol",
    [
        {"generator_rate_shape": 0.0},
        {"generator_connectivity_policy": "optional_cycle"},
        {"generator_normalization": "none"},
        {"exhaustive_tie_atol": -1e-10},
        {"exhaustive_tie_rtol": float("nan")},
    ],
)
def test_exact_protocol_rejects_unsupported_generator_and_search_choices(
    exact_protocol,
) -> None:
    with pytest.raises(ValidationError):
        LabConfig.model_validate(
            {
                "schema_version": 1,
                "experiment2": {"exact_protocol": exact_protocol},
            }
        )


def test_cuda_is_an_explicit_supported_runtime_device() -> None:
    config = LabConfig.model_validate(
        {
            "schema_version": 1,
            "runtime": {"device": "cuda"},
        }
    )
    assert config.runtime.device == "cuda"


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": True},
        {"protocol": {"root_seed": -1}},
        {"experiment1": {"checkpoints": [0, 0]}},
        {"experiment1": {"cuts": [], "sentinel_cuts": []}},
        {"experiment1": {"boundary_paths": {"directions": []}}},
        {"experiment1": {"snapping": {"alphas": [-1.0, float("nan")]}}},
        {"experiment1": {"probe_banks": {"equal_weight_per_image": False}}},
        {"experiment1": {"probe_banks": {"require_disjoint_fit_banks": False}}},
        {"runtime": {"workers": 1.0}},
        {"runtime": {"workers": 65}},
        {"runtime": {"deterministic": "false"}},
        {"runtime": {"deterministic": False}},
        {"runtime": {"device": "auto"}},
        {"inputs": {"tracking2": {"read_only": 1}}},
        {"inputs": {"tracking2": {"manifest_sha256": "not-a-digest"}}},
        {"inputs": {"tracking2_vgg": {"read_only": False}}},
        {"experiment1": {"plateau_protocol": {"local_surface_grid_points": 8}}},
        {"experiment1": {"plateau_protocol": {"three_anchor_axis_min": 2.0}}},
        {"experiment1": {"plateau_protocol": {"orientation_frame_ms": 1001}}},
        {"experiment1": {"synthetic_plateau_task": {"intervention_block": 4}}},
        {"experiment1": {"mechanical_protocol": {"protocol_version": True}}},
        {"experiment1": {"mechanical_protocol": {"input_batch_size": True}}},
        {"experiment2": {"exact_protocol": {"random_relabel": 1}}},
        {"experiment1": {"boundary_paths": {"finite_difference_delta_r": "0.01"}}},
        {"experiment2": {"generator_density": "0.5"}},
        {"experiment1": {"bootstrap": {"interval": "basic"}}},
        {"report": {"self_contained": False}},
        {"report": {"embed_spec": False}},
        {"experiment2": {"oracle_factor_sizes": [2, 5]}},
        {"experiment2": {"exact_protocol": {"max_states": 9}}},
        {"experiment2": {"exact_instances": 2}},
    ],
)
def test_invalid_or_unsupported_protocol_axes_are_rejected(payload) -> None:
    with pytest.raises(ValidationError):
        LabConfig.model_validate({"schema_version": 1, **payload})


def test_duplicate_yaml_keys_are_rejected(tmp_path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nprotocol:\n  root_seed: 1\n  root_seed: 2\n")

    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_config(path)


def test_gate_overrides_are_structured_and_scoped_per_gate() -> None:
    authorization = {
        "target_gate": "mechanical",
        "scope": "gate",
        "mode": "diagnostic_only",
        "reason": "Inspect downstream diagnostics after known JVP backend failure.",
        "authorized_by": "researcher@example.test",
        "recorded_at": "2026-08-16T22:00:00-07:00",
    }
    config = LabConfig.model_validate(
        {"schema_version": 1, "gates": {"overrides": {"mechanical": authorization}}}
    )
    assert config.gates.overrides.mechanical is not None
    assert config.gates.overrides.synthetic_exact is None

    wrong_target = {**authorization, "target_gate": "synthetic_exact"}
    with pytest.raises(ValidationError, match="must target"):
        LabConfig.model_validate(
            {"schema_version": 1, "gates": {"overrides": {"mechanical": wrong_target}}}
        )


def test_easy_sampled_regime_must_be_part_of_declared_sweep() -> None:
    with pytest.raises(ValidationError, match="easy_rho"):
        LabConfig.model_validate(
            {"schema_version": 1, "experiment2": {"sampling": {"easy_rho": 0.123}}}
        )


def test_confirmatory_mode_cannot_be_a_nominal_label() -> None:
    with pytest.raises(ValidationError, match="confirmatory mode is not yet supported"):
        LabConfig.model_validate({"schema_version": 1, "protocol": {"mode": "confirmatory"}})


def test_confirmation_training_recipe_is_explicit_and_seed_aligned() -> None:
    config = load_config("configs/pilot.yaml")
    recipe = config.experiment1.confirmation_training
    assert recipe.status == "unfrozen"
    assert recipe.training_seeds == config.gates.confirmation.training_seeds
    assert recipe.milestones == (30, 60, 90)

    with pytest.raises(ValidationError, match="training seeds must equal"):
        LabConfig.model_validate(
            {
                "schema_version": 1,
                "experiment1": {"confirmation_training": {"training_seeds": [1, 2, 3]}},
            }
        )
