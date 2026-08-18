from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import voronoi_lab.pipeline as pipeline_module
from voronoi_lab.exp1.plateau_collection import (
    ActivationInterventionAdapter,
    PlateauCollectionSettings,
    collect_plateau_checkpoint,
    make_resnet_intervention_adapter,
    make_vgg_intervention_adapter,
    package_plateau_checkpoint,
)
from voronoi_lab.exp1.surface_geometry import (
    ThreeAnchorSlice,
    empirical_covariance_gaussian,
    plane_pullback_jacobian_frobenius,
)
from voronoi_lab.pipeline import PipelineError


class TinyResidual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.readout = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.readout.weight.copy_(torch.tensor(((1.0, 0.0), (0.0, 1.0), (-1.0, -1.0))))

    def suffix(self, values: torch.Tensor) -> torch.Tensor:
        return self.readout(values)

    @staticmethod
    def transition(values: torch.Tensor) -> torch.Tensor:
        return 2.0 * values


def _adapter() -> ActivationInterventionAdapter:
    model = TinyResidual()
    return ActivationInterventionAdapter(
        architecture="test_residual",
        cut_name="block1",
        activation_shape=(2,),
        encode_batch=lambda values: values,
        suffix_batch=model.suffix,
        next_transition_batch=model.transition,
        spatial_site=None,
        residual_identity=True,
    )


def test_real_model_adapters_select_the_matched_same_resolution_transitions() -> None:
    resnet_blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(8))
    resnet_model = SimpleNamespace(blocks=resnet_blocks)
    tracking = SimpleNamespace(
        encode=lambda model, values, cut: values,
        suffix=lambda model, values, cut: values,
    )
    resnet = make_resnet_intervention_adapter(resnet_model, tracking)
    assert resnet.cut_name == "stage2.block1"
    assert resnet.activation_shape == (128, 16, 16)
    assert resnet.spatial_site == (8, 8)
    assert resnet.next_transition_batch is resnet_blocks[3]
    assert resnet.residual_identity is True
    assert resnet.device == "cpu"

    vgg_stages = torch.nn.ModuleList(
        torch.nn.ModuleList(torch.nn.Identity() for _ in range(count)) for count in (3, 3, 5, 5, 5)
    )
    vgg_model = SimpleNamespace(stages=vgg_stages)
    vgg = make_vgg_intervention_adapter(vgg_model, tracking)
    assert vgg.cut_name == "stage2.conv1"
    assert vgg.activation_shape == (128, 16, 16)
    assert vgg.spatial_site == (8, 8)
    assert vgg.next_transition_batch is vgg_stages[1][1]
    assert vgg.residual_identity is False
    assert vgg.device == "cpu"


def test_explicit_adapter_device_must_match_model_state_without_moving_it() -> None:
    class ParameterizedResNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))
            self.blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(8))

    model = ParameterizedResNet()
    tracking = SimpleNamespace(
        encode=lambda model, values, cut: values,
        suffix=lambda model, values, cut: values,
    )
    with pytest.raises(ValueError, match="differs from model state devices"):
        make_resnet_intervention_adapter(model, tracking, device="cuda:0")
    assert next(model.parameters()).device.type == "cpu"


def test_collect_plateau_checkpoint_preserves_contexts_and_raw_fields() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(12, 2)).astype(np.float32)
    test = rng.normal(size=(8, 2)).astype(np.float32)
    labels = np.asarray((0, 1, 2, 0, 1, 2, 0, 1), dtype=np.int64)
    settings = PlateauCollectionSettings(
        root_seed=11,
        covariance_fit_images=12,
        centers_per_kind=4,
        perturbation_directions_per_center=2,
        perturbation_steps=5,
        local_surface_grid_points=5,
        three_anchor_grid_points=5,
        hutchinson_probes=3,
        intervention_batch_size=8,
    )
    result = collect_plateau_checkpoint(
        _adapter(),
        epoch=5,
        train_images=train,
        train_image_ids=np.arange(100, 112),
        test_images=test,
        test_image_ids=np.arange(200, 208),
        test_labels=labels,
        settings=settings,
    )

    assert result.metadata["epoch"] == 5
    assert result.metadata["schema_version"] == 2
    assert result.metadata["compute_device"] == "cpu"
    assert result.metadata["persisted_array_policy"].startswith("CPU NumPy arrays")
    assert result.metadata["anchor_context_policy"] == "three separate frozen host contexts"
    assert all(
        array.dtype == np.float32
        for array in result.arrays.values()
        if np.issubdtype(array.dtype, np.floating)
    )
    assert result.arrays["path_logits"].shape == (2, 4, 2, 5, 3)
    assert result.arrays["anchor_logits_by_context"].shape == (5, 5, 3, 3)
    assert result.arrays["anchor_image_ids"].tolist() == [200, 201, 202]
    assert result.metadata["estimand_classification"]["path_response_l2_and_kl"].startswith(
        "architecture-adapted Heimersheim-Mendel"
    )
    np.testing.assert_allclose(
        result.arrays["hutchinson_transition_frobenius"],
        np.sqrt(8.0),
    )
    np.testing.assert_allclose(
        result.arrays["hutchinson_residual_frobenius"],
        np.sqrt(2.0),
    )
    np.testing.assert_allclose(
        result.arrays["local_transition_plane_jacobian"],
        np.sqrt(8.0),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        result.arrays["local_residual_update_plane_jacobian"],
        np.sqrt(2.0),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        result.arrays["anchor_transition_plane_jacobian_by_context"],
        np.sqrt(8.0),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        result.arrays["anchor_residual_update_plane_jacobian_by_context"],
        np.sqrt(2.0),
        rtol=1e-5,
    )
    assert result.metadata["plane_jacobian_display"] == {
        "local_array": "local_residual_update_plane_jacobian",
        "anchor_array": "anchor_residual_update_plane_jacobian_by_context",
        "estimand": "2D-plane-restricted ||D(T-I)||_F",
        "selection": "residual_update_for_residual_transition",
    }

    local_residual = result.arrays["local_transition_sites"] - result.arrays["local_grid_vectors"]
    local_expected = np.empty_like(result.arrays["local_residual_update_plane_jacobian"])
    for kind in range(2):
        for center in range(4):
            local_expected[kind, center] = plane_pullback_jacobian_frobenius(
                local_residual[kind, center],
                result.arrays["local_axis"],
                result.arrays["local_axis"],
                first_scale=float(result.metadata["robust_activation_scale"]),
                second_scale=float(result.metadata["robust_activation_scale"]),
            )
    np.testing.assert_allclose(
        result.arrays["local_residual_update_plane_jacobian"],
        local_expected,
    )

    anchor_plane = ThreeAnchorSlice.from_anchors(*result.arrays["anchor_vectors"])
    anchor_expected = np.empty_like(
        result.arrays["anchor_residual_update_plane_jacobian_by_context"]
    )
    for context in range(3):
        anchor_expected[context] = plane_pullback_jacobian_frobenius(
            result.arrays["anchor_transition_sites_by_context"][:, :, context]
            - result.arrays["anchor_grid_vectors"],
            result.arrays["anchor_axis"],
            result.arrays["anchor_axis"],
            first_scale=anchor_plane.alpha_scale,
            second_scale=anchor_plane.beta_scale,
        )
    np.testing.assert_allclose(
        result.arrays["anchor_residual_update_plane_jacobian_by_context"],
        anchor_expected,
    )

    coefficients = result.arrays["gaussian_direction_coefficients"]
    expected_targets = (
        empirical_covariance_gaussian(
            train,
            coefficients.reshape(-1, len(train)),
        )
        .reshape(4, 2, 2)
        .astype(np.float32)
    )
    np.testing.assert_allclose(result.arrays["gaussian_direction_targets"], expected_targets)
    for kind in range(2):
        for center in range(4):
            for direction_index in range(2):
                base = result.arrays["local_bases"][kind, center]
                target = expected_targets[center, direction_index]
                delta = target - base
                scale = result.arrays["path_direction_norms"][kind, center, direction_index]
                expected_last_vector = base + delta * (scale / np.linalg.norm(delta))
                observed_last_vector = result.arrays["path_intervention_vectors"][
                    kind, center, direction_index, -1
                ]
                np.testing.assert_allclose(observed_last_vector, expected_last_vector, atol=2e-6)
    np.testing.assert_array_equal(
        result.arrays["path_transition_sites"],
        2.0 * result.arrays["path_intervention_vectors"],
    )
    np.testing.assert_array_equal(
        result.arrays["local_transition_sites"],
        2.0 * result.arrays["local_grid_vectors"],
    )


def test_collection_is_deterministic_and_pickle_free() -> None:
    rng = np.random.default_rng(9)
    train = rng.normal(size=(10, 2)).astype(np.float32)
    test = rng.normal(size=(6, 2)).astype(np.float32)
    kwargs = {
        "adapter": _adapter(),
        "epoch": 1,
        "train_images": train,
        "train_image_ids": np.arange(10),
        "test_images": test,
        "test_image_ids": np.arange(20, 26),
        "test_labels": np.asarray((0, 1, 2, 0, 1, 2)),
        "settings": PlateauCollectionSettings(
            root_seed=3,
            covariance_fit_images=10,
            centers_per_kind=3,
            perturbation_directions_per_center=1,
            perturbation_steps=3,
            local_surface_grid_points=3,
            three_anchor_grid_points=3,
            hutchinson_probes=2,
            intervention_batch_size=16,
        ),
    }
    first = collect_plateau_checkpoint(**kwargs)
    second = collect_plateau_checkpoint(**kwargs)
    assert first.metadata_bytes() == second.metadata_bytes()
    for name in first.arrays:
        np.testing.assert_array_equal(first.arrays[name], second.arrays[name])

    files = package_plateau_checkpoint(first)
    second_files = package_plateau_checkpoint(second)
    assert files == second_files
    inventory = json.loads(files["inventory.json"])
    assert inventory["files"]["arrays.npz"]["size_bytes"] == len(files["arrays.npz"])
    with np.load(io.BytesIO(files["arrays.npz"]), allow_pickle=False) as archive:
        assert set(archive.files) == set(first.arrays)


def test_hutchinson_probes_measure_j_transpose_and_residual_branch() -> None:
    matrix = torch.tensor(((1.0, 2.0), (-3.0, 0.5)), dtype=torch.float32)
    model = TinyResidual()
    adapter = ActivationInterventionAdapter(
        architecture="test_residual",
        cut_name="block1",
        activation_shape=(2,),
        encode_batch=lambda values: values,
        suffix_batch=model.suffix,
        next_transition_batch=lambda values: values @ matrix.T,
        spatial_site=None,
        residual_identity=True,
    )
    rng = np.random.default_rng(19)
    train = rng.normal(size=(10, 2)).astype(np.float32)
    test = rng.normal(size=(6, 2)).astype(np.float32)
    result = collect_plateau_checkpoint(
        adapter,
        epoch=0,
        train_images=train,
        train_image_ids=np.arange(10),
        test_images=test,
        test_image_ids=np.arange(20, 26),
        test_labels=np.asarray((0, 1, 2, 0, 1, 2)),
        settings=PlateauCollectionSettings(
            root_seed=23,
            covariance_fit_images=10,
            centers_per_kind=3,
            perturbation_directions_per_center=1,
            perturbation_steps=3,
            local_surface_grid_points=3,
            three_anchor_grid_points=3,
            hutchinson_probes=4,
            intervention_batch_size=16,
        ),
    )
    probes = result.arrays["hutchinson_probes"]
    expected_raw = np.sum((probes @ matrix.numpy()) ** 2, axis=1)
    expected_adjusted = np.sum((probes @ (matrix.numpy() - np.eye(2))) ** 2, axis=1)
    np.testing.assert_allclose(
        result.arrays["hutchinson_transition_squared_norm_probes"],
        np.broadcast_to(expected_raw, (2, 3, 4)),
    )
    np.testing.assert_allclose(
        result.arrays["hutchinson_residual_squared_norm_probes"],
        np.broadcast_to(expected_adjusted, (2, 3, 4)),
    )


def test_spatial_interventions_retain_each_full_host_context() -> None:
    def suffix(values: torch.Tensor) -> torch.Tensor:
        sums = values.sum(dim=(2, 3))
        return torch.stack((sums[:, 0], sums[:, 1], sums[:, 0] - sums[:, 1]), dim=1)

    adapter = ActivationInterventionAdapter(
        architecture="test_spatial_residual",
        cut_name="stage2.block1",
        activation_shape=(2, 3, 3),
        encode_batch=lambda values: values,
        suffix_batch=suffix,
        next_transition_batch=lambda values: values,
        spatial_site=(1, 1),
        residual_identity=True,
    )
    rng = np.random.default_rng(29)
    train = rng.normal(size=(10, 2, 3, 3)).astype(np.float32)
    test = rng.normal(size=(6, 2, 3, 3)).astype(np.float32)
    # Make the non-intervened context visibly different for all three anchors.
    test[0] += 0.0
    test[1] += 10.0
    test[2] -= 7.0
    result = collect_plateau_checkpoint(
        adapter,
        epoch=0,
        train_images=train,
        train_image_ids=np.arange(10),
        test_images=test,
        test_image_ids=np.arange(20, 26),
        test_labels=np.asarray((0, 1, 2, 0, 1, 2)),
        settings=PlateauCollectionSettings(
            root_seed=31,
            covariance_fit_images=10,
            centers_per_kind=3,
            perturbation_directions_per_center=1,
            perturbation_steps=3,
            local_surface_grid_points=3,
            three_anchor_grid_points=3,
            hutchinson_probes=2,
            intervention_batch_size=16,
        ),
    )
    np.testing.assert_array_equal(result.arrays["anchor_host_activations"], test[:3])
    expected_references = suffix(torch.from_numpy(test[:3])).detach().numpy()
    np.testing.assert_allclose(result.arrays["anchor_reference_logits"], expected_references)
    same_grid_point = result.arrays["anchor_logits_by_context"][0, 0]
    assert not np.allclose(same_grid_point[0], same_grid_point[1])
    assert not np.allclose(same_grid_point[0], same_grid_point[2])
    np.testing.assert_allclose(result.arrays["hutchinson_residual_frobenius"], 0.0)
    np.testing.assert_allclose(
        result.arrays["local_transition_plane_jacobian"],
        np.sqrt(2.0),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(result.arrays["local_residual_update_plane_jacobian"], 0.0)
    np.testing.assert_allclose(
        result.arrays["anchor_transition_plane_jacobian_by_context"],
        np.sqrt(2.0),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.arrays["anchor_residual_update_plane_jacobian_by_context"],
        0.0,
    )


def test_nonresidual_collection_keeps_raw_fields_without_residual_update_fields() -> None:
    model = TinyResidual()
    adapter = ActivationInterventionAdapter(
        architecture="test_nonresidual",
        cut_name="layer1",
        activation_shape=(2,),
        encode_batch=lambda values: values,
        suffix_batch=model.suffix,
        next_transition_batch=model.transition,
        spatial_site=None,
        residual_identity=False,
    )
    rng = np.random.default_rng(41)
    train = rng.normal(size=(10, 2)).astype(np.float32)
    test = rng.normal(size=(6, 2)).astype(np.float32)
    result = collect_plateau_checkpoint(
        adapter,
        epoch=0,
        train_images=train,
        train_image_ids=np.arange(10),
        test_images=test,
        test_image_ids=np.arange(20, 26),
        test_labels=np.asarray((0, 1, 2, 0, 1, 2)),
        settings=PlateauCollectionSettings(
            root_seed=43,
            covariance_fit_images=10,
            centers_per_kind=3,
            perturbation_directions_per_center=1,
            perturbation_steps=3,
            local_surface_grid_points=3,
            three_anchor_grid_points=3,
            hutchinson_probes=2,
            intervention_batch_size=16,
        ),
    )

    assert "local_transition_plane_jacobian" in result.arrays
    assert "anchor_transition_plane_jacobian_by_context" in result.arrays
    assert "local_residual_update_plane_jacobian" not in result.arrays
    assert "anchor_residual_update_plane_jacobian_by_context" not in result.arrays
    assert "hutchinson_residual_frobenius" not in result.arrays
    assert result.metadata["plane_jacobian_display"] == {
        "local_array": "local_transition_plane_jacobian",
        "anchor_array": "anchor_transition_plane_jacobian_by_context",
        "estimand": "2D-plane-restricted ||DT||_F",
        "selection": "raw_transition_for_nonresidual_transition",
    }
    pipeline_module._validate_plateau_checkpoint_plane_contract(
        result.npz_bytes(),
        result.metadata,
        residual_identity=False,
        label="nonresidual fixture",
    )

    corrupted_arrays = dict(result.arrays)
    corrupted_arrays["local_transition_plane_jacobian"] = (
        corrupted_arrays["local_transition_plane_jacobian"] + 1.0
    )
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **corrupted_arrays)
    with pytest.raises(PipelineError, match="local raw transition plane field does not replay"):
        pipeline_module._validate_plateau_checkpoint_plane_contract(
            buffer.getvalue(),
            result.metadata,
            residual_identity=False,
            label="corrupted nonresidual fixture",
        )
