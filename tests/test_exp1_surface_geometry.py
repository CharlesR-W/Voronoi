from __future__ import annotations

import numpy as np
import pytest

from voronoi_lab.exp1.surface_geometry import (
    OrthonormalPerturbationPlane,
    ThreeAnchorSlice,
    base_response_fields,
    contextual_output_distance_fields,
    empirical_covariance_gaussian,
    output_distance_fields,
    path_directional_jacobian_norm,
    path_response_fields,
    plane_pullback_jacobian_frobenius,
)


def test_three_anchor_slice_reconstructs_all_anchors() -> None:
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([2.0, 0.0, 1.0])
    c = np.array([0.5, 3.0, 1.0])
    plane = ThreeAnchorSlice.from_anchors(a, b, c)

    np.testing.assert_allclose(plane.grid([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])[0, 0], a)
    np.testing.assert_allclose(plane.origin + plane.alpha_axis, b)
    np.testing.assert_allclose(
        plane.origin + plane.c_alpha * plane.alpha_axis + plane.beta_axis,
        c,
    )
    np.testing.assert_allclose(
        plane.anchor_coordinates,
        np.array([[0.0, 0.0], [1.0, 0.0], [0.25, 1.0]]),
    )


def test_three_anchor_slice_rejects_collinear_anchors() -> None:
    with pytest.raises(ValueError, match="collinear"):
        ThreeAnchorSlice.from_anchors([0.0, 0.0], [1.0, 0.0], [2.0, 0.0])


def test_orthonormal_perturbation_plane_uses_native_l2_scale() -> None:
    plane = OrthonormalPerturbationPlane.from_targets(
        np.zeros(3),
        np.array([2.0, 0.0, 0.0]),
        np.array([1.0, 3.0, 0.0]),
        scale=4.0,
    )
    grid = plane.grid([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(grid[1, 2], [4.0, 0.0, 0.0])
    np.testing.assert_allclose(grid[2, 1], [0.0, 4.0, 0.0])


def test_empirical_gaussian_uses_bank_covariance_factorization() -> None:
    bank = np.array([[0.0, 2.0], [2.0, 0.0], [4.0, 4.0]])
    coefficients = np.eye(3)
    samples = empirical_covariance_gaussian(bank, coefficients)
    expected = bank.mean(axis=0) + (bank - bank.mean(axis=0)) / np.sqrt(2.0)
    np.testing.assert_allclose(samples, expected)


def test_output_fields_preserve_raw_distances_and_rgb_definition() -> None:
    outputs = np.zeros((3, 3, 2))
    outputs[..., 0] = np.arange(3)[None, :]
    outputs[..., 1] = np.arange(3)[:, None]
    references = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    distances, rgb = output_distance_fields(outputs, references)

    np.testing.assert_allclose(distances[0, 0], [0.0, 2.0, 2.0])
    assert rgb.shape == (3, 3, 3)
    assert rgb[0, 0, 0] == 1.0
    assert np.all((rgb >= 0.0) & (rgb <= 1.0))


def test_plane_pullback_jacobian_matches_linear_map() -> None:
    first = np.linspace(-1.0, 1.0, 7)
    second = np.linspace(-2.0, 2.0, 9)
    x, y = np.meshgrid(first, second, indexing="xy")
    outputs = np.stack((2.0 * x + 3.0 * y, -x + 4.0 * y), axis=-1)

    observed = plane_pullback_jacobian_frobenius(
        outputs,
        first,
        second,
        first_scale=2.0,
        second_scale=5.0,
    )
    expected = np.sqrt((2.0 / 2.0) ** 2 + (-1.0 / 2.0) ** 2 + (3.0 / 5.0) ** 2 + (4.0 / 5.0) ** 2)
    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_base_response_is_zero_at_the_base_and_kl_is_nonnegative() -> None:
    logits = np.zeros((3, 3, 3))
    logits[1, 1] = np.array([1.0, -1.0, 0.5])
    l2, kl = base_response_fields(logits, logits[1, 1])
    assert l2[1, 1] == 0.0
    assert kl[1, 1] == pytest.approx(0.0, abs=1e-15)
    assert np.all(kl >= 0)


def test_contextual_output_distances_do_not_mix_host_contexts() -> None:
    axis = np.linspace(-1.0, 1.0, 5)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    outputs = np.stack(
        (
            np.stack((xx, yy), axis=-1),
            np.stack((xx + 10.0, yy), axis=-1),
            np.stack((xx, yy - 10.0), axis=-1),
        ),
        axis=2,
    )
    references = np.asarray(((0.0, 0.0), (10.0, 0.0), (0.0, -10.0)))
    distances, rgb = contextual_output_distance_fields(outputs, references)
    assert distances.shape == (5, 5, 3)
    assert rgb.shape == (5, 5, 3)
    np.testing.assert_allclose(distances[:, :, 0], distances[:, :, 1])
    np.testing.assert_allclose(distances[:, :, 0], distances[:, :, 2])


def test_path_response_and_directional_jacobian_fields() -> None:
    coefficients = np.linspace(0.0, 1.0, 7)
    slopes = np.asarray(((2.0, -1.0), (3.0, 4.0)))
    intercepts = np.asarray(((0.5, -0.5), (1.0, 2.0)))
    outputs = intercepts[:, None, :] + coefficients[None, :, None] * slopes[:, None, :]
    l2, kl = path_response_fields(outputs, intercepts)
    assert l2.shape == kl.shape == (2, 7)
    assert np.all(kl >= 0)
    observed = path_directional_jacobian_norm(
        outputs,
        coefficients,
        np.asarray((2.0, 5.0)),
    )
    expected = np.asarray((np.sqrt(5.0) / 2.0, 1.0))[:, None]
    np.testing.assert_allclose(observed, np.broadcast_to(expected, observed.shape), atol=1e-12)
