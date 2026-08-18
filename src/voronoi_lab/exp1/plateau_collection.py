"""Checkpoint-local activation-plateau and stable-region data collection.

This module keeps three estimands separate:

* downstream response along real and covariance-matched-Gaussian paths;
* the three-context RGB distance construction from the stable-regions paper;
* a new CNN diagnostic: the next-transition Jacobian restricted to a saved 2D
  activation plane, plus a Hutchinson Frobenius estimate at each center.

The collector stores the full frozen host activation maps used for every
intervention.  A sampled site vector alone is insufficient to replay a CNN
intervention because neighbouring spatial activations affect later convolutions.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from voronoi_lab.core import JSONLike, SeedDeriver, canonical_json_bytes, sha256_bytes

from .surface_geometry import (
    OrthonormalPerturbationPlane,
    ThreeAnchorSlice,
    base_response_fields,
    contextual_output_distance_fields,
    empirical_covariance_gaussian,
    median_pair_distance,
    path_directional_jacobian_norm,
    path_response_fields,
    plane_pullback_jacobian_frobenius,
)
from .tracking2 import Tracking2Adapter, resolve_cut
from .tracking2_vgg import Tracking2VGGAdapter, resolve_vgg_cut

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]
PLATEAU_CHECKPOINT_SCHEMA_VERSION = 2


class PlateauCollectionError(RuntimeError):
    """Raised when a declared plateau collection cannot be reproduced safely."""


@dataclass(frozen=True, slots=True)
class ActivationInterventionAdapter:
    """Minimal model boundary needed by the collector."""

    architecture: str
    cut_name: str
    activation_shape: tuple[int, ...]
    encode_batch: Callable[[Any], Any]
    suffix_batch: Callable[[Any], Any]
    next_transition_batch: Callable[[Any], Any]
    spatial_site: tuple[int, int] | None
    residual_identity: bool
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.architecture.strip() or not self.cut_name.strip():
            raise ValueError("adapter architecture and cut name cannot be blank")
        try:
            canonical_device = str(_torch().device(self.device))
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"adapter device is invalid: {self.device!r}") from error
        object.__setattr__(self, "device", canonical_device)
        if not self.activation_shape or any(size < 1 for size in self.activation_shape):
            raise ValueError("adapter activation shape must be positive")
        if self.spatial_site is None:
            if len(self.activation_shape) != 1:
                raise ValueError("whole-vector interventions require a 1D activation")
        else:
            if len(self.activation_shape) != 3:
                raise ValueError("spatial interventions require a CHW activation")
            row, column = self.spatial_site
            if not (0 <= row < self.activation_shape[1] and 0 <= column < self.activation_shape[2]):
                raise ValueError("adapter spatial site lies outside the activation map")

    @property
    def vector_width(self) -> int:
        return self.activation_shape[0]

    def site_vectors(self, activations: Any) -> Any:
        if self.spatial_site is None:
            return activations
        row, column = self.spatial_site
        return activations[:, :, row, column]

    def patch(self, host: Any, vectors: Any) -> Any:
        if self.spatial_site is None:
            return vectors
        row, column = self.spatial_site
        patched = host.unsqueeze(0).expand(len(vectors), *host.shape).clone()
        patched[:, :, row, column] = vectors
        return patched


def _model_device(model: Any, explicit: str | None) -> str:
    """Resolve one model device without moving model state implicitly."""

    torch = _torch()
    if explicit is not None:
        requested = str(torch.device(explicit))
    else:
        devices: set[str] = set()
        for accessor_name in ("parameters", "buffers"):
            accessor = getattr(model, accessor_name, None)
            if callable(accessor):
                devices.update(str(tensor.device) for tensor in accessor())
        if len(devices) > 1:
            raise ValueError(f"model state spans multiple devices: {sorted(devices)!r}")
        requested = next(iter(devices), "cpu")
    state_devices: set[str] = set()
    for accessor_name in ("parameters", "buffers"):
        accessor = getattr(model, accessor_name, None)
        if callable(accessor):
            state_devices.update(str(tensor.device) for tensor in accessor())
    if state_devices and state_devices != {requested}:
        raise ValueError(
            f"adapter device {requested!r} differs from model state devices "
            f"{sorted(state_devices)!r}"
        )
    return requested


def make_resnet_intervention_adapter(
    model: Any,
    tracking2: Tracking2Adapter,
    cut: str = "stage2.block1",
    *,
    device: str | None = None,
) -> ActivationInterventionAdapter:
    """Adapt the matched same-resolution ResNet transition."""

    spec = resolve_cut(cut)
    if spec.index + 1 >= len(model.blocks):
        raise ValueError("the selected ResNet cut has no following residual block")
    next_block = model.blocks[spec.index + 1]
    if resolve_cut(spec.index + 1).activation_shape != spec.activation_shape:
        raise ValueError("the selected ResNet transition changes activation shape")
    site = (spec.height // 2, spec.width // 2)
    return ActivationInterventionAdapter(
        architecture="tracking2_resnet18_v2_width64",
        cut_name=spec.name,
        activation_shape=spec.activation_shape,
        encode_batch=lambda images: tracking2.encode(model, images, spec),
        suffix_batch=lambda activations: tracking2.suffix(model, activations, spec),
        next_transition_batch=next_block,
        spatial_site=site,
        residual_identity=True,
        device=_model_device(model, device),
    )


def make_vgg_intervention_adapter(
    model: Any,
    tracking2: Tracking2VGGAdapter,
    cut: str = "stage2.conv1",
    *,
    device: str | None = None,
) -> ActivationInterventionAdapter:
    """Adapt the matched same-resolution VGG convolutional transition."""

    spec = resolve_vgg_cut(cut)
    next_spec = resolve_vgg_cut(spec.index + 1)
    if next_spec.activation_shape != spec.activation_shape:
        raise ValueError("the selected VGG transition changes activation shape")
    # The audited comparison is deliberately fixed to stage2.conv1 -> conv2.
    if spec.name != "stage2.conv1":
        raise ValueError("only the audited VGG stage2.conv1 transition is supported")
    next_module = model.stages[1][1]
    site = (spec.height // 2, spec.width // 2)
    return ActivationInterventionAdapter(
        architecture="tracking2_vgg19_bn_width1_classifier512",
        cut_name=spec.name,
        activation_shape=spec.activation_shape,
        encode_batch=lambda images: tracking2.encode(model, images, spec),
        suffix_batch=lambda activations: tracking2.suffix(model, activations, spec),
        next_transition_batch=next_module,
        spatial_site=site,
        residual_identity=False,
        device=_model_device(model, device),
    )


def make_synthetic_intervention_adapter(
    model: Any,
    cut: int | str = 1,
    *,
    device: str | None = None,
) -> ActivationInterventionAdapter:
    """Adapt one same-width transition of the synthetic residual MLP."""

    if isinstance(cut, str):
        try:
            index = model.cut_names.index(cut)
        except ValueError as error:
            raise ValueError(f"unknown synthetic cut: {cut!r}") from error
    elif isinstance(cut, bool) or not isinstance(cut, int) or not 0 <= cut < len(model.blocks):
        raise ValueError("synthetic cut must be a valid block index or name")
    else:
        index = cut
    if index + 1 >= len(model.blocks):
        raise ValueError("the selected synthetic cut has no following residual block")
    name = model.cut_names[index]
    return ActivationInterventionAdapter(
        architecture="synthetic_normalization_free_residual_mlp",
        cut_name=name,
        activation_shape=(model.config.width,),
        encode_batch=lambda values: model.encode_to_block(values, index),
        suffix_batch=lambda activations: model.forward_from_block(activations, index),
        next_transition_batch=model.blocks[index + 1],
        spatial_site=None,
        residual_identity=True,
        device=_model_device(model, device),
    )


@dataclass(frozen=True, slots=True)
class PlateauCollectionSettings:
    root_seed: int
    covariance_fit_images: int = 256
    centers_per_kind: int = 4
    perturbation_directions_per_center: int = 8
    perturbation_steps: int = 37
    perturbation_max_scale: float = 1.0
    local_surface_grid_points: int = 17
    local_surface_extent: float = 0.75
    three_anchor_grid_points: int = 21
    three_anchor_axis_min: float = -0.25
    three_anchor_axis_max: float = 1.25
    hutchinson_probes: int = 8
    intervention_batch_size: int = 64

    def __post_init__(self) -> None:
        positive = (
            self.covariance_fit_images,
            self.centers_per_kind,
            self.perturbation_directions_per_center,
            self.perturbation_steps,
            self.local_surface_grid_points,
            self.three_anchor_grid_points,
            self.hutchinson_probes,
            self.intervention_batch_size,
        )
        if self.root_seed < 0 or any(value < 1 for value in positive):
            raise ValueError("collection seed/counts must be nonnegative/positive")
        if self.centers_per_kind < 3:
            raise ValueError("at least three real and fake centers are required")
        if self.perturbation_steps < 3:
            raise ValueError("at least three path steps are required")
        for value in (self.local_surface_grid_points, self.three_anchor_grid_points):
            if value < 3 or value % 2 == 0:
                raise ValueError("surface grids must be odd and at least 3")
        if self.perturbation_max_scale <= 0 or self.local_surface_extent <= 0:
            raise ValueError("perturbation scales must be positive")
        if self.three_anchor_axis_min >= self.three_anchor_axis_max:
            raise ValueError("three-anchor axis bounds are reversed")


@dataclass(frozen=True, slots=True)
class PlateauCheckpointResult:
    metadata: Mapping[str, JSONLike]
    arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.metadata.get("schema_version") != PLATEAU_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("plateau checkpoint metadata has the wrong schema")
        if not self.arrays:
            raise ValueError("plateau checkpoint result cannot be empty")
        normalized: dict[str, np.ndarray] = {}
        for name, value in self.arrays.items():
            if not name or "/" in name:
                raise ValueError("plateau array names must be simple identifiers")
            array = np.asarray(value)
            if array.dtype.hasobject or array.size == 0:
                raise ValueError(f"plateau array {name!r} is empty or object-valued")
            if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
                raise ValueError(f"plateau array {name!r} contains non-finite values")
            normalized[name] = array
        object.__setattr__(self, "arrays", normalized)

    def npz_bytes(self) -> bytes:
        stream = io.BytesIO()
        np.savez_compressed(stream, **self.arrays)
        return stream.getvalue()

    def metadata_bytes(self) -> bytes:
        return canonical_json_bytes(dict(self.metadata))


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency
        raise PlateauCollectionError("plateau collection requires PyTorch") from error
    return torch


def _as_tensor(
    values: ArrayLike,
    *,
    like: Any | None = None,
    device: str | None = None,
) -> Any:
    torch = _torch()
    if like is not None and device is not None:
        raise ValueError("tensor conversion cannot specify both like and device")
    if like is None:
        return torch.as_tensor(values, dtype=torch.float32, device=device or "cpu")
    return torch.as_tensor(values, dtype=like.dtype, device=like.device)


def _site_vectors_numpy(
    adapter: ActivationInterventionAdapter,
    activations: Float32Array,
) -> Float32Array:
    """Extract persisted site vectors without copying saved arrays to an accelerator."""

    if adapter.spatial_site is None:
        return activations
    row, column = adapter.spatial_site
    return activations[:, :, row, column]


def _encode_all(
    adapter: ActivationInterventionAdapter,
    images: ArrayLike,
    *,
    batch_size: int,
) -> Float32Array:
    torch = _torch()
    tensor = _as_tensor(images, device=adapter.device)
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tensor), batch_size):
            encoded = adapter.encode_batch(tensor[start : start + batch_size])
            batches.append(encoded.detach().cpu().numpy().astype(np.float32, copy=False))
    values = np.concatenate(batches, axis=0)
    if values.shape[1:] != adapter.activation_shape:
        raise PlateauCollectionError(
            f"encoded activation shape {values.shape[1:]} differs from {adapter.activation_shape}"
        )
    return values


def _evaluate_vectors(
    adapter: ActivationInterventionAdapter,
    host: ArrayLike,
    vectors: ArrayLike,
    *,
    batch_size: int,
) -> tuple[Float32Array, Float32Array]:
    torch = _torch()
    host_tensor = _as_tensor(host, device=adapter.device)
    vector_tensor = _as_tensor(vectors, like=host_tensor)
    if vector_tensor.ndim != 2 or vector_tensor.shape[1] != adapter.vector_width:
        raise PlateauCollectionError("intervention vectors have the wrong shape")
    logits: list[np.ndarray] = []
    transition_sites: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(vector_tensor), batch_size):
            current = vector_tensor[start : start + batch_size]
            patched = adapter.patch(host_tensor, current)
            downstream = adapter.suffix_batch(patched)
            transitioned = adapter.next_transition_batch(patched)
            site = adapter.site_vectors(transitioned)
            if downstream.ndim != 2 or downstream.shape[0] != len(current):
                raise PlateauCollectionError(
                    "the suffix must return one flat logit vector per intervention"
                )
            if site.shape != (len(current), adapter.vector_width):
                raise PlateauCollectionError(
                    "the next transition must return one same-width site vector per intervention"
                )
            logits.append(downstream.detach().cpu().numpy().astype(np.float32, copy=False))
            transition_sites.append(site.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(logits, axis=0), np.concatenate(transition_sites, axis=0)


def _hutchinson_at_center(
    adapter: ActivationInterventionAdapter,
    host: ArrayLike,
    center: ArrayLike,
    probes: NDArray[np.float32],
) -> tuple[Float32Array, Float32Array | None]:
    torch = _torch()
    host_tensor = _as_tensor(host, device=adapter.device)
    center_tensor = _as_tensor(center, like=host_tensor).clone().requires_grad_(True)
    patched = adapter.patch(host_tensor, center_tensor.unsqueeze(0))
    transitioned = adapter.site_vectors(adapter.next_transition_batch(patched))[0]
    if transitioned.shape != center_tensor.shape:
        raise PlateauCollectionError("next transition changed the intervention-vector width")
    raw: list[float] = []
    adjusted: list[float] = []
    for index, probe in enumerate(probes):
        probe_tensor = _as_tensor(probe, like=center_tensor)
        gradient = torch.autograd.grad(
            torch.sum(transitioned * probe_tensor),
            center_tensor,
            retain_graph=index + 1 < len(probes),
        )[0]
        raw.append(float(torch.sum(gradient * gradient).detach().cpu()))
        if adapter.residual_identity:
            residual_gradient = gradient - probe_tensor
            adjusted.append(float(torch.sum(residual_gradient * residual_gradient).detach().cpu()))
    return np.asarray(raw, dtype=np.float32), (
        np.asarray(adjusted, dtype=np.float32) if adapter.residual_identity else None
    )


def _select_anchor_positions(labels: Int64Array) -> tuple[int, int, int]:
    selected: list[int] = []
    seen: set[int] = set()
    for position, label in enumerate(labels):
        if int(label) not in seen:
            selected.append(position)
            seen.add(int(label))
        if len(selected) == 3:
            return tuple(selected)  # type: ignore[return-value]
    raise PlateauCollectionError("the fixed intervention bank lacks three distinct labels")


def _random_plan(
    settings: PlateauCollectionSettings,
    bank_size: int,
    width: int,
) -> dict[str, np.ndarray]:
    if bank_size < max(3, settings.centers_per_kind):
        raise PlateauCollectionError("covariance bank is too small for the declared protocol")
    seed = SeedDeriver(settings.root_seed, ("exp1", "plateau_collection", "v1"))
    center_rng = np.random.default_rng(seed.derive("gaussian_centers", bank_size))
    direction_rng = np.random.default_rng(seed.derive("gaussian_direction_targets", bank_size))
    targets_rng = np.random.default_rng(seed.derive("targets", bank_size))
    probe_rng = np.random.default_rng(seed.derive("hutchinson", width))
    kinds = 2
    return {
        "gaussian_center_coefficients": center_rng.standard_normal(
            (settings.centers_per_kind, bank_size)
        ).astype(np.float32),
        # The same Gaussian target draw is used for the matched real and fake
        # base at a given (center, direction) coordinate.  Coefficients, rather
        # than ambient vectors, are shared across architectures/checkpoints so
        # that each model applies the same semantic random draw to its own
        # checkpoint-local empirical covariance.
        "gaussian_direction_coefficients": direction_rng.standard_normal(
            (
                settings.centers_per_kind,
                settings.perturbation_directions_per_center,
                bank_size,
            )
        ).astype(np.float32),
        "plane_target_positions": targets_rng.integers(
            0,
            bank_size,
            size=(kinds, settings.centers_per_kind, 2),
            dtype=np.int64,
        ),
        "hutchinson_probes": probe_rng.choice(
            np.asarray((-1.0, 1.0), dtype=np.float32),
            size=(settings.hutchinson_probes, width),
        ).astype(np.float32),
    }


def _nondegenerate_plane(
    base: Float32Array,
    bank: Float32Array,
    target_positions: NDArray[np.int64],
    *,
    scale: float,
) -> tuple[OrthonormalPerturbationPlane, tuple[int, int]]:
    # The deterministic fallback walks through the fixed bank if a sampled pair
    # is accidentally collinear. The chosen positions are recorded by the caller.
    first = int(target_positions[0])
    second = int(target_positions[1])
    for offset in range(len(bank)):
        first_position = (first + offset) % len(bank)
        second_position = (second + 2 * offset) % len(bank)
        try:
            plane = OrthonormalPerturbationPlane.from_targets(
                base,
                bank[first_position],
                bank[second_position],
                scale=scale,
            )
            return plane, (first_position, second_position)
        except ValueError:
            continue
    raise PlateauCollectionError("could not construct a nondegenerate perturbation plane")


def collect_plateau_checkpoint(
    adapter: ActivationInterventionAdapter,
    *,
    epoch: int,
    train_images: ArrayLike,
    train_image_ids: ArrayLike,
    test_images: ArrayLike,
    test_image_ids: ArrayLike,
    test_labels: ArrayLike,
    settings: PlateauCollectionSettings,
) -> PlateauCheckpointResult:
    """Collect raw, replayable data for one architecture/checkpoint shard."""

    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    train_ids = np.asarray(train_image_ids, dtype=np.int64)
    test_ids = np.asarray(test_image_ids, dtype=np.int64)
    labels = np.asarray(test_labels, dtype=np.int64)
    train_values = np.asarray(train_images, dtype=np.float32)
    test_values = np.asarray(test_images, dtype=np.float32)
    if train_ids.ndim != 1 or len(train_values) != len(train_ids):
        raise ValueError("training images and ids must align")
    if test_ids.ndim != 1 or labels.shape != test_ids.shape or len(test_values) != len(test_ids):
        raise ValueError("test images, ids, and labels must align")
    if len(np.unique(train_ids)) != len(train_ids) or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError("image ids must be unique within each split")
    fit_count = settings.covariance_fit_images
    if len(train_ids) < fit_count or len(test_ids) < settings.centers_per_kind:
        raise PlateauCollectionError("materialized banks are smaller than the protocol")

    train_hosts = _encode_all(
        adapter,
        train_values[:fit_count],
        batch_size=settings.intervention_batch_size,
    )
    test_hosts = _encode_all(
        adapter,
        test_values,
        batch_size=settings.intervention_batch_size,
    )
    train_sites = _site_vectors_numpy(adapter, train_hosts).astype(np.float32, copy=False)
    test_sites = _site_vectors_numpy(adapter, test_hosts).astype(np.float32, copy=False)
    if train_sites.shape != (fit_count, adapter.vector_width):
        raise PlateauCollectionError("covariance site bank has an unexpected shape")

    plan = _random_plan(settings, fit_count, adapter.vector_width)
    gaussian_centers = empirical_covariance_gaussian(
        train_sites,
        plan["gaussian_center_coefficients"],
    ).astype(np.float32)
    center_count = settings.centers_per_kind
    direction_count = settings.perturbation_directions_per_center
    direction_coefficients = plan["gaussian_direction_coefficients"]
    gaussian_direction_targets = (
        empirical_covariance_gaussian(
            train_sites,
            direction_coefficients.reshape(center_count * direction_count, fit_count),
        )
        .reshape(center_count, direction_count, adapter.vector_width)
        .astype(np.float32)
    )
    real_centers = test_sites[:center_count]
    center_hosts = test_hosts[:center_count]
    bases = np.stack((real_centers, gaussian_centers), axis=0).astype(np.float32)
    robust_scale = median_pair_distance(train_sites)

    example_logits, _example_transition = _evaluate_vectors(
        adapter,
        center_hosts[0],
        real_centers[0:1],
        batch_size=1,
    )
    logit_width = int(example_logits.shape[1])

    path_coefficients = np.linspace(
        0.0,
        settings.perturbation_max_scale,
        settings.perturbation_steps,
        dtype=np.float32,
    )
    path_logits = np.empty(
        (
            2,
            center_count,
            direction_count,
            settings.perturbation_steps,
            logit_width,
        ),
        dtype=np.float32,
    )
    path_transition = np.empty((*path_logits.shape[:-1], adapter.vector_width), dtype=np.float32)
    path_vectors = np.empty_like(path_transition)
    path_direction_norms = np.empty(path_logits.shape[:3], dtype=np.float32)
    base_logits = np.empty((*path_logits.shape[:3], path_logits.shape[-1]), dtype=np.float32)
    real_base_parity: list[float] = []
    for kind in range(2):
        for center_index in range(center_count):
            host = center_hosts[center_index]
            base = bases[kind, center_index]
            repeated_base_logits, _ = _evaluate_vectors(
                adapter,
                host,
                base[None, :],
                batch_size=1,
            )
            if kind == 0:
                clean_logits = (
                    adapter.suffix_batch(_as_tensor(host[None, ...])).detach().cpu().numpy()[0]
                )
                real_base_parity.append(
                    float(np.max(np.abs(clean_logits - repeated_base_logits[0])))
                )
            for direction_index in range(direction_count):
                target = gaussian_direction_targets[center_index, direction_index]
                delta = target - base
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm <= 1e-12:
                    raise PlateauCollectionError("a fixed perturbation target equals its center")
                base_norm = float(np.linalg.norm(base))
                native_norm = base_norm if base_norm > 1e-12 else robust_scale
                direction = delta * (native_norm / delta_norm)
                vectors = (base[None, :] + path_coefficients[:, None] * direction[None, :]).astype(
                    np.float32
                )
                logits, transition = _evaluate_vectors(
                    adapter,
                    host,
                    vectors,
                    batch_size=settings.intervention_batch_size,
                )
                path_logits[kind, center_index, direction_index] = logits
                path_transition[kind, center_index, direction_index] = transition
                path_vectors[kind, center_index, direction_index] = vectors
                path_direction_norms[kind, center_index, direction_index] = native_norm
                base_logits[kind, center_index, direction_index] = repeated_base_logits[0]

    path_l2, path_kl = path_response_fields(path_logits, base_logits)
    path_logit_jacobian = path_directional_jacobian_norm(
        path_logits,
        path_coefficients,
        path_direction_norms,
    )
    path_transition_jacobian = path_directional_jacobian_norm(
        path_transition,
        path_coefficients,
        path_direction_norms,
    )

    local_axis = np.linspace(
        -settings.local_surface_extent,
        settings.local_surface_extent,
        settings.local_surface_grid_points,
        dtype=np.float32,
    )
    grid = settings.local_surface_grid_points
    local_logits = np.empty((2, center_count, grid, grid, logit_width), dtype=np.float32)
    local_transition = np.empty(
        (2, center_count, grid, grid, adapter.vector_width), dtype=np.float32
    )
    local_grid_vectors = np.empty_like(local_transition)
    local_directions = np.empty((2, center_count, 2, adapter.vector_width), dtype=np.float32)
    local_l2 = np.empty((2, center_count, grid, grid), dtype=np.float32)
    local_kl = np.empty_like(local_l2)
    local_logit_jacobian = np.empty_like(local_l2)
    local_transition_jacobian = np.empty_like(local_l2)
    local_residual_update_jacobian = np.empty_like(local_l2) if adapter.residual_identity else None
    hutch_raw = np.empty((2, center_count, settings.hutchinson_probes), dtype=np.float32)
    hutch_adjusted = np.empty_like(hutch_raw) if adapter.residual_identity else None
    plane_targets = plan["plane_target_positions"]
    plane_targets_used = np.empty_like(plane_targets)
    for kind in range(2):
        for center_index in range(center_count):
            base = bases[kind, center_index]
            host = center_hosts[center_index]
            plane, used_positions = _nondegenerate_plane(
                base,
                train_sites,
                plane_targets[kind, center_index],
                scale=robust_scale,
            )
            plane_targets_used[kind, center_index] = used_positions
            local_directions[kind, center_index, 0] = plane.first
            local_directions[kind, center_index, 1] = plane.second
            vectors = (
                plane.grid(local_axis, local_axis)
                .reshape(-1, adapter.vector_width)
                .astype(np.float32)
            )
            logits, transition = _evaluate_vectors(
                adapter,
                host,
                vectors,
                batch_size=settings.intervention_batch_size,
            )
            logits_grid = logits.reshape(grid, grid, logit_width)
            transition_grid = transition.reshape(grid, grid, adapter.vector_width)
            local_logits[kind, center_index] = logits_grid
            local_transition[kind, center_index] = transition_grid
            local_grid_vectors[kind, center_index] = vectors.reshape(
                grid, grid, adapter.vector_width
            )
            base_position = grid // 2
            l2, kl = base_response_fields(logits_grid, logits_grid[base_position, base_position])
            local_l2[kind, center_index] = l2
            local_kl[kind, center_index] = kl
            local_logit_jacobian[kind, center_index] = plane_pullback_jacobian_frobenius(
                logits_grid,
                local_axis,
                local_axis,
                first_scale=robust_scale,
                second_scale=robust_scale,
            )
            local_transition_jacobian[kind, center_index] = plane_pullback_jacobian_frobenius(
                transition_grid,
                local_axis,
                local_axis,
                first_scale=robust_scale,
                second_scale=robust_scale,
            )
            if local_residual_update_jacobian is not None:
                residual_update_grid = transition_grid - local_grid_vectors[kind, center_index]
                local_residual_update_jacobian[kind, center_index] = (
                    plane_pullback_jacobian_frobenius(
                        residual_update_grid,
                        local_axis,
                        local_axis,
                        first_scale=robust_scale,
                        second_scale=robust_scale,
                    )
                )
            raw, adjusted = _hutchinson_at_center(
                adapter,
                host,
                base,
                plan["hutchinson_probes"],
            )
            hutch_raw[kind, center_index] = raw
            if hutch_adjusted is not None and adjusted is not None:
                hutch_adjusted[kind, center_index] = adjusted

    anchor_positions = _select_anchor_positions(labels)
    anchor_position_array = np.asarray(anchor_positions, dtype=np.int64)
    anchor_hosts = test_hosts[anchor_position_array]
    anchor_vectors = test_sites[anchor_position_array]
    anchor_plane = ThreeAnchorSlice.from_anchors(*anchor_vectors)
    anchor_axis = np.linspace(
        settings.three_anchor_axis_min,
        settings.three_anchor_axis_max,
        settings.three_anchor_grid_points,
        dtype=np.float32,
    )
    anchor_grid = anchor_plane.grid(anchor_axis, anchor_axis).astype(np.float32)
    anchor_logits = np.empty((len(anchor_axis), len(anchor_axis), 3, logit_width), dtype=np.float32)
    anchor_transition = np.empty(
        (len(anchor_axis), len(anchor_axis), 3, adapter.vector_width), dtype=np.float32
    )
    reference_logits = np.empty((3, logit_width), dtype=np.float32)
    reference_transition = np.empty((3, adapter.vector_width), dtype=np.float32)
    flattened_anchor_grid = anchor_grid.reshape(-1, adapter.vector_width)
    for context_index in range(3):
        logits, transition = _evaluate_vectors(
            adapter,
            anchor_hosts[context_index],
            flattened_anchor_grid,
            batch_size=settings.intervention_batch_size,
        )
        anchor_logits[:, :, context_index] = logits.reshape(
            len(anchor_axis), len(anchor_axis), logit_width
        )
        anchor_transition[:, :, context_index] = transition.reshape(
            len(anchor_axis), len(anchor_axis), adapter.vector_width
        )
        ref_logits, ref_transition = _evaluate_vectors(
            adapter,
            anchor_hosts[context_index],
            anchor_vectors[context_index : context_index + 1],
            batch_size=1,
        )
        reference_logits[context_index] = ref_logits[0]
        reference_transition[context_index] = ref_transition[0]
    anchor_distances, anchor_rgb_per_frame = contextual_output_distance_fields(
        anchor_logits,
        reference_logits,
    )
    anchor_logit_jacobian = np.empty((3, len(anchor_axis), len(anchor_axis)), dtype=np.float32)
    anchor_transition_jacobian = np.empty_like(anchor_logit_jacobian)
    anchor_residual_update_jacobian = (
        np.empty_like(anchor_logit_jacobian) if adapter.residual_identity else None
    )
    for context_index in range(3):
        anchor_logit_jacobian[context_index] = plane_pullback_jacobian_frobenius(
            anchor_logits[:, :, context_index],
            anchor_axis,
            anchor_axis,
            first_scale=anchor_plane.alpha_scale,
            second_scale=anchor_plane.beta_scale,
        )
        anchor_transition_jacobian[context_index] = plane_pullback_jacobian_frobenius(
            anchor_transition[:, :, context_index],
            anchor_axis,
            anchor_axis,
            first_scale=anchor_plane.alpha_scale,
            second_scale=anchor_plane.beta_scale,
        )
        if anchor_residual_update_jacobian is not None:
            residual_update_grid = anchor_transition[:, :, context_index] - anchor_grid
            anchor_residual_update_jacobian[context_index] = plane_pullback_jacobian_frobenius(
                residual_update_grid,
                anchor_axis,
                anchor_axis,
                first_scale=anchor_plane.alpha_scale,
                second_scale=anchor_plane.beta_scale,
            )

    arrays: dict[str, np.ndarray] = {
        "train_image_ids": train_ids[:fit_count],
        "train_site_vectors": train_sites,
        "center_image_ids": test_ids[:center_count],
        "center_labels": labels[:center_count],
        "center_host_activations": center_hosts,
        "center_vectors_real": real_centers,
        "center_vectors_gaussian": gaussian_centers,
        "gaussian_center_coefficients": plan["gaussian_center_coefficients"],
        "gaussian_direction_coefficients": direction_coefficients,
        "gaussian_direction_targets": gaussian_direction_targets,
        "path_coefficients": path_coefficients,
        "path_direction_norms": path_direction_norms,
        "path_intervention_vectors": path_vectors,
        "path_logits": path_logits,
        "path_transition_sites": path_transition,
        "path_response_l2": path_l2.astype(np.float32),
        "path_response_kl": path_kl.astype(np.float32),
        "path_logit_directional_jacobian": path_logit_jacobian.astype(np.float32),
        "path_transition_directional_jacobian": path_transition_jacobian.astype(np.float32),
        "local_axis": local_axis,
        "local_bases": bases,
        "local_plane_target_positions_requested": plane_targets,
        "local_plane_target_positions_used": plane_targets_used,
        "local_directions": local_directions,
        "local_grid_vectors": local_grid_vectors,
        "local_logits": local_logits,
        "local_transition_sites": local_transition,
        "local_response_l2": local_l2,
        "local_response_kl": local_kl,
        "local_logit_plane_jacobian": local_logit_jacobian,
        "local_transition_plane_jacobian": local_transition_jacobian,
        "hutchinson_probes": plan["hutchinson_probes"],
        "hutchinson_transition_squared_norm_probes": hutch_raw,
        "hutchinson_transition_frobenius": np.sqrt(hutch_raw.mean(axis=-1)),
        "anchor_positions_in_test_bank": anchor_position_array,
        "anchor_image_ids": test_ids[anchor_position_array],
        "anchor_labels": labels[anchor_position_array],
        "anchor_host_activations": anchor_hosts,
        "anchor_vectors": anchor_vectors,
        "anchor_coordinates": anchor_plane.anchor_coordinates.astype(np.float32),
        "anchor_axis": anchor_axis,
        "anchor_grid_vectors": anchor_grid,
        "anchor_reference_logits": reference_logits,
        "anchor_reference_transition_sites": reference_transition,
        "anchor_logits_by_context": anchor_logits,
        "anchor_transition_sites_by_context": anchor_transition,
        "anchor_output_distances": anchor_distances.astype(np.float32),
        "anchor_rgb_per_frame": anchor_rgb_per_frame.astype(np.float32),
        "anchor_logit_plane_jacobian_by_context": anchor_logit_jacobian,
        "anchor_transition_plane_jacobian_by_context": anchor_transition_jacobian,
    }
    if hutch_adjusted is not None:
        arrays["hutchinson_residual_squared_norm_probes"] = hutch_adjusted
        arrays["hutchinson_residual_frobenius"] = np.sqrt(hutch_adjusted.mean(axis=-1))
    if local_residual_update_jacobian is not None:
        arrays["local_residual_update_plane_jacobian"] = local_residual_update_jacobian
    if anchor_residual_update_jacobian is not None:
        arrays["anchor_residual_update_plane_jacobian_by_context"] = anchor_residual_update_jacobian

    array_inventory: dict[str, JSONLike] = {
        name: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for name, value in sorted(arrays.items())
    }
    metadata: dict[str, JSONLike] = {
        "schema_version": PLATEAU_CHECKPOINT_SCHEMA_VERSION,
        "architecture": adapter.architecture,
        "epoch": epoch,
        "cut": adapter.cut_name,
        "compute_device": adapter.device,
        "persisted_array_policy": "CPU NumPy arrays; all floating arrays use float32",
        "collection_settings": asdict(settings),
        "random_plan_namespace": ["exp1", "plateau_collection", "v1"],
        "activation_shape": list(adapter.activation_shape),
        "spatial_site": None if adapter.spatial_site is None else list(adapter.spatial_site),
        "host_context_policy": (
            "whole_residual_vector" if adapter.spatial_site is None else "full_chw_context_saved"
        ),
        "replay_policy": (
            "all float32 host activations and path/local/anchor intervention vectors are saved; "
            "the parent stage must bind this shard to exact dataset and checkpoint artifacts"
        ),
        "lineage_scope": "exploratory_single_seed_checkpoint",
        "residual_identity": adapter.residual_identity,
        "center_kinds": ["real", "empirical_covariance_gaussian"],
        "center_host_policy": "same indexed real host for matched real/fake centers",
        "path_direction_policy": (
            "paired empirical-covariance-Gaussian target draws; target-minus-base is "
            "rescaled to the base L2 norm"
        ),
        "path_pairing_policy": (
            "the same saved Gaussian target coefficients are used for matched real/fake "
            "centers and across architectures/checkpoints with the same bank size"
        ),
        "anchor_context_policy": "three separate frozen host contexts",
        "anchor_selection_policy": "first three distinct labels in fixed intervention order",
        "robust_activation_scale": robust_scale,
        "real_base_roundtrip_max_absolute_error": max(real_base_parity),
        "estimands": {
            "path_response": (
                "logit L2 and KL(base||perturbed) along straight paths toward "
                "covariance-Gaussian targets, with unit coefficient equal to one base L2 norm"
            ),
            "path_directional_jacobian": (
                "finite-difference output directional derivative norm per unit activation L2"
            ),
            "three_anchor_rgb": "per-context logit L2 with per-frame channel normalization",
            "plane_jacobian": (
                "finite-difference ||DT||_F restricted to two orthonormal native-L2 "
                "directions; retained as the raw next-transition field"
            ),
            "center_jacobian": (
                "Hutchinson estimate from saved output-space Rademacher probes of the "
                "next-transition site-to-site Frobenius norm"
            ),
            "residual_adjustment": (
                "finite-difference ||D(T-I)||_F local and anchor plane fields, plus "
                "||(J-I)^T v||^2 probes and sqrt(mean), are also stored"
                if adapter.residual_identity
                else "not applicable"
            ),
        },
        "plane_jacobian_display": {
            "local_array": (
                "local_residual_update_plane_jacobian"
                if adapter.residual_identity
                else "local_transition_plane_jacobian"
            ),
            "anchor_array": (
                "anchor_residual_update_plane_jacobian_by_context"
                if adapter.residual_identity
                else "anchor_transition_plane_jacobian_by_context"
            ),
            "estimand": (
                "2D-plane-restricted ||D(T-I)||_F"
                if adapter.residual_identity
                else "2D-plane-restricted ||DT||_F"
            ),
            "selection": (
                "residual_update_for_residual_transition"
                if adapter.residual_identity
                else "raw_transition_for_nonresidual_transition"
            ),
        },
        "estimand_classification": {
            "path_response_l2_and_kl": (
                "architecture-adapted Heimersheim-Mendel straight-mode analogue"
            ),
            "three_anchor_rgb": "architecture-adapted Janiak-et-al Appendix-C analogue",
            "path_and_plane_jacobians": "new hybrid CNN diagnostic",
            "local_real_fake_surfaces": "new hybrid CNN diagnostic",
            "hutchinson_center_jacobian": "new hybrid CNN diagnostic",
        },
        "source_separation_note": (
            "Only the response values along covariance-Gaussian straight paths and the "
            "three-context RGB construction are source-method analogues. All Jacobian "
            "fields and the paired local real/fake surfaces are explicitly new hybrid "
            "diagnostics."
        ),
        "array_inventory": array_inventory,
    }
    return PlateauCheckpointResult(metadata, arrays)


def package_plateau_checkpoint(result: PlateauCheckpointResult) -> dict[str, bytes]:
    """Return the two pickle-free payload files and a byte-level inventory."""

    arrays = result.npz_bytes()
    metadata = result.metadata_bytes()
    inventory: dict[str, JSONLike] = {
        "schema_version": 1,
        "files": {
            "arrays.npz": {"sha256": sha256_bytes(arrays), "size_bytes": len(arrays)},
            "metadata.json": {"sha256": sha256_bytes(metadata), "size_bytes": len(metadata)},
        },
    }
    return {
        "arrays.npz": arrays,
        "metadata.json": metadata,
        "inventory.json": canonical_json_bytes(inventory),
    }


__all__ = [
    "PLATEAU_CHECKPOINT_SCHEMA_VERSION",
    "ActivationInterventionAdapter",
    "PlateauCheckpointResult",
    "PlateauCollectionError",
    "PlateauCollectionSettings",
    "collect_plateau_checkpoint",
    "make_resnet_intervention_adapter",
    "make_synthetic_intervention_adapter",
    "make_vgg_intervention_adapter",
    "package_plateau_checkpoint",
]
