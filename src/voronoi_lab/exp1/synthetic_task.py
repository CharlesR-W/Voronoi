"""Deterministic residual-stream baseline for Experiment 1.

The task is deliberately small: three isotropic Gaussian classes in two input
dimensions and a normalization-free residual MLP.  It provides a controlled
training trajectory with the same checkpoint axis as the exploratory CIFAR
models, while keeping all serialized arrays pickle-free.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import nn


class SyntheticTaskError(RuntimeError):
    """Base class for invalid synthetic-task inputs or artifacts."""


class SyntheticArtifactError(SyntheticTaskError):
    """A serialized synthetic-task artifact failed strict validation."""


@dataclass(frozen=True, slots=True)
class GaussianMixtureConfig:
    """Parameters for the deterministic three-class, two-dimensional task."""

    seed: int = 20260817
    train_samples_per_class: int = 256
    test_samples_per_class: int = 128
    radius: float = 2.5
    standard_deviation: float = 0.65

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("train_samples_per_class", "test_samples_per_class"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("radius", "standard_deviation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class ResidualMLPConfig:
    """Architecture of the normalization-free residual baseline."""

    input_dim: int = 2
    width: int = 32
    blocks: int = 4
    classes: int = 3

    def __post_init__(self) -> None:
        for name in ("input_dim", "width", "blocks", "classes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.input_dim != 2:
            raise ValueError("the synthetic Gaussian-mixture task requires input_dim == 2")
        if self.classes != 3:
            raise ValueError("the synthetic Gaussian-mixture task requires classes == 3")


@dataclass(frozen=True, slots=True)
class SyntheticTrainingConfig:
    """Deterministic CPU SGD protocol and requested checkpoint axis."""

    seed: int = 20260818
    epochs: int = 100
    checkpoint_epochs: tuple[int, ...] = (0, 1, 5, 20, 100)
    batch_size: int = 64
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if not self.checkpoint_epochs:
            raise ValueError("checkpoint_epochs cannot be empty")
        if any(
            isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0
            for epoch in self.checkpoint_epochs
        ):
            raise ValueError("checkpoint_epochs must contain non-negative integers")
        if tuple(sorted(set(self.checkpoint_epochs))) != self.checkpoint_epochs:
            raise ValueError("checkpoint_epochs must be unique and strictly increasing")
        if self.checkpoint_epochs[0] != 0:
            raise ValueError("checkpoint_epochs must include epoch 0")
        if self.checkpoint_epochs[-1] != self.epochs:
            raise ValueError("checkpoint_epochs must end at epochs")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        for name in ("learning_rate", "momentum", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must lie in [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    train_inputs: np.ndarray
    train_labels: np.ndarray
    test_inputs: np.ndarray
    test_labels: np.ndarray
    class_centers: np.ndarray

    def __post_init__(self) -> None:
        _validate_dataset(self)


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    epoch: int
    state: MappingProxyType[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class SyntheticTrainingRun:
    dataset_config: GaussianMixtureConfig
    model_config: ResidualMLPConfig
    training_config: SyntheticTrainingConfig
    dataset: SyntheticDataset
    metrics: tuple[EpochMetrics, ...]
    checkpoints: tuple[CheckpointSnapshot, ...]

    @property
    def checkpoint_epochs(self) -> tuple[int, ...]:
        return tuple(checkpoint.epoch for checkpoint in self.checkpoints)

    def model_at(self, epoch: int) -> SyntheticResidualMLP:
        """Reconstruct one checkpoint with exact key, shape, and dtype checks."""

        for checkpoint in self.checkpoints:
            if checkpoint.epoch == epoch:
                model = _new_model(self.model_config, self.training_config.seed)
                _load_numpy_state(model, checkpoint.state)
                model.eval()
                return model
        available = ", ".join(str(item) for item in self.checkpoint_epochs)
        raise KeyError(f"checkpoint epoch {epoch} is unavailable; available: {available}")


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointFile:
    epoch: int
    file: ArtifactFile
    train_accuracy: float
    test_accuracy: float


@dataclass(frozen=True, slots=True)
class SyntheticArtifactInventory:
    root: Path
    dataset_file: ArtifactFile
    metrics_file: ArtifactFile
    checkpoints: tuple[CheckpointFile, ...]

    @property
    def checkpoint_epochs(self) -> tuple[int, ...]:
        return tuple(checkpoint.epoch for checkpoint in self.checkpoints)


@dataclass(frozen=True, slots=True)
class LoadedSyntheticArtifact:
    root: Path
    dataset_config: GaussianMixtureConfig
    model_config: ResidualMLPConfig
    training_config: SyntheticTrainingConfig
    dataset: SyntheticDataset
    metrics: tuple[EpochMetrics, ...]
    inventory: SyntheticArtifactInventory

    def load_model(self, epoch: int) -> SyntheticResidualMLP:
        for checkpoint in self.inventory.checkpoints:
            if checkpoint.epoch == epoch:
                state = _read_state_npz(self.root / checkpoint.file.path)
                model = _new_model(self.model_config, self.training_config.seed)
                _load_numpy_state(model, state)
                model.eval()
                return model
        available = ", ".join(str(item) for item in self.inventory.checkpoint_epochs)
        raise KeyError(f"checkpoint epoch {epoch} is unavailable; available: {available}")


class ResidualMLPBlock(nn.Module):
    """A normalization-free residual block with a bounded nonlinear branch."""

    def __init__(self, width: int, residual_scale: float) -> None:
        super().__init__()
        self.input = nn.Linear(width, width, device="cpu", dtype=torch.float32)
        self.output = nn.Linear(width, width, device="cpu", dtype=torch.float32)
        self.residual_scale = float(residual_scale)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        branch = self.output(torch.tanh(self.input(activation)))
        return activation + self.residual_scale * branch


class SyntheticResidualMLP(nn.Module):
    """Four-block residual-stream model with explicit split-forward methods."""

    def __init__(self, config: ResidualMLPConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = ResidualMLPConfig()
        self.config = config
        self.input_projection = nn.Linear(
            config.input_dim,
            config.width,
            device="cpu",
            dtype=torch.float32,
        )
        residual_scale = 1.0 / math.sqrt(config.blocks)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(config.width, residual_scale) for _ in range(config.blocks)
        )
        self.classifier = nn.Linear(
            config.width,
            config.classes,
            device="cpu",
            dtype=torch.float32,
        )

    @property
    def cut_names(self) -> tuple[str, ...]:
        return tuple(f"block{index + 1}" for index in range(len(self.blocks)))

    def _cut_index(self, cut: int | str) -> int:
        if isinstance(cut, bool):
            raise ValueError("cut must be a block index or name")
        if isinstance(cut, str):
            try:
                return self.cut_names.index(cut)
            except ValueError as exc:
                raise ValueError(f"unknown cut {cut!r}; expected one of {self.cut_names}") from exc
        if not isinstance(cut, int) or not 0 <= cut < len(self.blocks):
            raise ValueError(f"cut index must lie in [0, {len(self.blocks) - 1}]")
        return cut

    def encode_to_block(self, inputs: torch.Tensor, cut: int | str) -> torch.Tensor:
        index = self._cut_index(cut)
        activation = self.input_projection(inputs)
        for block in self.blocks[: index + 1]:
            activation = block(activation)
        return activation

    def forward_from_block(self, activation: torch.Tensor, cut: int | str) -> torch.Tensor:
        index = self._cut_index(cut)
        for block in self.blocks[index + 1 :]:
            activation = block(activation)
        return self.classifier(activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_from_block(
            self.encode_to_block(inputs, len(self.blocks) - 1),
            len(self.blocks) - 1,
        )


def generate_gaussian_mixture(config: GaussianMixtureConfig) -> SyntheticDataset:
    """Generate shuffled train/test splits from independent deterministic streams."""

    angles = np.arange(3, dtype=np.float64) * (2.0 * np.pi / 3.0)
    centers = config.radius * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    seed_sequence = np.random.SeedSequence(config.seed)
    train_seed, test_seed = seed_sequence.spawn(2)

    def split(
        samples_per_class: int,
        seed: np.random.SeedSequence,
    ) -> tuple[np.ndarray, np.ndarray]:
        generator = np.random.default_rng(seed)
        labels = np.repeat(np.arange(3, dtype=np.int64), samples_per_class)
        noise = generator.normal(
            loc=0.0,
            scale=config.standard_deviation,
            size=(len(labels), 2),
        )
        inputs = centers[labels] + noise
        permutation = generator.permutation(len(labels))
        return inputs[permutation].astype(np.float32), labels[permutation]

    train_inputs, train_labels = split(config.train_samples_per_class, train_seed)
    test_inputs, test_labels = split(config.test_samples_per_class, test_seed)
    return SyntheticDataset(
        train_inputs=train_inputs,
        train_labels=train_labels,
        test_inputs=test_inputs,
        test_labels=test_labels,
        class_centers=centers.astype(np.float32),
    )


def train_synthetic_residual_task(
    dataset: SyntheticDataset,
    *,
    dataset_config: GaussianMixtureConfig,
    model_config: ResidualMLPConfig | None = None,
    training_config: SyntheticTrainingConfig | None = None,
) -> SyntheticTrainingRun:
    """Train the residual MLP on CPU and retain the requested state trajectory."""

    if model_config is None:
        model_config = ResidualMLPConfig()
    if training_config is None:
        training_config = SyntheticTrainingConfig()
    _validate_dataset_against_configs(dataset, dataset_config, model_config)
    model = _new_model(model_config, training_config.seed)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=training_config.learning_rate,
        momentum=training_config.momentum,
        weight_decay=training_config.weight_decay,
    )
    train_inputs = torch.from_numpy(dataset.train_inputs)
    train_labels = torch.from_numpy(dataset.train_labels)
    test_inputs = torch.from_numpy(dataset.test_inputs)
    test_labels = torch.from_numpy(dataset.test_labels)
    permutation_rng = np.random.default_rng(
        np.random.SeedSequence([training_config.seed, 0x534744])
    )
    metrics: list[EpochMetrics] = []
    checkpoints: list[CheckpointSnapshot] = []

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for epoch in range(training_config.epochs + 1):
            train_loss, train_accuracy = _evaluate(model, train_inputs, train_labels)
            test_loss, test_accuracy = _evaluate(model, test_inputs, test_labels)
            metrics.append(
                EpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_accuracy=train_accuracy,
                    test_loss=test_loss,
                    test_accuracy=test_accuracy,
                )
            )
            if epoch in training_config.checkpoint_epochs:
                checkpoints.append(
                    CheckpointSnapshot(
                        epoch=epoch,
                        state=MappingProxyType(_numpy_state(model)),
                    )
                )
            if epoch == training_config.epochs:
                break

            model.train()
            permutation = permutation_rng.permutation(len(train_inputs))
            for start in range(0, len(permutation), training_config.batch_size):
                batch_indices = torch.from_numpy(
                    permutation[start : start + training_config.batch_size]
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(train_inputs[batch_indices])
                loss = torch.nn.functional.cross_entropy(logits, train_labels[batch_indices])
                loss.backward()
                optimizer.step()
    finally:
        torch.use_deterministic_algorithms(deterministic_before)

    return SyntheticTrainingRun(
        dataset_config=dataset_config,
        model_config=model_config,
        training_config=training_config,
        dataset=dataset,
        metrics=tuple(metrics),
        checkpoints=tuple(checkpoints),
    )


def run_synthetic_residual_task(
    *,
    dataset_config: GaussianMixtureConfig | None = None,
    model_config: ResidualMLPConfig | None = None,
    training_config: SyntheticTrainingConfig | None = None,
) -> SyntheticTrainingRun:
    """Generate the task and train its checkpoint trajectory."""

    if dataset_config is None:
        dataset_config = GaussianMixtureConfig()
    if model_config is None:
        model_config = ResidualMLPConfig()
    if training_config is None:
        training_config = SyntheticTrainingConfig()
    dataset = generate_gaussian_mixture(dataset_config)
    return train_synthetic_residual_task(
        dataset,
        dataset_config=dataset_config,
        model_config=model_config,
        training_config=training_config,
    )


def write_synthetic_artifact(
    root: str | Path,
    run: SyntheticTrainingRun,
) -> SyntheticArtifactInventory:
    """Write a self-describing, hash-indexed artifact without pickle payloads."""

    target = Path(root)
    if target.exists() and any(target.iterdir()):
        raise SyntheticArtifactError(f"artifact directory must be empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = target / "checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)

    dataset_path = target / "dataset.npz"
    _write_npz(
        dataset_path,
        {
            "train_inputs": run.dataset.train_inputs,
            "train_labels": run.dataset.train_labels,
            "test_inputs": run.dataset.test_inputs,
            "test_labels": run.dataset.test_labels,
            "class_centers": run.dataset.class_centers,
        },
    )
    dataset_file = _file_record(target, dataset_path)

    metrics_path = target / "training_progress.npz"
    _write_npz(metrics_path, _metrics_arrays(run.metrics))
    metrics_file = _file_record(target, metrics_path)
    metric_by_epoch = {metric.epoch: metric for metric in run.metrics}

    checkpoint_files: list[CheckpointFile] = []
    for checkpoint in run.checkpoints:
        path = checkpoint_directory / f"epoch_{checkpoint.epoch:05d}.npz"
        _write_npz(path, dict(checkpoint.state))
        metric = metric_by_epoch[checkpoint.epoch]
        checkpoint_files.append(
            CheckpointFile(
                epoch=checkpoint.epoch,
                file=_file_record(target, path),
                train_accuracy=metric.train_accuracy,
                test_accuracy=metric.test_accuracy,
            )
        )

    expected_state = _numpy_state(_new_model(run.model_config, run.training_config.seed))
    manifest = {
        "schema_version": 1,
        "task": "three_class_2d_gaussian_mixture",
        "architecture": "normalization_free_residual_mlp",
        "dataset_config": asdict(run.dataset_config),
        "model_config": asdict(run.model_config),
        "training_config": {
            **asdict(run.training_config),
            "checkpoint_epochs": list(run.training_config.checkpoint_epochs),
        },
        "state_schema": [
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            for name, array in expected_state.items()
        ],
        "dataset_file": asdict(dataset_file),
        "metrics_file": asdict(metrics_file),
        "checkpoints": [
            {
                "epoch": checkpoint.epoch,
                "file": asdict(checkpoint.file),
                "train_accuracy": checkpoint.train_accuracy,
                "test_accuracy": checkpoint.test_accuracy,
            }
            for checkpoint in checkpoint_files
        ],
    }
    manifest_path = target / "inventory.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return SyntheticArtifactInventory(
        root=target,
        dataset_file=dataset_file,
        metrics_file=metrics_file,
        checkpoints=tuple(checkpoint_files),
    )


def load_synthetic_artifact(root: str | Path) -> LoadedSyntheticArtifact:
    """Load and validate every declared byte, array, metric, and model checkpoint."""

    target = Path(root)
    manifest_path = target / "inventory.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticArtifactError(f"could not read artifact inventory: {exc}") from exc
    _require_keys(
        manifest,
        {
            "schema_version",
            "task",
            "architecture",
            "dataset_config",
            "model_config",
            "training_config",
            "state_schema",
            "dataset_file",
            "metrics_file",
            "checkpoints",
        },
        label="inventory",
    )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise SyntheticArtifactError("inventory.schema_version must equal 1")
    if manifest["task"] != "three_class_2d_gaussian_mixture":
        raise SyntheticArtifactError("inventory declares an unsupported task")
    if manifest["architecture"] != "normalization_free_residual_mlp":
        raise SyntheticArtifactError("inventory declares an unsupported architecture")

    try:
        dataset_config = GaussianMixtureConfig(**manifest["dataset_config"])
        model_config = ResidualMLPConfig(**manifest["model_config"])
        training_values = dict(manifest["training_config"])
        training_values["checkpoint_epochs"] = tuple(training_values["checkpoint_epochs"])
        training_config = SyntheticTrainingConfig(**training_values)
    except (TypeError, ValueError, KeyError) as exc:
        raise SyntheticArtifactError(f"invalid task configuration: {exc}") from exc

    dataset_file = _parse_file(manifest["dataset_file"], label="dataset_file")
    metrics_file = _parse_file(manifest["metrics_file"], label="metrics_file")
    checkpoint_files = _parse_checkpoints(manifest["checkpoints"])
    if dataset_file.path != "dataset.npz":
        raise SyntheticArtifactError("dataset_file must point to dataset.npz")
    if metrics_file.path != "training_progress.npz":
        raise SyntheticArtifactError("metrics_file must point to training_progress.npz")
    if any(
        checkpoint.file.path != f"checkpoints/epoch_{checkpoint.epoch:05d}.npz"
        for checkpoint in checkpoint_files
    ):
        raise SyntheticArtifactError("checkpoint paths do not match their epochs")
    if tuple(item.epoch for item in checkpoint_files) != training_config.checkpoint_epochs:
        raise SyntheticArtifactError("checkpoint inventory does not match checkpoint_epochs")
    declared_files = {
        "inventory.json",
        dataset_file.path,
        metrics_file.path,
        *(checkpoint.file.path for checkpoint in checkpoint_files),
    }
    observed_files = {
        path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()
    }
    if observed_files != declared_files:
        raise SyntheticArtifactError("artifact file set differs from its inventory")
    for record in (dataset_file, metrics_file):
        _validate_file(target, record)
    for checkpoint in checkpoint_files:
        _validate_file(target, checkpoint.file)

    dataset_arrays = _read_npz(
        target / dataset_file.path,
        expected_keys={
            "train_inputs",
            "train_labels",
            "test_inputs",
            "test_labels",
            "class_centers",
        },
    )
    dataset = SyntheticDataset(**dataset_arrays)
    _validate_dataset_against_configs(dataset, dataset_config, model_config)
    metrics = _metrics_from_arrays(_read_metrics_npz(target / metrics_file.path))
    if tuple(metric.epoch for metric in metrics) != tuple(range(training_config.epochs + 1)):
        raise SyntheticArtifactError("training progress must cover every epoch from 0 to epochs")

    expected_model = _new_model(model_config, training_config.seed)
    expected_state = _numpy_state(expected_model)
    _validate_state_schema(manifest["state_schema"], expected_state)
    metric_by_epoch = {metric.epoch: metric for metric in metrics}
    for checkpoint in checkpoint_files:
        state = _read_state_npz(target / checkpoint.file.path)
        _load_numpy_state(expected_model, state)
        metric = metric_by_epoch[checkpoint.epoch]
        if (
            checkpoint.train_accuracy != metric.train_accuracy
            or checkpoint.test_accuracy != metric.test_accuracy
        ):
            raise SyntheticArtifactError("checkpoint accuracy does not match training progress")

    inventory = SyntheticArtifactInventory(
        root=target,
        dataset_file=dataset_file,
        metrics_file=metrics_file,
        checkpoints=checkpoint_files,
    )
    return LoadedSyntheticArtifact(
        root=target,
        dataset_config=dataset_config,
        model_config=model_config,
        training_config=training_config,
        dataset=dataset,
        metrics=metrics,
        inventory=inventory,
    )


def _new_model(config: ResidualMLPConfig, seed: int) -> SyntheticResidualMLP:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SyntheticResidualMLP(config).cpu()


def _evaluate(
    model: SyntheticResidualMLP,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).to(torch.float64).mean()
    return float(loss.item()), float(accuracy.item())


def _numpy_state(model: nn.Module) -> dict[str, np.ndarray]:
    return {
        name: tensor.detach().cpu().numpy().copy() for name, tensor in model.state_dict().items()
    }


def _load_numpy_state(model: nn.Module, state: Any) -> None:
    expected = model.state_dict()
    if not hasattr(state, "keys") or set(state) != set(expected):
        raise SyntheticArtifactError("checkpoint state keys do not match the model")
    tensors: dict[str, torch.Tensor] = {}
    for name, expected_tensor in expected.items():
        value = np.asarray(state[name])
        if value.dtype.hasobject:
            raise SyntheticArtifactError(f"checkpoint tensor {name!r} has object dtype")
        if tuple(value.shape) != tuple(expected_tensor.shape):
            raise SyntheticArtifactError(f"checkpoint tensor {name!r} has the wrong shape")
        if str(value.dtype) != str(expected_tensor.detach().cpu().numpy().dtype):
            raise SyntheticArtifactError(f"checkpoint tensor {name!r} has the wrong dtype")
        if not np.all(np.isfinite(value)):
            raise SyntheticArtifactError(f"checkpoint tensor {name!r} is non-finite")
        tensors[name] = torch.from_numpy(value.copy())
    model.load_state_dict(tensors, strict=True)


def _validate_dataset(dataset: SyntheticDataset) -> None:
    arrays = {
        "train_inputs": dataset.train_inputs,
        "train_labels": dataset.train_labels,
        "test_inputs": dataset.test_inputs,
        "test_labels": dataset.test_labels,
        "class_centers": dataset.class_centers,
    }
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray):
            raise ValueError(f"{name} must be a NumPy array")
        if array.dtype.hasobject:
            raise ValueError(f"{name} cannot have object dtype")
    for split in ("train", "test"):
        inputs = arrays[f"{split}_inputs"]
        labels = arrays[f"{split}_labels"]
        if inputs.ndim != 2 or inputs.shape[1] != 2 or inputs.dtype != np.float32:
            raise ValueError(f"{split}_inputs must have float32 shape (samples, 2)")
        if labels.ndim != 1 or len(labels) != len(inputs) or labels.dtype != np.int64:
            raise ValueError(f"{split}_labels must be int64 with one value per sample")
        if not np.all(np.isfinite(inputs)):
            raise ValueError(f"{split}_inputs must be finite")
        if np.any(labels < 0) or np.any(labels >= 3):
            raise ValueError(f"{split}_labels must lie in [0, 2]")
    if dataset.class_centers.shape != (3, 2) or dataset.class_centers.dtype != np.float32:
        raise ValueError("class_centers must have float32 shape (3, 2)")
    if not np.all(np.isfinite(dataset.class_centers)):
        raise ValueError("class_centers must be finite")


def _validate_dataset_against_configs(
    dataset: SyntheticDataset,
    dataset_config: GaussianMixtureConfig,
    model_config: ResidualMLPConfig,
) -> None:
    _validate_dataset(dataset)
    if len(dataset.train_inputs) != 3 * dataset_config.train_samples_per_class:
        raise ValueError("training dataset size does not match dataset_config")
    if len(dataset.test_inputs) != 3 * dataset_config.test_samples_per_class:
        raise ValueError("test dataset size does not match dataset_config")
    if dataset.train_inputs.shape[1] != model_config.input_dim:
        raise ValueError("dataset input width does not match model_config")
    expected = generate_gaussian_mixture(dataset_config)
    for name in (
        "train_inputs",
        "train_labels",
        "test_inputs",
        "test_labels",
        "class_centers",
    ):
        if not np.array_equal(getattr(dataset, name), getattr(expected, name)):
            raise ValueError(f"dataset {name} does not match its deterministic configuration")


def _metrics_arrays(metrics: tuple[EpochMetrics, ...]) -> dict[str, np.ndarray]:
    return {
        "epoch": np.asarray([metric.epoch for metric in metrics], dtype=np.int64),
        "train_loss": np.asarray([metric.train_loss for metric in metrics], dtype=np.float64),
        "train_accuracy": np.asarray(
            [metric.train_accuracy for metric in metrics], dtype=np.float64
        ),
        "test_loss": np.asarray([metric.test_loss for metric in metrics], dtype=np.float64),
        "test_accuracy": np.asarray([metric.test_accuracy for metric in metrics], dtype=np.float64),
    }


def _metrics_from_arrays(arrays: dict[str, np.ndarray]) -> tuple[EpochMetrics, ...]:
    length = len(arrays["epoch"])
    if length == 0 or any(array.ndim != 1 or len(array) != length for array in arrays.values()):
        raise SyntheticArtifactError("training progress arrays must be non-empty aligned vectors")
    if arrays["epoch"].dtype != np.int64:
        raise SyntheticArtifactError("training progress epoch must have int64 dtype")
    for name in ("train_loss", "train_accuracy", "test_loss", "test_accuracy"):
        if arrays[name].dtype != np.float64 or not np.all(np.isfinite(arrays[name])):
            raise SyntheticArtifactError(f"training progress {name} must be finite float64")
    if np.any(arrays["train_loss"] < 0) or np.any(arrays["test_loss"] < 0):
        raise SyntheticArtifactError("training progress losses cannot be negative")
    if np.any((arrays["train_accuracy"] < 0) | (arrays["train_accuracy"] > 1)) or np.any(
        (arrays["test_accuracy"] < 0) | (arrays["test_accuracy"] > 1)
    ):
        raise SyntheticArtifactError("training progress accuracies must lie in [0, 1]")
    return tuple(
        EpochMetrics(
            epoch=int(arrays["epoch"][index]),
            train_loss=float(arrays["train_loss"][index]),
            train_accuracy=float(arrays["train_accuracy"][index]),
            test_loss=float(arrays["test_loss"][index]),
            test_accuracy=float(arrays["test_accuracy"][index]),
        )
        for index in range(length)
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if any(not isinstance(name, str) or not name for name in arrays):
        raise SyntheticArtifactError("NPZ array names must be non-empty strings")
    if any(np.asarray(array).dtype.hasobject for array in arrays.values()):
        raise SyntheticArtifactError("object arrays are not permitted in synthetic artifacts")
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _read_npz(path: Path, *, expected_keys: set[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                raise SyntheticArtifactError(f"NPZ array set differs from schema: {path}")
            return {name: archive[name].copy() for name in archive.files}
    except SyntheticArtifactError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise SyntheticArtifactError(f"could not read numeric NPZ {path}: {exc}") from exc


def _read_state_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise SyntheticArtifactError(f"checkpoint NPZ is empty: {path}")
            return {name: archive[name].copy() for name in archive.files}
    except SyntheticArtifactError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise SyntheticArtifactError(f"could not read checkpoint NPZ {path}: {exc}") from exc


def _read_metrics_npz(path: Path) -> dict[str, np.ndarray]:
    return _read_npz(
        path,
        expected_keys={"epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"},
    )


def _file_record(root: Path, path: Path) -> ArtifactFile:
    return ArtifactFile(
        path=path.relative_to(root).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_file(value: Any, *, label: str) -> ArtifactFile:
    _require_keys(value, {"path", "size_bytes", "sha256"}, label=label)
    path = value["path"]
    if (
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
    ):
        raise SyntheticArtifactError(f"{label}.path must be a safe relative path")
    size = value["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise SyntheticArtifactError(f"{label}.size_bytes must be a positive integer")
    digest = value["sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise SyntheticArtifactError(f"{label}.sha256 must be a lowercase SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise SyntheticArtifactError(f"{label}.sha256 must be a lowercase SHA-256 digest") from exc
    if digest.lower() != digest:
        raise SyntheticArtifactError(f"{label}.sha256 must be lowercase")
    return ArtifactFile(path=path, size_bytes=size, sha256=digest)


def _parse_checkpoints(value: Any) -> tuple[CheckpointFile, ...]:
    if not isinstance(value, list) or not value:
        raise SyntheticArtifactError("checkpoints must be a non-empty list")
    result: list[CheckpointFile] = []
    for index, row in enumerate(value):
        label = f"checkpoints[{index}]"
        _require_keys(
            row,
            {"epoch", "file", "train_accuracy", "test_accuracy"},
            label=label,
        )
        epoch = row["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise SyntheticArtifactError(f"{label}.epoch must be a non-negative integer")
        accuracies: list[float] = []
        for name in ("train_accuracy", "test_accuracy"):
            accuracy = row[name]
            if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
                raise SyntheticArtifactError(f"{label}.{name} must be a number")
            accuracy = float(accuracy)
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise SyntheticArtifactError(f"{label}.{name} must lie in [0, 1]")
            accuracies.append(accuracy)
        result.append(
            CheckpointFile(
                epoch=epoch,
                file=_parse_file(row["file"], label=f"{label}.file"),
                train_accuracy=accuracies[0],
                test_accuracy=accuracies[1],
            )
        )
    epochs = tuple(item.epoch for item in result)
    if tuple(sorted(set(epochs))) != epochs:
        raise SyntheticArtifactError("checkpoint epochs must be unique and strictly increasing")
    return tuple(result)


def _require_keys(value: Any, expected: set[str], *, label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise SyntheticArtifactError(f"{label} keys differ from the strict schema")


def _validate_file(root: Path, record: ArtifactFile) -> None:
    path = root / record.path
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SyntheticArtifactError(f"missing declared artifact file {record.path}") from exc
    if size != record.size_bytes:
        raise SyntheticArtifactError(f"size mismatch for artifact file {record.path}")
    if _sha256_file(path) != record.sha256:
        raise SyntheticArtifactError(f"SHA-256 mismatch for artifact file {record.path}")


def _validate_state_schema(value: Any, expected: dict[str, np.ndarray]) -> None:
    if not isinstance(value, list) or len(value) != len(expected):
        raise SyntheticArtifactError("state_schema does not match the model")
    observed: dict[str, tuple[tuple[int, ...], str]] = {}
    for index, row in enumerate(value):
        _require_keys(row, {"name", "shape", "dtype"}, label=f"state_schema[{index}]")
        name = row["name"]
        shape = row["shape"]
        dtype = row["dtype"]
        if not isinstance(name, str) or not isinstance(shape, list) or not isinstance(dtype, str):
            raise SyntheticArtifactError("state_schema has invalid field types")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape):
            raise SyntheticArtifactError("state_schema shapes must contain non-negative integers")
        if name in observed:
            raise SyntheticArtifactError("state_schema names must be unique")
        observed[name] = (tuple(shape), dtype)
    expected_schema = {
        name: (tuple(array.shape), str(array.dtype)) for name, array in expected.items()
    }
    if observed != expected_schema:
        raise SyntheticArtifactError("state_schema does not match the model")


__all__ = [
    "ArtifactFile",
    "CheckpointFile",
    "CheckpointSnapshot",
    "EpochMetrics",
    "GaussianMixtureConfig",
    "LoadedSyntheticArtifact",
    "ResidualMLPConfig",
    "SyntheticArtifactError",
    "SyntheticArtifactInventory",
    "SyntheticDataset",
    "SyntheticResidualMLP",
    "SyntheticTaskError",
    "SyntheticTrainingConfig",
    "SyntheticTrainingRun",
    "generate_gaussian_mixture",
    "load_synthetic_artifact",
    "run_synthetic_residual_task",
    "train_synthetic_residual_task",
    "write_synthetic_artifact",
]
