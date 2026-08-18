from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from voronoi_lab.exp1.synthetic_task import (  # noqa: E402
    GaussianMixtureConfig,
    ResidualMLPConfig,
    SyntheticArtifactError,
    SyntheticResidualMLP,
    SyntheticTrainingConfig,
    generate_gaussian_mixture,
    load_synthetic_artifact,
    run_synthetic_residual_task,
    write_synthetic_artifact,
)


def _small_run():
    return run_synthetic_residual_task(
        dataset_config=GaussianMixtureConfig(
            seed=11,
            train_samples_per_class=48,
            test_samples_per_class=24,
            standard_deviation=0.5,
        ),
        training_config=SyntheticTrainingConfig(
            seed=12,
            epochs=5,
            checkpoint_epochs=(0, 1, 3, 5),
            batch_size=24,
        ),
    )


def test_gaussian_mixture_is_deterministic_balanced_and_split() -> None:
    config = GaussianMixtureConfig(
        seed=7,
        train_samples_per_class=9,
        test_samples_per_class=4,
    )
    first = generate_gaussian_mixture(config)
    second = generate_gaussian_mixture(config)

    np.testing.assert_array_equal(first.train_inputs, second.train_inputs)
    np.testing.assert_array_equal(first.train_labels, second.train_labels)
    np.testing.assert_array_equal(first.test_inputs, second.test_inputs)
    assert first.train_inputs.dtype == first.test_inputs.dtype == np.float32
    assert first.train_labels.dtype == first.test_labels.dtype == np.int64
    np.testing.assert_array_equal(np.bincount(first.train_labels), [9, 9, 9])
    np.testing.assert_array_equal(np.bincount(first.test_labels), [4, 4, 4])
    assert not np.array_equal(first.train_inputs[:12], first.test_inputs)


def test_residual_model_has_exact_split_roundtrip_at_every_block() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        model = SyntheticResidualMLP(ResidualMLPConfig(width=32, blocks=4)).eval()
    inputs = torch.randn(13, 2)
    expected = model(inputs)

    assert model.cut_names == ("block1", "block2", "block3", "block4")
    assert not any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()
    )
    assert not any(isinstance(module, torch.nn.LayerNorm) for module in model.modules())
    for index, name in enumerate(model.cut_names):
        encoded = model.encode_to_block(inputs, index)
        observed = model.forward_from_block(encoded, name)
        assert encoded.shape == (13, 32)
        assert torch.equal(observed, expected)


def test_training_is_deterministic_and_records_progress_and_checkpoints() -> None:
    first = _small_run()
    second = _small_run()

    assert first.training_config.checkpoint_epochs == (0, 1, 3, 5)
    assert first.checkpoint_epochs == (0, 1, 3, 5)
    assert len(first.metrics) == 6
    assert first.metrics == second.metrics
    for left, right in zip(first.checkpoints, second.checkpoints, strict=True):
        assert left.epoch == right.epoch
        assert tuple(left.state) == tuple(right.state)
        for name in left.state:
            np.testing.assert_array_equal(left.state[name], right.state[name])

    assert first.metrics[-1].train_accuracy > 0.95
    assert first.metrics[-1].test_accuracy > 0.95
    assert first.metrics[-1].test_loss < first.metrics[0].test_loss
    assert SyntheticTrainingConfig().checkpoint_epochs == (0, 1, 5, 20, 100)


def test_pickle_free_artifact_roundtrip_includes_inventory_and_models(tmp_path: Path) -> None:
    run = _small_run()
    inventory = write_synthetic_artifact(tmp_path / "synthetic", run)
    loaded = load_synthetic_artifact(tmp_path / "synthetic")

    assert inventory.checkpoint_epochs == loaded.inventory.checkpoint_epochs == (0, 1, 3, 5)
    assert loaded.metrics == run.metrics
    np.testing.assert_array_equal(loaded.dataset.train_inputs, run.dataset.train_inputs)
    assert [item.test_accuracy for item in inventory.checkpoints] == [
        run.metrics[epoch].test_accuracy for epoch in run.checkpoint_epochs
    ]

    inputs = torch.from_numpy(run.dataset.test_inputs[:8])
    for epoch in run.checkpoint_epochs:
        expected = run.model_at(epoch)(inputs)
        observed = loaded.load_model(epoch)(inputs)
        assert torch.equal(observed, expected)

    with np.load(tmp_path / "synthetic" / "dataset.npz", allow_pickle=False) as archive:
        assert all(not archive[name].dtype.hasobject for name in archive.files)
    manifest = json.loads((tmp_path / "synthetic" / "inventory.json").read_text())
    assert [row["epoch"] for row in manifest["checkpoints"]] == [0, 1, 3, 5]
    assert manifest["state_schema"]


def test_artifact_loader_rejects_changed_or_undeclared_bytes(tmp_path: Path) -> None:
    run = _small_run()
    root = tmp_path / "synthetic"
    write_synthetic_artifact(root, run)
    checkpoint = root / "checkpoints" / "epoch_00003.npz"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")

    with pytest.raises(SyntheticArtifactError, match="size mismatch"):
        load_synthetic_artifact(root)

    root = tmp_path / "second"
    write_synthetic_artifact(root, run)
    (root / "undeclared.txt").write_text("not part of the artifact")
    with pytest.raises(SyntheticArtifactError, match="file set"):
        load_synthetic_artifact(root)
