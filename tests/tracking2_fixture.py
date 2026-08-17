"""Self-contained, hash-pinned Tracking2 input fixture construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from voronoi_lab.config import LabConfig

_MODULE_NAMES = (
    "stage0",
    "stage1.resblk1",
    "stage1.resblk2",
    "stage2.resblk1",
    "stage2.resblk2",
    "stage3.resblk1",
    "stage3.resblk2",
    "stage4.resblk1",
    "stage4.resblk2",
    "final_linear",
)


def _write_reference(root: Path, relative: str, content: bytes) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": relative,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def with_tracking2_fixture(config: LabConfig, tmp_path: Path) -> LabConfig:
    """Write a complete tiny external input tree and return its signed config."""

    external = tmp_path / "external"
    external.mkdir()
    epochs = tuple(config.experiment1.checkpoints)
    baseline = {"loss": 1.0, "accuracy": 0.8, "error": 0.2}
    sources = ("random", *(str(epoch) for epoch in epochs))
    interventions = []
    for module_index, module in enumerate(_MODULE_NAMES):
        for source in sources:
            loss = 1.1 if source == "random" else 1.0
            accuracy = 0.7 if source == "random" else 0.8
            error = 1.0 - accuracy
            interventions.append(
                {
                    "module_index": module_index,
                    "module": module,
                    "source": source,
                    "loss": loss,
                    "accuracy": accuracy,
                    "error": error,
                    "delta_loss": loss - baseline["loss"],
                    "delta_error": error - baseline["error"],
                }
            )
    transplant = {
        "schema_version": 1,
        "experiment": "resnet18_block_criticality",
        "status": "MEASURED",
        "config": {
            "output": "artifacts/resnet_criticality/seed0",
            "data_root": "data",
            "fake_data": False,
            "train_size": 8,
            "test_size": 4,
            "epochs": epochs[-1],
            "batch_size": 4,
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "checkpoint_epochs": list(epochs),
            "lr_milestones": [1],
            "width": 64,
            "seed": 0,
            "device": "cpu",
            "amp": False,
        },
        "device": "cpu",
        "module_names": list(_MODULE_NAMES),
        "baseline": baseline,
        "training": [{"epoch": epoch, **baseline} for epoch in epochs],
        "interventions": interventions,
        "runtime_seconds": 1.0,
    }
    transplant_bytes = json.dumps(transplant, allow_nan=False).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "name": "test-fixture",
        "lineage_quality": "exploratory_legacy",
        "lineage_note": "Synthetic test fixture with bounded lineage.",
        "root": ".",
        "observed_repository_revision": "b" * 40,
        "read_only": True,
        "architecture": {
            "module": "tracking2.models",
            "class_name": "InstrumentedResNet18V2",
            "width": 64,
            "num_classes": 10,
            "source": _write_reference(
                external,
                "src/tracking2/models.py",
                b"# fixture model\n",
            ),
            "state_dict_tensors": 1,
            "state_dict_parameters": 1,
            "state_dict_dtype": "float32",
        },
        "training": {
            "seed": 0,
            "train_size": 8,
            "test_size": 4,
            "epochs": epochs[-1],
            "batch_size": 4,
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "checkpoint_epochs": list(epochs),
            "lr_milestones": [1],
            "width": 64,
            "device": "cpu",
            "amp": False,
            "target_epoch": epochs[-1],
        },
        "checkpoints": [
            {
                "epoch": epoch,
                **_write_reference(
                    external,
                    f"artifacts/checkpoint_epoch{epoch}.pt",
                    f"checkpoint-{epoch}".encode("ascii"),
                ),
            }
            for epoch in epochs
        ],
        "datasets": {
            "backend": "parquet",
            "train": _write_reference(external, "data/train.parquet", b"train"),
            "test": _write_reference(external, "data/test.parquet", b"test"),
        },
        "transplant": {
            "schema_version": 1,
            "experiment": "resnet18_block_criticality",
            "status": "MEASURED",
            "file": _write_reference(
                external,
                "artifacts/resnet_criticality.json",
                transplant_bytes,
            ),
        },
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    inputs = config.inputs.tracking2.model_copy(
        update={
            "root": Path("external"),
            "manifest": Path("manifest.yaml"),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
    )
    return config.model_copy(
        update={"inputs": config.inputs.model_copy(update={"tracking2": inputs})}
    )


__all__ = ["with_tracking2_fixture"]
