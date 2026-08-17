from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from voronoi_lab.exp1.tracking2 import (
    CUT_SPECS,
    ExternalInputValidationError,
    Tracking2Adapter,
    Tracking2InputManifest,
    Tracking2ManifestError,
    TransplantArtifactError,
    load_tracking2_manifest,
    resolve_cut,
)

MODULE_NAMES = (
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


def _write(root: Path, relative: str, data: bytes) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": relative,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _manifest_payload(
    root: Path,
    *,
    source: bytes = b"# verified model fixture\n",
    checkpoint: bytes = b"checkpoint fixture",
    transplant: bytes = b"{}",
    tensor_count: int = 1,
    parameter_count: int = 1,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "test-tracking2-input",
        "lineage_quality": "exploratory_legacy",
        "lineage_note": "Synthetic unit-test fixture with deliberately bounded lineage.",
        "root": str(root),
        "observed_repository_revision": "a" * 40,
        "read_only": True,
        "architecture": {
            "module": "tracking2.models",
            "class_name": "InstrumentedResNet18V2",
            "width": 64,
            "num_classes": 10,
            "source": _write(root, "src/tracking2/models.py", source),
            "state_dict_tensors": tensor_count,
            "state_dict_parameters": parameter_count,
            "state_dict_dtype": "float32",
        },
        "training": {
            "seed": 0,
            "train_size": 8,
            "test_size": 4,
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "checkpoint_epochs": [1],
            "lr_milestones": [1],
            "width": 64,
            "device": "cpu",
            "amp": False,
            "target_epoch": 1,
        },
        "checkpoints": [
            {
                "epoch": 1,
                **_write(root, "artifacts/checkpoint_epoch1.pt", checkpoint),
            }
        ],
        "datasets": {
            "backend": "parquet",
            "train": _write(root, "data/train.parquet", b"train"),
            "test": _write(root, "data/test.parquet", b"test"),
        },
        "transplant": {
            "schema_version": 1,
            "experiment": "resnet18_block_criticality",
            "status": "MEASURED",
            "file": _write(root, "artifacts/resnet_criticality.json", transplant),
        },
    }


def _legacy_transplant_bytes() -> bytes:
    baseline = {"loss": 1.0, "accuracy": 0.8, "error": 0.2}
    sources = ("random", "1")
    interventions = []
    for module_index, module in enumerate(MODULE_NAMES):
        for source in sources:
            loss = 1.5 if source == "random" else 1.0
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
    payload = {
        "schema_version": 1,
        "experiment": "resnet18_block_criticality",
        "status": "MEASURED",
        "config": {
            "output": "artifacts/resnet_criticality/seed0",
            "data_root": "data",
            "fake_data": False,
            "train_size": 8,
            "test_size": 4,
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "checkpoint_epochs": [1],
            "lr_milestones": [1],
            "width": 64,
            "seed": 0,
            "device": "cpu",
            "amp": False,
        },
        "device": "cpu",
        "module_names": list(MODULE_NAMES),
        "baseline": baseline,
        "training": [{"epoch": 1, **baseline}],
        "interventions": interventions,
        "runtime_seconds": 1.0,
    }
    return json.dumps(payload, allow_nan=False).encode()


def test_checked_in_manifest_parses_strictly() -> None:
    manifest = load_tracking2_manifest("configs/inputs/tracking2_seed0.yaml")
    assert manifest.lineage_quality == "exploratory_legacy"
    assert tuple(row.epoch for row in manifest.checkpoints) == (0, 1, 5, 20, 100)
    assert manifest.architecture.source.path == Path("src/tracking2/models.py")


def test_manifest_rejects_unknown_and_duplicate_keys(tmp_path: Path) -> None:
    payload = _manifest_payload(tmp_path)
    payload["unexpected"] = True
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(Tracking2ManifestError, match="unexpected"):
        load_tracking2_manifest(path)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema_version: 1\nschema_version: 1\n")
    with pytest.raises(Tracking2ManifestError, match="duplicate YAML key"):
        load_tracking2_manifest(duplicate)


def test_manifest_requires_distinct_train_and_test_source_files(tmp_path: Path) -> None:
    payload = _manifest_payload(tmp_path)
    payload["datasets"]["test"] = dict(payload["datasets"]["train"])
    payload["datasets"]["test"]["path"] = "data/copy-of-train.parquet"
    with pytest.raises(ValidationError, match="distinct content hashes"):
        Tracking2InputManifest.model_validate(payload)

    payload = _manifest_payload(tmp_path)
    payload["datasets"]["test"]["path"] = payload["datasets"]["train"]["path"]
    with pytest.raises(ValidationError, match="distinct paths"):
        Tracking2InputManifest.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "coerced_value"),
    (("amp", "false"), ("learning_rate", "0.05"), ("weight_decay", "0.0")),
)
def test_manifest_rejects_coercive_legacy_scalar_types(
    tmp_path: Path, field: str, coerced_value: object
) -> None:
    payload = _manifest_payload(tmp_path)
    payload["training"][field] = coerced_value
    with pytest.raises(ValidationError):
        Tracking2InputManifest.model_validate(payload)


@pytest.mark.parametrize(
    ("field_path", "value"),
    (("schema_version", True), ("read_only", 1), ("transplant.schema_version", True)),
)
def test_manifest_rejects_boolean_integer_literal_aliases(
    tmp_path: Path, field_path: str, value: object
) -> None:
    payload = _manifest_payload(tmp_path)
    if "." in field_path:
        parent, child = field_path.split(".")
        payload[parent][child] = value
    else:
        payload[field_path] = value
    with pytest.raises(ValidationError):
        Tracking2InputManifest.model_validate(payload)


def test_transplant_json_rejects_coercive_scalar_types(tmp_path: Path) -> None:
    transplant = json.loads(_legacy_transplant_bytes())
    transplant["config"]["amp"] = "false"
    payload = _manifest_payload(tmp_path, transplant=json.dumps(transplant).encode())
    adapter = Tracking2Adapter(Tracking2InputManifest.model_validate(payload))

    with pytest.raises(TransplantArtifactError, match="invalid transplant artifact"):
        adapter.transplant_rows()


def test_size_and_hash_mismatches_are_rejected(tmp_path: Path) -> None:
    manifest = Tracking2InputManifest.model_validate(_manifest_payload(tmp_path))
    adapter = Tracking2Adapter(manifest)
    train_path = tmp_path / manifest.datasets.train.path

    train_path.write_bytes(b"x")
    with pytest.raises(ExternalInputValidationError, match="size mismatch"):
        adapter.validate_file(manifest.datasets.train)

    train_path.write_bytes(b"TRAIN")
    assert train_path.stat().st_size == manifest.datasets.train.size_bytes
    with pytest.raises(ExternalInputValidationError, match="SHA-256 mismatch"):
        adapter.validate_file(manifest.datasets.train)


def test_cut_mapping_is_complete_and_unambiguous() -> None:
    assert len(CUT_SPECS) == 8
    assert [cut.index for cut in CUT_SPECS] == list(range(8))
    assert [cut.transplant_module_index for cut in CUT_SPECS] == list(range(1, 9))
    assert [cut.activation_shape for cut in CUT_SPECS] == [
        (64, 32, 32),
        (64, 32, 32),
        (128, 16, 16),
        (128, 16, 16),
        (256, 8, 8),
        (256, 8, 8),
        (512, 4, 4),
        (512, 4, 4),
    ]
    assert resolve_cut("stage2.block2") is CUT_SPECS[3]
    assert resolve_cut("stage2.resblk2") is CUT_SPECS[3]
    assert resolve_cut(3) is CUT_SPECS[3]
    assert [cut.index for cut in CUT_SPECS if cut.is_stage_end] == [1, 3, 5, 7]
    with pytest.raises(ValueError, match=r"\[0, 7\]"):
        resolve_cut(8)


def test_legacy_transplants_normalize_to_typed_residual_rows(tmp_path: Path) -> None:
    payload = _manifest_payload(tmp_path, transplant=_legacy_transplant_bytes())
    adapter = Tracking2Adapter(Tracking2InputManifest.model_validate(payload))

    rows = adapter.transplant_rows()

    assert len(rows) == 16
    assert rows[0].cut_name == "stage1.block1"
    assert rows[0].tracking2_module_name == "stage1.resblk1"
    assert rows[0].transplant_module_index == 1
    assert rows[0].source_kind == "random"
    assert rows[0].source_epoch is None
    assert rows[1].source_kind == "checkpoint"
    assert rows[1].source_epoch == 1
    assert rows[-1].cut_index == 7


def test_tiny_verified_torch_model_load_and_split_api(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source = b"""\
from __future__ import annotations
import torch
from torch import nn

class InstrumentedResNet18V2(nn.Module):
    def __init__(self, num_classes=10, width=64):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(8))
        self.block_names = (
            'stage1.resblk1', 'stage1.resblk2',
            'stage2.resblk1', 'stage2.resblk2',
            'stage3.resblk1', 'stage3.resblk2',
            'stage4.resblk1', 'stage4.resblk2',
        )

    def encode_to_block(self, x, block_index):
        return x

    def forward_from_block(self, representation, block_index):
        return representation * self.scale
"""
    checkpoint_buffer = io.BytesIO()
    torch.save({"scale": torch.tensor([2.0], dtype=torch.float32)}, checkpoint_buffer)
    payload = _manifest_payload(
        tmp_path,
        source=source,
        checkpoint=checkpoint_buffer.getvalue(),
        tensor_count=1,
        parameter_count=1,
    )
    adapter = Tracking2Adapter(Tracking2InputManifest.model_validate(payload))

    model = adapter.load_model(1)
    values = torch.tensor([3.0])

    assert not model.training
    assert torch.equal(adapter.encode(model, values, "stage1.block1"), values)
    assert torch.equal(adapter.suffix(model, values, 0), torch.tensor([6.0]))
    assert torch.equal(adapter.next_block(model, values, 0), values)
    with pytest.raises(ValueError, match="no next residual block"):
        adapter.next_block(model, values, 7)
