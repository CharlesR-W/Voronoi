from __future__ import annotations

import io
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from voronoi_lab.exp1.tracking2_vgg import (  # noqa: E402
    Tracking2VGGAdapter,
    load_tracking2_vgg_manifest,
    resolve_vgg_cut,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "configs/inputs/tracking2_vgg_seed0.yaml"
EXTERNAL_ROOT = PROJECT_ROOT.parent / "Experiments/Tracking2"


def test_vgg_manifest_declares_matched_stage_end_cut() -> None:
    manifest = load_tracking2_vgg_manifest(MANIFEST)
    assert manifest.architecture.class_name == "InstrumentedVGG19"
    assert tuple(item.epoch for item in manifest.checkpoints) == (0, 1, 5, 20, 100)
    assert resolve_vgg_cut("stage2.conv2").activation_shape == (128, 16, 16)


@pytest.mark.skipif(not EXTERNAL_ROOT.is_dir(), reason="local Tracking2 checkout unavailable")
def test_vgg_adapter_validates_and_strict_loads_real_checkpoint() -> None:
    adapter = Tracking2VGGAdapter.from_yaml(MANIFEST, root_override=EXTERNAL_ROOT)
    validated = adapter.validate_all()
    assert len(validated) == 9
    model = adapter.load_model(0, device="cpu")
    images = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    with torch.no_grad():
        activation = adapter.encode(model, images, "stage2.conv2")
        split_logits = adapter.suffix(model, activation, "stage2.conv2")
        full_logits = model(images)
    assert activation.shape == (1, 128, 16, 16)
    torch.testing.assert_close(split_logits, full_logits, rtol=0, atol=0)


def test_vgg_checkpoint_dtype_inventory_rejects_tampering(tmp_path: Path) -> None:
    manifest = load_tracking2_vgg_manifest(MANIFEST)
    state = {"value": torch.ones(1)}
    buffer = io.BytesIO()
    torch.save(state, buffer)
    checkpoint = manifest.checkpoints[0]
    path = tmp_path / checkpoint.path
    path.parent.mkdir(parents=True)
    path.write_bytes(buffer.getvalue())
    # The adapter rejects this before attempting to construct a model because
    # both its size and digest differ from the signed external reference.
    adapter = Tracking2VGGAdapter(manifest, root_override=tmp_path)
    with pytest.raises(Exception, match="does not match manifest"):
        adapter.load_model(0)
