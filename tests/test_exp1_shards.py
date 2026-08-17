from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from voronoi_lab.core import ArtifactStore
from voronoi_lab.exp1.banks import make_site_bank
from voronoi_lab.exp1.shards import (
    gather_site_vectors,
    put_activation_shard,
    read_activation_shard,
)


def test_gather_preserves_site_bank_order_and_artifact_is_idempotent(tmp_path) -> None:
    site_bank = make_site_bank(
        [12, 4], height=2, width=2, max_sites_per_image=3, root_seed=2, namespace="cut"
    )
    activation = np.arange(2 * 5 * 2 * 2, dtype=np.float32).reshape(2, 5, 2, 2)
    unbound = gather_site_vectors(activation, [4, 12], site_bank)
    shard = replace(
        unbound,
        checkpoint=20,
        cut="stage2.block2",
        bank_id="bank-abc",
        shard_index=3,
    )
    for index, (image_id, row, column) in enumerate(
        zip(site_bank.image_ids, site_bank.rows, site_bank.columns, strict=True)
    ):
        batch_index = 0 if image_id == 4 else 1
        assert np.array_equal(shard.values[index], activation[batch_index, :, row, column])

    store = ArtifactStore(tmp_path / "artifacts")
    left = put_activation_shard(store, shard)
    right = put_activation_shard(store, shard)
    assert left.artifact_id == right.artifact_id
    restored = read_activation_shard(store, left.artifact_id)
    assert np.array_equal(restored.values, shard.values)
    assert restored.metadata == shard.metadata


def test_gather_rejects_missing_images() -> None:
    bank = make_site_bank(
        [1, 2], height=2, width=2, max_sites_per_image=1, root_seed=1, namespace="x"
    )
    with pytest.raises(ValueError, match="absent"):
        gather_site_vectors(np.zeros((1, 3, 2, 2)), [1], bank)


def test_float64_activation_precision_is_preserved(tmp_path) -> None:
    bank = make_site_bank(
        [7], height=2, width=2, max_sites_per_image=2, root_seed=3, namespace="f64"
    )
    activation = np.arange(12, dtype=np.float64).reshape(1, 3, 2, 2)
    shard = replace(
        gather_site_vectors(activation, [7], bank),
        checkpoint=1,
        cut="stage1.block1",
        bank_id="float64-bank",
    )

    assert shard.values.dtype == np.float64
    store = ArtifactStore(tmp_path / "artifacts")
    restored = read_activation_shard(store, put_activation_shard(store, shard).artifact_id)
    assert restored.values.dtype == np.float64
    assert restored.metadata["dtype"] == "float64"
    np.testing.assert_array_equal(restored.values, shard.values)


def test_activation_shards_reject_undeclared_precision() -> None:
    bank = make_site_bank(
        [7], height=1, width=1, max_sites_per_image=1, root_seed=3, namespace="bad-dtype"
    )
    with pytest.raises(ValueError, match="float32 or float64"):
        gather_site_vectors(np.ones((1, 2, 1, 1), dtype=np.float16), [7], bank)


def test_activation_publisher_composes_parent_shard_lineage(tmp_path) -> None:
    bank = make_site_bank(
        [4], height=1, width=1, max_sites_per_image=1, root_seed=1, namespace="lineage"
    )
    shard = replace(
        gather_site_vectors(np.ones((1, 2, 1, 1), dtype=np.float32), [4], bank),
        checkpoint=5,
        cut="stage1.block2",
        bank_id="bank-lineage",
    )
    store = ArtifactStore(tmp_path / "artifacts")
    reference = put_activation_shard(
        store,
        shard,
        metadata={"parent_stage_signature": "a" * 64, "shard_key": {"chunk": 0}},
    )
    assert reference.manifest.metadata["parent_stage_signature"] == "a" * 64
    assert reference.manifest.metadata["dtype"] == "float32"
    with pytest.raises(ValueError, match="conflicts"):
        put_activation_shard(store, shard, metadata={"checkpoint": 100})
