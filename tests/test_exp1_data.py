from __future__ import annotations

import io

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
Image = pytest.importorskip("PIL.Image")

from voronoi_lab.core import sha256_file  # noqa: E402
from voronoi_lab.exp1.data import CifarParquetSource, InputRecipe  # noqa: E402


def _png(value: int) -> bytes:
    image = Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _fixture_parquet(tmp_path):
    path = tmp_path / "cifar.parquet"
    images = [{"bytes": _png(index * 20), "path": None} for index in range(6)]
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table(
        {
            "img": pa.array(images, type=image_type),
            "label": pa.array([0, 1, 2, 3, 4, 5], type=pa.int64()),
        }
    )
    pq.write_table(table, path, row_group_size=2)
    return path


def _clean_recipe() -> InputRecipe:
    return InputRecipe(
        "clean",
        kind="clean",
        crop_padding=0,
        flip_probability=0.0,
        brightness_fraction=0.0,
    )


def test_recipe_version_rejects_boolean_alias() -> None:
    with pytest.raises(ValueError, match="version"):
        InputRecipe(
            "clean",
            kind="clean",
            crop_padding=0,
            flip_probability=0.0,
            brightness_fraction=0.0,
            recipe_version=True,
        )


def test_materialization_preserves_requested_order_and_is_deterministic(tmp_path) -> None:
    path = _fixture_parquet(tmp_path)
    source = CifarParquetSource(
        path,
        split="train",
        expected_sha256=sha256_file(path),
        expected_size=path.stat().st_size,
    )
    recipe = InputRecipe(
        "fixed-crop",
        kind="crop_flip",
        crop_padding=4,
        flip_probability=0.5,
        brightness_fraction=0.0,
    )
    left = source.materialize([5, 1, 3], recipe=recipe, root_seed=11)
    right = source.materialize([5, 1, 3], recipe=recipe, root_seed=11)
    assert left.labels.tolist() == [5, 1, 3]
    assert np.array_equal(left.tensors, right.tensors)
    assert left.bank_id == right.bank_id
    assert left.to_npz_bytes() == right.to_npz_bytes()


def test_clean_normalization_and_hash_rejection(tmp_path) -> None:
    path = _fixture_parquet(tmp_path)
    source = CifarParquetSource(path, split="test")
    bank = source.materialize([0], recipe=_clean_recipe(), root_seed=0)
    assert bank.tensors.shape == (1, 3, 32, 32)
    assert np.all(np.isfinite(bank.tensors))
    with pytest.raises(ValueError, match="SHA-256"):
        CifarParquetSource(path, split="test", expected_sha256="0" * 64)


def test_invalid_or_duplicate_rows_are_rejected(tmp_path) -> None:
    source = CifarParquetSource(_fixture_parquet(tmp_path), split="test")
    with pytest.raises(ValueError, match="unique"):
        source.materialize([1, 1], recipe=_clean_recipe(), root_seed=0)


def test_source_uses_the_exact_bytes_that_were_hash_validated(tmp_path) -> None:
    path = _fixture_parquet(tmp_path)
    source = CifarParquetSource(
        path,
        split="test",
        expected_sha256=sha256_file(path),
        expected_size=path.stat().st_size,
    )

    # Replacing the external pathname after construction must not change the
    # validated source snapshot consumed by this reader.
    replacement = tmp_path / "replacement.parquet"
    images = [{"bytes": _png(240), "path": None} for _ in range(6)]
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    pq.write_table(
        pa.table(
            {
                "img": pa.array(images, type=image_type),
                "label": pa.array([9, 9, 9, 9, 9, 9], type=pa.int64()),
            }
        ),
        replacement,
        row_group_size=2,
    )
    replacement.replace(path)

    bank = source.materialize([0, 5], recipe=_clean_recipe(), root_seed=0)
    assert bank.labels.tolist() == [0, 5]
    assert bank.source_sha256 == source.sha256
