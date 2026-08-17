"""Deterministic CIFAR bank materialization from hash-pinned Parquet inputs."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from voronoi_lab.core import SeedDeriver, canonical_hash, sha256_bytes

CIFAR10_MEAN = np.asarray((0.4914, 0.4822, 0.4465), dtype=np.float32)
CIFAR10_STD = np.asarray((0.2470, 0.2435, 0.2616), dtype=np.float32)


def _pyarrow() -> tuple[object, object]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError("Parquet banks require the 'resnet' optional dependency") from error
    return pa, pq


@dataclass(frozen=True, slots=True)
class InputRecipe:
    """A reproducible input-derived state recipe, applied once and persisted."""

    name: str
    kind: Literal["clean", "crop_flip", "mild_color"]
    crop_padding: int
    flip_probability: float
    brightness_fraction: float
    recipe_version: Literal[1] = 1

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("recipe name cannot be blank")
        if type(self.recipe_version) is not int or self.recipe_version != 1:
            raise ValueError("unsupported input recipe version")
        if self.crop_padding < 0:
            raise ValueError("crop_padding must be nonnegative")
        if not 0 <= self.flip_probability <= 1:
            raise ValueError("flip_probability must be in [0, 1]")
        if not 0 <= self.brightness_fraction < 1:
            raise ValueError("brightness_fraction must be in [0, 1)")
        if self.kind == "clean" and (
            self.crop_padding != 0 or self.flip_probability != 0 or self.brightness_fraction != 0
        ):
            raise ValueError("clean recipe parameters must all be zero")
        if self.kind == "crop_flip" and self.brightness_fraction != 0:
            raise ValueError("crop_flip brightness_fraction must be zero")
        if self.kind == "mild_color" and (self.crop_padding != 0 or self.flip_probability != 0):
            raise ValueError("mild_color crop/flip parameters must be zero")

    def to_dict(self) -> dict[str, object]:
        return {
            "brightness_fraction": self.brightness_fraction,
            "crop_padding": self.crop_padding,
            "flip_probability": self.flip_probability,
            "kind": self.kind,
            "name": self.name,
            "recipe_version": self.recipe_version,
        }


@dataclass(frozen=True, slots=True)
class MaterializedImageBank:
    image_ids: NDArray[np.int64]
    labels: NDArray[np.int64]
    tensors: NDArray[np.float32]
    split: Literal["train", "test"]
    source_sha256: str
    recipe: InputRecipe
    root_seed: int

    def __post_init__(self) -> None:
        image_ids = np.asarray(self.image_ids, dtype=np.int64)
        labels = np.asarray(self.labels, dtype=np.int64)
        tensors = np.asarray(self.tensors, dtype=np.float32)
        if image_ids.ndim != 1 or labels.shape != image_ids.shape:
            raise ValueError("image ids and labels must be matching vectors")
        if tensors.shape != (len(image_ids), 3, 32, 32):
            raise ValueError("CIFAR tensors must have shape (images, 3, 32, 32)")
        if len(np.unique(image_ids)) != len(image_ids) or np.any(image_ids < 0):
            raise ValueError("image ids must be unique and nonnegative")
        if np.any(labels < 0) or np.any(labels > 9) or not np.all(np.isfinite(tensors)):
            raise ValueError("labels or tensors are invalid")
        if len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a full digest")
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "tensors", tensors)

    @property
    def tensor_sha256(self) -> str:
        header = json.dumps(
            {
                "dtype": self.tensors.dtype.str,
                "shape": self.tensors.shape,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return sha256_bytes(header + self.tensors.tobytes(order="C"))

    @property
    def bank_id(self) -> str:
        return canonical_hash(
            {
                "image_ids": self.image_ids.tolist(),
                "labels": self.labels.tolist(),
                "recipe": self.recipe.to_dict(),
                "root_seed": self.root_seed,
                "source_sha256": self.source_sha256,
                "split": self.split,
                "tensor_sha256": self.tensor_sha256,
            }
        )

    def to_npz_bytes(self) -> bytes:
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            image_ids=self.image_ids,
            labels=self.labels,
            tensors=self.tensors,
            metadata=np.asarray(
                json.dumps(
                    {
                        "bank_id": self.bank_id,
                        "recipe": self.recipe.to_dict(),
                        "root_seed": self.root_seed,
                        "source_sha256": self.source_sha256,
                        "split": self.split,
                        "tensor_sha256": self.tensor_sha256,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        )
        return buffer.getvalue()


class CifarParquetSource:
    """Read only requested row ids from the audited HuggingFace-style files."""

    def __init__(
        self,
        path: str | Path,
        *,
        split: Literal["train", "test"],
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.split = split
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        # Retain the exact validated bytes. Hashing a pathname and reopening it later
        # would allow replacement or in-place mutation between validation and use.
        self._source_bytes = self.path.read_bytes()
        if expected_size is not None and len(self._source_bytes) != expected_size:
            raise ValueError(f"Parquet byte size does not match manifest: {self.path}")
        self.sha256 = sha256_bytes(self._source_bytes)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError(f"Parquet SHA-256 does not match manifest: {self.path}")
        pa, pq = _pyarrow()
        self._file = pq.ParquetFile(pa.BufferReader(self._source_bytes))
        if set(self._file.schema_arrow.names) != {"img", "label"}:
            raise ValueError("CIFAR Parquet must contain exactly img and label columns")

    @property
    def size(self) -> int:
        return int(self._file.metadata.num_rows)

    def materialize(
        self,
        image_ids: ArrayLike,
        *,
        recipe: InputRecipe,
        root_seed: int,
    ) -> MaterializedImageBank:
        requested = np.asarray(image_ids, dtype=np.int64)
        if requested.ndim != 1 or len(requested) == 0:
            raise ValueError("image_ids must be a nonempty vector")
        if (
            len(np.unique(requested)) != len(requested)
            or np.any(requested < 0)
            or np.any(requested >= self.size)
        ):
            raise ValueError("image_ids must be unique valid row indices")
        raw_by_index = self._read_rows(requested)
        tensors = np.empty((len(requested), 3, 32, 32), dtype=np.float32)
        labels = np.empty(len(requested), dtype=np.int64)
        seeds = SeedDeriver(root_seed, ("input_recipe", recipe.name, self.split))
        for output_index, image_id in enumerate(requested):
            image_bytes, label = raw_by_index[int(image_id)]
            rgb = _decode_png(image_bytes)
            augmented = _apply_recipe(rgb, recipe, seeds.derive(int(image_id)))
            tensors[output_index] = _normalize(augmented)
            labels[output_index] = label
        return MaterializedImageBank(
            requested,
            labels,
            tensors,
            self.split,
            self.sha256,
            recipe,
            root_seed,
        )

    def _read_rows(self, requested: NDArray[np.int64]) -> dict[int, tuple[bytes, int]]:
        needed = set(int(index) for index in requested)
        result: dict[int, tuple[bytes, int]] = {}
        row_start = 0
        for group_index in range(self._file.metadata.num_row_groups):
            row_count = self._file.metadata.row_group(group_index).num_rows
            relevant = sorted(
                index for index in needed if row_start <= index < row_start + row_count
            )
            if relevant:
                rows = self._file.read_row_group(group_index, columns=["img", "label"]).to_pylist()
                for index in relevant:
                    row = rows[index - row_start]
                    image = row["img"]
                    if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
                        raise ValueError(f"row {index} does not contain embedded image bytes")
                    result[index] = (image["bytes"], int(row["label"]))
                    needed.remove(index)
            row_start += row_count
            if not needed:
                break
        if needed:
            raise ValueError(f"Parquet rows could not be read: {sorted(needed)}")
        return result


def _decode_png(data: bytes) -> NDArray[np.uint8]:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError("image decoding requires the 'resnet' optional dependency") from error
    with Image.open(io.BytesIO(data)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (32, 32, 3):
        raise ValueError(f"expected a 32x32 RGB CIFAR image, got {rgb.shape}")
    return rgb


def _apply_recipe(image: NDArray[np.uint8], recipe: InputRecipe, seed: int) -> NDArray[np.uint8]:
    if recipe.kind == "clean":
        return image.copy()
    rng = np.random.default_rng(seed)
    if recipe.kind == "crop_flip":
        padded = np.pad(
            image,
            (
                (recipe.crop_padding, recipe.crop_padding),
                (recipe.crop_padding, recipe.crop_padding),
                (0, 0),
            ),
            mode="constant",
        )
        extent = 2 * recipe.crop_padding + 1
        row = int(rng.integers(extent))
        column = int(rng.integers(extent))
        result = padded[row : row + 32, column : column + 32]
        if rng.random() < recipe.flip_probability:
            result = result[:, ::-1]
        return np.ascontiguousarray(result)
    if recipe.kind == "mild_color":
        factor = 1.0 + rng.uniform(-recipe.brightness_fraction, recipe.brightness_fraction)
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown recipe kind: {recipe.kind}")  # pragma: no cover


def _normalize(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    scaled = image.astype(np.float32) / 255.0
    normalized = (scaled - CIFAR10_MEAN) / CIFAR10_STD
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False)
