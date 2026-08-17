"""Immutable activation shards and deterministic site-vector extraction."""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from voronoi_lab.core import ArtifactRef, ArtifactStore, JSONLike

from .banks import SiteBank

ACTIVATION_SHARD_SCHEMA_VERSION = 2


def _npy_bytes(values: ArrayLike) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values), allow_pickle=False)
    return buffer.getvalue()


def _read_npy(data: bytes) -> NDArray[Any]:
    return np.load(io.BytesIO(data), allow_pickle=False)


@dataclass(frozen=True, slots=True)
class ActivationShard:
    values: NDArray[np.float32] | NDArray[np.float64]
    image_ids: NDArray[np.int64]
    rows: NDArray[np.int64]
    columns: NDArray[np.int64]
    weights: NDArray[np.float64]
    checkpoint: int
    cut: str
    bank_id: str
    shard_index: int

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        image_ids = np.asarray(self.image_ids, dtype=np.int64)
        rows = np.asarray(self.rows, dtype=np.int64)
        columns = np.asarray(self.columns, dtype=np.int64)
        weights = np.asarray(self.weights, dtype=np.float64)
        count = len(values)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("activation values must have shape (sites, channels)")
        if values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("activation values must use declared float32 or float64 precision")
        if not (image_ids.shape == rows.shape == columns.shape == weights.shape == (count,)):
            raise ValueError("activation shard metadata vectors must match the site count")
        if count == 0 or np.any(image_ids < 0) or np.any(rows < 0) or np.any(columns < 0):
            raise ValueError("activation shard indices must be nonempty and nonnegative")
        if (
            np.any(weights <= 0)
            or not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("activation shard weights/values are invalid")
        if self.checkpoint < 0 or self.shard_index < 0 or not self.cut or not self.bank_id:
            raise ValueError("activation shard identity fields are invalid")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "weights", weights)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "bank_id": self.bank_id,
            "channels": self.values.shape[1],
            "checkpoint": self.checkpoint,
            "coordinate_space": "native",
            "cut": self.cut,
            "dtype": self.values.dtype.name,
            "schema_version": ACTIVATION_SHARD_SCHEMA_VERSION,
            "shard_index": self.shard_index,
            "site_count": len(self.values),
        }


def put_activation_shard(
    store: ArtifactStore,
    shard: ActivationShard,
    *,
    metadata: Mapping[str, JSONLike] | None = None,
) -> ArtifactRef:
    """Publish one shard with optional parent-run identity metadata.

    ``metadata`` is the integration point for :class:`ShardContext` identity. Any
    overlapping activation fields must agree, so callers cannot disguise the
    checkpoint, cut, bank, shard index, or numeric precision of the payload.
    """

    merged_metadata: dict[str, JSONLike] = dict(metadata or {})
    conflicts = [
        key
        for key, value in shard.metadata.items()
        if key in merged_metadata and merged_metadata[key] != value
    ]
    if conflicts:
        raise ValueError(
            "activation shard metadata conflicts with supplied lineage: "
            + ", ".join(sorted(conflicts))
        )
    merged_metadata.update(shard.metadata)  # type: ignore[arg-type]

    return store.put_files(
        {
            "columns.npy": _npy_bytes(shard.columns),
            "image_ids.npy": _npy_bytes(shard.image_ids),
            "rows.npy": _npy_bytes(shard.rows),
            "values.npy": _npy_bytes(shard.values),
            "weights.npy": _npy_bytes(shard.weights),
        },
        kind="exp1/activation-shard",
        metadata=merged_metadata,
        media_types={
            "columns.npy": "application/x-npy",
            "image_ids.npy": "application/x-npy",
            "rows.npy": "application/x-npy",
            "values.npy": "application/x-npy",
            "weights.npy": "application/x-npy",
        },
    )


def read_activation_shard(store: ArtifactStore, artifact_id: str) -> ActivationShard:
    reference = store.get(artifact_id, verify=True)
    if reference.manifest.kind != "exp1/activation-shard":
        raise ValueError("artifact is not an Experiment 1 activation shard")
    metadata = reference.manifest.metadata
    if metadata.get("schema_version") != ACTIVATION_SHARD_SCHEMA_VERSION:
        raise ValueError("unsupported activation shard schema")
    shard = ActivationShard(
        values=_read_npy(store.read_bytes(artifact_id, "values.npy")),
        image_ids=_read_npy(store.read_bytes(artifact_id, "image_ids.npy")),
        rows=_read_npy(store.read_bytes(artifact_id, "rows.npy")),
        columns=_read_npy(store.read_bytes(artifact_id, "columns.npy")),
        weights=_read_npy(store.read_bytes(artifact_id, "weights.npy")),
        checkpoint=int(metadata["checkpoint"]),
        cut=str(metadata["cut"]),
        bank_id=str(metadata["bank_id"]),
        shard_index=int(metadata["shard_index"]),
    )
    expected = shard.metadata
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(
            "activation shard metadata does not match payload: " + ", ".join(mismatches)
        )
    return shard


def gather_site_vectors(
    activation: Any,
    batch_image_ids: ArrayLike,
    site_bank: SiteBank,
) -> ActivationShard:
    """Gather BCHW channel vectors in the persisted SiteBank order.

    The returned identity fields are placeholders; callers use ``dataclasses.replace``
    or construct the final shard with checkpoint/cut/bank identity before publication.
    """

    if hasattr(activation, "detach"):
        values = activation.detach().cpu().numpy()
    else:
        values = np.asarray(activation)
    values = np.asarray(values)
    if values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("activation must use float32 or float64 precision")
    batch_ids = np.asarray(batch_image_ids, dtype=np.int64)
    if values.ndim != 4 or batch_ids.shape != (values.shape[0],):
        raise ValueError("activation must be BCHW with one image id per batch item")
    if values.shape[2:] != (site_bank.height, site_bank.width):
        raise ValueError("activation grid does not match the site bank")
    if len(np.unique(batch_ids)) != len(batch_ids):
        raise ValueError("batch image ids must be unique")
    batch_lookup = {int(image_id): index for index, image_id in enumerate(batch_ids)}
    try:
        batch_indices = np.asarray(
            [batch_lookup[int(image_id)] for image_id in site_bank.image_ids], dtype=np.int64
        )
    except KeyError as error:
        missing = error.args[0]
        raise ValueError(f"site bank references image absent from batch: {missing}") from error
    gathered = values[
        batch_indices,
        :,
        site_bank.rows,
        site_bank.columns,
    ]
    return ActivationShard(
        values=gathered,
        image_ids=site_bank.image_ids,
        rows=site_bank.rows,
        columns=site_bank.columns,
        weights=site_bank.weights,
        checkpoint=0,
        cut="unbound",
        bank_id="unbound",
        shard_index=0,
    )
