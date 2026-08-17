from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from voronoi_lab.core import (
    ArtifactFile,
    ArtifactStore,
    ArtifactValidationError,
    ArtifactVerificationError,
)


def test_artifact_store_deduplicates_canonical_json_and_verifies(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    first = store.put_json({"b": [2, 3], "a": 1}, kind="metrics", metadata={"stage": "static"})
    second = store.put_json({"a": 1, "b": (2, 3)}, kind="metrics", metadata={"stage": "static"})

    assert first.artifact_id == second.artifact_id
    assert first.path == second.path
    assert store.read_json(first.artifact_id) == {"a": 1, "b": [2, 3]}
    assert store.verify(first.artifact_id).manifest == first.manifest
    assert list(store.staging_dir.iterdir()) == []
    stored_manifest = json.loads((first.path / "manifest.json").read_text())
    assert stored_manifest["artifact_id"] == first.artifact_id


def test_artifact_store_supports_nested_multi_file_objects(tmp_path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source bytes")
    store = ArtifactStore(tmp_path / "store")

    reference = store.put_files(
        {"arrays/part-0.bin": source, "summary.txt": b"summary"},
        kind="stage-output/v1",
        metadata={"shape": [2, 4]},
        media_types={"summary.txt": "text/plain"},
    )

    assert reference.payload_path("arrays/part-0.bin").read_bytes() == b"source bytes"
    assert store.read_bytes(reference.artifact_id, "summary.txt") == b"summary"
    assert [entry.path for entry in reference.manifest.files] == [
        "arrays/part-0.bin",
        "summary.txt",
    ]
    assert reference.manifest.core_dict()["metadata"] == {"shape": [2, 4]}


@pytest.mark.parametrize(
    "name",
    ["", "../escape", "/absolute", "nested/../escape", "manifest.json", "win\\path"],
)
def test_artifact_store_rejects_unsafe_payload_paths(tmp_path, name: str) -> None:
    store = ArtifactStore(tmp_path / "store")
    with pytest.raises(ArtifactValidationError):
        store.put_bytes(b"payload", filename=name)


def test_artifact_store_detects_payload_tampering(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    reference = store.put_bytes(b"original", filename="payload.bin")
    payload = reference.payload_path("payload.bin")
    payload.chmod(0o644)
    payload.write_bytes(b"tampered")

    with pytest.raises(ArtifactVerificationError, match="checksum mismatch"):
        store.verify(reference.artifact_id)
    with pytest.raises(ArtifactVerificationError):
        store.put_bytes(b"original", filename="payload.bin")


def test_artifact_store_detects_undeclared_payloads(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    reference = store.put_bytes(b"original")
    extra = reference.path / "files" / "extra.bin"
    extra.write_bytes(b"not declared")

    with pytest.raises(ArtifactVerificationError, match="inventory mismatch"):
        store.verify(reference.artifact_id)


def test_metadata_and_kind_are_part_of_artifact_identity(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    base = store.put_bytes(b"same", kind="one", metadata={"version": 1})
    changed_kind = store.put_bytes(b"same", kind="two", metadata={"version": 1})
    changed_metadata = store.put_bytes(b"same", kind="one", metadata={"version": 2})

    assert len({base.artifact_id, changed_kind.artifact_id, changed_metadata.artifact_id}) == 3


def test_concurrent_publishers_converge_on_one_verified_object(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")

    def publish(_index: int) -> str:
        return store.put_bytes(b"shared", kind="concurrency-test").artifact_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        artifact_ids = list(executor.map(publish, range(32)))

    assert len(set(artifact_ids)) == 1
    store.verify(artifact_ids[0])
    object_dirs = [path for path in store.objects_dir.glob("*/*") if path.is_dir()]
    assert len(object_dirs) == 1


def test_get_rejects_path_traversal_as_an_artifact_id(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "store")
    with pytest.raises(ArtifactValidationError, match="SHA-256"):
        store.get("../manifest.json")


def test_manifest_snapshots_metadata_and_normalizes_payload_names(tmp_path) -> None:
    metadata = {"axes": [1, 2]}
    store = ArtifactStore(tmp_path / "store")
    reference = store.put_bytes(
        b"payload", filename="e\N{COMBINING ACUTE ACCENT}.bin", metadata=metadata
    )
    original_id = reference.artifact_id

    metadata["axes"].append(3)

    assert reference.manifest.artifact_id == original_id
    assert reference.manifest.metadata == {"axes": (1, 2)}
    assert reference.manifest.files[0].path == "é.bin"
    direct_file = ArtifactFile(path="e\N{COMBINING ACUTE ACCENT}.bin", size=0, sha256="0" * 64)
    assert direct_file.path == "é.bin"
    with pytest.raises(TypeError):
        reference.manifest.metadata["new"] = True  # type: ignore[index]


@pytest.mark.parametrize("payload", [b'{"x":NaN}', b'{"x":1,"x":2}'])
def test_read_json_rejects_noncanonical_json_values(tmp_path, payload: bytes) -> None:
    store = ArtifactStore(tmp_path / "store")
    reference = store.put_bytes(
        payload,
        filename="data.json",
        kind="json",
        media_type="application/json",
    )

    with pytest.raises(ArtifactVerificationError, match="not valid"):
        store.read_json(reference.artifact_id)
