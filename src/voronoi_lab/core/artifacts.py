"""Immutable, content-addressed experiment artifacts.

An artifact is a directory containing a canonical manifest and one or more
payload files.  Its identifier is the SHA-256 hash of the manifest core, which
in turn contains every payload checksum.  Objects are assembled in a staging
directory and renamed into place atomically; an existing object is verified
rather than overwritten.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, TypeAlias

from .hashing import (
    JSONLike,
    JSONValue,
    canonical_hash,
    canonical_json_bytes,
    freeze_json,
    sha256_file,
    thaw_json,
)

ARTIFACT_SCHEMA_VERSION = 1
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_KIND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_COPY_CHUNK_SIZE = 1024 * 1024

FileSource: TypeAlias = bytes | bytearray | memoryview | Path


class ArtifactError(RuntimeError):
    """Base class for artifact-store failures."""


class ArtifactValidationError(ArtifactError):
    """Raised for malformed manifests, identifiers, or payload paths."""


class ArtifactVerificationError(ArtifactError):
    """Raised when stored bytes do not match their immutable manifest."""


class ArtifactCollisionError(ArtifactError):
    """Raised if an existing object occupies an expected content address."""


def _canonical_metadata(metadata: Mapping[str, JSONLike] | None) -> dict[str, JSONValue]:
    raw: Mapping[str, JSONLike] = {} if metadata is None else metadata
    normalized = json.loads(canonical_json_bytes(raw))
    if not isinstance(normalized, dict):  # Mapping input guarantees this; retain a hard boundary.
        raise ArtifactValidationError("artifact metadata must be a JSON object")
    return normalized


def _strict_json_loads(data: bytes) -> JSONValue:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number is forbidden: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
        result: dict[str, JSONValue] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    return json.loads(
        data,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def _validate_kind(kind: str) -> str:
    if not isinstance(kind, str) or not _KIND_RE.fullmatch(kind):
        raise ArtifactValidationError("kind must match [A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
    return kind


def _validate_digest(digest: str, *, label: str = "artifact_id") -> str:
    if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
        raise ArtifactValidationError(f"{label} must be a lowercase SHA-256 hex digest")
    return digest


def _validate_payload_path(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ArtifactValidationError("payload paths must be non-empty POSIX paths")
    path = PurePosixPath(unicodedata.normalize("NFC", name))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactValidationError(f"unsafe payload path: {name!r}")
    if path.parts[0] == "manifest.json":
        raise ArtifactValidationError("manifest.json is reserved by the artifact store")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """One checksummed payload declared by an artifact manifest."""

    path: str
    size: int
    sha256: str
    media_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _validate_payload_path(self.path))
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ArtifactValidationError("artifact file size must be a non-negative integer")
        _validate_digest(self.sha256, label="file sha256")
        if self.media_type is not None and (
            not isinstance(self.media_type, str) or not self.media_type.strip()
        ):
            raise ArtifactValidationError("media_type must be a non-empty string when supplied")

    def to_dict(self) -> dict[str, JSONValue]:
        result: dict[str, JSONValue] = {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
        }
        if self.media_type is not None:
            result["media_type"] = self.media_type
        return result


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Validated immutable artifact manifest."""

    artifact_id: str
    kind: str
    files: tuple[ArtifactFile, ...]
    metadata: Mapping[str, JSONLike]
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_digest(self.artifact_id)
        _validate_kind(self.kind)
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != ARTIFACT_SCHEMA_VERSION
        ):
            raise ArtifactValidationError(
                f"unsupported artifact schema version {self.schema_version}"
            )
        files = tuple(self.files)
        if not all(isinstance(entry, ArtifactFile) for entry in files):
            raise ArtifactValidationError("manifest files must be ArtifactFile objects")
        object.__setattr__(self, "files", files)
        canonical_metadata = _canonical_metadata(self.metadata)
        object.__setattr__(self, "metadata", freeze_json(canonical_metadata))
        names = [entry.path for entry in self.files]
        if not names:
            raise ArtifactValidationError("an artifact must contain at least one payload file")
        if names != sorted(names) or len(names) != len(set(names)):
            raise ArtifactValidationError("artifact files must be unique and sorted by path")
        expected = canonical_hash(self.core_dict())
        if self.artifact_id != expected:
            raise ArtifactValidationError(
                f"artifact id does not match manifest content: expected {expected}"
            )

    def core_dict(self) -> dict[str, JSONValue]:
        return {
            "artifact_schema_version": self.schema_version,
            "files": [entry.to_dict() for entry in self.files],
            "kind": self.kind,
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, JSONValue]:
        return {"artifact_id": self.artifact_id, **self.core_dict()}

    @classmethod
    def from_dict(cls, value: object) -> ArtifactManifest:
        if not isinstance(value, dict):
            raise ArtifactValidationError("artifact manifest must be a JSON object")
        expected_keys = {
            "artifact_id",
            "artifact_schema_version",
            "files",
            "kind",
            "metadata",
        }
        if set(value) != expected_keys:
            raise ArtifactValidationError(
                f"artifact manifest keys must be exactly {sorted(expected_keys)}"
            )
        raw_files = value["files"]
        if not isinstance(raw_files, list):
            raise ArtifactValidationError("artifact manifest files must be a list")
        files: list[ArtifactFile] = []
        for raw_file in raw_files:
            if not isinstance(raw_file, dict):
                raise ArtifactValidationError("artifact file entries must be objects")
            allowed_keys = {"path", "sha256", "size", "media_type"}
            if not {"path", "sha256", "size"}.issubset(raw_file) or not set(raw_file).issubset(
                allowed_keys
            ):
                raise ArtifactValidationError("malformed artifact file entry")
            files.append(
                ArtifactFile(
                    path=raw_file["path"],
                    size=raw_file["size"],
                    sha256=raw_file["sha256"],
                    media_type=raw_file.get("media_type"),
                )
            )
        metadata = value["metadata"]
        if not isinstance(metadata, dict):
            raise ArtifactValidationError("artifact metadata must be an object")
        return cls(
            artifact_id=value["artifact_id"],
            kind=value["kind"],
            files=tuple(files),
            metadata=metadata,
            schema_version=value["artifact_schema_version"],
        )


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A resolved artifact object and its manifest."""

    path: Path
    manifest: ArtifactManifest

    @property
    def artifact_id(self) -> str:
        return self.manifest.artifact_id

    def payload_path(self, name: str) -> Path:
        normalized = _validate_payload_path(name)
        declared = {entry.path for entry in self.manifest.files}
        if normalized not in declared:
            raise ArtifactValidationError(f"payload is not declared by manifest: {normalized}")
        return self.path / "files" / normalized


class ArtifactStore:
    """Local immutable artifact store with atomic object publication."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects_dir = self.root / "objects"
        self.staging_dir = self.root / ".staging"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def object_path(self, artifact_id: str) -> Path:
        digest = _validate_digest(artifact_id)
        return self.objects_dir / digest[:2] / digest

    def put_bytes(
        self,
        data: bytes | bytearray | memoryview,
        *,
        filename: str = "payload.bin",
        kind: str = "blob",
        metadata: Mapping[str, JSONLike] | None = None,
        media_type: str | None = None,
    ) -> ArtifactRef:
        media_types = None if media_type is None else {filename: media_type}
        return self.put_files(
            {filename: data}, kind=kind, metadata=metadata, media_types=media_types
        )

    def put_json(
        self,
        value: JSONLike,
        *,
        filename: str = "data.json",
        kind: str = "json",
        metadata: Mapping[str, JSONLike] | None = None,
    ) -> ArtifactRef:
        return self.put_bytes(
            canonical_json_bytes(value),
            filename=filename,
            kind=kind,
            metadata=metadata,
            media_type="application/json",
        )

    def put_file(
        self,
        source: str | Path,
        *,
        filename: str | None = None,
        kind: str = "file",
        metadata: Mapping[str, JSONLike] | None = None,
        media_type: str | None = None,
    ) -> ArtifactRef:
        source_path = Path(source)
        target_name = source_path.name if filename is None else filename
        media_types = None if media_type is None else {target_name: media_type}
        return self.put_files(
            {target_name: source_path},
            kind=kind,
            metadata=metadata,
            media_types=media_types,
        )

    def put_files(
        self,
        files: Mapping[str, FileSource],
        *,
        kind: str,
        metadata: Mapping[str, JSONLike] | None = None,
        media_types: Mapping[str, str] | None = None,
    ) -> ArtifactRef:
        """Publish one or more files as a single immutable object."""

        _validate_kind(kind)
        if not files:
            raise ArtifactValidationError("an artifact must contain at least one payload")
        normalized_sources: dict[str, FileSource] = {}
        for name, source in files.items():
            normalized = _validate_payload_path(name)
            if normalized in normalized_sources:
                raise ArtifactValidationError(f"duplicate normalized payload path: {normalized}")
            if not isinstance(source, (bytes, bytearray, memoryview, Path)):
                raise ArtifactValidationError(
                    f"payload {normalized!r} must be bytes-like or pathlib.Path"
                )
            if isinstance(source, Path) and not source.is_file():
                raise ArtifactValidationError(f"payload source is not a file: {source}")
            normalized_sources[normalized] = source
        source_names = set(normalized_sources)
        for name in source_names:
            parents = PurePosixPath(name).parents
            if any(parent.as_posix() in source_names for parent in parents[:-1]):
                raise ArtifactValidationError(
                    f"payload path conflicts with another payload file: {name}"
                )

        normalized_media_types: dict[str, str] = {}
        if media_types is not None:
            for name, media_type in media_types.items():
                normalized = _validate_payload_path(name)
                if normalized not in normalized_sources:
                    raise ArtifactValidationError(
                        f"media type supplied for unknown payload: {normalized}"
                    )
                if not isinstance(media_type, str) or not media_type.strip():
                    raise ArtifactValidationError("media types must be non-empty strings")
                normalized_media_types[normalized] = media_type

        canonical_metadata = _canonical_metadata(metadata)
        temporary = Path(tempfile.mkdtemp(prefix="object-", dir=self.staging_dir))
        try:
            payload_root = temporary / "files"
            payload_root.mkdir()
            manifest_files: list[ArtifactFile] = []
            for name in sorted(normalized_sources):
                destination = payload_root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                self._write_source(normalized_sources[name], destination)
                manifest_files.append(
                    ArtifactFile(
                        path=name,
                        size=destination.stat().st_size,
                        sha256=sha256_file(destination),
                        media_type=normalized_media_types.get(name),
                    )
                )

            core: dict[str, JSONValue] = {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "files": [entry.to_dict() for entry in manifest_files],
                "kind": kind,
                "metadata": canonical_metadata,
            }
            artifact_id = canonical_hash(core)
            manifest = ArtifactManifest(
                artifact_id=artifact_id,
                kind=kind,
                files=tuple(manifest_files),
                metadata=canonical_metadata,
            )
            self._write_bytes(temporary / "manifest.json", canonical_json_bytes(manifest.to_dict()))
            self._sync_directory_tree(payload_root)
            self._sync_directory(temporary)

            destination = self.object_path(artifact_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._sync_directory(self.objects_dir)
            with self._publication_lock(destination):
                if destination.exists():
                    if not destination.is_dir():
                        raise ArtifactCollisionError(
                            f"artifact address is occupied by a non-directory: {artifact_id}"
                        )
                    return self.verify(artifact_id)
                try:
                    os.rename(temporary, destination)
                except OSError as exc:
                    if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                        raise
                    raise ArtifactCollisionError(
                        f"artifact address appeared during atomic publication: {artifact_id}"
                    ) from exc
                self._sync_directory(destination.parent)
                self._make_read_only(destination)
                return ArtifactRef(path=destination, manifest=manifest)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def get(self, artifact_id: str, *, verify: bool = True) -> ArtifactRef:
        """Resolve an artifact, optionally verifying every payload checksum."""

        return self.verify(artifact_id) if verify else self._load(artifact_id)

    def verify(self, artifact_id: str) -> ArtifactRef:
        """Verify manifest identity, payload inventory, sizes, and checksums."""

        reference = self._load(artifact_id)
        payload_root = reference.path / "files"
        expected = {entry.path: entry for entry in reference.manifest.files}
        if payload_root.is_symlink() or not payload_root.is_dir():
            raise ArtifactVerificationError(f"artifact payload directory is missing: {artifact_id}")
        top_level = {child.name for child in reference.path.iterdir()}
        if top_level != {"files", "manifest.json"}:
            raise ArtifactVerificationError(
                f"artifact object inventory mismatch: {sorted(top_level)}"
            )
        for path in payload_root.rglob("*"):
            if path.is_symlink():
                raise ArtifactVerificationError(
                    f"artifact payload may not be a symbolic link: {path.relative_to(payload_root)}"
                )
            if not path.is_file() and not path.is_dir():
                raise ArtifactVerificationError(
                    f"artifact payload has an unsupported filesystem entry: "
                    f"{path.relative_to(payload_root)}"
                )
        expected_directories = {
            parent.as_posix() for name in expected for parent in PurePosixPath(name).parents[:-1]
        }
        actual_directories = {
            path.relative_to(payload_root).as_posix()
            for path in payload_root.rglob("*")
            if path.is_dir()
        }
        if actual_directories != expected_directories:
            raise ArtifactVerificationError(
                "artifact directory inventory mismatch; "
                f"expected={sorted(expected_directories)}, actual={sorted(actual_directories)}"
            )
        actual = {
            path.relative_to(payload_root).as_posix()
            for path in payload_root.rglob("*")
            if path.is_file()
        }
        if actual != set(expected):
            missing = sorted(set(expected) - actual)
            extra = sorted(actual - set(expected))
            raise ArtifactVerificationError(
                f"artifact payload inventory mismatch; missing={missing}, extra={extra}"
            )
        for name, entry in expected.items():
            payload = payload_root / name
            size = payload.stat().st_size
            if size != entry.size:
                raise ArtifactVerificationError(
                    f"artifact payload size mismatch for {name}: expected {entry.size}, got {size}"
                )
            digest = sha256_file(payload)
            if digest != entry.sha256:
                raise ArtifactVerificationError(
                    f"artifact payload checksum mismatch for {name}: "
                    f"expected {entry.sha256}, got {digest}"
                )
        return reference

    def read_bytes(self, artifact_id: str, filename: str, *, verify: bool = True) -> bytes:
        reference = self.get(artifact_id, verify=verify)
        return reference.payload_path(filename).read_bytes()

    def read_json(self, artifact_id: str, filename: str = "data.json") -> JSONValue:
        try:
            return _strict_json_loads(self.read_bytes(artifact_id, filename))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ArtifactVerificationError(
                f"declared JSON payload is not valid UTF-8 JSON: {filename}"
            ) from exc

    def open(self, artifact_id: str, filename: str, *, verify: bool = True) -> BinaryIO:
        """Open a declared payload read-only."""

        reference = self.get(artifact_id, verify=verify)
        return reference.payload_path(filename).open("rb")

    def _load(self, artifact_id: str) -> ArtifactRef:
        digest = _validate_digest(artifact_id)
        path = self.object_path(digest)
        manifest_path = path / "manifest.json"
        if path.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
            raise ArtifactVerificationError(f"artifact manifest is missing: {digest}")
        try:
            manifest_bytes = manifest_path.read_bytes()
            raw = _strict_json_loads(manifest_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ArtifactVerificationError(f"artifact manifest is unreadable: {digest}") from exc
        if manifest_bytes != canonical_json_bytes(raw):
            raise ArtifactVerificationError(f"artifact manifest is not canonical JSON: {digest}")
        try:
            manifest = ArtifactManifest.from_dict(raw)
        except ArtifactValidationError as exc:
            raise ArtifactVerificationError(f"invalid artifact manifest: {digest}: {exc}") from exc
        if manifest.artifact_id != digest:
            raise ArtifactVerificationError(
                f"artifact directory id {digest} differs from manifest id {manifest.artifact_id}"
            )
        return ArtifactRef(path=path, manifest=manifest)

    @staticmethod
    def _write_source(source: FileSource, destination: Path) -> None:
        if isinstance(source, Path):
            with source.open("rb") as source_handle, destination.open("xb") as target_handle:
                while chunk := source_handle.read(_COPY_CHUNK_SIZE):
                    target_handle.write(chunk)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            return
        ArtifactStore._write_bytes(destination, bytes(source))

    @staticmethod
    def _write_bytes(destination: Path, data: bytes) -> None:
        with destination.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _sync_directory(path: Path) -> None:
        if not hasattr(os, "O_DIRECTORY"):
            return
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _sync_directory_tree(cls, root: Path) -> None:
        directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
        for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
            cls._sync_directory(directory)

    @staticmethod
    @contextmanager
    def _publication_lock(destination: Path) -> Iterator[None]:
        """Serialize cooperating publishers without leaving stale lock ownership."""

        lock_path = destination.parent / ".publish.lock"
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _make_read_only(path: Path) -> None:
        """Make published object contents read-only as a best-effort guardrail."""

        try:
            for child in sorted(path.rglob("*"), reverse=True):
                child.chmod(0o755 if child.is_dir() else 0o444)
            path.chmod(0o755)
        except OSError:
            # Checksums and no-overwrite publication are the correctness boundary;
            # filesystem modes are only an additional local guardrail.
            return
