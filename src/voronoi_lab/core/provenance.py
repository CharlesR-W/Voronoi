"""Dependency-light capture of runtime, source, and input provenance."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import stat
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .hashing import JSONLike, JSONValue, canonical_hash, freeze_json, sha256_bytes, sha256_file

PROVENANCE_SCHEMA_VERSION = 1
_LOGICAL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SENSITIVE_ENVIRONMENT_FRAGMENTS = (
    "AUTH",
    "COOKIE",
    "CREDENTIAL",
    "KEY",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)


class ProvenanceError(RuntimeError):
    """Raised when requested provenance cannot be captured safely or exactly."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _require_exact_object(
    value: object,
    *,
    keys: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ProvenanceError(f"{label} must be an object")
    if set(value) != keys:
        raise ProvenanceError(f"{label} keys must be exactly {sorted(keys)}")
    return value


def _require_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProvenanceError(f"{label} must be a non-empty string")
    return value


def _require_absolute_path(value: object, *, label: str) -> str:
    text = _require_nonempty_string(value, label=label)
    if not Path(text).is_absolute():
        raise ProvenanceError(f"{label} must be an absolute path")
    return text


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ProvenanceError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class InputFileProvenance:
    logical_name: str
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.logical_name, str) or not _LOGICAL_NAME_RE.fullmatch(
            self.logical_name
        ):
            raise ProvenanceError(f"invalid input logical name: {self.logical_name!r}")
        _require_absolute_path(self.path, label="input provenance path")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ProvenanceError("input provenance size must be a non-negative integer")
        _require_sha256(self.sha256, label="input provenance sha256")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "logical_name": self.logical_name,
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, value: object) -> InputFileProvenance:
        raw = _require_exact_object(
            value,
            keys={"logical_name", "path", "sha256", "size"},
            label="input provenance",
        )
        return cls(
            logical_name=raw["logical_name"],  # type: ignore[arg-type]
            path=raw["path"],  # type: ignore[arg-type]
            size=raw["size"],  # type: ignore[arg-type]
            sha256=raw["sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class GitProvenance:
    root: str
    commit: str | None
    branch: str | None
    dirty: bool
    workspace_sha256: str
    status_entry_count: int

    def __post_init__(self) -> None:
        _require_absolute_path(self.root, label="Git provenance root")
        if self.commit is not None and (
            not isinstance(self.commit, str) or not _GIT_COMMIT_RE.fullmatch(self.commit)
        ):
            raise ProvenanceError("Git provenance commit must be a lowercase Git object id")
        if self.branch is not None and (
            not isinstance(self.branch, str) or not self.branch or "\0" in self.branch
        ):
            raise ProvenanceError("Git provenance branch must be a non-empty string or null")
        if not isinstance(self.dirty, bool):
            raise ProvenanceError("Git provenance dirty flag must be boolean")
        _require_sha256(self.workspace_sha256, label="Git workspace sha256")
        if (
            isinstance(self.status_entry_count, bool)
            or not isinstance(self.status_entry_count, int)
            or self.status_entry_count < 0
        ):
            raise ProvenanceError("Git status_entry_count must be a non-negative integer")
        if self.dirty is not (self.status_entry_count > 0):
            raise ProvenanceError("Git dirty flag does not match status_entry_count")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "branch": self.branch,
            "commit": self.commit,
            "dirty": self.dirty,
            "root": self.root,
            "status_entry_count": self.status_entry_count,
            "workspace_sha256": self.workspace_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> GitProvenance:
        raw = _require_exact_object(
            value,
            keys={
                "branch",
                "commit",
                "dirty",
                "root",
                "status_entry_count",
                "workspace_sha256",
            },
            label="Git provenance",
        )
        return cls(
            root=raw["root"],  # type: ignore[arg-type]
            commit=raw["commit"],  # type: ignore[arg-type]
            branch=raw["branch"],  # type: ignore[arg-type]
            dirty=raw["dirty"],  # type: ignore[arg-type]
            workspace_sha256=raw["workspace_sha256"],  # type: ignore[arg-type]
            status_entry_count=raw["status_entry_count"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    python_version: str
    python_implementation: str
    executable: str
    platform: str
    machine: str
    processor: str
    cpu_count: int | None
    cwd: str

    def __post_init__(self) -> None:
        for label, value in (
            ("python_version", self.python_version),
            ("python_implementation", self.python_implementation),
            ("platform", self.platform),
            ("machine", self.machine),
        ):
            _require_nonempty_string(value, label=f"runtime {label}")
        if not isinstance(self.processor, str):
            raise ProvenanceError("runtime processor must be a string")
        _require_absolute_path(self.executable, label="runtime executable")
        _require_absolute_path(self.cwd, label="runtime cwd")
        if self.cpu_count is not None and (
            isinstance(self.cpu_count, bool)
            or not isinstance(self.cpu_count, int)
            or self.cpu_count < 1
        ):
            raise ProvenanceError("runtime cpu_count must be a positive integer or null")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "cpu_count": self.cpu_count,
            "cwd": self.cwd,
            "executable": self.executable,
            "machine": self.machine,
            "platform": self.platform,
            "processor": self.processor,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> RuntimeProvenance:
        raw = _require_exact_object(
            value,
            keys={
                "cpu_count",
                "cwd",
                "executable",
                "machine",
                "platform",
                "processor",
                "python_implementation",
                "python_version",
            },
            label="runtime provenance",
        )
        return cls(
            python_version=raw["python_version"],  # type: ignore[arg-type]
            python_implementation=raw["python_implementation"],  # type: ignore[arg-type]
            executable=raw["executable"],  # type: ignore[arg-type]
            platform=raw["platform"],  # type: ignore[arg-type]
            machine=raw["machine"],  # type: ignore[arg-type]
            processor=raw["processor"],  # type: ignore[arg-type]
            cpu_count=raw["cpu_count"],  # type: ignore[arg-type]
            cwd=raw["cwd"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class Provenance:
    captured_at: str
    runtime: RuntimeProvenance
    git: GitProvenance | None
    inputs: tuple[InputFileProvenance, ...]
    environment: Mapping[str, JSONLike]
    packages: Mapping[str, JSONLike]
    schema_version: int = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != PROVENANCE_SCHEMA_VERSION
        ):
            raise ProvenanceError(f"unsupported provenance schema: {self.schema_version}")
        captured_at = _require_nonempty_string(
            self.captured_at,
            label="provenance captured_at",
        )
        try:
            captured_time = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ProvenanceError("provenance captured_at must be ISO-8601") from exc
        if captured_time.tzinfo is None or captured_time.utcoffset() is None:
            raise ProvenanceError("provenance captured_at must include a timezone")
        if not isinstance(self.runtime, RuntimeProvenance):
            raise ProvenanceError("provenance runtime must be RuntimeProvenance")
        if self.git is not None and not isinstance(self.git, GitProvenance):
            raise ProvenanceError("provenance git must be GitProvenance or null")
        if not isinstance(self.inputs, Sequence) or isinstance(self.inputs, (str, bytes)):
            raise ProvenanceError("provenance inputs must be a sequence")
        inputs = tuple(self.inputs)
        if not all(isinstance(item, InputFileProvenance) for item in inputs):
            raise ProvenanceError("provenance inputs must contain InputFileProvenance")
        logical_names = [item.logical_name for item in inputs]
        if logical_names != sorted(logical_names) or len(logical_names) != len(set(logical_names)):
            raise ProvenanceError("provenance inputs must be unique and sorted by logical_name")
        object.__setattr__(self, "inputs", inputs)
        if not isinstance(self.environment, Mapping) or not all(
            isinstance(key, str)
            and _ENVIRONMENT_NAME_RE.fullmatch(key)
            and (value is None or isinstance(value, str))
            for key, value in self.environment.items()
        ):
            raise ProvenanceError(
                "provenance environment must map valid variable names to strings or null"
            )
        if any(
            any(fragment in key.upper() for fragment in _SENSITIVE_ENVIRONMENT_FRAGMENTS)
            for key in self.environment
        ):
            raise ProvenanceError("provenance environment contains a secret-like variable name")
        if not isinstance(self.packages, Mapping) or not all(
            isinstance(key, str) and bool(key) and isinstance(value, str) and bool(value)
            for key, value in self.packages.items()
        ):
            raise ProvenanceError(
                "provenance packages must map non-empty package names to versions"
            )
        environment = freeze_json(self.environment)
        packages = freeze_json(self.packages)
        if not isinstance(environment, Mapping) or not isinstance(packages, Mapping):
            raise ProvenanceError("provenance environment and packages must be mappings")
        object.__setattr__(self, "environment", environment)
        object.__setattr__(self, "packages", packages)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "captured_at": self.captured_at,
            "environment": dict(self.environment),
            "git": None if self.git is None else self.git.to_dict(),
            "inputs": [item.to_dict() for item in self.inputs],
            "packages": dict(self.packages),
            "provenance_schema_version": self.schema_version,
            "runtime": self.runtime.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Provenance:
        raw = _require_exact_object(
            value,
            keys={
                "captured_at",
                "environment",
                "git",
                "inputs",
                "packages",
                "provenance_schema_version",
                "runtime",
            },
            label="provenance",
        )
        raw_inputs = raw["inputs"]
        if not isinstance(raw_inputs, list):
            raise ProvenanceError("provenance inputs must be a list")
        raw_environment = raw["environment"]
        raw_packages = raw["packages"]
        if not isinstance(raw_environment, dict) or not isinstance(raw_packages, dict):
            raise ProvenanceError("provenance environment and packages must be objects")
        raw_git = raw["git"]
        return cls(
            captured_at=raw["captured_at"],  # type: ignore[arg-type]
            runtime=RuntimeProvenance.from_dict(raw["runtime"]),
            git=None if raw_git is None else GitProvenance.from_dict(raw_git),
            inputs=tuple(InputFileProvenance.from_dict(item) for item in raw_inputs),
            environment=raw_environment,  # type: ignore[arg-type]
            packages=raw_packages,  # type: ignore[arg-type]
            schema_version=raw["provenance_schema_version"],  # type: ignore[arg-type]
        )

    @property
    def source_identity(self) -> dict[str, JSONLike]:
        """Return the cache-significant source/runtime identity derived from this record."""

        git = self.git
        source_files = [
            {
                "logical_name": item.logical_name,
                "sha256": item.sha256,
                "size": item.size,
            }
            for item in self.inputs
        ]
        return {
            "environment": {
                "machine": self.runtime.machine,
                "packages_sha256": canonical_hash(self.packages),
                "platform": self.runtime.platform,
                "python_implementation": self.runtime.python_implementation,
                "python_version": self.runtime.python_version,
            },
            "git_branch": None if git is None else git.branch,
            "git_commit": None if git is None else git.commit,
            "git_dirty": False if git is None else git.dirty,
            "source_files_sha256": canonical_hash(source_files),
            "workspace_sha256": None if git is None else git.workspace_sha256,
        }

    @property
    def fingerprint(self) -> str:
        """Hash the complete provenance record, including capture time."""

        return canonical_hash(self.to_dict())


def capture_provenance(
    *,
    repo_root: str | Path | None = None,
    input_files: Mapping[str, str | Path] | None = None,
    environment_keys: Iterable[str] = (),
    include_packages: bool = True,
    git_timeout_seconds: float = 10.0,
) -> Provenance:
    """Capture a self-contained provenance snapshot without importing heavy libraries.

    Environment capture is opt-in and secret-like variable names are rejected.
    Input files are hashed by content and sorted by logical name.  If
    ``repo_root`` is supplied but is not a Git worktree, :class:`ProvenanceError`
    is raised rather than silently recording misleading source identity.
    """

    if git_timeout_seconds <= 0:
        raise ValueError("git_timeout_seconds must be positive")
    runtime = RuntimeProvenance(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        executable=str(Path(sys.executable).resolve()),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count(),
        cwd=str(Path.cwd().resolve()),
    )
    inputs = _capture_inputs(input_files)
    environment = _capture_environment(environment_keys)
    packages = _capture_packages() if include_packages else {}
    git = None if repo_root is None else _capture_git(Path(repo_root), git_timeout_seconds)
    return Provenance(
        captured_at=_utc_now(),
        runtime=runtime,
        git=git,
        inputs=inputs,
        environment=environment,
        packages=packages,
    )


def _capture_inputs(
    input_files: Mapping[str, str | Path] | None,
) -> tuple[InputFileProvenance, ...]:
    if input_files is None:
        return ()
    captured: list[InputFileProvenance] = []
    validated_items: list[tuple[str, str | Path]] = []
    for logical_name, raw_path in input_files.items():
        if not isinstance(logical_name, str) or not _LOGICAL_NAME_RE.fullmatch(logical_name):
            raise ProvenanceError(f"invalid input logical name: {logical_name!r}")
        validated_items.append((logical_name, raw_path))
    for logical_name, raw_path in sorted(validated_items):
        path = Path(raw_path).resolve(strict=True)
        if not path.is_file():
            raise ProvenanceError(f"provenance input is not a regular file: {path}")
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ProvenanceError(f"provenance input changed while it was being hashed: {path}")
        captured.append(
            InputFileProvenance(
                logical_name=logical_name,
                path=str(path),
                size=after.st_size,
                sha256=digest,
            )
        )
    return tuple(captured)


def _capture_environment(keys: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for key in sorted(set(keys)):
        if not isinstance(key, str) or not _ENVIRONMENT_NAME_RE.fullmatch(key):
            raise ProvenanceError(f"invalid environment variable name: {key!r}")
        upper = key.upper()
        if any(fragment in upper for fragment in _SENSITIVE_ENVIRONMENT_FRAGMENTS):
            raise ProvenanceError(f"refusing to capture secret-like environment variable: {key}")
        result[key] = os.environ.get(key)
    return result


def _capture_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = distribution.version
    return dict(sorted(packages.items()))


def _run_git(root: Path, arguments: list[str], timeout: float) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ProvenanceError(
            f"unable to capture Git provenance with {' '.join(arguments)!r}: {exc}"
        ) from exc
    return completed.stdout


def _capture_git(requested_root: Path, timeout: float) -> GitProvenance:
    root_output = _run_git(requested_root, ["rev-parse", "--show-toplevel"], timeout)
    root = Path(root_output.decode("utf-8").strip()).resolve(strict=True)
    for _attempt in range(3):
        commit = _run_git(root, ["rev-parse", "--verify", "HEAD"], timeout)
        branch_output = _run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"], timeout)
        status = _run_git(root, ["status", "--porcelain=v1", "-z"], timeout)
        diff = _run_git(root, ["diff", "--binary", "HEAD"], timeout)
        untracked = _run_git(root, ["ls-files", "--others", "--exclude-standard", "-z"], timeout)
        try:
            untracked_manifest = _hash_untracked_files(root, untracked)
        except OSError:
            continue
        if commit != _run_git(root, ["rev-parse", "--verify", "HEAD"], timeout):
            continue
        if branch_output != _run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"], timeout):
            continue
        if status != _run_git(root, ["status", "--porcelain=v1", "-z"], timeout):
            continue
        if diff != _run_git(root, ["diff", "--binary", "HEAD"], timeout):
            continue
        confirmed_untracked = _run_git(
            root, ["ls-files", "--others", "--exclude-standard", "-z"], timeout
        )
        if untracked != confirmed_untracked:
            continue
        try:
            confirmed_untracked_manifest = _hash_untracked_files(
                root,
                confirmed_untracked,
            )
        except OSError:
            continue
        if canonical_hash(untracked_manifest) != canonical_hash(confirmed_untracked_manifest):
            continue
        branch_text = branch_output.decode("utf-8").strip()
        workspace_sha256 = sha256_bytes(
            b"git-workspace-v1\0"
            + status
            + b"\0"
            + diff
            + b"\0"
            + canonical_hash(untracked_manifest).encode("ascii")
        )
        return GitProvenance(
            root=str(root),
            commit=commit.decode("ascii").strip(),
            branch=None if branch_text == "HEAD" else branch_text,
            dirty=bool(status),
            workspace_sha256=workspace_sha256,
            status_entry_count=_count_porcelain_entries(status),
        )
    raise ProvenanceError("Git worktree changed repeatedly while provenance was captured")


def _hash_untracked_files(root: Path, output: bytes) -> list[dict[str, JSONValue]]:
    manifest: list[dict[str, JSONValue]] = []
    for encoded_path in sorted(item for item in output.split(b"\0") if item):
        path = root / os.fsdecode(encoded_path)
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            target = os.readlink(path)
            target_bytes = os.fsencode(target)
            after = path.lstat()
            if _stat_identity(before) != _stat_identity(after) or target != os.readlink(path):
                raise OSError(f"symbolic link changed while hashing: {path}")
            manifest.append(
                {
                    "kind": "symlink",
                    "path_bytes_hex": encoded_path.hex(),
                    "sha256": sha256_bytes(target_bytes),
                    "size": len(target_bytes),
                }
            )
        elif stat.S_ISREG(before.st_mode):
            digest = sha256_file(path)
            after = path.lstat()
            if _stat_identity(before) != _stat_identity(after):
                raise OSError(f"file changed while hashing: {path}")
            manifest.append(
                {
                    "kind": "file",
                    "path_bytes_hex": encoded_path.hex(),
                    "sha256": digest,
                    "size": after.st_size,
                }
            )
        else:
            raise ProvenanceError(f"unsupported untracked filesystem entry: {path}")
    return manifest


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_mode, value.st_ino, value.st_size, value.st_mtime_ns)


def _count_porcelain_entries(status: bytes) -> int:
    entries = [item for item in status.split(b"\0") if item]
    count = 0
    index = 0
    while index < len(entries):
        record = entries[index]
        count += 1
        renamed_or_copied = len(record) >= 2 and (
            record[0:1] in {b"R", b"C"} or record[1:2] in {b"R", b"C"}
        )
        index += 2 if renamed_or_copied else 1
    return count
