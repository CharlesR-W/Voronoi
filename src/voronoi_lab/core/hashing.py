"""Deterministic JSON serialization, hashing, and seed derivation.

The functions in this module intentionally accept only JSON-like values.  That
keeps cache keys and random streams independent of object reprs, dictionary
insertion order, and Python's process-randomized ``hash`` implementation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

JSONScalar: TypeAlias = bool | int | float | str | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONLike: TypeAlias = JSONScalar | Sequence["JSONLike"] | Mapping[str, "JSONLike"]
SeedRoot: TypeAlias = int | str | bytes

_HASH_CHUNK_SIZE = 1024 * 1024
_SEED_DOMAIN = b"voronoi-lab-seed-v1\0"


class CanonicalJSONError(ValueError):
    """Raised when a value cannot be represented by the canonical JSON format."""


def _normalize(value: JSONLike, *, location: str = "$") -> JSONValue:
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str):
            return unicodedata.normalize("NFC", value)
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJSONError(f"{location}: non-finite floats are not permitted")
        return value

    if isinstance(value, Mapping):
        normalized: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJSONError(f"{location}: object keys must be strings")
            canonical_key = unicodedata.normalize("NFC", key)
            if canonical_key in normalized:
                raise CanonicalJSONError(
                    f"{location}: keys collide after Unicode normalization: {key!r}"
                )
            normalized[canonical_key] = _normalize(item, location=f"{location}.{canonical_key}")
        return normalized

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        return [
            _normalize(item, location=f"{location}[{index}]") for index, item in enumerate(value)
        ]

    raise CanonicalJSONError(
        f"{location}: unsupported value type {type(value).__qualname__}; "
        "convert it to JSON primitives explicitly"
    )


def _freeze_normalized_json(value: JSONValue) -> JSONLike:
    """Recursively freeze normalized JSON so seed namespaces cannot drift."""

    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_normalized_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_normalized_json(item) for item in value)
    return value


def freeze_json(value: JSONLike) -> JSONLike:
    """Return an immutable, canonical snapshot of a JSON-like value."""

    return _freeze_normalized_json(_normalize(value))


def thaw_json(value: JSONLike) -> JSONValue:
    """Return a detached plain-dict/list canonical snapshot."""

    return _normalize(value)


def canonical_json_bytes(value: JSONLike) -> bytes:
    """Serialize a JSON-like value to deterministic UTF-8 bytes.

    Object keys are sorted, insignificant whitespace is omitted, strings are
    normalized to NFC, and NaN/infinity are rejected.  Tuples and other
    non-byte sequences serialize as JSON arrays.
    """

    normalized = _normalize(value)
    try:
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:  # Defensive: _normalize should catch these.
        raise CanonicalJSONError(str(exc)) from exc
    return encoded.encode("utf-8")


def canonical_json_text(value: JSONLike) -> str:
    """Return the canonical JSON representation as text."""

    return canonical_json_bytes(value).decode("utf-8")


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    """Return a lowercase SHA-256 hex digest for a byte buffer."""

    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    """Stream a file and return its lowercase SHA-256 hex digest."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: JSONLike) -> str:
    """Hash a value's canonical JSON representation with SHA-256."""

    return sha256_bytes(canonical_json_bytes(value))


def _seed_key(root_seed: SeedRoot) -> bytes:
    if isinstance(root_seed, bool):
        raise TypeError("root_seed must be an int, str, or bytes, not bool")
    if isinstance(root_seed, int):
        if root_seed < 0:
            raise ValueError("integer root_seed must be non-negative")
        return b"int\0" + str(root_seed).encode("ascii")
    if isinstance(root_seed, str):
        return b"str\0" + unicodedata.normalize("NFC", root_seed).encode("utf-8")
    if isinstance(root_seed, bytes):
        return b"bytes\0" + root_seed
    raise TypeError("root_seed must be an int, str, or bytes")


def derive_seed(root_seed: SeedRoot, *coordinates: JSONLike, bits: int = 64) -> int:
    """Derive a deterministic integer seed from semantic coordinates.

    HMAC gives every coordinate tuple an independent-looking stream while
    preserving reproducibility.  Type tags make roots such as ``1`` and
    ``"1"`` distinct.  ``bits`` must be byte-aligned and no larger than the
    SHA-256 output.
    """

    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError("bits must be an integer")
    if bits <= 0 or bits > 256 or bits % 8:
        raise ValueError("bits must be a positive multiple of 8 no larger than 256")

    message = _SEED_DOMAIN + canonical_json_bytes(list(coordinates))
    digest = hmac.new(_seed_key(root_seed), message, hashlib.sha256).digest()
    return int.from_bytes(digest[: bits // 8], byteorder="big", signed=False)


@dataclass(frozen=True, slots=True)
class SeedDeriver:
    """A namespaced deterministic seed factory."""

    root_seed: SeedRoot
    namespace: tuple[JSONLike, ...] = ()

    def __post_init__(self) -> None:
        _seed_key(self.root_seed)
        normalized = freeze_json(list(self.namespace))
        assert isinstance(normalized, tuple)
        object.__setattr__(self, "namespace", normalized)

    def child(self, *coordinates: JSONLike) -> SeedDeriver:
        """Return a new factory beneath additional semantic coordinates."""

        return SeedDeriver(self.root_seed, (*self.namespace, *coordinates))

    def derive(self, *coordinates: JSONLike, bits: int = 64) -> int:
        """Derive a seed within this namespace."""

        return derive_seed(self.root_seed, *self.namespace, *coordinates, bits=bits)
