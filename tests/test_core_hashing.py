from __future__ import annotations

import hashlib

import pytest

from voronoi_lab.core import (
    CanonicalJSONError,
    SeedDeriver,
    canonical_hash,
    canonical_json_bytes,
    derive_seed,
    sha256_file,
)


def test_canonical_json_is_order_independent_and_normalizes_sequences_and_unicode() -> None:
    composed = "é"
    decomposed = "e\N{COMBINING ACUTE ACCENT}"
    left = {"z": (1, True, None), decomposed: "value"}
    right = {composed: "value", "z": [1, True, None]}

    expected = '{"z":[1,true,null],"é":"value"}'.encode()
    assert canonical_json_bytes(left) == expected
    assert canonical_json_bytes(right) == expected
    assert canonical_hash(left) == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(CanonicalJSONError, match="non-finite"):
        canonical_json_bytes({"bad": value})


def test_canonical_json_rejects_non_string_keys_and_implicit_object_reprs() -> None:
    with pytest.raises(CanonicalJSONError, match="keys must be strings"):
        canonical_json_bytes({1: "value"})  # type: ignore[dict-item]
    with pytest.raises(CanonicalJSONError, match="unsupported value type"):
        canonical_json_bytes({"bad": object()})  # type: ignore[dict-item]


def test_canonical_json_detects_unicode_key_collisions() -> None:
    with pytest.raises(CanonicalJSONError, match="collide"):
        canonical_json_bytes({"é": 1, "e\N{COMBINING ACUTE ACCENT}": 2})


def test_seed_derivation_is_stable_namespaced_and_type_sensitive() -> None:
    direct = derive_seed(1234, "probe", {"cut": 2}, bits=64)
    namespaced = SeedDeriver(1234).child("probe").derive({"cut": 2}, bits=64)

    assert direct == namespaced
    assert direct == 6630124293588212588
    assert derive_seed(1234, "probe", {"cut": 3}) != direct
    assert derive_seed("1234", "probe", {"cut": 2}) != direct
    assert derive_seed(b"1234", "probe", {"cut": 2}) != direct
    assert 0 <= derive_seed(1234, "probe", bits=32) < 2**32


def test_seed_validation_rejects_ambiguous_or_invalid_roots_and_widths() -> None:
    with pytest.raises(TypeError, match="not bool"):
        derive_seed(True, "x")
    with pytest.raises(ValueError, match="non-negative"):
        derive_seed(-1, "x")
    with pytest.raises(ValueError, match="positive multiple"):
        derive_seed(1, "x", bits=7)
    with pytest.raises(CanonicalJSONError):
        SeedDeriver(1, (float("nan"),))


def test_seed_deriver_snapshots_mutable_namespace_coordinates() -> None:
    coordinate = {"cuts": [1, 2]}
    deriver = SeedDeriver(7, (coordinate,))
    before = deriver.derive("probe")

    coordinate["cuts"].append(3)

    assert deriver.derive("probe") == before
    with pytest.raises(TypeError):
        deriver.namespace[0]["new"] = True  # type: ignore[index]


def test_sha256_file_streams_exact_bytes(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    payload = b"abc" * 500_000
    path.write_bytes(payload)

    assert sha256_file(path, chunk_size=997) == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="positive"):
        sha256_file(path, chunk_size=0)
