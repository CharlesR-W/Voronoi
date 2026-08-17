from __future__ import annotations

import copy
import hashlib
import os
import subprocess

import pytest

import voronoi_lab.core.provenance as provenance_module
from voronoi_lab.core import Provenance, ProvenanceError, capture_provenance


def test_capture_provenance_hashes_inputs_and_only_requested_environment(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "checkpoint.bin"
    input_path.write_bytes(b"checkpoint bytes")
    monkeypatch.setenv("VORONOI_TEST_DEVICE", "cpu")
    monkeypatch.setenv("UNREQUESTED_VALUE", "must not leak")

    provenance = capture_provenance(
        input_files={"model/checkpoint": input_path},
        environment_keys=["VORONOI_TEST_DEVICE", "MISSING_SAFE_VALUE"],
        include_packages=False,
    )

    assert provenance.git is None
    assert provenance.environment == {
        "MISSING_SAFE_VALUE": None,
        "VORONOI_TEST_DEVICE": "cpu",
    }
    assert provenance.packages == {}
    assert provenance.inputs[0].logical_name == "model/checkpoint"
    assert provenance.inputs[0].size == len(b"checkpoint bytes")
    assert provenance.inputs[0].sha256 == hashlib.sha256(b"checkpoint bytes").hexdigest()
    assert "UNREQUESTED_VALUE" not in provenance.to_dict()["environment"]
    assert len(provenance.fingerprint) == 64
    assert Provenance.from_dict(provenance.to_dict()) == provenance
    assert Provenance.from_dict(provenance.to_dict()).source_identity == provenance.source_identity


def test_persisted_provenance_schema_is_exact_and_nested_values_are_validated(tmp_path) -> None:
    provenance = capture_provenance(include_packages=False)
    payload = provenance.to_dict()

    with pytest.raises(ProvenanceError, match="keys must be exactly"):
        Provenance.from_dict({})

    missing = dict(payload)
    missing.pop("runtime")
    with pytest.raises(ProvenanceError, match="keys must be exactly"):
        Provenance.from_dict(missing)

    extra = {**payload, "unexpected": True}
    with pytest.raises(ProvenanceError, match="keys must be exactly"):
        Provenance.from_dict(extra)

    invalid_runtime = copy.deepcopy(payload)
    invalid_runtime["runtime"] = {}
    with pytest.raises(ProvenanceError, match="runtime provenance keys"):
        Provenance.from_dict(invalid_runtime)

    invalid_packages = copy.deepcopy(payload)
    invalid_packages["packages"] = {"package": 1}
    with pytest.raises(ProvenanceError, match="packages must map"):
        Provenance.from_dict(invalid_packages)

    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"input")
    with_input = capture_provenance(
        input_files={"input": input_path},
        include_packages=False,
    ).to_dict()
    with_input["inputs"][0]["sha256"] = "not-a-digest"
    with pytest.raises(ProvenanceError, match="SHA-256"):
        Provenance.from_dict(with_input)


@pytest.mark.parametrize("name", ["API_TOKEN", "DB_PASSWORD", "SSH_AUTH_SOCK", "PRIVATE_KEY"])
def test_capture_provenance_refuses_secret_like_environment_names(name: str) -> None:
    with pytest.raises(ProvenanceError, match="secret-like"):
        capture_provenance(environment_keys=[name], include_packages=False)


def test_capture_git_provenance_fingerprints_dirty_content(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("initial\n")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "initial"], check=True)

    clean = capture_provenance(repo_root=repo, include_packages=False)
    assert clean.git is not None
    assert not clean.git.dirty
    assert len(clean.git.commit or "") == 40
    assert Provenance.from_dict(clean.to_dict()) == clean

    tracked.write_text("changed\n")
    (repo / "untracked.txt").write_text("new\n")
    dirty = capture_provenance(repo_root=repo, include_packages=False)
    assert dirty.git is not None
    assert dirty.git.dirty
    assert dirty.git.status_entry_count == 2
    assert dirty.git.workspace_sha256 != clean.git.workspace_sha256


def test_capture_git_retries_when_untracked_content_changes_between_snapshots(
    tmp_path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("tracked\n")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "initial"], check=True)
    untracked = repo / "untracked.txt"
    untracked.write_text("first\n")

    original = provenance_module._hash_untracked_files
    calls = 0

    def mutate_after_first_snapshot(root, output):
        nonlocal calls
        calls += 1
        manifest = original(root, output)
        if calls == 1:
            untracked.write_text("second\n")
        return manifest

    monkeypatch.setattr(provenance_module, "_hash_untracked_files", mutate_after_first_snapshot)
    captured = capture_provenance(repo_root=repo, include_packages=False)
    monkeypatch.setattr(provenance_module, "_hash_untracked_files", original)
    stable = capture_provenance(repo_root=repo, include_packages=False)

    assert calls >= 3
    assert captured.git is not None and stable.git is not None
    assert captured.git.workspace_sha256 == stable.git.workspace_sha256


def test_capture_provenance_rejects_non_git_root(tmp_path) -> None:
    with pytest.raises(ProvenanceError, match="Git provenance"):
        capture_provenance(repo_root=tmp_path, include_packages=False)


def test_provenance_environment_capture_does_not_require_variable_to_exist(monkeypatch) -> None:
    monkeypatch.delenv("VORONOI_SAFE_MISSING", raising=False)
    provenance = capture_provenance(
        environment_keys=["VORONOI_SAFE_MISSING"], include_packages=False
    )
    assert provenance.environment["VORONOI_SAFE_MISSING"] is None
    assert os.environ.get("VORONOI_SAFE_MISSING") is None
