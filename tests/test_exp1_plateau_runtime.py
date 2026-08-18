from __future__ import annotations

from types import SimpleNamespace

import pytest

from voronoi_lab import stage_handlers


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    @staticmethod
    def current_device() -> int:
        return 2

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 2
        return "contract-test GPU"

    @staticmethod
    def get_device_capability(index: int) -> tuple[int, int]:
        assert index == 2
        return (8, 9)


class _FakeCudnn:
    deterministic = False
    benchmark = True

    @staticmethod
    def version() -> int:
        return 90100


class _FakeTorch:
    __version__ = "2.contract"

    def __init__(self, *, cuda_available: bool) -> None:
        self.cuda = _FakeCuda(cuda_available)
        self.backends = SimpleNamespace(cudnn=_FakeCudnn())
        self.version = SimpleNamespace(cuda="12.contract")
        self._deterministic = False

    def use_deterministic_algorithms(self, enabled: bool) -> None:
        self._deterministic = enabled

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self._deterministic


def test_plateau_cpu_runtime_is_explicit_and_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch = _FakeTorch(cuda_available=False)
    actual, provenance = stage_handlers._configure_plateau_torch_runtime(
        "cpu",
        torch_module=torch,
    )

    assert actual == "cpu"
    assert provenance == {
        "requested_device": "cpu",
        "actual_device": "cpu",
        "torch_version": "2.contract",
        "torch_cuda_build_version": "12.contract",
        "cuda_available": False,
        "cuda_device_index": None,
        "cuda_device_name": None,
        "cuda_compute_capability": None,
        "cudnn_version": 90100,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": None,
        "persisted_arrays": "cpu_numpy_float32",
    }


def test_plateau_cuda_runtime_requires_an_available_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(stage_handlers.StageHandlerError, match=r"cuda\.is_available"):
        stage_handlers._configure_plateau_torch_runtime(
            "cuda",
            torch_module=_FakeTorch(cuda_available=False),
        )


def test_plateau_cuda_runtime_resolves_index_and_records_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch = _FakeTorch(cuda_available=True)
    actual, provenance = stage_handlers._configure_plateau_torch_runtime(
        "cuda",
        torch_module=torch,
    )

    assert actual == "cuda:2"
    assert provenance["actual_device"] == "cuda:2"
    assert provenance["cuda_device_name"] == "contract-test GPU"
    assert provenance["cuda_compute_capability"] == [8, 9]
    assert provenance["deterministic_algorithms"] is True
    assert provenance["cudnn_deterministic"] is True
    assert provenance["cudnn_benchmark"] is False
    assert provenance["cublas_workspace_config"] == ":4096:8"


def test_plateau_cuda_runtime_rejects_nondeterministic_cublas_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(stage_handlers.StageHandlerError, match="deterministic CUDA"):
        stage_handlers._configure_plateau_torch_runtime(
            "cuda",
            torch_module=_FakeTorch(cuda_available=True),
        )
