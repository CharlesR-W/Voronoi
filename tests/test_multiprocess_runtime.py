from __future__ import annotations

import multiprocessing
from pathlib import Path

from voronoi_lab.core import ArtifactStore, RunIndex


def _publish_from_process(
    artifact_root: str,
    index_path: str,
    run_id: str,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    try:
        start.wait(timeout=10)
        store = ArtifactStore(artifact_root)
        reference = store.put_json(
            {"schema_version": 1, "value": "shared"},
            filename="shared.json",
            kind="fixture/shared",
        )
        index = RunIndex(index_path)
        index.register_run(
            run_id,
            config_hash="a" * 64,
            provenance_artifact_id=reference.artifact_id,
            metadata={"worker": run_id},
        )
        results.put(("ok", reference.artifact_id))
    except BaseException as error:  # pragma: no cover - child failure is asserted by parent
        results.put(("error", f"{type(error).__name__}: {error}"))


def test_artifact_and_run_index_initialization_are_multiprocess_safe(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    artifact_root = str(tmp_path / "artifacts")
    index_path = str(tmp_path / "runs" / "index.sqlite")
    processes = [
        context.Process(
            target=_publish_from_process,
            args=(artifact_root, index_path, f"worker-{index}", start, results),
        )
        for index in range(3)
    ]
    for process in processes:
        process.start()
    start.set()
    observed = [results.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    assert {status for status, _value in observed} == {"ok"}
    artifact_ids = {value for _status, value in observed}
    assert len(artifact_ids) == 1
    reference = ArtifactStore(artifact_root).get(artifact_ids.pop(), verify=True)
    assert reference.manifest.kind == "fixture/shared"
    index = RunIndex(index_path)
    assert all(index.get_run(f"worker-{worker}") is not None for worker in range(3))
