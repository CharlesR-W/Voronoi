from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from tests.tracking2_fixture import with_tracking2_fixture
from voronoi_lab import stage_handlers
from voronoi_lab.config import LabConfig
from voronoi_lab.core import ArtifactStore, GateStatus, RunIndex, canonical_hash
from voronoi_lab.execution import ExecutionError, ExperimentRunner, StageContext
from voronoi_lab.pipeline import (
    DEFAULT_STAGES,
    ImplementationStatus,
    PipelineError,
    stage_signature,
    validate_stage_output,
)


def _small_config() -> LabConfig:
    config = LabConfig()
    probe_banks = config.experiment1.probe_banks.model_copy(
        update={
            "fit_train_images": 3,
            "independent_fit_train_images": 2,
            "geometry_test_images": 4,
            "intervention_test_images": 2,
            "max_sites_per_image": 2,
        }
    )
    bootstrap = config.experiment1.bootstrap.model_copy(update={"resamples": 3})
    experiment1 = config.experiment1.model_copy(
        update={
            "cuts": ("stage1.block1",),
            "sentinel_cuts": ("stage1.block1",),
            "probe_banks": probe_banks,
            "bootstrap": bootstrap,
        }
    )
    exact_protocol = config.experiment2.exact_protocol.model_copy(
        update={
            "max_states": 4,
            "random_relabel": True,
            "generator_rate_shape": 3.0,
            "exhaustive_tie_atol": 2e-10,
            "exhaustive_tie_rtol": 3e-10,
        }
    )
    experiment2 = config.experiment2.model_copy(
        update={
            "exact_instances": 2,
            "exact_protocol": exact_protocol,
            "oracle_factor_sizes": (2, 2),
            "train_primitives": 4,
            "unary_weight": 1.7,
        }
    )
    synthetic_gate = config.gates.synthetic.model_copy(update={"noiseless_instances": 2})
    coarse_gate = config.gates.coarse.model_copy(update={"required_passing_sentinel_cuts": 1})
    functional_gate = config.gates.functional.model_copy(update={"required_passing_cuts": 1})
    confirmation_gate = config.gates.confirmation.model_copy(update={"required_passing_cuts": 1})
    gates = config.gates.model_copy(
        update={
            "coarse": coarse_gate,
            "functional": functional_gate,
            "synthetic": synthetic_gate,
            "confirmation": confirmation_gate,
        }
    )
    runtime = config.runtime.model_copy(update={"workers": 2})
    return config.model_copy(
        update={
            "experiment1": experiment1,
            "experiment2": experiment2,
            "gates": gates,
            "runtime": runtime,
        }
    )


def _context(tmp_path: Path, config: LabConfig, stage_name: str) -> StageContext:
    store = ArtifactStore(tmp_path / "artifacts")
    index = RunIndex(tmp_path / "runs.sqlite")
    source_identity = {"git_commit": "fixture", "workspace_sha256": "fixture"}
    config_hash = canonical_hash(config.model_dump(mode="json"))
    run_id = f"fixture-{config_hash[:12]}"
    provenance = store.put_json(
        {"schema_version": 1, "fixture": True},
        filename="provenance.json",
        kind="fixture/provenance",
    )
    index.register_run(
        run_id,
        config_hash=config_hash,
        provenance_artifact_id=provenance.artifact_id,
        metadata={"source_identity": source_identity},
    )
    return StageContext(
        config=config,
        store=store,
        index=index,
        run_id=run_id,
        project_root=tmp_path,
        source_identity=source_identity,
        stage_spec=DEFAULT_STAGES.get(stage_name),
    )


def _claim_parent(context: StageContext) -> None:
    signature = stage_signature(
        context.stage_spec,
        context.config,
        upstream_artifact_ids={},
        source_identity=context.source_identity,
    )
    context.index.claim_stage(
        context.run_id,
        context.stage_spec.name,
        signature,
        owner_token="fixture-parent",
    )


class _Dumpable(SimpleNamespace):
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return dict(self.payload)


def test_inputs_handler_can_be_exercised_with_a_read_only_adapter_fixture(
    tmp_path, monkeypatch
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("fixture: true\n")
    reference = SimpleNamespace(path=Path("fixture.bin"), size_bytes=7, sha256="a" * 64)
    test_reference = SimpleNamespace(path=Path("test-fixture.bin"), size_bytes=8, sha256="d" * 64)
    checkpoint = SimpleNamespace(epoch=0, path=Path("epoch0.pt"), size_bytes=9, sha256="b" * 64)
    architecture = _Dumpable(
        payload={"width": 64, "source": {"path": "fixture.bin"}},
        width=64,
        source=reference,
    )
    training = _Dumpable(payload={"train_size": 10, "test_size": 8})
    manifest = _Dumpable(
        payload={"fixture": True},
        architecture=architecture,
        checkpoints=(checkpoint,),
        datasets=SimpleNamespace(train=reference, test=test_reference),
        transplant=SimpleNamespace(file=reference),
        lineage_note="fixture lineage",
        lineage_quality="exploratory_legacy",
        observed_repository_revision="c" * 40,
        training=training,
    )
    validated = {
        "model_source": tmp_path / "fixture.bin",
        "checkpoint_epoch0": tmp_path / "epoch0.pt",
        "dataset_train": tmp_path / "train.parquet",
        "dataset_test": tmp_path / "test.parquet",
        "transplant": tmp_path / "transplant.json",
    }
    adapter = SimpleNamespace(
        manifest=manifest,
        root=tmp_path,
        validate_all=lambda: validated,
        read_validated_bytes=lambda _reference: b"fixture-transplant",
        normalize_transplant_bytes=lambda _raw: (_Dumpable(payload={"cut_index": 0}),),
    )
    monkeypatch.setattr(
        stage_handlers,
        "_tracking2_adapter",
        lambda _context: (manifest_path, adapter),
    )
    monkeypatch.setattr(
        stage_handlers,
        "parse_tracking2_manifest_bytes",
        lambda *_args, **_kwargs: manifest,
    )
    config = _small_config()
    inputs = config.inputs.tracking2.model_copy(
        update={"manifest_sha256": stage_handlers.sha256_file(manifest_path)}
    )
    config = config.model_copy(
        update={"inputs": config.inputs.model_copy(update={"tracking2": inputs})}
    )
    context = _context(tmp_path, config, "inputs.tracking2")
    reference_artifact = stage_handlers.handle_inputs_tracking2(context, {})
    payload = context.store.read_json(reference_artifact.artifact_id, "inputs.json")
    assert payload["read_only"] is True
    assert payload["training"] == {"test_size": 8, "train_size": 10}
    assert len(payload["validated_files"]) == 5
    assert payload["transplant_rows"] == [{"cut_index": 0}]
    assert payload["external_root"] == "../Experiments/Tracking2"
    assert payload["manifest"]["path"] == "configs/inputs/tracking2_seed0.yaml"
    assert all("resolved_path" not in entry for entry in payload["validated_files"].values())

    relocated_root = tmp_path / "relocated"
    relocated_root.mkdir()
    relocated_manifest = relocated_root / "manifest.yaml"
    relocated_manifest.write_text(manifest_path.read_text())
    relocated_adapter = SimpleNamespace(
        manifest=manifest,
        root=relocated_root,
        validate_all=lambda: {name: relocated_root / path.name for name, path in validated.items()},
        read_validated_bytes=adapter.read_validated_bytes,
        normalize_transplant_bytes=adapter.normalize_transplant_bytes,
    )
    monkeypatch.setattr(
        stage_handlers,
        "_tracking2_adapter",
        lambda _context: (relocated_manifest, relocated_adapter),
    )
    relocated_context = _context(relocated_root, config, "inputs.tracking2")
    relocated_artifact = stage_handlers.handle_inputs_tracking2(relocated_context, {})
    assert relocated_artifact.artifact_id == reference_artifact.artifact_id


def test_tracking2_boundary_rejects_checkpoint_and_bank_axis_mismatch(
    tmp_path, monkeypatch
) -> None:
    config = _small_config()
    monkeypatch.setattr(
        stage_handlers,
        "sha256_file",
        lambda _path: config.inputs.tracking2.manifest_sha256,
    )
    manifest = SimpleNamespace(
        architecture=SimpleNamespace(width=64),
        checkpoints=(SimpleNamespace(epoch=0),),
        training=SimpleNamespace(train_size=10, test_size=8),
    )
    adapter = SimpleNamespace(manifest=manifest)
    monkeypatch.setattr(
        stage_handlers.Tracking2Adapter,
        "from_yaml",
        lambda *_args, **_kwargs: adapter,
    )
    with pytest.raises(stage_handlers.StageHandlerError, match="checkpoints"):
        stage_handlers.tracking2_adapter_from_config(config, tmp_path)

    matching_manifest = SimpleNamespace(
        architecture=SimpleNamespace(width=64),
        checkpoints=tuple(SimpleNamespace(epoch=value) for value in config.experiment1.checkpoints),
        training=SimpleNamespace(train_size=4, test_size=8),
    )
    adapter.manifest = matching_manifest
    with pytest.raises(stage_handlers.StageHandlerError, match="fit banks exceed"):
        stage_handlers.tracking2_adapter_from_config(config, tmp_path)


def test_input_preflight_rejects_stale_cache_after_external_manifest_changes(
    tmp_path,
) -> None:
    config = _small_config()
    banks = config.experiment1.probe_banks.model_copy(
        update={"geometry_test_images": 2, "intervention_test_images": 1}
    )
    config = config.model_copy(
        update={"experiment1": config.experiment1.model_copy(update={"probe_banks": banks})}
    )
    config = with_tracking2_fixture(config, tmp_path)
    producer = ExperimentRunner(
        config,
        project_root=tmp_path,
        handlers={"inputs.tracking2": stage_handlers.handle_inputs_tracking2},
        run_id="input-preflight-producer",
    )
    first = producer.run(["inputs.tracking2"])["inputs.tracking2"]

    model_path = tmp_path / "external" / "src" / "tracking2" / "models.py"
    replacement = b"# coherently changed fixture model\n"
    model_path.write_bytes(replacement)
    manifest_path = tmp_path / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["architecture"]["source"].update(
        {
            "size_bytes": len(replacement),
            "sha256": hashlib.sha256(replacement).hexdigest(),
        }
    )
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    stale = ExperimentRunner(
        config,
        project_root=tmp_path,
        handlers={"inputs.tracking2": stage_handlers.handle_inputs_tracking2},
        run_id="input-preflight-stale-consumer",
    )
    with pytest.raises(ExecutionError, match="manifest SHA-256"):
        stale.run(["inputs.tracking2"])
    assert stale.index.get_stage(stale.run_id, "inputs.tracking2") is None

    updated_inputs = config.inputs.tracking2.model_copy(
        update={"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    )
    updated = config.model_copy(
        update={"inputs": config.inputs.model_copy(update={"tracking2": updated_inputs})}
    )
    consumer = ExperimentRunner(
        updated,
        project_root=tmp_path,
        handlers={"inputs.tracking2": stage_handlers.handle_inputs_tracking2},
        run_id="input-preflight-updated-consumer",
    )
    second = consumer.run(["inputs.tracking2"])["inputs.tracking2"]
    assert second.artifact_id != first.artifact_id
    record = consumer.index.get_stage(consumer.run_id, "inputs.tracking2")
    assert record is not None and record.metadata["cache_hit"] is False


def test_probe_handler_freezes_indices_sites_and_bootstraps_deterministically(tmp_path) -> None:
    config = _small_config()
    context = _context(tmp_path, config, "exp1.probe_banks")
    input_ref = context.store.put_json(
        {"training": {"train_size": 10, "test_size": 8}},
        filename="inputs.json",
        kind="fixture/inputs",
    )
    dependencies = {"inputs.tracking2": input_ref}
    first = stage_handlers.handle_exp1_probe_banks(context, dependencies)
    second = stage_handlers.handle_exp1_probe_banks(context, dependencies)
    assert first.artifact_id == second.artifact_id

    payload = context.store.read_json(first.artifact_id, "plan.json")
    assert payload["cuts"] == ["stage1.block1"]
    assert payload["roles"]["geometry"]["bootstrap_shape"] == [3, 4]
    assert stage_handlers._probe_determinism_check(
        context,
        {"training": {"train_size": 10, "test_size": 8}},
        first,
    )

    copied_files = {
        entry.path: context.store.read_bytes(first.artifact_id, entry.path)
        for entry in first.manifest.files
    }
    copied_files["indices/geometry.npy"] += b"changed"
    changed_ref = context.store.put_files(
        copied_files,
        kind="fixture/changed-probe",
        media_types={"plan.json": "application/json"},
    )
    assert not stage_handlers._probe_determinism_check(
        context,
        {"training": {"train_size": 10, "test_size": 8}},
        changed_ref,
    )
    assert set(payload["roles"]["geometry"]["site_files"]) == {"stage1.block1"}
    site_file = payload["roles"]["geometry"]["site_files"]["stage1.block1"]["file"]
    sites = np.load(io.BytesIO(context.store.read_bytes(first.artifact_id, site_file)))
    assert sites.shape == (8, 3)
    assert len(np.unique(sites[:, 0])) == 4

    alternate_exp1 = config.experiment1.model_copy(
        update={"cuts": ("stage4.block2",), "sentinel_cuts": ("stage4.block2",)}
    )
    alternate = config.model_copy(update={"experiment1": alternate_exp1})
    alternate_context = _context(tmp_path, alternate, "exp1.probe_banks")
    alternate_ref = stage_handlers.handle_exp1_probe_banks(alternate_context, dependencies)
    alternate_plan = alternate_context.store.read_json(alternate_ref.artifact_id, "plan.json")
    assert alternate_ref.artifact_id != first.artifact_id
    assert alternate_plan["roles"]["geometry"]["site_files"]["stage4.block2"][
        "activation_shape"
    ] == [512, 4, 4]


def test_mechanical_gate_preserves_missing_metrics_but_fails_incomplete_jvp(tmp_path) -> None:
    config = _small_config()
    context = _context(tmp_path, config, "gate.mechanical")
    observations = {
        "probe_banks": {
            "artifact_valid": True,
            "deterministic": True,
            "distinct_train_test_sources": True,
        },
        "geometry": {
            "roundtrip": {"relative_rms_error": 0.0},
            "centroid_reconstruction": {"relative_rms_error": 0.0},
            "boundary": {"passed": True},
            "mixture_gaussian": {"passed": True},
        },
        "resnet": {
            "identity_exact": None,
            "jvp_cuts_completed": 0,
            "jvp_median_relative_error": None,
            "jvp_p95_relative_error": None,
        },
        "synthetic_invariants": {"passed": True},
    }
    mechanical_ref = context.store.put_json(
        observations,
        filename="mechanical.json",
        kind="fixture/mechanical",
    )
    probe_ref = context.store.put_json(
        {"schema_version": 1}, filename="plan.json", kind="fixture/probe"
    )
    gate_ref = stage_handlers.handle_gate_mechanical(
        context,
        {"exp1.mechanical": mechanical_ref, "exp1.probe_banks": probe_ref},
    )
    result = context.store.read_json(gate_ref.artifact_id, "gate.json")
    assert result["status"] == GateStatus.FAIL.value
    unavailable = {
        check["name"] for check in result["checks"] if check["status"] == "NOT_EVALUABLE"
    }
    assert unavailable == {"identity", "jvp_median", "jvp_p95"}
    failed = {check["name"] for check in result["checks"] if check["status"] == "FAIL"}
    assert failed == {"jvp_completion"}


def test_mechanical_gate_contract_rejects_gate_observation_different_from_bound_upstream(
    tmp_path, monkeypatch
) -> None:
    config = _small_config()
    banks = config.experiment1.probe_banks.model_copy(
        update={"geometry_test_images": 2, "intervention_test_images": 1}
    )
    config = config.model_copy(
        update={"experiment1": config.experiment1.model_copy(update={"probe_banks": banks})}
    )
    config = with_tracking2_fixture(config, tmp_path)
    input_context = _context(tmp_path, config, "inputs.tracking2")
    input_ref = stage_handlers.handle_inputs_tracking2(input_context, {})
    store = input_context.store
    probe_context = _context(tmp_path, config, "exp1.probe_banks")
    probe_ref = stage_handlers.handle_exp1_probe_banks(
        probe_context, {"inputs.tracking2": input_ref}
    )
    epsilon = config.experiment1.mechanical_protocol.jvp_epsilon_float32
    zero = [0.0] * 10
    unit = [1.0, *zero[1:]]
    passing_resnet = {
        "actual_device": "cpu",
        "identity_exact": True,
        "identity_max_absolute_error": 0.0,
        "identity_logits_by_cut": {
            cut: {"full_logits": zero, "split_logits": zero} for cut in config.experiment1.cuts
        },
        "identity_per_cut": {cut: 0.0 for cut in config.experiment1.cuts},
        "jvp_by_cut": {
            cut: {
                "automatic_jvp": unit,
                "epsilon": epsilon,
                "finite_difference_jvp": unit,
                "relative_error": 0.0,
            }
            for cut in config.experiment1.sentinel_cuts
        },
        "jvp_cuts_completed": len(config.experiment1.sentinel_cuts),
        "jvp_failures": {},
        "jvp_median_relative_error": 0.0,
        "jvp_p95_relative_error": 0.0,
    }
    monkeypatch.setattr(
        stage_handlers,
        "_run_resnet_mechanics",
        lambda _context: (passing_resnet, []),
    )
    mechanical_context = _context(tmp_path, config, "exp1.mechanical")
    mechanical_ref = stage_handlers.handle_exp1_mechanical(
        mechanical_context,
        {
            "inputs.tracking2": input_ref,
            "exp1.probe_banks": probe_ref,
        },
    )
    gate_context = _context(tmp_path, config, "gate.mechanical")
    gate_ref = stage_handlers.handle_gate_mechanical(
        gate_context,
        {
            "exp1.mechanical": mechanical_ref,
            "exp1.probe_banks": probe_ref,
        },
    )
    assert (
        validate_stage_output(
            gate_ref,
            gate_context.stage_spec,
            store,
            config=config,
        ).status
        is GateStatus.PASS
    )

    forged = copy.deepcopy(store.read_json(gate_ref.artifact_id, "gate.json"))
    roundtrip = next(check for check in forged["checks"] if check["name"] == "roundtrip")
    candidate = config.gates.mechanical.roundtrip_relative_rms_max / 2
    if candidate == roundtrip["observed"]:
        candidate /= 2
    roundtrip["observed"] = candidate
    forged_ref = store.put_json(
        forged,
        filename="gate.json",
        kind="gate/mechanical",
        metadata=gate_ref.manifest.metadata,
    )
    with pytest.raises(PipelineError, match="does not match its bound upstream evidence"):
        validate_stage_output(
            forged_ref,
            gate_context.stage_spec,
            store,
            config=config,
        )


def test_configured_exact_handler_and_gate_pass_tiny_fixture(tmp_path) -> None:
    config = _small_config()
    exact_context = _context(tmp_path, config, "exp2.exact")
    _claim_parent(exact_context)
    exact_ref = stage_handlers.handle_exp2_exact(exact_context, {})
    exact = exact_context.store.read_json(exact_ref.artifact_id, "exact.json")
    assert exact["instances"] == 2
    assert exact["density"] == config.experiment2.generator_density
    assert exact["train_primitives"] == config.experiment2.train_primitives
    assert exact["heldout_primitives"] == config.experiment2.heldout_primitives
    assert exact["rho"] == config.experiment2.exact_protocol.rho
    assert exact["factor_sizes"] == [2, 2]
    assert exact["random_relabel"] is True
    assert exact["unary_weight"] == 1.7
    assert exact["generator_rate_shape"] == 3.0
    assert exact["generator_connectivity_policy"] == "mandatory_directed_cycle"
    assert exact["generator_normalization"] == "unit_mean_exit_rate"
    assert exact["exhaustive_tie_atol"] == 2e-10
    assert exact["exhaustive_tie_rtol"] == 3e-10
    assert len(exact["ordered_instance_artifact_ids"]) == 2
    reducer = exact_context.store.read_json(exact["reducer_artifact_id"], "shards.json")
    assert [row["artifact_id"] for row in reducer["ordered_shards"]] == exact[
        "ordered_instance_artifact_ids"
    ]
    assert [row["instance_index"] for row in exact["instance_results"]] == [0, 1]
    assert all(row["observed_generator_family"] for row in exact["instance_results"])
    first_shard = exact_context.store.read_json(
        exact["ordered_instance_artifact_ids"][0], "instance.json"
    )
    assert first_shard["protocol"]["train_primitives"] == 4
    assert first_shard["protocol"]["generator_rate_shape"] == 3.0
    assert first_shard["protocol"]["exhaustive_tie_atol"] == 2e-10
    assert first_shard["protocol"]["exhaustive_tie_rtol"] == 3e-10
    assert first_shard["protocol"]["random_relabel"] is True

    forged = dict(exact)
    forged["exact_instances"] = 0
    forged["exact_tuple_recovery_fraction"] = 0.0
    forged["aggregate"] = {
        **forged["aggregate"],
        "exact_instances": 0,
        "exact_tuple_recovery_fraction": 0.0,
    }
    forged_ref = exact_context.store.put_json(
        forged,
        filename="exact.json",
        kind="stage/exp2-exact",
        metadata=exact_ref.manifest.metadata,
    )
    with pytest.raises(PipelineError, match="aggregate does not match"):
        validate_stage_output(
            forged_ref,
            exact_context.stage_spec,
            exact_context.store,
            config=config,
        )

    forged_evidence = copy.deepcopy(exact)
    forged_row = forged_evidence["instance_results"][0]
    forged_row["selected_labeling"][0], forged_row["selected_labeling"][1] = (
        forged_row["selected_labeling"][1],
        forged_row["selected_labeling"][0],
    )
    forged_row["selected_labeling_hash"] = canonical_hash(forged_row["selected_labeling"])
    forged_row["exact"] = True
    forged_row["train_support_error"] = 0.0
    forged_row["heldout_support_error"] = 0.0
    forged_evidence_ref = exact_context.store.put_json(
        forged_evidence,
        filename="exact.json",
        kind="stage/exp2-exact",
        metadata=exact_ref.manifest.metadata,
    )
    with pytest.raises(PipelineError, match="deterministic replay"):
        validate_stage_output(
            forged_evidence_ref,
            exact_context.stage_spec,
            exact_context.store,
            config=config,
        )

    gate_context = _context(tmp_path, config, "gate.synthetic_exact")
    gate_ref = stage_handlers.handle_gate_synthetic_exact(gate_context, {"exp2.exact": exact_ref})
    gate = gate_context.store.read_json(gate_ref.artifact_id, "gate.json")
    assert gate["status"] == GateStatus.PASS.value
    assert (
        validate_stage_output(
            gate_ref,
            gate_context.stage_spec,
            gate_context.store,
            config=config,
        ).status
        is GateStatus.PASS
    )

    forged_gate = copy.deepcopy(gate)
    support = next(
        check for check in forged_gate["checks"] if check["name"] == "support_component_error"
    )
    candidate = config.gates.synthetic.relative_support_error_max / 2
    if candidate == support["observed"]:
        candidate /= 2
    support["observed"] = candidate
    forged_gate_ref = gate_context.store.put_json(
        forged_gate,
        filename="gate.json",
        kind="gate/synthetic-exact",
        metadata=gate_ref.manifest.metadata,
    )
    with pytest.raises(PipelineError, match="does not match its bound upstream evidence"):
        validate_stage_output(
            forged_gate_ref,
            gate_context.stage_spec,
            gate_context.store,
            config=config,
        )


def test_exact_handler_reuses_completed_instance_shards_on_parent_retry(
    tmp_path, monkeypatch
) -> None:
    config = _small_config()
    context = _context(tmp_path, config, "exp2.exact")
    _claim_parent(context)
    calls: list[dict[str, object]] = []
    original = stage_handlers.run_oracle_exhaustive_instance

    def counted(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(stage_handlers, "run_oracle_exhaustive_instance", counted)
    first = stage_handlers.handle_exp2_exact(context, {})
    assert sorted(call["instance_index"] for call in calls) == [0, 1]
    assert all(call["generator_rate_shape"] == 3.0 for call in calls)
    assert all(call["exhaustive_tie_atol"] == 2e-10 for call in calls)
    assert all(call["exhaustive_tie_rtol"] == 3e-10 for call in calls)
    calls.clear()
    second = stage_handlers.handle_exp2_exact(context, {})
    assert calls == []
    assert second.artifact_id == first.artifact_id
    shards = [
        record
        for record in context.index.list_stages(context.run_id)
        if record.metadata.get("parent_stage") == "exp2.exact"
    ]
    assert len(shards) == 2
    assert all(record.attempts == 1 for record in shards)


def test_exact_handler_workers_zero_is_strictly_sequential(tmp_path, monkeypatch) -> None:
    config = _small_config()
    experiment2 = config.experiment2.model_copy(update={"exact_instances": 1})
    runtime = config.runtime.model_copy(update={"workers": 0})
    config = config.model_copy(update={"experiment2": experiment2, "runtime": runtime})
    context = _context(tmp_path, config, "exp2.exact")
    _claim_parent(context)

    class ForbiddenPool:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("workers=0 must not construct a thread pool")

    monkeypatch.setattr(stage_handlers, "ThreadPoolExecutor", ForbiddenPool)
    reference = stage_handlers.handle_exp2_exact(context, {})
    exact = context.store.read_json(reference.artifact_id, "exact.json")
    assert exact["instances"] == 1
    assert len(exact["ordered_instance_artifact_ids"]) == 1


def test_default_handlers_exactly_cover_runnable_default_stages() -> None:
    runnable = {
        stage.name
        for stage in DEFAULT_STAGES.topological_order()
        if stage.implementation is ImplementationStatus.RUNNABLE
    }
    assert set(stage_handlers.default_handlers()) == runnable


def test_report_stage_rejects_non_embedded_spec_even_if_validation_is_bypassed(
    tmp_path,
) -> None:
    config = LabConfig()
    report = config.report.model_copy(update={"embed_spec": False})
    config = config.model_copy(update={"report": report})
    context = _context(tmp_path, config, "report.build")
    with pytest.raises(stage_handlers.StageHandlerError, match="specification"):
        stage_handlers.handle_report_build(context, {})
