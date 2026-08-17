from __future__ import annotations

import copy
from dataclasses import replace

import pytest

import voronoi_lab.pipeline as pipeline_module
from tests.tracking2_fixture import with_tracking2_fixture
from voronoi_lab import stage_handlers
from voronoi_lab.config import GateOverrideAuthorization, LabConfig, load_config
from voronoi_lab.core import (
    ArtifactStore,
    GateCheck,
    GateEvaluator,
    GateRule,
    canonical_hash,
    canonical_json_bytes,
)
from voronoi_lab.execution import ExperimentRunner
from voronoi_lab.exp1.probe_artifact import build_probe_bank_files
from voronoi_lab.mechanical import replay_synthetic_invariants, replay_toy_geometry
from voronoi_lab.pipeline import (
    DEFAULT_STAGES,
    ImplementationStatus,
    PipelineError,
    StageRegistry,
    StageSpec,
    StageValidationContext,
    expected_gate_override_authorization,
    expected_gate_rule,
    stage_signature,
    validate_stage_output,
)


def test_default_plan_exposes_real_compute_scale_and_implementation_boundary() -> None:
    config = load_config("configs/pilot.yaml")
    plans = DEFAULT_STAGES.plan(config)
    by_name = {plan.name: plan for plan in plans}
    # Per checkpoint/cut: ceil(2000/64) for each of three large banks plus
    # ceil(256/64) for interventions, rather than one crash-prone shard per bank.
    assert by_name["exp1.activations"].estimated_shards == 5 * 8 * 3 * (32 + 32 + 32 + 4)
    assert by_name["exp1.codebooks"].estimated_shards == 5 * 8 * 2 * 3
    assert by_name["exp1.boundary_paths"].estimated_shards == 5 * 4 * 2 * 3 * 4
    assert by_name["exp1.activations"].implementation is ImplementationStatus.PLANNED
    assert by_name["exp2.exact"].implementation is ImplementationStatus.RUNNABLE


def test_every_runnable_default_stage_declares_a_strict_output_contract() -> None:
    runnable = [
        stage
        for stage in DEFAULT_STAGES.topological_order()
        if stage.implementation is ImplementationStatus.RUNNABLE
    ]

    assert runnable
    assert all(stage.expected_artifact_kind for stage in runnable)
    assert all(stage.required_payload_paths for stage in runnable)
    assert all(stage.result_schema_version == 1 for stage in runnable)
    assert all(stage.payload_schema_id for stage in runnable)
    assert {stage.name for stage in runnable if stage.gate_payload_path is not None} == {
        "gate.mechanical",
        "gate.synthetic_exact",
    }
    assert {
        stage.name: stage.expected_gate_id
        for stage in runnable
        if stage.expected_gate_id is not None
    } == {
        "gate.mechanical": "mechanical",
        "gate.synthetic_exact": "synthetic_exact",
    }
    assert {
        stage.name: (stage.gate_evidence_dependency, stage.gate_evidence_payload_path)
        for stage in runnable
        if stage.gate_payload_path is not None
    } == {
        "gate.mechanical": ("exp1.mechanical", "mechanical.json"),
        "gate.synthetic_exact": ("exp2.exact", "exact.json"),
    }


def test_validation_context_binds_every_semantic_input_and_never_caches_failures(
    tmp_path,
    monkeypatch,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    stage = StageSpec(
        "fixture",
        "validation memo fixture",
        implementation=ImplementationStatus.RUNNABLE,
        expected_artifact_kind="stage/fixture",
        required_payload_paths=("data.json",),
        result_schema_version=1,
    )
    registry = StageRegistry((stage,))
    source = {"revision": "source-a"}
    artifact = store.put_json(
        {"schema_version": 1, "value": 1},
        kind="stage/fixture",
        metadata={"result_schema_version": 1, "source_identity": source},
    )
    context = StageValidationContext()
    calls = 0
    forced_failure = True
    original = pipeline_module._validate_named_payload_schema

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if forced_failure:
            raise PipelineError("forced semantic failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "_validate_named_payload_schema", counted)
    config = LabConfig()
    for _ in range(2):
        with pytest.raises(PipelineError, match="forced semantic failure"):
            validate_stage_output(
                artifact,
                stage,
                store,
                config=config,
                registry=registry,
                source_identity=source,
                validation_context=context,
            )
    assert calls == 2

    forced_failure = False
    for _ in range(2):
        assert (
            validate_stage_output(
                artifact,
                stage,
                store,
                config=config,
                registry=registry,
                source_identity=source,
                validation_context=context,
            )
            is None
        )
    assert calls == 3

    changed_config = config.model_copy(
        update={"protocol": config.protocol.model_copy(update={"root_seed": 91})}
    )
    validate_stage_output(
        artifact,
        stage,
        store,
        config=changed_config,
        registry=registry,
        source_identity=source,
        validation_context=context,
    )
    assert calls == 4
    validate_stage_output(
        artifact,
        stage,
        store,
        config=config,
        registry=registry,
        source_identity={"revision": "source-b"},
        validation_context=context,
    )
    assert calls == 5

    evolved_stage = StageSpec(
        "fixture",
        "evolved validation memo fixture",
        implementation=ImplementationStatus.RUNNABLE,
        stage_version=2,
        expected_artifact_kind="stage/fixture",
        required_payload_paths=("data.json",),
        result_schema_version=1,
    )
    validate_stage_output(
        artifact,
        evolved_stage,
        store,
        config=config,
        registry=StageRegistry((evolved_stage,)),
        source_identity=source,
        validation_context=context,
    )
    assert calls == 6
    expanded_registry = StageRegistry(
        (
            stage,
            StageSpec("extra", "registry identity fixture"),
        )
    )
    validate_stage_output(
        artifact,
        stage,
        store,
        config=config,
        registry=expanded_registry,
        source_identity=source,
        validation_context=context,
    )
    assert calls == 7

    payload_path = artifact.payload_path("data.json")
    payload_path.chmod(0o600)
    payload_path.write_text('{"schema_version":1,"value":2}', encoding="utf-8")
    with pytest.raises(PipelineError, match="unavailable or corrupt"):
        validate_stage_output(
            artifact,
            stage,
            store,
            config=config,
            registry=registry,
            source_identity=source,
            validation_context=context,
        )
    assert calls == 7


def test_target_closure_is_topologically_ordered() -> None:
    names = [stage.name for stage in DEFAULT_STAGES.topological_order(["gate.mechanical"])]
    assert names == [
        "inputs.tracking2",
        "exp1.probe_banks",
        "exp1.mechanical",
        "gate.mechanical",
    ]


def test_signature_changes_only_for_declared_inputs() -> None:
    config = load_config("configs/pilot.yaml")
    stage = DEFAULT_STAGES.get("exp2.exact")
    base = stage_signature(stage, config, upstream_artifact_ids={}, source_identity={"git": "x"})
    changed_report = config.model_copy(
        update={"report": config.report.model_copy(update={"self_contained": False})}
    )
    same = stage_signature(
        stage, changed_report, upstream_artifact_ids={}, source_identity={"git": "x"}
    )
    assert same == base
    changed_seed = config.model_copy(
        update={"protocol": config.protocol.model_copy(update={"root_seed": 9})}
    )
    different = stage_signature(
        stage, changed_seed, upstream_artifact_ids={}, source_identity={"git": "x"}
    )
    assert different != base


def test_producer_signatures_exclude_downstream_gates_and_runtime_policy() -> None:
    config = load_config("configs/pilot.yaml")
    source = {"git": "x"}
    input_id = "a" * 64
    probe_id = "b" * 64
    mechanical_upstream = {
        "inputs.tracking2": input_id,
        "exp1.probe_banks": probe_id,
    }

    mechanical = DEFAULT_STAGES.get("exp1.mechanical")
    mechanical_base = stage_signature(
        mechanical,
        config,
        upstream_artifact_ids=mechanical_upstream,
        source_identity=source,
    )
    changed_dtype = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"dtype": "float64"})}
    )
    assert (
        stage_signature(
            mechanical,
            changed_dtype,
            upstream_artifact_ids=mechanical_upstream,
            source_identity=source,
        )
        != mechanical_base
    )
    changed_downstream_codebook = config.model_copy(
        update={
            "experiment1": config.experiment1.model_copy(
                update={"codebooks": config.experiment1.codebooks.model_copy(update={"n_init": 11})}
            )
        }
    )
    assert (
        stage_signature(
            mechanical,
            changed_downstream_codebook,
            upstream_artifact_ids=mechanical_upstream,
            source_identity=source,
        )
        == mechanical_base
    )

    changed_mechanical_threshold = config.model_copy(
        update={
            "gates": config.gates.model_copy(
                update={
                    "mechanical": config.gates.mechanical.model_copy(
                        update={"roundtrip_relative_rms_max": 2e-6}
                    )
                }
            )
        }
    )
    assert (
        stage_signature(
            mechanical,
            changed_mechanical_threshold,
            upstream_artifact_ids=mechanical_upstream,
            source_identity=source,
        )
        == mechanical_base
    )
    mechanical_gate = DEFAULT_STAGES.get("gate.mechanical")
    mechanical_gate_base = stage_signature(
        mechanical_gate,
        config,
        upstream_artifact_ids={
            "exp1.mechanical": "c" * 64,
            "exp1.probe_banks": probe_id,
        },
        source_identity=source,
    )
    assert (
        stage_signature(
            mechanical_gate,
            changed_mechanical_threshold,
            upstream_artifact_ids={
                "exp1.mechanical": "c" * 64,
                "exp1.probe_banks": probe_id,
            },
            source_identity=source,
        )
        != mechanical_gate_base
    )
    authorization = GateOverrideAuthorization(
        target_gate="mechanical",
        reason="diagnostic contract test",
        authorized_by="test-suite",
        recorded_at="2026-08-16T12:00:00-07:00",
    )
    changed_override = config.model_copy(
        update={
            "gates": config.gates.model_copy(
                update={
                    "overrides": config.gates.overrides.model_copy(
                        update={"mechanical": authorization}
                    )
                }
            )
        }
    )
    assert (
        stage_signature(
            mechanical_gate,
            changed_override,
            upstream_artifact_ids={
                "exp1.mechanical": "c" * 64,
                "exp1.probe_banks": probe_id,
            },
            source_identity=source,
        )
        != mechanical_gate_base
    )

    exact = DEFAULT_STAGES.get("exp2.exact")
    exact_base = stage_signature(exact, config, upstream_artifact_ids={}, source_identity=source)
    changed_workers = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"workers": 2})}
    )
    assert (
        stage_signature(exact, changed_workers, upstream_artifact_ids={}, source_identity=source)
        == exact_base
    )
    changed_sampled_only = config.model_copy(
        update={
            "experiment2": config.experiment2.model_copy(
                update={"sampling": config.experiment2.sampling.model_copy(update={"tau": 0.2})}
            )
        }
    )
    assert (
        stage_signature(
            exact, changed_sampled_only, upstream_artifact_ids={}, source_identity=source
        )
        == exact_base
    )
    changed_synthetic_threshold = config.model_copy(
        update={
            "gates": config.gates.model_copy(
                update={
                    "synthetic": config.gates.synthetic.model_copy(
                        update={"relative_support_error_max": 2e-8}
                    )
                }
            )
        }
    )
    assert (
        stage_signature(
            exact,
            changed_synthetic_threshold,
            upstream_artifact_ids={},
            source_identity=source,
        )
        == exact_base
    )
    synthetic_gate = DEFAULT_STAGES.get("gate.synthetic_exact")
    assert stage_signature(
        synthetic_gate,
        changed_synthetic_threshold,
        upstream_artifact_ids={"exp2.exact": "c" * 64},
        source_identity=source,
    ) != stage_signature(
        synthetic_gate,
        config,
        upstream_artifact_ids={"exp2.exact": "c" * 64},
        source_identity=source,
    )


def test_output_contract_is_part_of_the_stage_signature() -> None:
    config = load_config("configs/pilot.yaml")
    stage = DEFAULT_STAGES.get("exp2.exact")
    base = stage_signature(stage, config, upstream_artifact_ids={}, source_identity={})
    revised = replace(stage, expected_artifact_kind="stage/exp2-exact-v2")

    assert stage_signature(revised, config, upstream_artifact_ids={}, source_identity={}) != base

    gate_stage = DEFAULT_STAGES.get("gate.synthetic_exact")
    gate_base = stage_signature(
        gate_stage,
        config,
        upstream_artifact_ids={"exp2.exact": "a" * 64},
        source_identity={},
    )
    swapped_gate = replace(gate_stage, expected_gate_id="different")
    assert (
        stage_signature(
            swapped_gate,
            config,
            upstream_artifact_ids={"exp2.exact": "a" * 64},
            source_identity={},
        )
        != gate_base
    )


def test_stage_output_contract_rejects_wrong_kind_missing_payload_and_schema(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    stage = StageSpec(
        "fixture",
        "contract fixture",
        implementation=ImplementationStatus.RUNNABLE,
        expected_artifact_kind="stage/fixture",
        required_payload_paths=("result.json",),
        result_schema_version=1,
    )
    valid = store.put_json(
        {"schema_version": 1},
        filename="result.json",
        kind="stage/fixture",
        metadata={"result_schema_version": 1},
    )
    assert validate_stage_output(valid, stage, store) is None

    wrong_kind = store.put_json(
        {"schema_version": 1},
        filename="result.json",
        kind="stage/not-fixture",
        metadata={"result_schema_version": 1},
    )
    with pytest.raises(PipelineError, match="has kind"):
        validate_stage_output(wrong_kind, stage, store)

    missing_payload = store.put_json(
        {"schema_version": 1},
        filename="other.json",
        kind="stage/fixture",
        metadata={"result_schema_version": 1},
    )
    with pytest.raises(PipelineError, match="missing required payloads"):
        validate_stage_output(missing_payload, stage, store)

    wrong_schema = store.put_json(
        {"schema_version": 1},
        filename="result.json",
        kind="stage/fixture",
        metadata={"result_schema_version": 2},
    )
    with pytest.raises(PipelineError, match="result_schema_version"):
        validate_stage_output(wrong_schema, stage, store)

    empty_payload = store.put_json(
        {},
        filename="result.json",
        kind="stage/fixture",
        metadata={"result_schema_version": 1},
    )
    with pytest.raises(PipelineError, match=r"schema_version None; expected 1"):
        validate_stage_output(empty_payload, stage, store)

    wrong_payload_schema = store.put_json(
        {"schema_version": 2},
        filename="result.json",
        kind="stage/fixture",
        metadata={"result_schema_version": 1},
    )
    with pytest.raises(PipelineError, match=r"schema_version 2; expected 1"):
        validate_stage_output(wrong_payload_schema, stage, store)

    array_payload = store.put_json(
        [],
        filename="result.json",
        kind="stage/fixture",
        metadata={"result_schema_version": 1},
    )
    with pytest.raises(PipelineError, match="must be an object"):
        validate_stage_output(array_payload, stage, store)

    wrong_media_type = store.put_bytes(
        b'{"schema_version":1}',
        filename="result.json",
        kind="stage/fixture",
        metadata={"result_schema_version": 1},
        media_type="text/plain",
    )
    with pytest.raises(PipelineError, match="media type 'text/plain'"):
        validate_stage_output(wrong_media_type, stage, store)


@pytest.mark.parametrize(
    ("stage_name", "json_path", "extra_files"),
    [
        ("inputs.tracking2", "inputs.json", {}),
        ("exp1.probe_banks", "plan.json", {}),
        ("exp1.mechanical", "mechanical.json", {}),
        ("exp2.exact", "exact.json", {}),
        ("report.build", "report_payload.json", {"report.html": b"<!doctype html>"}),
    ],
)
def test_default_non_gate_contracts_reject_semantic_json_shells(
    tmp_path, stage_name, json_path, extra_files
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    stage = DEFAULT_STAGES.get(stage_name)
    artifact = store.put_files(
        {
            json_path: canonical_json_bytes({"schema_version": 1}),
            **extra_files,
        },
        kind=stage.expected_artifact_kind,
        metadata={"result_schema_version": 1},
        media_types={json_path: "application/json"},
    )

    with pytest.raises(PipelineError):
        validate_stage_output(artifact, stage, store)


def test_inputs_contract_binds_manifest_digest_to_signed_config(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    config = load_config("configs/pilot.yaml")
    payload = {
        "schema_version": 1,
        "architecture": {"model": "fixture"},
        "dataset_isolation": {
            "passed": True,
            "train_test_distinct_hashes": True,
            "train_test_distinct_paths": True,
        },
        "external_root": config.inputs.tracking2.root.as_posix(),
        "lineage_note": "fixture",
        "lineage_quality": "exploratory_legacy",
        "manifest": {
            "path": config.inputs.tracking2.manifest.as_posix(),
            "sha256": "f" * 64,
        },
        "observed_repository_revision": "a" * 40,
        "read_only": True,
        "training": {"train_size": 8, "test_size": 4},
        "transplant_rows": [{"fixture": True}],
        "validated_files": {
            "fixture": {
                "declared_path": "fixture.bin",
                "sha256": "b" * 64,
                "size_bytes": 1,
            }
        },
    }
    artifact = store.put_files(
        {
            "inputs.json": canonical_json_bytes(payload),
            "manifest.yaml": b"schema_version: 1\n",
            "transplant.json": b'{"schema_version":1}',
        },
        kind="stage/inputs-tracking2",
        metadata={"result_schema_version": 1},
        media_types={
            "inputs.json": "application/json",
            "manifest.yaml": "application/yaml",
            "transplant.json": "application/json",
        },
    )

    with pytest.raises(PipelineError, match=r"signed declaration|signed digest"):
        validate_stage_output(
            artifact,
            DEFAULT_STAGES.get("inputs.tracking2"),
            store,
            config=config,
        )


def test_mechanical_contract_rejects_jvp_aggregates_without_completed_cuts(tmp_path) -> None:
    config = load_config("configs/pilot.yaml")
    banks = config.experiment1.probe_banks.model_copy(
        update={
            "fit_train_images": 2,
            "independent_fit_train_images": 2,
            "geometry_test_images": 2,
            "intervention_test_images": 1,
            "max_sites_per_image": 1,
        }
    )
    bootstrap = config.experiment1.bootstrap.model_copy(update={"resamples": 2})
    experiment1 = config.experiment1.model_copy(
        update={"probe_banks": banks, "bootstrap": bootstrap}
    )
    config = config.model_copy(update={"experiment1": experiment1})
    config = with_tracking2_fixture(config, tmp_path)
    runner = ExperimentRunner(
        config,
        project_root=tmp_path,
        handlers={"inputs.tracking2": stage_handlers.handle_inputs_tracking2},
        run_id="pipeline-input-fixture",
    )
    input_reference = runner.run(["inputs.tracking2"])["inputs.tracking2"]
    store = runner.store
    stage = DEFAULT_STAGES.get("exp1.mechanical")
    input_payload = store.read_json(
        input_reference.artifact_id,
        "inputs.json",
    )
    forged_input_payload = copy.deepcopy(input_payload)
    forged_input_payload["architecture"] = {}
    forged_input = store.put_files(
        {
            "inputs.json": canonical_json_bytes(forged_input_payload),
            "manifest.yaml": store.read_bytes(input_reference.artifact_id, "manifest.yaml"),
            "transplant.json": store.read_bytes(input_reference.artifact_id, "transplant.json"),
        },
        kind="stage/inputs-tracking2",
        metadata={"result_schema_version": 1},
        media_types={
            "inputs.json": "application/json",
            "manifest.yaml": "application/yaml",
            "transplant.json": "application/json",
        },
    )
    with pytest.raises(PipelineError, match="preserved raw evidence"):
        validate_stage_output(
            forged_input,
            DEFAULT_STAGES.get("inputs.tracking2"),
            store,
            config=config,
        )
    probe_files, _ = build_probe_bank_files(config, input_payload)
    probe_reference = store.put_files(
        probe_files,
        kind="stage/exp1-probe-banks",
        metadata={
            "result_schema_version": 1,
            "upstream_artifacts": {"inputs.tracking2": input_reference.artifact_id},
        },
        media_types={"plan.json": "application/json"},
    )
    assert (
        validate_stage_output(
            probe_reference,
            DEFAULT_STAGES.get("exp1.probe_banks"),
            store,
            config=config,
        )
        is None
    )
    corrupted_probe_files = dict(probe_files)
    corrupted_probe_files["indices/geometry.npy"] = b"not-an-npy-array"
    corrupted_probe = store.put_files(
        corrupted_probe_files,
        kind="stage/exp1-probe-banks",
        metadata={
            "result_schema_version": 1,
            "upstream_artifacts": {"inputs.tracking2": input_reference.artifact_id},
        },
        media_types={"plan.json": "application/json"},
    )
    with pytest.raises(PipelineError, match="deterministic replay"):
        validate_stage_output(
            corrupted_probe,
            DEFAULT_STAGES.get("exp1.probe_banks"),
            store,
            config=config,
        )
    mechanical_metadata = {
        "result_schema_version": 1,
        "upstream_artifacts": {
            "inputs.tracking2": input_reference.artifact_id,
            "exp1.probe_banks": probe_reference.artifact_id,
        },
    }
    artifact = store.put_json(
        {
            "schema_version": 1,
            "protocol": config.experiment1.mechanical_protocol.model_dump(mode="json"),
            "geometry": replay_toy_geometry(
                config.protocol.root_seed,
                rms_epsilon=config.experiment1.state_metric.rms_epsilon,
            ),
            "probe_banks": {
                "artifact_valid": True,
                "deterministic": True,
                "distinct_train_test_sources": True,
            },
            "resnet": {
                "actual_device": "cpu",
                "identity_exact": None,
                "identity_logits_by_cut": {},
                "identity_max_absolute_error": None,
                "identity_per_cut": {},
                "jvp_by_cut": {},
                "jvp_cuts_completed": 0,
                "jvp_failures": {},
                "jvp_median_relative_error": 0.0,
                "jvp_p95_relative_error": 0.0,
            },
            "synthetic_invariants": replay_synthetic_invariants(config.protocol.root_seed),
            "warnings": [],
        },
        filename="mechanical.json",
        kind="stage/exp1-mechanical",
        metadata=mechanical_metadata,
    )

    with pytest.raises(PipelineError, match="require every sentinel cut"):
        validate_stage_output(artifact, stage, store, config=config)

    payload = store.read_json(artifact.artifact_id, "mechanical.json")
    inconsistent_geometry = copy.deepcopy(payload)
    inconsistent_geometry["geometry"]["boundary"]["passed"] = False
    inconsistent_geometry_ref = store.put_json(
        inconsistent_geometry,
        filename="mechanical.json",
        kind="stage/exp1-mechanical",
        metadata=mechanical_metadata,
    )
    with pytest.raises(PipelineError, match="deterministic replay"):
        validate_stage_output(inconsistent_geometry_ref, stage, store, config=config)

    output_values = (
        config.experiment1.mechanical_protocol.input_batch_size
        * input_payload["architecture"]["num_classes"]
    )
    zero = [0.0] * output_values
    unit = [1.0, *zero[1:]]
    epsilon = config.experiment1.mechanical_protocol.jvp_epsilon_float32
    payload["resnet"] = {
        "actual_device": "cpu",
        "identity_exact": True,
        "identity_logits_by_cut": {
            cut: {"full_logits": zero, "split_logits": unit} for cut in config.experiment1.cuts
        },
        "identity_max_absolute_error": 0.0,
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
    forged_identity = store.put_json(
        payload,
        filename="mechanical.json",
        kind="stage/exp1-mechanical",
        metadata=mechanical_metadata,
    )
    with pytest.raises(PipelineError, match="identity errors disagree with saved raw logits"):
        validate_stage_output(forged_identity, stage, store, config=config)

    payload["resnet"]["identity_logits_by_cut"] = {
        cut: {"full_logits": zero, "split_logits": zero} for cut in config.experiment1.cuts
    }
    first_sentinel = config.experiment1.sentinel_cuts[0]
    payload["resnet"]["jvp_by_cut"][first_sentinel]["finite_difference_jvp"] = zero
    forged_jvp = store.put_json(
        payload,
        filename="mechanical.json",
        kind="stage/exp1-mechanical",
        metadata=mechanical_metadata,
    )
    with pytest.raises(PipelineError, match="JVP errors disagree with saved raw outputs"):
        validate_stage_output(forged_jvp, stage, store, config=config)


def test_gate_output_contract_reconstructs_and_cross_checks_gate_result(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    stage = StageSpec(
        "gate.fixture",
        "gate contract fixture",
        implementation=ImplementationStatus.RUNNABLE,
        expected_artifact_kind="gate/fixture",
        required_payload_paths=("gate.json",),
        result_schema_version=1,
        gate_payload_path="gate.json",
        expected_gate_id="fixture",
    )
    result = GateEvaluator().evaluate(
        GateRule(
            gate_id="fixture",
            checks=(GateCheck("required", "required", "is_true"),),
        ),
        {"required": True},
    )
    valid = store.put_json(
        result.to_dict(),
        filename="gate.json",
        kind="gate/fixture",
        metadata={
            "gate_id": result.gate_id,
            "gate_status": result.status.value,
            "natural_status": result.natural_status.value,
            "result_schema_version": 1,
        },
    )
    assert validate_stage_output(valid, stage, store) == result

    overridden_upstream = GateEvaluator().evaluate(
        GateRule(
            gate_id="upstream",
            checks=(GateCheck("required", "required", "is_true"),),
        ),
        {"required": False},
        override_reason="diagnostic continuation",
    )
    inherited_result = GateEvaluator().evaluate(
        GateRule(
            gate_id="fixture",
            dependencies=("upstream",),
            checks=(GateCheck("required", "required", "is_true"),),
        ),
        {"required": True},
        dependencies={"upstream": overridden_upstream},
    )
    inherited = store.put_json(
        inherited_result.to_dict(),
        filename="gate.json",
        kind="gate/fixture",
        metadata={
            "gate_id": inherited_result.gate_id,
            "gate_status": inherited_result.status.value,
            "inherited_gate_overrides": [
                item.to_dict() for item in inherited_result.override_lineage
            ],
            "natural_status": inherited_result.natural_status.value,
            "result_schema_version": 1,
        },
    )
    assert validate_stage_output(inherited, stage, store) == inherited_result

    dropped_lineage = store.put_json(
        result.to_dict(),
        filename="gate.json",
        kind="gate/fixture",
        metadata={
            "gate_id": result.gate_id,
            "gate_status": result.status.value,
            "inherited_gate_overrides": [
                item.to_dict() for item in overridden_upstream.override_lineage
            ],
            "natural_status": result.natural_status.value,
            "result_schema_version": 1,
        },
    )
    with pytest.raises(PipelineError, match="does not preserve inherited override lineage"):
        validate_stage_output(dropped_lineage, stage, store)
    inconsistent = result.to_dict()
    inconsistent["override_reason"] = "undeclared override"
    invalid = store.put_json(
        inconsistent,
        filename="gate.json",
        kind="gate/fixture",
        metadata={
            "gate_id": result.gate_id,
            "gate_status": result.status.value,
            "natural_status": result.natural_status.value,
            "result_schema_version": 1,
        },
    )
    with pytest.raises(PipelineError, match="invalid GateResult"):
        validate_stage_output(invalid, stage, store)

    swapped_result = GateEvaluator().evaluate(
        GateRule(
            gate_id="different",
            checks=(GateCheck("required", "required", "is_true"),),
        ),
        {"required": True},
    )
    swapped = store.put_json(
        swapped_result.to_dict(),
        filename="gate.json",
        kind="gate/fixture",
        metadata={
            "gate_id": swapped_result.gate_id,
            "gate_status": swapped_result.status.value,
            "natural_status": swapped_result.natural_status.value,
            "result_schema_version": 1,
        },
    )
    with pytest.raises(PipelineError, match=r"gate_id 'different'; expected 'fixture'"):
        validate_stage_output(swapped, stage, store)


def test_default_gate_contract_rejects_a_well_formed_but_fabricated_pass(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    config = load_config("configs/pilot.yaml")
    stage = DEFAULT_STAGES.get("gate.mechanical")
    expected = expected_gate_rule(stage, config)
    fabricated = GateEvaluator().evaluate(
        GateRule(
            gate_id="mechanical",
            checks=(GateCheck("fabricated", "fabricated", "is_true"),),
        ),
        {"fabricated": True},
    )
    artifact = store.put_json(
        fabricated.to_dict(),
        filename="gate.json",
        kind="gate/mechanical",
        metadata={
            "gate_id": fabricated.gate_id,
            "gate_status": fabricated.status.value,
            # Even a forged metadata signature cannot substitute a different
            # serialized check/rule contract.
            "gate_rule_signature": canonical_hash(expected.to_dict()),
            "inherited_gate_overrides": [],
            "natural_status": fabricated.natural_status.value,
            "result_schema_version": 1,
        },
    )

    with pytest.raises(PipelineError, match="check count"):
        validate_stage_output(artifact, stage, store, gate_rule=expected)


def test_gate_contract_binds_override_to_signed_authorization(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    config = load_config("configs/pilot.yaml")
    stage = DEFAULT_STAGES.get("gate.mechanical")
    rule = expected_gate_rule(stage, config)
    rogue = GateEvaluator().evaluate(rule, {}, override_reason="rogue continuation")
    base_metadata = {
        "gate_id": rogue.gate_id,
        "gate_status": rogue.status.value,
        "gate_rule_signature": canonical_hash(rule.to_dict()),
        "inherited_gate_overrides": [],
        "natural_status": rogue.natural_status.value,
        "result_schema_version": 1,
    }
    unauthorized = store.put_json(
        rogue.to_dict(),
        filename="gate.json",
        kind="gate/mechanical",
        metadata={**base_metadata, "override_authorization": None},
    )
    with pytest.raises(PipelineError, match="without configured authorization"):
        validate_stage_output(
            unauthorized,
            stage,
            store,
            gate_rule=rule,
            gate_override_authorization=None,
        )

    authorization = GateOverrideAuthorization(
        target_gate="mechanical",
        reason="authorized diagnostic continuation",
        authorized_by="test-suite",
        recorded_at="2026-08-17T12:00:00-07:00",
    )
    authorized_config = config.model_copy(
        update={
            "gates": config.gates.model_copy(
                update={
                    "overrides": config.gates.overrides.model_copy(
                        update={"mechanical": authorization}
                    )
                }
            )
        }
    )
    expected_authorization = expected_gate_override_authorization(stage, authorized_config)
    wrong_reason = store.put_json(
        rogue.to_dict(),
        filename="gate.json",
        kind="gate/mechanical",
        metadata={**base_metadata, "override_authorization": expected_authorization},
    )
    with pytest.raises(PipelineError, match="reason differs"):
        validate_stage_output(
            wrong_reason,
            stage,
            store,
            gate_rule=rule,
            gate_override_authorization=expected_authorization,
        )


def test_gate_contract_requires_an_expected_gate_identity() -> None:
    with pytest.raises(PipelineError, match="must be declared together"):
        StageSpec(
            "gate.fixture",
            "incomplete gate contract",
            required_payload_paths=("gate.json",),
            result_schema_version=1,
            gate_payload_path="gate.json",
        )
    with pytest.raises(PipelineError, match="must be declared together"):
        StageSpec(
            "gate.fixture",
            "incomplete evidence contract",
            dependencies=("evidence",),
            required_payload_paths=("gate.json",),
            result_schema_version=1,
            gate_payload_path="gate.json",
            expected_gate_id="fixture",
            gate_evidence_dependency="evidence",
        )
    with pytest.raises(PipelineError, match="direct stage dependency"):
        StageSpec(
            "gate.fixture",
            "indirect evidence contract",
            required_payload_paths=("gate.json",),
            result_schema_version=1,
            gate_payload_path="gate.json",
            expected_gate_id="fixture",
            gate_evidence_dependency="evidence",
            gate_evidence_payload_path="evidence.json",
        )


def test_registry_rejects_unknown_dependencies_and_cycles() -> None:
    with pytest.raises(PipelineError, match="unknown dependencies"):
        StageRegistry((StageSpec("a", "a", dependencies=("missing",)),))
    with pytest.raises(PipelineError, match="not declared by producer"):
        StageRegistry(
            (
                StageSpec(
                    "evidence",
                    "evidence",
                    required_payload_paths=("actual.json",),
                    result_schema_version=1,
                ),
                StageSpec(
                    "gate.fixture",
                    "gate",
                    dependencies=("evidence",),
                    required_payload_paths=("gate.json",),
                    result_schema_version=1,
                    gate_payload_path="gate.json",
                    expected_gate_id="fixture",
                    gate_evidence_dependency="evidence",
                    gate_evidence_payload_path="missing.json",
                ),
            )
        )
    with pytest.raises(PipelineError, match="cycle"):
        StageRegistry(
            (
                StageSpec("a", "a", dependencies=("b",)),
                StageSpec("b", "b", dependencies=("a",)),
            )
        )


def test_stage_specs_align_names_and_versions_with_run_index_identifiers() -> None:
    with pytest.raises(PipelineError, match="stage names must match"):
        StageSpec("bad stage", "invalid name")
    with pytest.raises(PipelineError, match="stage names must match"):
        StageSpec("bad/stage", "invalid name")
    with pytest.raises(PipelineError, match="positive integer"):
        StageSpec("valid", "invalid version", stage_version=True)
    with pytest.raises(PipelineError, match="config_paths must be a sequence"):
        StageSpec("valid", "invalid config paths", config_paths="runtime.dtype")  # type: ignore[arg-type]
    with pytest.raises(PipelineError, match="must be unique"):
        StageSpec(
            "valid",
            "duplicate config paths",
            config_paths=("runtime.dtype", "runtime.dtype"),
        )
    with pytest.raises(PipelineError, match="dotted configuration paths"):
        StageSpec("valid", "invalid config path", config_paths=("runtime..dtype",))


def test_planned_join_and_functional_gate_retain_direct_evidence_dependencies() -> None:
    assert set(DEFAULT_STAGES.get("exp1.transplant_join").dependencies) == {
        "gate.functional",
        "inputs.tracking2",
        "exp1.static_geometry",
        "exp1.snapping_recovery",
    }
    assert set(DEFAULT_STAGES.get("gate.functional").dependencies) == {
        "exp1.snapping_recovery",
        "gate.coarse",
    }
    assert set(DEFAULT_STAGES.get("exp1.confirmation").dependencies) == {
        "gate.functional",
        "inputs.tracking2",
    }
    assert DEFAULT_STAGES.get("exp1.mechanical").config_paths == (
        "protocol.root_seed",
        "runtime.dtype",
        "experiment1.cuts",
        "experiment1.sentinel_cuts",
        "experiment1.mechanical_protocol",
        "experiment1.state_metric.rms_epsilon",
    )
    assert DEFAULT_STAGES.get("gate.mechanical").config_paths == (
        "gates.mechanical",
        "gates.overrides.mechanical",
    )
    assert DEFAULT_STAGES.get("exp2.exact").config_paths == (
        "protocol.root_seed",
        "experiment2.oracle_factor_sizes",
        "experiment2.train_primitives",
        "experiment2.heldout_primitives",
        "experiment2.generator_density",
        "experiment2.unary_weight",
        "experiment2.exact_instances",
        "experiment2.support_penalties",
        "experiment2.exact_protocol",
    )
    assert DEFAULT_STAGES.get("gate.synthetic_exact").config_paths == (
        "gates.synthetic.noiseless_instances",
        "gates.synthetic.exact_tuple_recovery_fraction_min",
        "gates.synthetic.relative_support_error_max",
        "gates.overrides.synthetic_exact",
    )
