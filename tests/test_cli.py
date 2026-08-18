from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from tests.tracking2_fixture import with_tracking2_fixture
from voronoi_lab import cli, stage_handlers
from voronoi_lab.config import GateOverrideAuthorization, LabConfig
from voronoi_lab.core import RunIndex
from voronoi_lab.execution import ExperimentRunner, artifact_metadata
from voronoi_lab.exp1.probe_artifact import build_probe_bank_files
from voronoi_lab.exp1.torch_mechanics import summarize_resnet_mechanical_evidence
from voronoi_lab.mechanical import replay_toy_geometry


def _run_mechanical_gate(tmp_path, *, identity, override=False, run_id="fixture-run"):
    config = LabConfig()
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
    if override:
        authorization = GateOverrideAuthorization(
            target_gate="mechanical",
            reason="diagnostic continuation",
            authorized_by="test-suite",
            recorded_at="2026-08-17T00:00:00-07:00",
        )
        overrides = config.gates.overrides.model_copy(update={"mechanical": authorization})
        config = config.model_copy(
            update={"gates": config.gates.model_copy(update={"overrides": overrides})}
        )

    def probe_banks(context, dependencies):
        input_payload = context.store.read_json(
            dependencies["inputs.tracking2"].artifact_id, "inputs.json"
        )
        files, _ = build_probe_bank_files(context.config, input_payload)
        return context.store.put_files(
            files,
            kind="stage/exp1-probe-banks",
            metadata={
                **artifact_metadata(context, dependencies),
                "result_schema_version": 1,
            },
            media_types={"plan.json": "application/json"},
        )

    def mechanical(context, dependencies):
        input_payload = context.store.read_json(
            dependencies["inputs.tracking2"].artifact_id, "inputs.json"
        )
        output_values = (
            context.config.experiment1.mechanical_protocol.input_batch_size
            * input_payload["architecture"]["num_classes"]
        )
        zero = [0.0] * output_values
        identity_logits_by_cut = {}
        if identity is not None:
            split = list(zero)
            if not identity:
                split[0] = 1.0
            identity_logits_by_cut = {
                cut: {"full_logits": zero, "split_logits": split}
                for cut in context.config.experiment1.cuts
            }
        jvp_evidence = {
            cut: {
                "automatic_jvp": [1.0, *zero[1:]],
                "finite_difference_jvp": [1.0, *zero[1:]],
            }
            for cut in context.config.experiment1.sentinel_cuts
        }
        summary = summarize_resnet_mechanical_evidence(
            identity_logits_by_cut,
            jvp_evidence,
            identity_cuts=context.config.experiment1.cuts,
            jvp_cuts=context.config.experiment1.sentinel_cuts,
            denominator_floor=context.config.experiment1.mechanical_protocol.denominator_floor,
        )
        epsilon = (
            context.config.experiment1.mechanical_protocol.jvp_epsilon_float64
            if context.config.runtime.dtype == "float64"
            else context.config.experiment1.mechanical_protocol.jvp_epsilon_float32
        )
        jvp_by_cut = {
            cut: {
                **raw,
                "epsilon": epsilon,
                "relative_error": summary.jvp_relative_error_by_cut[cut],
            }
            for cut, raw in jvp_evidence.items()
        }
        payload = {
            "schema_version": 1,
            "protocol": context.config.experiment1.mechanical_protocol.model_dump(mode="json"),
            "probe_banks": {
                "artifact_valid": True,
                "deterministic": True,
                "distinct_train_test_sources": True,
            },
            "geometry": replay_toy_geometry(
                context.config.protocol.root_seed,
                rms_epsilon=context.config.experiment1.state_metric.rms_epsilon,
            ),
            "resnet": {
                "actual_device": "cpu",
                "identity_exact": summary.identity_exact,
                "identity_logits_by_cut": identity_logits_by_cut,
                "identity_max_absolute_error": summary.identity_max_absolute_error,
                "identity_per_cut": summary.identity_per_cut,
                "jvp_by_cut": jvp_by_cut,
                "jvp_cuts_completed": summary.jvp_cuts_completed,
                "jvp_failures": {},
                "jvp_median_relative_error": summary.jvp_median_relative_error,
                "jvp_p95_relative_error": summary.jvp_p95_relative_error,
            },
            "warnings": [],
        }
        return context.store.put_json(
            payload,
            filename="mechanical.json",
            kind="stage/exp1-mechanical",
            metadata={
                **artifact_metadata(context, dependencies),
                "result_schema_version": 1,
            },
        )

    runner = ExperimentRunner(
        config,
        project_root=tmp_path,
        handlers={
            "inputs.tracking2": stage_handlers.handle_inputs_tracking2,
            "exp1.probe_banks": probe_banks,
            "exp1.mechanical": mechanical,
            "gate.mechanical": stage_handlers.handle_gate_mechanical,
        },
        run_id=run_id,
    )
    reference = runner.run(["gate.mechanical"])["gate.mechanical"]
    return runner, reference


def test_validate_and_plan_emit_machine_readable_output(capsys) -> None:
    assert cli.main(["validate", "--json"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "VALID"
    assert "gate.mechanical" in validated["runnable_stages"]

    assert cli.main(["plan", "--stage", "gate.mechanical", "--json"]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert [stage["name"] for stage in planned["stages"]] == [
        "inputs.tracking2",
        "exp1.probe_banks",
        "exp1.mechanical",
        "gate.mechanical",
    ]


def test_validate_inputs_uses_the_same_strict_adapter_boundary(
    tmp_path, monkeypatch, capsys
) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("fixture: true\n")
    vgg_manifest = tmp_path / "vgg-manifest.yaml"
    vgg_manifest.write_text("fixture: vgg\n")
    adapter = SimpleNamespace(
        root=tmp_path / "external",
        validate_all=lambda: {"model_source": tmp_path / "models.py"},
        transplant_rows=lambda: (object(), object()),
    )
    vgg_adapter = SimpleNamespace(
        root=tmp_path / "vgg-external",
        manifest=SimpleNamespace(lineage_quality="exploratory_legacy"),
        validate_all=lambda: {
            "model_source": tmp_path / "models.py",
            "training_record": tmp_path / "criticality.json",
        },
        read_training_record=lambda: {
            "schema_version": 1,
            "experiment": "vgg_checkpoint_training",
        },
    )
    calls: list[tuple[str, int, Path]] = []

    def adapter_from_config(config, root):
        calls.append(("resnet", config.schema_version, root))
        return manifest, adapter

    def vgg_adapter_from_config(config, root):
        calls.append(("vgg", config.schema_version, root))
        return vgg_manifest, vgg_adapter

    monkeypatch.setattr(cli, "tracking2_adapter_from_config", adapter_from_config)
    monkeypatch.setattr(cli, "tracking2_vgg_adapter_from_config", vgg_adapter_from_config)
    assert cli.main(["validate", "--inputs", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [call[0] for call in calls] == ["resnet", "vgg"]
    assert all(call[1] == 1 for call in calls)
    assert result["tracking2"]["transplant_rows"] == 2
    assert result["tracking2_vgg"]["lineage_quality"] == "exploratory_legacy"
    assert set(result["tracking2_vgg"]["validated_files"]) == {
        "model_source",
        "training_record",
    }


def test_run_requires_a_target_and_gate_failure_has_strict_exit(monkeypatch, capsys) -> None:
    assert cli.main(["run", "--json"]) == cli.EXIT_USAGE
    missing = json.loads(capsys.readouterr().out)
    assert missing["status"] == "ERROR"

    class _Store:
        @staticmethod
        def read_json(_artifact_id: str, _filename: str) -> dict[str, str]:
            return {"status": "FAIL"}

    class _Runner:
        def __init__(self, *_args, **_kwargs) -> None:
            self.run_id = "fixture-run"
            self.store = _Store()
            self.receipt_artifact_id = None

        @staticmethod
        def run(_targets):
            return {"gate.mechanical": SimpleNamespace(artifact_id="a" * 64)}

    monkeypatch.setattr(cli, "ExperimentRunner", _Runner)
    monkeypatch.setattr(cli, "default_handlers", lambda: {})
    status = cli.main(["run", "--stage", "gate.mechanical", "--json"])
    assert status == cli.EXIT_GATE_BLOCKED
    result = json.loads(capsys.readouterr().out)
    assert result["gate_statuses"] == {"gate.mechanical": "FAIL"}


def test_artifact_verify_and_gate_inspect(tmp_path, capsys) -> None:
    _runner, artifact = _run_mechanical_gate(tmp_path, identity=None)

    root_args = ["--project-root", str(tmp_path)]
    assert cli.main([*root_args, "artifact", "verify", artifact.artifact_id, "--json"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["artifacts"][0]["status"] == "VERIFIED"

    status = cli.main(
        [
            *root_args,
            "gate",
            "inspect",
            "fixture-run",
            "--require-pass",
            "--require-gate",
            "gate.mechanical",
            "--json",
        ]
    )
    assert status == cli.EXIT_GATE_BLOCKED
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["gates"][0]["gate"]["status"] == "NOT_EVALUABLE"


def test_run_receipt_cli_semantically_verifies_and_inspect_lists_history(tmp_path, capsys) -> None:
    runner, _artifact = _run_mechanical_gate(tmp_path, identity=True)
    assert runner.receipt_artifact_id is not None
    root_args = ["--project-root", str(tmp_path)]

    assert (
        cli.main(
            [
                *root_args,
                "run",
                "receipt",
                runner.receipt_artifact_id,
                "--json",
            ]
        )
        == 0
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["status"] == "INTEGRITY_VERIFIED"
    assert verified["run_id"] == "fixture-run"
    assert verified["stages"][-1]["stage"] == "gate.mechanical"
    assert verified["stages"][-1]["attempt_history"][-1]["state"] == "COMPLETED"
    assert verified["integrity_validation"] == "PASSED"
    assert verified["config_compatibility"] == "CURRENT_COMPATIBLE"
    assert verified["provenance_compatibility"] == "CURRENT_COMPATIBLE"
    assert verified["source_compatibility"] == "NOT_CHECKED"
    assert verified["registry_compatibility"] == "MATCHED"
    assert verified["semantic_validation"] == "SKIPPED"
    assert "not supplied" in verified["semantic_validation_reason"]

    assert cli.main([*root_args, "run", "inspect", "fixture-run", "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["receipts"][-1]["artifact_id"] == runner.receipt_artifact_id


def test_gate_require_pass_does_not_treat_override_as_a_literal_pass(tmp_path, capsys) -> None:
    _runner, _artifact = _run_mechanical_gate(
        tmp_path,
        identity=False,
        override=True,
        run_id="override-run",
    )
    status = cli.main(
        [
            "--project-root",
            str(tmp_path),
            "gate",
            "inspect",
            "override-run",
            "--require-pass",
            "--require-gate",
            "gate.mechanical",
            "--json",
        ]
    )
    assert status == cli.EXIT_GATE_BLOCKED
    inspected = json.loads(capsys.readouterr().out)["gates"][0]
    assert inspected["gate"]["status"] == "OVERRIDDEN"
    assert inspected["override_authorization"]["authorized_by"] == "test-suite"


def test_gate_inspect_rejects_self_asserted_pass_artifact(tmp_path, capsys) -> None:
    runner, _valid = _run_mechanical_gate(tmp_path, identity=True, run_id="fake-run")
    store = runner.store
    artifact = store.put_json({"status": "PASS"}, filename="gate.json", kind="unrelated/blob")
    with sqlite3.connect(runner.index.path) as connection:
        connection.execute(
            "UPDATE run_stages SET artifact_id = ? WHERE run_id = ? AND stage_name = ?",
            (artifact.artifact_id, "fake-run", "gate.mechanical"),
        )

    status = cli.main(
        [
            "--project-root",
            str(tmp_path),
            "gate",
            "inspect",
            "fake-run",
            "--require-pass",
            "--require-gate",
            "gate.mechanical",
            "--json",
        ]
    )
    assert status == cli.EXIT_ERROR
    assert "incompatible stage identity" in json.loads(capsys.readouterr().out)["error"]


def test_run_inspect_and_explicit_crash_reclaim(tmp_path, capsys) -> None:
    index = RunIndex(tmp_path / "runs" / "index.sqlite")
    index.register_run("crashed-run", config_hash="a" * 64)
    index.claim_stage(
        "crashed-run",
        "fixture",
        "b" * 64,
        owner_token="dead-worker",
        metadata={"work": "fixture"},
    )
    root_args = ["--project-root", str(tmp_path)]

    assert cli.main([*root_args, "run", "inspect", "crashed-run", "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["stages"][0]["state"] == "RUNNING"
    assert inspected["stages"][0]["owner_token"] == "dead-worker"
    assert inspected["stages"][0]["attempt_history"][0]["event_kind"] == "CLAIMED"

    assert (
        cli.main(
            [
                *root_args,
                "run",
                "reclaim",
                "crashed-run",
                "fixture",
                "--reason",
                "worker host was terminated",
                "--json",
            ]
        )
        == 0
    )
    reclaimed = json.loads(capsys.readouterr().out)
    assert reclaimed["status"] == "RECLAIMED_FOR_RETRY"
    assert reclaimed["stage"]["state"] == "FAILED"
    assert reclaimed["stage"]["owner_token"] is None
    assert reclaimed["stage"]["metadata"]["recovery"]["previous_owner_token"] == ("dead-worker")
    assert [event["event_kind"] for event in reclaimed["stage"]["attempt_history"]] == [
        "CLAIMED",
        "RECLAIMED",
        "FINISHED",
    ]
    assert reclaimed["stage"]["attempt_history"][1]["message"] == (
        "reclaimed: worker host was terminated"
    )


def test_report_build_uses_configured_mockup_boundary(tmp_path, monkeypatch, capsys) -> None:
    output = tmp_path / "mockup.html"

    def fake_build_report(destination, *, mode, payload_path, readme_path):
        assert mode == "mockup"
        assert payload_path is None
        assert Path(readme_path).name == "README.md"
        path = Path(destination)
        path.write_text("MOCKUP")
        return path

    monkeypatch.setattr(cli, "build_report", fake_build_report)
    assert cli.main(["report", "build", "--output", str(output), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "mockup"
    assert result["output"] == str(output)
    assert output.read_text() == "MOCKUP"


def test_cli_disables_unverified_real_report_payloads(tmp_path, capsys) -> None:
    payload = tmp_path / "self-asserted.json"
    payload.write_text("{}", encoding="utf-8")
    status = cli.main(["report", "build", "--mode", "real", "--payload", str(payload), "--json"])
    assert status == cli.EXIT_USAGE
    error = json.loads(capsys.readouterr().out)
    assert "verified report-payload artifact" in error["error"]
