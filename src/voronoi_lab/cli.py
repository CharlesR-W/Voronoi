"""Command-line entry point for planning, running, and auditing experiments."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from voronoi_lab.config import LabConfig, load_config
from voronoi_lab.core import (
    ArtifactStore,
    GateStatus,
    RunIndex,
    StageState,
    canonical_hash,
    sha256_file,
    thaw_json,
)
from voronoi_lab.execution import (
    ExperimentRunner,
    load_verified_run_identity,
    verify_recorded_stage,
)
from voronoi_lab.pipeline import (
    DEFAULT_STAGES,
    PipelineError,
    StageValidationContext,
    expected_gate_override_authorization,
    expected_gate_rule,
    validate_stage_output,
)
from voronoi_lab.receipts import verify_run_receipt
from voronoi_lab.reporting import build_report
from voronoi_lab.stage_handlers import (
    default_handlers,
    tracking2_adapter_from_config,
    tracking2_vgg_adapter_from_config,
)

EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_GATE_BLOCKED = 3


class CLIError(RuntimeError):
    """A concise operational error suitable for command-line display."""


def _emit(value: object, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return
    if isinstance(value, str):
        print(value)
        return
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _project_root(args: argparse.Namespace) -> Path:
    return Path(args.project_root).expanduser().resolve()


def _path_from_root(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _config(args: argparse.Namespace) -> tuple[Path, LabConfig]:
    root = _project_root(args)
    path = _path_from_root(root, args.config)
    config = load_config(path)
    if config.experiment2.exact_instances != config.gates.synthetic.noiseless_instances:
        raise CLIError("experiment2.exact_instances must equal gates.synthetic.noiseless_instances")
    if not config.report.self_contained:
        raise CLIError("only self-contained reports are currently supported")
    if not config.report.embed_spec:
        raise CLIError("report.embed_spec must be true for the current report builder")
    return path, config


def _validate_command(args: argparse.Namespace) -> int:
    config_path, config = _config(args)
    plans = DEFAULT_STAGES.plan(config)
    result: dict[str, Any] = {
        "config_hash": canonical_hash(config.model_dump(mode="json")),
        "config_path": str(config_path),
        "runnable_stages": [plan.name for plan in plans if plan.implementation.value == "RUNNABLE"],
        "schema_version": config.schema_version,
        "status": "VALID",
    }
    if args.inputs:
        root = _project_root(args)
        manifest_path, adapter = tracking2_adapter_from_config(config, root)
        validated = adapter.validate_all()
        rows = adapter.transplant_rows()
        result["tracking2"] = {
            "external_root": str(adapter.root),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "transplant_rows": len(rows),
            "validated_files": {name: str(path) for name, path in sorted(validated.items())},
        }
        vgg_manifest_path, vgg_adapter = tracking2_vgg_adapter_from_config(config, root)
        vgg_validated = vgg_adapter.validate_all()
        training_record = vgg_adapter.read_training_record()
        result["tracking2_vgg"] = {
            "criticality_experiment": training_record["experiment"],
            "external_root": str(vgg_adapter.root),
            "lineage_quality": vgg_adapter.manifest.lineage_quality,
            "manifest_path": str(vgg_manifest_path),
            "manifest_sha256": sha256_file(vgg_manifest_path),
            "validated_files": {name: str(path) for name, path in sorted(vgg_validated.items())},
        }
    _emit(result, as_json=args.json)
    return 0


def _plan_command(args: argparse.Namespace) -> int:
    _, config = _config(args)
    targets = None if not args.stage else tuple(args.stage)
    plans = DEFAULT_STAGES.plan(config, targets)
    result = {
        "stages": [
            {
                "dependencies": list(plan.dependencies),
                "description": plan.description,
                "estimated_shards": plan.estimated_shards,
                "implementation": plan.implementation.value,
                "name": plan.name,
            }
            for plan in plans
        ],
        "targets": None if targets is None else list(targets),
    }
    if args.json:
        _emit(result, as_json=True)
    else:
        for plan in result["stages"]:
            dependencies = ",".join(plan["dependencies"]) or "-"
            print(
                f"{plan['name']:<28} {plan['implementation']:<8} "
                f"shards={plan['estimated_shards']:<4} deps={dependencies}"
            )
    return 0


def _gate_status_from_artifact(runner: ExperimentRunner, artifact_id: str) -> str:
    value = runner.store.read_json(artifact_id, "gate.json")
    if not isinstance(value, dict) or not isinstance(value.get("status"), str):
        raise CLIError(f"gate artifact {artifact_id} has no status")
    return value["status"]


def _run_command(args: argparse.Namespace) -> int:
    if args.run_action == "inspect":
        return _run_inspect_command(args)
    if args.run_action == "reclaim":
        return _run_reclaim_command(args)
    if args.run_action == "receipt":
        return _run_receipt_command(args)
    if args.run_subject is not None or args.run_stage_name is not None or args.reason is not None:
        raise CLIError("run identity arguments require the inspect or reclaim action")
    _, config = _config(args)
    if args.until is not None and args.stage:
        raise CLIError("use either --stage or --until, not both")
    targets = tuple(args.stage or ([args.until] if args.until is not None else ()))
    if not targets:
        raise CLIError("run requires at least one --stage or one --until target")
    root = _project_root(args)
    runner = ExperimentRunner(
        config,
        project_root=root,
        artifact_root=args.artifact_root,
        run_index_path=args.run_index,
        handlers=default_handlers(),
        run_id=args.run_id,
    )
    references = runner.run(targets)
    gate_statuses = {
        name: _gate_status_from_artifact(runner, reference.artifact_id)
        for name, reference in references.items()
        if name.startswith("gate.")
    }
    result = {
        "artifacts": {name: reference.artifact_id for name, reference in references.items()},
        "gate_statuses": gate_statuses,
        "receipt_artifact_id": runner.receipt_artifact_id,
        "run_id": runner.run_id,
        "targets": list(targets),
    }
    _emit(result, as_json=args.json)
    allowed = {GateStatus.PASS.value, GateStatus.OVERRIDDEN.value}
    blocked = any(status not in allowed for status in gate_statuses.values())
    return EXIT_GATE_BLOCKED if blocked else 0


def _stage_record_dict(stage: Any) -> dict[str, Any]:
    return {
        "artifact_id": stage.artifact_id,
        "attempts": stage.attempts,
        "message": stage.message,
        "metadata": thaw_json(stage.metadata),
        "owner_token": stage.owner_token,
        "stage": stage.stage_name,
        "stage_signature": stage.stage_signature,
        "state": stage.state.value,
        "updated_at": stage.updated_at,
    }


def _attempt_event_dict(event: Any) -> dict[str, Any]:
    return {
        "attempt": event.attempt,
        "created_at": event.created_at,
        "event_kind": event.event_kind,
        "event_sequence": event.sequence,
        "message": event.message,
        "metadata": thaw_json(event.metadata),
        "owner_token": event.owner_token,
        "state": event.state.value,
    }


def _stage_record_with_history(index: RunIndex, stage: Any) -> dict[str, Any]:
    history = index.list_stage_attempts(stage.run_id, stage.stage_name)
    return {
        **_stage_record_dict(stage),
        "attempt_history": [_attempt_event_dict(event) for event in history],
        "attempt_history_status": (
            "UNAVAILABLE_PRE_V3" if stage.attempts > 0 and not history else "COMPLETE"
        ),
    }


def _run_inspect_command(args: argparse.Namespace) -> int:
    if args.run_subject is None or args.run_stage_name is not None:
        raise CLIError("usage: voronoi-lab run inspect RUN_ID")
    root = _project_root(args)
    index = RunIndex(_path_from_root(root, args.run_index))
    run = index.get_run(args.run_subject)
    if run is None:
        raise CLIError(f"unknown run id: {args.run_subject}")
    result = {
        "config_hash": run.config_hash,
        "created_at": run.created_at,
        "provenance_artifact_id": run.provenance_artifact_id,
        "receipts": [
            {
                "artifact_id": receipt.artifact_id,
                "created_at": receipt.created_at,
                "requested_targets": list(receipt.requested_targets),
                "sequence": receipt.sequence,
            }
            for receipt in index.list_receipts(run.run_id)
        ],
        "run_id": run.run_id,
        "stages": [
            _stage_record_with_history(index, stage) for stage in index.list_stages(run.run_id)
        ],
        "updated_at": run.updated_at,
    }
    _emit(result, as_json=args.json)
    return 0


def _run_receipt_command(args: argparse.Namespace) -> int:
    if args.run_subject is None or args.run_stage_name is not None:
        raise CLIError("usage: voronoi-lab run receipt RECEIPT_ARTIFACT_ID")
    root = _project_root(args)
    store = ArtifactStore(_path_from_root(root, args.artifact_root))
    verified = verify_run_receipt(store, args.run_subject)
    payload = verified.payload
    config = payload["config"]
    provenance = payload["provenance"]
    stages = payload["stages"]
    result = {
        "config_hash": config["sha256"],
        "config_schema_version": verified.config_schema_version,
        "config_compatibility": verified.config_compatibility,
        "integrity_validation": verified.integrity_validation,
        "provenance_artifact_id": provenance["artifact_id"],
        "provenance_schema_version": verified.provenance_schema_version,
        "provenance_compatibility": verified.provenance_compatibility,
        "source_compatibility": verified.source_compatibility,
        "receipt_artifact_id": verified.artifact_id,
        "requested_targets": payload["requested_targets"],
        "run_id": payload["run_id"],
        "stages": [
            {
                "artifact_id": stage["artifact_id"],
                "attempt_history": stage["attempt_history"],
                "attempt_history_status": stage["attempt_history_status"],
                "attempts": stage["attempts"],
                "cache": stage["cache"],
                "gate": stage["gate"],
                "override_authorization": stage["override_authorization"],
                "override_lineage": stage["override_lineage"],
                "stage": stage["stage_name"],
                "state": stage["state"],
            }
            for stage in stages
        ],
        "auxiliary_stage_records": payload["auxiliary_stage_records"],
        "registry_compatibility": verified.registry_compatibility,
        "semantic_validation": verified.semantic_validation,
        "semantic_validation_reason": verified.semantic_validation_reason,
        "status": (
            "VERIFIED"
            if verified.semantic_validation == "PASSED"
            and verified.config_compatibility == "CURRENT_COMPATIBLE"
            and verified.provenance_compatibility == "CURRENT_COMPATIBLE"
            else "INTEGRITY_VERIFIED"
        ),
    }
    _emit(result, as_json=args.json)
    return 0


def _run_reclaim_command(args: argparse.Namespace) -> int:
    if args.run_subject is None or args.run_stage_name is None:
        raise CLIError("usage: voronoi-lab run reclaim RUN_ID STAGE --reason TEXT")
    if args.reason is None or not args.reason.strip():
        raise CLIError("run reclaim requires a nonblank --reason")
    root = _project_root(args)
    index = RunIndex(_path_from_root(root, args.run_index))
    previous = index.get_stage(args.run_subject, args.run_stage_name)
    if previous is None:
        raise CLIError(f"unknown stage: {args.run_subject}/{args.run_stage_name}")
    recovery_owner = f"reclaimer-{uuid4().hex}"
    reclaimed = index.reclaim_stage(
        args.run_subject,
        args.run_stage_name,
        owner_token=recovery_owner,
        reason=args.reason,
    )
    released = index.finish_stage(
        args.run_subject,
        args.run_stage_name,
        owner_token=recovery_owner,
        state=StageState.FAILED,
        message=f"reclaimed crashed attempt: {args.reason.strip()}",
        metadata={
            **dict(reclaimed.metadata),
            "recovery": {
                "previous_owner_token": previous.owner_token,
                "reason": args.reason.strip(),
            },
        },
    )
    _emit(
        {
            "run_id": args.run_subject,
            "stage": _stage_record_with_history(index, released),
            "status": "RECLAIMED_FOR_RETRY",
        },
        as_json=args.json,
    )
    return 0


def _gate_inspect_command(args: argparse.Namespace) -> int:
    root = _project_root(args)
    index = RunIndex(_path_from_root(root, args.run_index))
    store = ArtifactStore(_path_from_root(root, args.artifact_root))
    run = index.get_run(args.run_id)
    if run is None:
        raise CLIError(f"unknown run id: {args.run_id}")
    verified_run = load_verified_run_identity(store, index, args.run_id)
    validation_context = StageValidationContext()
    gates: list[dict[str, Any]] = []
    for stage in index.list_stages(args.run_id):
        if not stage.stage_name.startswith("gate."):
            continue
        try:
            stage_spec = DEFAULT_STAGES.get(stage.stage_name)
        except PipelineError as error:
            raise CLIError(f"unknown gate stage in run index: {stage.stage_name}") from error
        entry: dict[str, Any] = {
            "artifact_id": stage.artifact_id,
            "attempts": stage.attempts,
            "message": stage.message,
            "stage": stage.stage_name,
            "state": stage.state.value,
        }
        if stage.state is not StageState.COMPLETED or stage.artifact_id is None:
            raise CLIError(f"gate stage {stage.stage_name} is not complete")
        reference = verify_recorded_stage(
            store,
            index,
            args.run_id,
            stage.stage_name,
            validation_context=validation_context,
        )
        try:
            gate_result = validate_stage_output(
                reference,
                stage_spec,
                store,
                gate_rule=expected_gate_rule(stage_spec, verified_run.config),
                gate_override_authorization=expected_gate_override_authorization(
                    stage_spec, verified_run.config
                ),
                config=verified_run.config,
                registry=DEFAULT_STAGES,
                source_identity=verified_run.source_identity,
                validation_context=validation_context,
            )
        except PipelineError as error:
            raise CLIError(f"invalid gate artifact for {stage.stage_name}: {error}") from error
        if gate_result is None:
            raise CLIError(f"stage {stage.stage_name} has no gate output contract")
        entry["gate"] = gate_result.to_dict()
        entry["override_authorization"] = thaw_json(
            reference.manifest.metadata.get("override_authorization")
        )
        gates.append(entry)
    if not gates:
        raise CLIError(f"run {args.run_id!r} has no gate stages")
    required_gates = tuple(args.require_gate or ())
    if required_gates and not args.require_pass:
        raise CLIError("--require-gate requires --require-pass")
    if args.require_pass and not required_gates:
        raise CLIError("--require-pass requires at least one explicit --require-gate")
    result = {"gates": gates, "required_gates": list(required_gates), "run_id": args.run_id}
    _emit(result, as_json=args.json)
    if not args.require_pass:
        return 0
    by_name = {entry["stage"]: entry for entry in gates}
    for required in required_gates:
        entry = by_name.get(required)
        gate = None if entry is None else entry.get("gate")
        if not isinstance(gate, dict) or gate.get("status") != GateStatus.PASS.value:
            return EXIT_GATE_BLOCKED
    return 0


def _artifact_verify_command(args: argparse.Namespace) -> int:
    root = _project_root(args)
    store = ArtifactStore(_path_from_root(root, args.artifact_root))
    verified: list[dict[str, Any]] = []
    for artifact_id in args.artifact_id:
        reference = store.verify(artifact_id)
        verified.append(
            {
                "artifact_id": artifact_id,
                "files": [entry.path for entry in reference.manifest.files],
                "kind": reference.manifest.kind,
                "status": "VERIFIED",
            }
        )
    _emit({"artifacts": verified}, as_json=args.json)
    return 0


def _report_build_command(args: argparse.Namespace) -> int:
    if args.mode != "mockup":
        raise CLIError(
            "real report builds are disabled until a verified report-payload artifact is "
            "assembled from typed run outputs"
        )
    _, config = _config(args)
    root = _project_root(args)
    if args.output is None:
        declared_output = config.report.mockup_output
        output = _path_from_root(root, declared_output)
    else:
        output = _path_from_root(root, args.output)
    payload = None if args.payload is None else _path_from_root(root, args.payload)
    readme = _path_from_root(root, args.readme)
    built = build_report(
        output,
        mode=args.mode,
        payload_path=payload,
        readme_path=readme,
    )
    result = {
        "mode": args.mode,
        "output": str(built),
        "sha256": sha256_file(built),
        "size_bytes": built.stat().st_size,
    }
    _emit(result, as_json=args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voronoi-lab",
        description="Reproducible, gated Voronoi residual-computation experiments.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="project root used to resolve configuration and artifact paths (default: cwd)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate strict configuration and inputs")
    validate.add_argument("-c", "--config", default="configs/pilot.yaml")
    validate.add_argument(
        "--inputs",
        action="store_true",
        help="also hash every external Tracking2 input (read-only but potentially slow)",
    )
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(handler=_validate_command)

    plan = commands.add_parser("plan", help="show a dependency-closed execution plan")
    plan.add_argument("-c", "--config", default="configs/pilot.yaml")
    plan.add_argument("--stage", action="append", help="target stage (repeatable)")
    plan.add_argument("--json", action="store_true")
    plan.set_defaults(handler=_plan_command)

    run = commands.add_parser("run", help="execute runnable stages with immutable caching")
    run.add_argument("run_action", nargs="?", choices=("inspect", "receipt", "reclaim"))
    run.add_argument("run_subject", nargs="?", metavar="RUN_ID")
    run.add_argument("run_stage_name", nargs="?", metavar="STAGE")
    run.add_argument("-c", "--config", default="configs/pilot.yaml")
    run.add_argument("--stage", action="append", help="target stage (repeatable)")
    run.add_argument("--until", help="single target stage and its dependency closure")
    run.add_argument("--run-id", help="explicit stable run identifier")
    run.add_argument("--artifact-root", default="artifacts")
    run.add_argument("--run-index", default="runs/index.sqlite")
    run.add_argument("--reason", help="required audit reason when reclaiming a crashed stage")
    run.add_argument("--json", action="store_true")
    run.set_defaults(handler=_run_command)

    gate = commands.add_parser("gate", help="inspect gate outcomes")
    gate_commands = gate.add_subparsers(dest="gate_command", required=True)
    gate_inspect = gate_commands.add_parser("inspect", help="inspect all gates for a run")
    gate_inspect.add_argument("run_id")
    gate_inspect.add_argument("--artifact-root", default="artifacts")
    gate_inspect.add_argument("--run-index", default="runs/index.sqlite")
    gate_inspect.add_argument("--require-pass", action="store_true")
    gate_inspect.add_argument(
        "--require-gate",
        action="append",
        help="gate stage that must be a literal PASS (repeatable; requires --require-pass)",
    )
    gate_inspect.add_argument("--json", action="store_true")
    gate_inspect.set_defaults(handler=_gate_inspect_command)

    artifact = commands.add_parser("artifact", help="audit immutable artifacts")
    artifact_commands = artifact.add_subparsers(dest="artifact_command", required=True)
    verify = artifact_commands.add_parser("verify", help="verify manifests and payload checksums")
    verify.add_argument("artifact_id", nargs="+")
    verify.add_argument("--artifact-root", default="artifacts")
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(handler=_artifact_verify_command)

    report = commands.add_parser("report", help="build self-contained HTML reports")
    report_commands = report.add_subparsers(dest="report_command", required=True)
    report_build = report_commands.add_parser(
        "build", help="build the self-contained MOCKUP (real mode is lineage-gated off)"
    )
    report_build.add_argument("-c", "--config", default="configs/pilot.yaml")
    report_build.add_argument("--mode", choices=("mockup", "real"), default="mockup")
    report_build.add_argument("--payload", help="saved ReportPayload JSON (required for real mode)")
    report_build.add_argument("--output", help="override configured output path")
    report_build.add_argument("--readme", default="README.md")
    report_build.add_argument("--json", action="store_true")
    report_build.set_defaults(handler=_report_build_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as error:  # CLI boundary: keep expected failures concise and nonzero.
        if getattr(args, "json", False):
            _emit(
                {"error": str(error), "error_type": type(error).__name__, "status": "ERROR"},
                as_json=True,
            )
        else:
            print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE if isinstance(error, CLIError) else EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
