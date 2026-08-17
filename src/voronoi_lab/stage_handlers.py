"""Concrete handlers for the deliberately small runnable pipeline surface.

Handlers consume only configuration plus immutable dependency artifacts and
publish immutable artifacts of their own.  Scientific outcome stages that are
still marked ``PLANNED`` in :mod:`voronoi_lab.pipeline` intentionally have no
handler here.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from voronoi_lab.config import LabConfig
from voronoi_lab.core import (
    ArtifactRef,
    GateEvaluator,
    GateResult,
    GateRule,
    GateStatus,
    JSONLike,
    SeedDeriver,
    StageState,
    canonical_hash,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from voronoi_lab.execution import StageContext, StageHandler, artifact_metadata
from voronoi_lab.exp1.probe_artifact import ProbeArtifactError, build_probe_bank_files
from voronoi_lab.exp1.torch_mechanics import (
    summarize_resnet_mechanical_evidence,
    validate_jvp,
    zero_intervention_parity,
)
from voronoi_lab.exp1.tracking2 import (
    Tracking2Adapter,
    parse_tracking2_manifest_bytes,
    resolve_cut,
)
from voronoi_lab.mechanical import replay_synthetic_invariants, replay_toy_geometry
from voronoi_lab.pipeline import expected_gate_rule, stage_config
from voronoi_lab.reporting.builder import render_report
from voronoi_lab.reporting.payload import make_mock_payload
from voronoi_lab.sharding import ShardContext, ShardExecutor, ShardKey, ShardReducer, ShardSpec
from voronoi_lab.synthetic import (
    OracleExhaustiveInstanceResult,
    run_oracle_exhaustive_instance,
    summarize_oracle_exhaustive_instances,
)

RESULT_SCHEMA_VERSION = 1


class StageHandlerError(RuntimeError):
    """Raised when a runnable handler cannot honor its declared contract."""


def _resolve(project_root: Path, declared: str | Path) -> Path:
    path = Path(declared).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def tracking2_adapter_from_config(
    config: LabConfig, project_root: Path
) -> tuple[Path, Tracking2Adapter]:
    """Construct the one supported adapter, rejecting silent legacy overrides."""

    inputs = config.inputs.tracking2
    if not inputs.read_only:
        raise StageHandlerError("Tracking2 inputs must remain read-only")
    if inputs.checkpoint_files or inputs.dataset_root is not None or inputs.transplant_results:
        raise StageHandlerError(
            "per-file Tracking2 overrides are not implemented; use one hash-pinned manifest"
        )
    manifest_path = _resolve(project_root, inputs.manifest)
    observed_manifest_hash = sha256_file(manifest_path)
    if observed_manifest_hash != inputs.manifest_sha256:
        raise StageHandlerError(
            "Tracking2 manifest SHA-256 does not match the signed configuration: "
            f"expected {inputs.manifest_sha256}, observed {observed_manifest_hash}"
        )
    root = _resolve(project_root, inputs.root)
    adapter = Tracking2Adapter.from_yaml(manifest_path, root_override=root)
    expected = f"preactivation_resnet18_v2_width{adapter.manifest.architecture.width}"
    if inputs.expected_model != expected:
        raise StageHandlerError(
            f"expected_model={inputs.expected_model!r} does not match manifest model {expected!r}"
        )
    manifest = adapter.manifest
    configured_epochs = tuple(config.experiment1.checkpoints)
    manifest_epochs = tuple(item.epoch for item in manifest.checkpoints)
    if configured_epochs != manifest_epochs:
        raise StageHandlerError(
            "experiment1.checkpoints must exactly match the Tracking2 manifest epochs"
        )
    banks = config.experiment1.probe_banks
    if banks.fit_train_images + banks.independent_fit_train_images > manifest.training.train_size:
        raise StageHandlerError("disjoint codebook fit banks exceed the manifest training split")
    if banks.intervention_nested_in_geometry:
        if banks.intervention_test_images > banks.geometry_test_images:
            raise StageHandlerError("nested intervention bank cannot exceed the geometry bank")
        required_test = banks.geometry_test_images
    else:
        required_test = banks.geometry_test_images + banks.intervention_test_images
    if required_test > manifest.training.test_size:
        raise StageHandlerError("declared held-out probe banks exceed the manifest test split")
    return manifest_path, adapter


def _tracking2_adapter(context: StageContext) -> tuple[Path, Tracking2Adapter]:
    return tracking2_adapter_from_config(context.config, context.project_root)


def _preflight_inputs_tracking2(context: StageContext) -> None:
    """Verify every current external byte before completed/cache reuse."""

    _manifest_path, adapter = _tracking2_adapter(context)
    adapter.validate_all()


def _stage_metadata(
    context: StageContext,
    dependencies: Mapping[str, ArtifactRef],
    **extra: JSONLike,
) -> dict[str, JSONLike]:
    return {
        **artifact_metadata(context, dependencies),
        "result_schema_version": RESULT_SCHEMA_VERSION,
        **extra,
    }


def _json_object(context: StageContext, reference: ArtifactRef, filename: str) -> dict[str, Any]:
    value = context.store.read_json(reference.artifact_id, filename)
    if not isinstance(value, dict):
        raise StageHandlerError(f"dependency payload {filename!r} must be a JSON object")
    return value


def _gate_with_optional_override(
    rule: GateRule,
    observations: Mapping[str, object],
    override_reason: str | None,
) -> GateResult:
    evaluator = GateEvaluator()
    natural = evaluator.evaluate(rule, observations)
    if override_reason is None or natural.status is GateStatus.PASS:
        return natural
    return evaluator.evaluate(rule, observations, override_reason=override_reason)


def handle_inputs_tracking2(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    """Validate all Tracking2 bytes and normalize the legacy transplant table."""

    if dependencies:
        raise StageHandlerError("inputs.tracking2 must not receive dependency artifacts")
    manifest_path, adapter = _tracking2_adapter(context)
    validated = adapter.validate_all()
    manifest = adapter.manifest
    references = {
        "model_source": manifest.architecture.source,
        **{f"checkpoint_epoch{item.epoch}": item for item in manifest.checkpoints},
        "dataset_train": manifest.datasets.train,
        "dataset_test": manifest.datasets.test,
        "transplant": manifest.transplant.file,
    }
    if set(validated) != set(references):
        raise StageHandlerError("Tracking2 adapter returned an unexpected validated-file inventory")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise StageHandlerError("could not snapshot the Tracking2 manifest") from error
    if sha256_bytes(manifest_bytes) != context.config.inputs.tracking2.manifest_sha256:
        raise StageHandlerError("Tracking2 manifest changed while the input stage was running")
    embedded_manifest = parse_tracking2_manifest_bytes(
        manifest_bytes, source=context.config.inputs.tracking2.manifest
    )
    if canonical_hash(embedded_manifest.model_dump(mode="json")) != canonical_hash(
        manifest.model_dump(mode="json")
    ):
        raise StageHandlerError("Tracking2 manifest changed while it was being parsed")
    transplant_bytes = adapter.read_validated_bytes(manifest.transplant.file)
    files = {
        name: {
            "declared_path": reference.path.as_posix(),
            "sha256": reference.sha256,
            "size_bytes": reference.size_bytes,
        }
        for name, reference in sorted(references.items())
    }
    rows = [
        row.model_dump(mode="json") for row in adapter.normalize_transplant_bytes(transplant_bytes)
    ]
    result: dict[str, JSONLike] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "architecture": manifest.architecture.model_dump(mode="json"),
        "dataset_isolation": {
            "passed": True,
            "train_test_distinct_hashes": (
                manifest.datasets.train.sha256 != manifest.datasets.test.sha256
            ),
            "train_test_distinct_paths": (
                manifest.datasets.train.path != manifest.datasets.test.path
            ),
        },
        # Keep immutable scientific input identity independent of the machine's
        # checkout location. Resolved operational paths belong in run provenance,
        # while these declarations are already signed by the stage configuration.
        "external_root": context.config.inputs.tracking2.root.as_posix(),
        "lineage_note": manifest.lineage_note,
        "lineage_quality": manifest.lineage_quality,
        "manifest": {
            "path": context.config.inputs.tracking2.manifest.as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "observed_repository_revision": manifest.observed_repository_revision,
        "read_only": True,
        "training": manifest.training.model_dump(mode="json"),
        "validated_files": files,
        "transplant_rows": rows,
    }
    return context.store.put_files(
        {
            "inputs.json": canonical_json_bytes(result),
            "manifest.yaml": manifest_bytes,
            "transplant.json": transplant_bytes,
        },
        kind="stage/inputs-tracking2",
        metadata=_stage_metadata(
            context,
            dependencies,
            input_count=len(validated),
            lineage_quality=manifest.lineage_quality,
        ),
        media_types={
            "inputs.json": "application/json",
            "manifest.yaml": "application/yaml",
            "transplant.json": "application/json",
        },
    )


def _build_probe_bank_files(
    context: StageContext, input_payload: Mapping[str, object]
) -> tuple[dict[str, bytes], dict[str, JSONLike]]:
    """Materialize the complete deterministic probe artifact before publication."""

    try:
        return build_probe_bank_files(context.config, input_payload)
    except (ProbeArtifactError, ValueError) as error:
        raise StageHandlerError(str(error)) from error


def handle_exp1_probe_banks(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    """Freeze image identities, per-cut sites, and held-out bootstrap draws."""

    try:
        input_ref = dependencies["inputs.tracking2"]
    except KeyError as error:
        raise StageHandlerError("exp1.probe_banks requires inputs.tracking2") from error
    input_payload = _json_object(context, input_ref, "inputs.json")
    files, plan_payload = _build_probe_bank_files(context, input_payload)
    return context.store.put_files(
        files,
        kind="stage/exp1-probe-banks",
        metadata=_stage_metadata(
            context,
            dependencies,
            role_count=len(plan_payload["roles"]),  # type: ignore[arg-type]
            cut_count=len(plan_payload["cuts"]),  # type: ignore[arg-type]
        ),
        media_types={"plan.json": "application/json"},
    )


def _probe_determinism_check(
    context: StageContext,
    input_payload: Mapping[str, object],
    saved_reference: ArtifactRef,
) -> bool:
    first, _ = _build_probe_bank_files(context, input_payload)
    second, _ = _build_probe_bank_files(context, input_payload)
    if first.keys() != second.keys() or any(first[name] != second[name] for name in first):
        return False
    saved_paths = {entry.path for entry in saved_reference.manifest.files}
    return saved_paths == set(first) and all(
        context.store.read_bytes(saved_reference.artifact_id, name) == expected
        for name, expected in first.items()
    )


def _empty_resnet_metrics() -> dict[str, JSONLike]:
    return {
        "actual_device": "cpu",
        "identity_exact": None,
        "identity_max_absolute_error": None,
        "identity_logits_by_cut": {},
        "identity_per_cut": {},
        "jvp_by_cut": {},
        "jvp_cuts_completed": 0,
        "jvp_failures": {},
        "jvp_median_relative_error": None,
        "jvp_p95_relative_error": None,
    }


def _run_resnet_mechanics(context: StageContext) -> tuple[dict[str, JSONLike], list[str]]:
    metrics = _empty_resnet_metrics()
    warnings: list[str] = []
    try:
        import torch

        _, adapter = _tracking2_adapter(context)
        target_epoch = adapter.manifest.training.target_epoch
        model = adapter.load_model(target_epoch, device="cpu")
        dtype = torch.float64 if context.config.runtime.dtype == "float64" else torch.float32
        model.to(dtype=dtype)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        protocol = context.config.experiment1.mechanical_protocol
        values_per_image = 3 * 32 * 32
        images = torch.linspace(
            protocol.input_min,
            protocol.input_max,
            protocol.input_batch_size * values_per_image,
            dtype=dtype,
        ).reshape(protocol.input_batch_size, 3, 32, 32)
        cuts = tuple(resolve_cut(name) for name in context.config.experiment1.cuts)
        parity = zero_intervention_parity(
            model,
            images,
            cuts,
            encode=adapter.encode,
            suffix=adapter.suffix,
        )
        identity_logits_by_cut: dict[str, dict[str, JSONLike]] = {
            name: {
                "full_logits": list(full),
                "split_logits": list(split),
            }
            for name, full, split in zip(
                parity.cuts,
                parity.full_logits,
                parity.split_logits,
                strict=True,
            )
        }

        direction_seeds = SeedDeriver(
            context.config.protocol.root_seed,
            ("exp1", "mechanical", "jvp_direction"),
        )
        jvp_outputs_by_cut: dict[str, dict[str, JSONLike]] = {}
        jvp_failures: dict[str, JSONLike] = {}
        epsilon = (
            protocol.jvp_epsilon_float64 if dtype == torch.float64 else protocol.jvp_epsilon_float32
        )
        for cut_name in context.config.experiment1.sentinel_cuts:
            cut = resolve_cut(cut_name)
            direction_generator = torch.Generator(device="cpu")
            direction_generator.manual_seed(direction_seeds.derive({"cut": cut.name}))
            with torch.no_grad():
                point = adapter.encode(model, images, cut).detach().clone()
            direction = torch.randn(
                point.shape,
                dtype=point.dtype,
                device=point.device,
                generator=direction_generator,
            )
            direction /= torch.sqrt(torch.mean(direction**2))
            try:
                check = validate_jvp(
                    lambda value, selected=cut: adapter.suffix(model, value, selected),
                    point,
                    direction,
                    epsilon=epsilon,
                    sample_axis=None,
                    denominator_floor=protocol.denominator_floor,
                )
            except (RuntimeError, TypeError, ValueError) as error:
                warnings.append(f"JVP unavailable at {cut.name}: {type(error).__name__}: {error}")
                jvp_failures[cut.name] = f"{type(error).__name__}: {error}"
                continue
            jvp_outputs_by_cut[cut.name] = {
                "epsilon": check.epsilon,
                "automatic_jvp": list(check.automatic_output),
                "finite_difference_jvp": list(check.finite_difference_output),
            }
        summary = summarize_resnet_mechanical_evidence(
            identity_logits_by_cut,
            jvp_outputs_by_cut,
            identity_cuts=context.config.experiment1.cuts,
            jvp_cuts=context.config.experiment1.sentinel_cuts,
            denominator_floor=protocol.denominator_floor,
        )
        metrics.update(
            {
                "identity_exact": summary.identity_exact,
                "identity_max_absolute_error": summary.identity_max_absolute_error,
                "identity_logits_by_cut": identity_logits_by_cut,
                "identity_per_cut": summary.identity_per_cut,
                "jvp_by_cut": {
                    cut_name: {
                        **raw,
                        "relative_error": summary.jvp_relative_error_by_cut[cut_name],
                    }
                    for cut_name, raw in jvp_outputs_by_cut.items()
                },
                "jvp_cuts_completed": summary.jvp_cuts_completed,
                "jvp_failures": jvp_failures,
                "jvp_median_relative_error": summary.jvp_median_relative_error,
                "jvp_p95_relative_error": summary.jvp_p95_relative_error,
            }
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        warnings.append(f"real ResNet mechanics unavailable: {type(error).__name__}: {error}")
    return metrics, warnings


def handle_exp1_mechanical(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    """Run deterministic toy invariants and bounded CPU checks on the pinned model."""

    try:
        input_ref = dependencies["inputs.tracking2"]
        probe_ref = dependencies["exp1.probe_banks"]
    except KeyError as error:
        raise StageHandlerError(
            "exp1.mechanical requires inputs.tracking2 and exp1.probe_banks"
        ) from error
    input_payload = _json_object(context, input_ref, "inputs.json")
    isolation = input_payload.get("dataset_isolation")
    distinct_train_test_sources = (
        isinstance(isolation, Mapping)
        and isolation.get("passed") is True
        and isolation.get("train_test_distinct_hashes") is True
        and isolation.get("train_test_distinct_paths") is True
    )
    geometry = replay_toy_geometry(
        context.config.protocol.root_seed,
        rms_epsilon=context.config.experiment1.state_metric.rms_epsilon,
    )
    resnet, warnings = _run_resnet_mechanics(context)
    result: dict[str, JSONLike] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol": context.config.experiment1.mechanical_protocol.model_dump(mode="json"),
        "geometry": geometry,
        "probe_banks": {
            "artifact_valid": True,
            "deterministic": _probe_determinism_check(context, input_payload, probe_ref),
            "distinct_train_test_sources": distinct_train_test_sources,
        },
        "resnet": resnet,
        "synthetic_invariants": replay_synthetic_invariants(context.config.protocol.root_seed),
        "warnings": warnings,
    }
    return context.store.put_json(
        result,
        filename="mechanical.json",
        kind="stage/exp1-mechanical",
        metadata=_stage_metadata(
            context,
            dependencies,
            evidence_class="mechanical_validation",
            warning_count=len(warnings),
        ),
    )


def _mechanical_rule(context: StageContext) -> GateRule:
    return expected_gate_rule(context.stage_spec, context.config)


def handle_gate_mechanical(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    try:
        reference = dependencies["exp1.mechanical"]
    except KeyError as error:
        raise StageHandlerError("gate.mechanical requires exp1.mechanical") from error
    observations = _json_object(context, reference, "mechanical.json")
    authorization = context.config.gates.overrides.mechanical
    rule = _mechanical_rule(context)
    result = _gate_with_optional_override(
        rule,
        observations,
        None if authorization is None else authorization.reason,
    )
    return context.store.put_json(
        result.to_dict(),
        filename="gate.json",
        kind="gate/mechanical",
        metadata=_stage_metadata(
            context,
            dependencies,
            gate_id=result.gate_id,
            gate_status=result.status.value,
            gate_rule_signature=canonical_hash(rule.to_dict()),
            natural_status=result.natural_status.value,
            override_authorization=(
                None if authorization is None else authorization.model_dump(mode="json")
            ),
        ),
    )


def handle_exp2_exact(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    if dependencies:
        raise StageHandlerError("exp2.exact must not receive dependency artifacts")
    parent_record = context.index.get_stage(context.run_id, context.stage_spec.name)
    if parent_record is None or parent_record.state is not StageState.RUNNING:
        raise StageHandlerError(
            "exp2.exact shards require their parent stage to be claimed RUNNING"
        )
    experiment = context.config.experiment2
    protocol = experiment.exact_protocol
    protocol_payload: dict[str, JSONLike] = {
        "delta": protocol.delta,
        "density": experiment.generator_density,
        "exhaustive_tie_atol": protocol.exhaustive_tie_atol,
        "exhaustive_tie_rtol": protocol.exhaustive_tie_rtol,
        "factor_sizes": list(experiment.oracle_factor_sizes),
        "generator_connectivity_policy": protocol.generator_connectivity_policy,
        "generator_normalization": protocol.generator_normalization,
        "generator_rate_shape": protocol.generator_rate_shape,
        "heldout_primitives": experiment.heldout_primitives,
        "max_states": protocol.max_states,
        "numeric_dtype": "float64",
        "penalties": list(experiment.support_penalties),
        "protocol_version": 1,
        "random_relabel": protocol.random_relabel,
        "rho": protocol.rho,
        "root_seed": context.config.protocol.root_seed,
        "seed_namespace": ["exp2", "exact", "oracle_exhaustive", "v1"],
        "support_policy": protocol.support_policy,
        "train_primitives": experiment.train_primitives,
        "unary_weight": experiment.unary_weight,
    }
    selected_config = stage_config(context.config, context.stage_spec.config_paths)
    specs = tuple(
        ShardSpec(
            parent_stage=context.stage_spec.name,
            parent_stage_signature=parent_record.stage_signature,
            key=ShardKey({"instance_index": instance_index}),
            artifact_kind="shard/exp2-exact-instance",
            stage_config=selected_config,
            source_identity=context.source_identity,
            upstream_artifacts={},
        )
        for instance_index in range(experiment.exact_instances)
    )

    def execute_instance(spec: ShardSpec) -> ArtifactRef:
        instance_index = spec.key.coordinates["instance_index"]
        if type(instance_index) is not int:
            raise StageHandlerError("exact shard instance index must be an integer")

        def publish(shard_context: ShardContext) -> ArtifactRef:
            instance = run_oracle_exhaustive_instance(
                seed=context.config.protocol.root_seed,
                instance_index=instance_index,
                factor_sizes=experiment.oracle_factor_sizes,
                train_primitives=experiment.train_primitives,
                heldout_primitives=experiment.heldout_primitives,
                density=experiment.generator_density,
                unary_weight=experiment.unary_weight,
                rho=protocol.rho,
                delta=protocol.delta,
                support_policy=protocol.support_policy,
                random_relabel=protocol.random_relabel,
                penalties=experiment.support_penalties,
                max_states=protocol.max_states,
                generator_rate_shape=protocol.generator_rate_shape,
                generator_connectivity_policy=protocol.generator_connectivity_policy,
                generator_normalization=protocol.generator_normalization,
                exhaustive_tie_atol=protocol.exhaustive_tie_atol,
                exhaustive_tie_rtol=protocol.exhaustive_tie_rtol,
            )
            return shard_context.store.put_json(
                {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "protocol": protocol_payload,
                    "instance": asdict(instance),
                },
                filename="instance.json",
                kind=spec.artifact_kind,
                metadata=shard_context.artifact_metadata(
                    {
                        "evidence_class": "synthetic_oracle_instance",
                        "instance_index": instance_index,
                        "observed_generator_family_hash": (instance.observed_generator_family_hash),
                        "result_schema_version": RESULT_SCHEMA_VERSION,
                    }
                ),
            )

        return ShardExecutor(
            context.store,
            context.index,
            context.run_id,
        ).execute(spec, publish)

    workers = context.config.runtime.workers
    if workers == 0:
        instance_refs = tuple(execute_instance(spec) for spec in specs)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            instance_refs = tuple(pool.map(execute_instance, specs))

    reduction = ShardReducer(context.store, context.index, context.run_id).publish(specs)
    reducer_ids = tuple(reference.artifact_id for reference in reduction.shard_artifacts)
    executed_ids = tuple(reference.artifact_id for reference in instance_refs)
    if executed_ids != reducer_ids:
        raise StageHandlerError("ordered reducer inputs differ from executed exact shards")

    instance_results: list[OracleExhaustiveInstanceResult] = []
    expected_fields = set(OracleExhaustiveInstanceResult.__dataclass_fields__)
    for expected_index, reference in enumerate(reduction.shard_artifacts):
        payload = _json_object(context, reference, "instance.json")
        raw_instance = payload.get("instance")
        if not isinstance(raw_instance, Mapping) or set(raw_instance) != expected_fields:
            raise StageHandlerError("exact instance shard has an invalid result schema")
        if canonical_hash(payload.get("protocol")) != canonical_hash(protocol_payload):
            raise StageHandlerError("exact instance shard has incompatible protocol evidence")
        try:
            instance = OracleExhaustiveInstanceResult(**raw_instance)  # type: ignore[arg-type]
        except TypeError as error:
            raise StageHandlerError("exact instance shard cannot be reconstructed") from error
        if instance.instance_index != expected_index:
            raise StageHandlerError("exact instance shard order does not match its global index")
        evidence_hashes = {
            "observed_generator_family_hash": canonical_hash(
                raw_instance["observed_generator_family"]  # type: ignore[arg-type]
            ),
            "realized_normalized_support_order_spectrum_hash": canonical_hash(
                raw_instance[  # type: ignore[arg-type]
                    "realized_normalized_support_order_spectrum"
                ]
            ),
            "truth_labeling_hash": canonical_hash(
                raw_instance["truth_labeling"]  # type: ignore[arg-type]
            ),
            "selected_labeling_hash": canonical_hash(
                raw_instance["selected_labeling"]  # type: ignore[arg-type]
            ),
        }
        if any(raw_instance[name] != digest for name, digest in evidence_hashes.items()):
            raise StageHandlerError("exact instance evidence content hash is inconsistent")
        if instance.observed_generator_family_hash != context.store.get(
            reference.artifact_id
        ).manifest.metadata.get("observed_generator_family_hash"):
            raise StageHandlerError("exact instance evidence hash metadata is inconsistent")
        instance_results.append(instance)

    smoke = summarize_oracle_exhaustive_instances(
        instance_results,
        seed=context.config.protocol.root_seed,
        factor_sizes=experiment.oracle_factor_sizes,
        train_primitives=experiment.train_primitives,
        heldout_primitives=experiment.heldout_primitives,
        density=experiment.generator_density,
        unary_weight=experiment.unary_weight,
        rho=protocol.rho,
        delta=protocol.delta,
        support_policy=protocol.support_policy,
        random_relabel=protocol.random_relabel,
        penalties=experiment.support_penalties,
        max_states=protocol.max_states,
        generator_rate_shape=protocol.generator_rate_shape,
        generator_connectivity_policy=protocol.generator_connectivity_policy,
        generator_normalization=protocol.generator_normalization,
        exhaustive_tie_atol=protocol.exhaustive_tie_atol,
        exhaustive_tie_rtol=protocol.exhaustive_tie_rtol,
    )
    aggregate: dict[str, JSONLike] = {
        "evaluated_labelings": smoke.evaluated_labelings,
        "exact_instances": smoke.exact_instances,
        "exact_tuple_recovery_fraction": smoke.exact_instances / smoke.instances,
        "max_best_objective": smoke.max_best_objective,
        "max_heldout_excess_objective": smoke.max_heldout_excess_objective,
        "worst_support_error": smoke.worst_support_error,
        "worst_train_support_error": smoke.worst_train_support_error,
    }
    result: dict[str, JSONLike] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        **asdict(smoke),
        "exact_tuple_recovery_fraction": smoke.exact_instances / smoke.instances,
        "numeric_dtype": "float64",
        "aggregate": aggregate,
        "ordered_instance_artifact_ids": list(reducer_ids),
        "reducer_artifact_id": reduction.reference.artifact_id,
    }
    return context.store.put_json(
        result,
        filename="exact.json",
        kind="stage/exp2-exact",
        metadata=_stage_metadata(
            context,
            dependencies,
            evidence_class="synthetic_oracle_validation",
            instances=experiment.exact_instances,
            reducer_artifact_id=reduction.reference.artifact_id,
        ),
    )


def _synthetic_exact_rule(context: StageContext) -> GateRule:
    return expected_gate_rule(context.stage_spec, context.config)


def handle_gate_synthetic_exact(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    try:
        reference = dependencies["exp2.exact"]
    except KeyError as error:
        raise StageHandlerError("gate.synthetic_exact requires exp2.exact") from error
    observations = _json_object(context, reference, "exact.json")
    authorization = context.config.gates.overrides.synthetic_exact
    rule = _synthetic_exact_rule(context)
    result = _gate_with_optional_override(
        rule,
        observations,
        None if authorization is None else authorization.reason,
    )
    return context.store.put_json(
        result.to_dict(),
        filename="gate.json",
        kind="gate/synthetic-exact",
        metadata=_stage_metadata(
            context,
            dependencies,
            gate_id=result.gate_id,
            gate_status=result.status.value,
            gate_rule_signature=canonical_hash(rule.to_dict()),
            natural_status=result.natural_status.value,
            override_authorization=(
                None if authorization is None else authorization.model_dump(mode="json")
            ),
        ),
    )


def handle_report_build(
    context: StageContext, dependencies: Mapping[str, ArtifactRef]
) -> ArtifactRef:
    """Publish the interpretation-first MOCKUP as an immutable report artifact."""

    if dependencies:
        raise StageHandlerError("report.build currently consumes no dependency artifacts")
    if not context.config.report.self_contained:
        raise StageHandlerError("report.build supports only self-contained reports")
    if not context.config.report.embed_spec:
        raise StageHandlerError("report.build requires the experiment specification to be embedded")
    readme_path = context.project_root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    payload = make_mock_payload()
    document = render_report(payload, readme)
    return context.store.put_files(
        {
            "report.html": document.encode("utf-8"),
            "report_payload.json": canonical_json_bytes(payload.model_dump(mode="json")),
            "spec.md": readme.encode("utf-8"),
        },
        kind="report/mockup",
        metadata=_stage_metadata(
            context,
            dependencies,
            evidence_mode="mockup",
            self_contained=True,
        ),
        media_types={
            "report.html": "text/html; charset=utf-8",
            "report_payload.json": "application/json",
            "spec.md": "text/markdown; charset=utf-8",
        },
    )


def default_handlers() -> dict[str, StageHandler]:
    """Return a fresh mapping for every stage declared runnable in the default DAG."""

    return {
        "inputs.tracking2": handle_inputs_tracking2,
        "exp1.probe_banks": handle_exp1_probe_banks,
        "exp1.mechanical": handle_exp1_mechanical,
        "gate.mechanical": handle_gate_mechanical,
        "exp2.exact": handle_exp2_exact,
        "gate.synthetic_exact": handle_gate_synthetic_exact,
        "report.build": handle_report_build,
    }


# ExperimentRunner recognizes this generic callable attribute and invokes it
# before considering completed-stage or cross-run cache reuse.
handle_inputs_tracking2.__dict__["__voronoi_stage_preflight__"] = _preflight_inputs_tracking2


__all__ = [
    "StageHandlerError",
    "default_handlers",
    "handle_exp1_mechanical",
    "handle_exp1_probe_banks",
    "handle_exp2_exact",
    "handle_gate_mechanical",
    "handle_gate_synthetic_exact",
    "handle_inputs_tracking2",
    "handle_report_build",
    "tracking2_adapter_from_config",
]
