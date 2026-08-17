from __future__ import annotations

import math

import pytest

from voronoi_lab.core import (
    CheckStatus,
    ComparisonOperator,
    GateCheck,
    GateCheckResult,
    GateEvaluationError,
    GateEvaluator,
    GateOverride,
    GateResult,
    GateRule,
    GateStatus,
    canonical_json_bytes,
    evaluate_dependencies,
)


def mechanical_rule() -> GateRule:
    return GateRule(
        gate_id="mechanical",
        checks=(
            GateCheck(
                name="identity",
                metric="identity_exact",
                operator=ComparisonOperator.IS_TRUE,
            ),
            GateCheck(
                name="roundtrip",
                metric="errors.roundtrip_rms",
                operator=ComparisonOperator.LT,
                threshold=1e-6,
            ),
            GateCheck(
                name="jvp_median",
                metric="errors.jvp_median",
                operator=ComparisonOperator.LT,
                threshold=1e-2,
            ),
        ),
    )


def passing_metrics() -> dict[str, object]:
    return {
        "identity_exact": True,
        "errors": {"roundtrip_rms": 1e-7, "jvp_median": 2e-3},
    }


def test_gate_evaluator_passes_and_serializes_a_declarative_rule() -> None:
    evaluator = GateEvaluator()
    rule = mechanical_rule()
    result = evaluator.evaluate(rule, passing_metrics())

    assert result.status is GateStatus.PASS
    assert result.natural_status is GateStatus.PASS
    assert result.passed_count == result.required_passes == 3
    assert result.can_proceed
    assert all(check.status is CheckStatus.PASS for check in result.checks)
    assert GateRule.from_dict(rule.to_dict()) == rule
    assert GateResult.from_dict(result.to_dict()) == result
    canonical_json_bytes(result.to_dict())


def test_result_deserializers_accept_only_their_exact_to_dict_schema() -> None:
    result = GateEvaluator().evaluate(mechanical_rule(), passing_metrics())
    check_payload = result.checks[0].to_dict()
    override = GateOverride(
        gate_id="mechanical",
        scope="gate",
        reason="diagnostic continuation",
    )

    assert GateCheckResult.from_dict(check_payload) == result.checks[0]
    assert GateOverride.from_dict(override.to_dict()) == override

    for parser, payload in (
        (GateCheckResult.from_dict, check_payload),
        (GateOverride.from_dict, override.to_dict()),
        (GateResult.from_dict, result.to_dict()),
    ):
        missing = dict(payload)
        missing.pop(next(iter(missing)))
        with pytest.raises(GateEvaluationError, match="keys must be exactly"):
            parser(missing)
        extra = {**payload, "unexpected": True}
        with pytest.raises(GateEvaluationError, match="keys must be exactly"):
            parser(extra)


def test_gate_result_deserialization_rejects_nonfinite_and_inconsistent_evidence() -> None:
    result = GateEvaluator().evaluate(mechanical_rule(), passing_metrics())

    nonfinite = result.to_dict()
    nonfinite["checks"][0]["observed"] = math.nan
    with pytest.raises(GateEvaluationError, match="finite canonical JSON"):
        GateResult.from_dict(nonfinite)

    wrong_check_status = result.to_dict()
    wrong_check_status["checks"][0]["status"] = "FAIL"
    with pytest.raises(GateEvaluationError, match="inconsistent with its observed"):
        GateResult.from_dict(wrong_check_status)

    wrong_count = result.to_dict()
    wrong_count["passed_count"] = 2
    with pytest.raises(GateEvaluationError, match="counts do not match"):
        GateResult.from_dict(wrong_count)

    wrong_status = result.to_dict()
    wrong_status["status"] = "FAIL"
    with pytest.raises(GateEvaluationError, match="status is inconsistent"):
        GateResult.from_dict(wrong_status)


def test_gate_result_deserialization_rejects_inconsistent_override_lineage() -> None:
    overridden = GateEvaluator().evaluate(
        mechanical_rule(),
        {
            "identity_exact": False,
            "errors": {"roundtrip_rms": 1.0, "jvp_median": 1.0},
        },
        override_reason="diagnostic continuation",
    )
    assert GateResult.from_dict(overridden.to_dict()) == overridden

    missing_lineage = overridden.to_dict()
    missing_lineage["override_lineage"] = []
    with pytest.raises(GateEvaluationError, match="override_reason does not match"):
        GateResult.from_dict(missing_lineage)

    wrong_reason = overridden.to_dict()
    wrong_reason["override_lineage"][0]["reason"] = "different reason"
    with pytest.raises(GateEvaluationError, match="override_reason does not match"):
        GateResult.from_dict(wrong_reason)


def test_gate_evaluator_distinguishes_failure_from_missing_data() -> None:
    evaluator = GateEvaluator()
    rule = mechanical_rule()
    failed = passing_metrics()
    failed["identity_exact"] = False

    failure = evaluator.evaluate(rule, failed)
    missing = evaluator.evaluate(rule, {"identity_exact": True})

    assert failure.status is GateStatus.FAIL
    assert failure.checks[0].status is CheckStatus.FAIL
    assert missing.status is GateStatus.NOT_EVALUABLE
    assert missing.passed_count == 1
    assert [check.status for check in missing.checks] == [
        CheckStatus.PASS,
        CheckStatus.NOT_EVALUABLE,
        CheckStatus.NOT_EVALUABLE,
    ]


def test_n_of_m_gate_handles_missing_metrics_logically() -> None:
    rule = GateRule(
        gate_id="coarse",
        checks=tuple(
            GateCheck(name=f"cut_{index}", metric=f"cuts.{index}", operator="is_true")
            for index in range(4)
        ),
        min_passes=2,
    )
    evaluator = GateEvaluator()

    assert evaluator.evaluate(rule, {"cuts": {"0": True, "1": True}}).status is GateStatus.PASS
    partial = evaluator.evaluate(rule, {"cuts": {"0": True, "1": False}})
    assert partial.status is GateStatus.NOT_EVALUABLE
    assert (
        evaluator.evaluate(rule, {"cuts": {"0": False, "1": False, "2": False}}).status
        is GateStatus.FAIL
    )


def test_non_counting_diagnostic_check_does_not_block_gate() -> None:
    rule = GateRule(
        gate_id="diagnostic",
        checks=(
            GateCheck(name="required", metric="required", operator="is_true"),
            GateCheck(
                name="optional",
                metric="optional",
                operator="is_true",
                counts_toward_gate=False,
            ),
        ),
    )
    result = GateEvaluator().evaluate(rule, {"required": True})

    assert result.status is GateStatus.PASS
    assert result.checks[1].status is CheckStatus.NOT_EVALUABLE


def test_dependency_failure_blocks_and_requires_an_explicit_reason_to_override() -> None:
    prerequisite = GateEvaluator().evaluate(
        GateRule(
            gate_id="coarse",
            checks=(GateCheck(name="effect", metric="effect", operator="is_true"),),
        ),
        {"effect": False},
    )
    downstream_rule = GateRule(
        gate_id="functional",
        dependencies=("coarse",),
        checks=(GateCheck(name="snap", metric="snap", operator="is_true"),),
    )

    blocked = GateEvaluator().evaluate(
        downstream_rule, {"snap": True}, dependencies={"coarse": prerequisite}
    )
    overridden = GateEvaluator().evaluate(
        downstream_rule,
        {"snap": True},
        dependencies={"coarse": prerequisite},
        dependency_override_reason="diagnostic run requested by investigator",
    )

    assert blocked.status is GateStatus.NOT_EVALUABLE
    assert blocked.natural_status is GateStatus.PASS
    assert blocked.blockers == ("coarse:FAIL",)
    assert not blocked.can_proceed
    assert overridden.status is GateStatus.OVERRIDDEN
    assert overridden.can_proceed
    assert overridden.dependency_override_reason == "diagnostic run requested by investigator"
    assert overridden.blockers == ("coarse:FAIL",)


def test_inherited_override_taints_a_passing_descendant_but_not_a_failure() -> None:
    overridden_parent = GateEvaluator().evaluate(
        GateRule(
            gate_id="coarse",
            checks=(GateCheck(name="effect", metric="effect", operator="is_true"),),
        ),
        {"effect": False},
        override_reason="exploratory continuation",
    )
    child_rule = GateRule(
        gate_id="functional",
        dependencies=("coarse",),
        checks=(GateCheck(name="snap", metric="snap", operator="is_true"),),
    )

    passing_child = GateEvaluator().evaluate(
        child_rule, {"snap": True}, dependencies={"coarse": overridden_parent}
    )
    failing_child = GateEvaluator().evaluate(
        child_rule, {"snap": False}, dependencies={"coarse": overridden_parent}
    )

    assert passing_child.status is GateStatus.OVERRIDDEN
    assert passing_child.inherited_overrides == ("coarse",)
    assert failing_child.status is GateStatus.FAIL
    assert failing_child.inherited_overrides == ("coarse",)


def test_direct_override_preserves_natural_failure() -> None:
    result = GateEvaluator().evaluate(
        mechanical_rule(),
        {"identity_exact": False, "errors": {"roundtrip_rms": 1.0, "jvp_median": 1.0}},
        override_reason="exercise report path only",
    )

    assert result.status is GateStatus.OVERRIDDEN
    assert result.natural_status is GateStatus.FAIL
    assert result.passed_count == 0


def test_dependency_override_does_not_override_the_current_gate() -> None:
    prerequisite = GateEvaluator().evaluate(
        GateRule(
            gate_id="coarse",
            checks=(GateCheck(name="effect", metric="effect", operator="is_true"),),
        ),
        {"effect": False},
    )
    child = GateRule(
        gate_id="functional",
        dependencies=("coarse",),
        checks=(GateCheck(name="snap", metric="snap", operator="is_true"),),
    )

    dependency_only = GateEvaluator().evaluate(
        child,
        {"snap": False},
        dependencies={"coarse": prerequisite},
        dependency_override_reason="inspect functional failure despite coarse failure",
    )
    both = GateEvaluator().evaluate(
        child,
        {"snap": False},
        dependencies={"coarse": prerequisite},
        dependency_override_reason="inspect functional failure despite coarse failure",
        override_reason="continue past the observed functional failure",
    )

    assert dependency_only.status is GateStatus.FAIL
    assert not dependency_only.can_proceed
    assert both.status is GateStatus.OVERRIDDEN
    assert both.can_proceed
    assert [entry.scope.value for entry in both.override_lineage] == ["dependencies", "gate"]


def test_override_lineage_survives_multiple_dependency_generations() -> None:
    first = GateEvaluator().evaluate(
        GateRule(
            gate_id="coarse",
            checks=(GateCheck(name="effect", metric="effect", operator="is_true"),),
        ),
        {"effect": False},
        override_reason="exploratory continuation",
    )
    second = GateEvaluator().evaluate(
        GateRule(
            gate_id="functional",
            dependencies=("coarse",),
            checks=(GateCheck(name="snap", metric="snap", operator="is_true"),),
        ),
        {"snap": True},
        dependencies={"coarse": first},
    )
    third = GateEvaluator().evaluate(
        GateRule(
            gate_id="real-algebra",
            dependencies=("functional",),
            checks=(GateCheck(name="compression", metric="compression", operator="is_true"),),
        ),
        {"compression": True},
        dependencies={"functional": second},
    )

    assert third.status is GateStatus.OVERRIDDEN
    assert third.inherited_overrides == ("coarse",)
    assert third.override_lineage[0].reason == "exploratory continuation"


def test_ancestral_override_survives_a_failing_child_and_later_dependency_override() -> None:
    coarse = GateEvaluator().evaluate(
        GateRule(
            gate_id="coarse",
            checks=(GateCheck(name="effect", metric="effect", operator="is_true"),),
        ),
        {"effect": False},
        override_reason="exploratory coarse continuation",
    )
    functional = GateEvaluator().evaluate(
        GateRule(
            gate_id="functional",
            dependencies=("coarse",),
            checks=(GateCheck(name="snap", metric="snap", operator="is_true"),),
        ),
        {"snap": False},
        dependencies={"coarse": coarse},
    )
    assert functional.status is GateStatus.FAIL

    algebra = GateEvaluator().evaluate(
        GateRule(
            gate_id="real-algebra",
            dependencies=("functional",),
            checks=(GateCheck(name="compression", metric="compression", operator="is_true"),),
        ),
        {"compression": True},
        dependencies={"functional": functional},
        dependency_override_reason="inspect algebra after functional failure",
    )

    assert algebra.status is GateStatus.OVERRIDDEN
    assert algebra.inherited_overrides == ("coarse",)
    assert [entry.gate_id for entry in algebra.override_lineage] == [
        "coarse",
        "real-algebra",
    ]


def test_invalid_observations_are_not_evaluable_instead_of_silently_failing() -> None:
    rule = GateRule(
        gate_id="finite",
        checks=(GateCheck(name="metric", metric="value", operator="lt", threshold=1.0),),
    )

    nan_result = GateEvaluator().evaluate(rule, {"value": math.nan})
    string_result = GateEvaluator().evaluate(rule, {"value": "0.1"})

    assert nan_result.status is GateStatus.NOT_EVALUABLE
    assert string_result.status is GateStatus.NOT_EVALUABLE
    assert "invalid" in nan_result.checks[0].detail


def test_gate_schema_and_override_reason_are_strict() -> None:
    with pytest.raises(GateEvaluationError, match="permits only"):
        GateCheck.from_dict({"name": "x", "metric": "x", "operator": "is_true", "typo": 1})
    with pytest.raises(GateEvaluationError, match="ascending"):
        GateCheck(name="range", metric="x", operator="between_closed", threshold=[2, 1])
    with pytest.raises(GateEvaluationError, match="non-empty"):
        evaluate_dependencies(["coarse"], {"coarse": GateStatus.FAIL}, override_reason="  ")
    with pytest.raises(GateEvaluationError, match="cannot depend"):
        GateRule(
            gate_id="self",
            dependencies=("self",),
            checks=(GateCheck(name="x", metric="x", operator="is_true"),),
        )
    with pytest.raises(GateEvaluationError, match="no GateResult reason"):
        evaluate_dependencies(["coarse"], {"coarse": GateStatus.OVERRIDDEN})


def test_gate_checks_snapshot_mutable_thresholds() -> None:
    threshold = [0.0, 1.0]
    check = GateCheck(name="range", metric="value", operator="between_closed", threshold=threshold)
    rule = GateRule(gate_id="frozen", checks=(check,))

    threshold[1] = 3.0

    assert GateEvaluator().evaluate(rule, {"value": 2.0}).status is GateStatus.FAIL
    assert check.to_dict()["threshold"] == [0.0, 1.0]
    with pytest.raises(TypeError):
        check.threshold[1] = 3.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("operator", "observed", "threshold", "expected"),
    [
        ("lt", 0.9, 1.0, True),
        ("le", 1.0, 1.0, True),
        ("gt", 1.1, 1.0, True),
        ("ge", 1.0, 1.0, True),
        ("eq", "pass", "pass", True),
        ("ne", "fail", "pass", True),
        ("between_closed", 1.0, [0.0, 1.0], True),
        ("is_false", False, None, True),
    ],
)
def test_supported_comparison_operators(operator, observed, threshold, expected) -> None:
    check = GateCheck(name="check", metric="value", operator=operator, threshold=threshold)
    rule = GateRule(gate_id="operators", checks=(check,))
    result = GateEvaluator().evaluate(rule, {"value": observed})
    assert (result.status is GateStatus.PASS) is expected
