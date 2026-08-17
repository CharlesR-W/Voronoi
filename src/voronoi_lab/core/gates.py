"""Declarative scientific gates and explicit dependency overrides."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .hashing import CanonicalJSONError, JSONLike, JSONValue, freeze_json, thaw_json

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")


class GateEvaluationError(ValueError):
    """Raised for an invalid gate declaration or unsafe override."""


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    OVERRIDDEN = "OVERRIDDEN"


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class ComparisonOperator(StrEnum):
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    EQ = "eq"
    NE = "ne"
    BETWEEN_CLOSED = "between_closed"
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"


class OverrideScope(StrEnum):
    GATE = "gate"
    DEPENDENCIES = "dependencies"


_NUMERIC_OPERATORS = {
    ComparisonOperator.LT,
    ComparisonOperator.LE,
    ComparisonOperator.GT,
    ComparisonOperator.GE,
}


def _validate_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise GateEvaluationError(f"{label} must match [A-Za-z0-9][A-Za-z0-9._/-]{{0,127}}")
    return value


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _json_normalize(value: JSONLike) -> JSONValue:
    return thaw_json(value)


def _freeze_gate_json(value: object, *, label: str) -> JSONLike:
    """Snapshot one JSON field while keeping gate errors at the public boundary."""

    try:
        return freeze_json(value)  # type: ignore[arg-type]
    except (CanonicalJSONError, TypeError) as exc:
        raise GateEvaluationError(f"{label} must be finite canonical JSON") from exc


def _require_exact_object(
    value: object,
    *,
    keys: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GateEvaluationError(f"{label} must be an object")
    if set(value) != keys:
        raise GateEvaluationError(f"{label} keys must be exactly {sorted(keys)}")
    return value


@dataclass(frozen=True, slots=True)
class GateCheck:
    """One metric comparison within a gate."""

    name: str
    metric: str
    operator: ComparisonOperator
    threshold: JSONValue = None
    counts_toward_gate: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        _validate_identifier(self.name, label="check name")
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise GateEvaluationError("check metric must be a non-empty string")
        try:
            operator = ComparisonOperator(self.operator)
        except ValueError as exc:
            raise GateEvaluationError(f"unknown comparison operator: {self.operator!r}") from exc
        object.__setattr__(self, "operator", operator)
        if not isinstance(self.counts_toward_gate, bool):
            raise GateEvaluationError("counts_toward_gate must be boolean")
        if not isinstance(self.description, str):
            raise GateEvaluationError("check description must be a string")
        normalized_threshold = freeze_json(self.threshold)
        object.__setattr__(self, "threshold", normalized_threshold)

        if operator in _NUMERIC_OPERATORS and not _is_finite_number(self.threshold):
            raise GateEvaluationError(
                f"operator {operator.value} requires a finite numeric threshold"
            )
        if operator is ComparisonOperator.BETWEEN_CLOSED:
            threshold = self.threshold
            if (
                not isinstance(threshold, Sequence)
                or isinstance(threshold, (str, bytes))
                or len(threshold) != 2
                or not all(_is_finite_number(item) for item in threshold)
                or threshold[0] > threshold[1]
            ):
                raise GateEvaluationError(
                    "between_closed requires a two-element ascending numeric threshold list"
                )
        if (
            operator in {ComparisonOperator.IS_TRUE, ComparisonOperator.IS_FALSE}
            and self.threshold is not None
        ):
            raise GateEvaluationError(f"operator {operator.value} does not accept a threshold")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "counts_toward_gate": self.counts_toward_gate,
            "description": self.description,
            "metric": self.metric,
            "name": self.name,
            "operator": self.operator.value,
            "threshold": thaw_json(self.threshold),
        }

    @classmethod
    def from_dict(cls, value: object) -> GateCheck:
        if not isinstance(value, dict):
            raise GateEvaluationError("gate check must be an object")
        required = {"name", "metric", "operator"}
        allowed = required | {"threshold", "counts_toward_gate", "description"}
        if not required.issubset(value) or not set(value).issubset(allowed):
            raise GateEvaluationError(
                f"gate check requires {sorted(required)} and permits only {sorted(allowed)}"
            )
        return cls(
            name=value["name"],
            metric=value["metric"],
            operator=value["operator"],
            threshold=value.get("threshold"),
            counts_toward_gate=value.get("counts_toward_gate", True),
            description=value.get("description", ""),
        )


@dataclass(frozen=True, slots=True)
class GateRule:
    """A deterministic N-of-M gate with prerequisite gate ids."""

    gate_id: str
    checks: tuple[GateCheck, ...]
    dependencies: tuple[str, ...] = ()
    min_passes: int | None = None
    description: str = ""

    def __post_init__(self) -> None:
        _validate_identifier(self.gate_id, label="gate_id")
        if not isinstance(self.checks, Sequence) or isinstance(self.checks, (str, bytes)):
            raise GateEvaluationError("gate checks must be a sequence of GateCheck objects")
        checks = tuple(self.checks)
        if not checks or not all(isinstance(check, GateCheck) for check in checks):
            raise GateEvaluationError("a gate must declare at least one check")
        object.__setattr__(self, "checks", checks)
        if not isinstance(self.dependencies, Sequence) or isinstance(
            self.dependencies, (str, bytes)
        ):
            raise GateEvaluationError("gate dependencies must be a sequence of strings")
        dependencies = tuple(self.dependencies)
        object.__setattr__(self, "dependencies", dependencies)
        check_names = [check.name for check in checks]
        if len(check_names) != len(set(check_names)):
            raise GateEvaluationError("gate check names must be unique")
        for dependency in dependencies:
            _validate_identifier(dependency, label="dependency gate id")
        if self.gate_id in dependencies:
            raise GateEvaluationError("a gate cannot depend on itself")
        if len(dependencies) != len(set(dependencies)):
            raise GateEvaluationError("gate dependencies must be unique")
        counted = sum(check.counts_toward_gate for check in checks)
        if counted == 0:
            raise GateEvaluationError("a gate must have at least one counting check")
        minimum = counted if self.min_passes is None else self.min_passes
        if isinstance(minimum, bool) or not isinstance(minimum, int) or not 1 <= minimum <= counted:
            raise GateEvaluationError(f"min_passes must be between 1 and {counted}")
        object.__setattr__(self, "min_passes", minimum)
        if not isinstance(self.description, str):
            raise GateEvaluationError("gate description must be a string")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "checks": [check.to_dict() for check in self.checks],
            "dependencies": list(self.dependencies),
            "description": self.description,
            "gate_id": self.gate_id,
            "min_passes": self.min_passes,
        }

    @classmethod
    def from_dict(cls, value: object) -> GateRule:
        if not isinstance(value, dict):
            raise GateEvaluationError("gate rule must be an object")
        required = {"gate_id", "checks"}
        allowed = required | {"dependencies", "min_passes", "description"}
        if not required.issubset(value) or not set(value).issubset(allowed):
            raise GateEvaluationError(
                f"gate rule requires {sorted(required)} and permits only {sorted(allowed)}"
            )
        checks = value["checks"]
        dependencies = value.get("dependencies", [])
        if not isinstance(checks, list):
            raise GateEvaluationError("gate checks must be a list")
        if not isinstance(dependencies, list) or not all(
            isinstance(item, str) for item in dependencies
        ):
            raise GateEvaluationError("gate dependencies must be a list of strings")
        return cls(
            gate_id=value["gate_id"],
            checks=tuple(GateCheck.from_dict(check) for check in checks),
            dependencies=tuple(dependencies),
            min_passes=value.get("min_passes"),
            description=value.get("description", ""),
        )


@dataclass(frozen=True, slots=True)
class GateCheckResult:
    name: str
    metric: str
    status: CheckStatus
    observed: JSONValue
    operator: ComparisonOperator
    threshold: JSONValue
    counts_toward_gate: bool
    detail: str

    def __post_init__(self) -> None:
        _validate_identifier(self.name, label="check result name")
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise GateEvaluationError("check result metric must be a non-empty string")
        try:
            status = CheckStatus(self.status)
            operator = ComparisonOperator(self.operator)
        except (TypeError, ValueError) as exc:
            raise GateEvaluationError("invalid check result status or operator") from exc
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "operator", operator)
        observed = _freeze_gate_json(self.observed, label="check result observed value")
        threshold = _freeze_gate_json(self.threshold, label="check result threshold")
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "threshold", threshold)
        if not isinstance(self.counts_toward_gate, bool):
            raise GateEvaluationError("check result counting flag must be boolean")
        if not isinstance(self.detail, str) or not self.detail:
            raise GateEvaluationError("check result detail must be a non-empty string")
        # A serialized result is evidence, not merely a status label.  Rebuild the
        # corresponding check so operator/threshold constraints are revalidated.
        GateCheck(
            name=self.name,
            metric=self.metric,
            operator=operator,
            threshold=threshold,
            counts_toward_gate=self.counts_toward_gate,
        )
        if status is CheckStatus.NOT_EVALUABLE:
            if observed is not None:
                raise GateEvaluationError(
                    "a NOT_EVALUABLE check result must not retain an observed value"
                )
        else:
            try:
                comparison_passed = _compare(operator, thaw_json(observed), threshold)
            except (TypeError, ValueError) as exc:
                raise GateEvaluationError(
                    "an evaluable check result must contain a compatible observed value"
                ) from exc
            expected_status = CheckStatus.PASS if comparison_passed else CheckStatus.FAIL
            if status is not expected_status:
                raise GateEvaluationError(
                    "check result status is inconsistent with its observed comparison"
                )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "counts_toward_gate": self.counts_toward_gate,
            "detail": self.detail,
            "metric": self.metric,
            "name": self.name,
            "observed": thaw_json(self.observed),
            "operator": self.operator.value,
            "status": self.status.value,
            "threshold": thaw_json(self.threshold),
        }

    @classmethod
    def from_dict(cls, value: object) -> GateCheckResult:
        raw = _require_exact_object(
            value,
            keys={
                "counts_toward_gate",
                "detail",
                "metric",
                "name",
                "observed",
                "operator",
                "status",
                "threshold",
            },
            label="gate check result",
        )
        return cls(
            name=raw["name"],  # type: ignore[arg-type]
            metric=raw["metric"],  # type: ignore[arg-type]
            status=raw["status"],  # type: ignore[arg-type]
            observed=raw["observed"],  # type: ignore[arg-type]
            operator=raw["operator"],  # type: ignore[arg-type]
            threshold=raw["threshold"],  # type: ignore[arg-type]
            counts_toward_gate=raw["counts_toward_gate"],  # type: ignore[arg-type]
            detail=raw["detail"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class GateOverride:
    """One explicit override decision retained across the full gate lineage."""

    gate_id: str
    scope: OverrideScope
    reason: str
    targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.gate_id, label="override gate id")
        try:
            scope = OverrideScope(self.scope)
        except (TypeError, ValueError) as exc:
            raise GateEvaluationError(f"invalid override scope: {self.scope!r}") from exc
        object.__setattr__(self, "scope", scope)
        reason = _normalize_override_reason(self.reason)
        if reason is None:
            raise GateEvaluationError("an override reason is required")
        object.__setattr__(self, "reason", reason)
        if not isinstance(self.targets, Sequence) or isinstance(self.targets, (str, bytes)):
            raise GateEvaluationError("override targets must be a sequence of gate ids")
        targets = tuple(self.targets)
        for target in targets:
            _validate_identifier(target, label="override target gate id")
        if len(targets) != len(set(targets)):
            raise GateEvaluationError("override target gate ids must be unique")
        if scope is OverrideScope.DEPENDENCIES and not targets:
            raise GateEvaluationError("a dependency override must identify its target gates")
        if scope is OverrideScope.GATE and targets:
            raise GateEvaluationError("a direct gate override cannot have dependency targets")
        object.__setattr__(self, "targets", targets)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "gate_id": self.gate_id,
            "reason": self.reason,
            "scope": self.scope.value,
            "targets": list(self.targets),
        }

    @classmethod
    def from_dict(cls, value: object) -> GateOverride:
        raw = _require_exact_object(
            value,
            keys={"gate_id", "reason", "scope", "targets"},
            label="gate override",
        )
        targets = raw["targets"]
        if not isinstance(targets, list):
            raise GateEvaluationError("gate override targets must be a list")
        return cls(
            gate_id=raw["gate_id"],  # type: ignore[arg-type]
            scope=raw["scope"],  # type: ignore[arg-type]
            reason=raw["reason"],  # type: ignore[arg-type]
            targets=tuple(targets),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DependencyDecision:
    allowed: bool
    blockers: tuple[str, ...]
    inherited_overrides: tuple[str, ...]
    override_lineage: tuple[GateOverride, ...]
    override_reason: str | None

    @property
    def overridden(self) -> bool:
        return bool(self.inherited_overrides or (self.blockers and self.override_reason))


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    status: GateStatus
    natural_status: GateStatus
    checks: tuple[GateCheckResult, ...]
    passed_count: int
    required_passes: int
    blockers: tuple[str, ...] = ()
    inherited_overrides: tuple[str, ...] = ()
    override_reason: str | None = None
    dependency_override_reason: str | None = None
    override_lineage: tuple[GateOverride, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.gate_id, label="gate result id")
        try:
            status = GateStatus(self.status)
            natural_status = GateStatus(self.natural_status)
        except (TypeError, ValueError) as exc:
            raise GateEvaluationError("invalid gate result status") from exc
        if natural_status is GateStatus.OVERRIDDEN:
            raise GateEvaluationError("natural_status cannot itself be OVERRIDDEN")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "natural_status", natural_status)
        if not isinstance(self.checks, Sequence) or isinstance(self.checks, (str, bytes)):
            raise GateEvaluationError("gate result checks must be a sequence")
        checks = tuple(self.checks)
        if not checks or not all(isinstance(check, GateCheckResult) for check in checks):
            raise GateEvaluationError("gate result checks must be GateCheckResult objects")
        if len({check.name for check in checks}) != len(checks):
            raise GateEvaluationError("gate result check names must be unique")
        object.__setattr__(self, "checks", checks)
        if (
            isinstance(self.passed_count, bool)
            or not isinstance(self.passed_count, int)
            or self.passed_count < 0
        ):
            raise GateEvaluationError("passed_count must be a non-negative integer")
        if (
            isinstance(self.required_passes, bool)
            or not isinstance(self.required_passes, int)
            or self.required_passes < 1
        ):
            raise GateEvaluationError("invalid required_passes or passed_count")
        counted = [check for check in checks if check.counts_toward_gate]
        actual_passes = sum(check.status is CheckStatus.PASS for check in counted)
        missing_count = sum(check.status is CheckStatus.NOT_EVALUABLE for check in counted)
        if self.required_passes > len(counted) or self.passed_count != actual_passes:
            raise GateEvaluationError("gate result counts do not match its check results")
        if actual_passes >= self.required_passes:
            expected_natural = GateStatus.PASS
        elif actual_passes + missing_count < self.required_passes:
            expected_natural = GateStatus.FAIL
        else:
            expected_natural = GateStatus.NOT_EVALUABLE
        if natural_status is not expected_natural:
            raise GateEvaluationError("natural_status does not match the check results")
        if not isinstance(self.blockers, Sequence) or isinstance(self.blockers, (str, bytes)):
            raise GateEvaluationError("gate blockers must be a sequence")
        if not isinstance(self.inherited_overrides, Sequence) or isinstance(
            self.inherited_overrides, (str, bytes)
        ):
            raise GateEvaluationError("inherited overrides must be a sequence")
        blockers = tuple(self.blockers)
        inherited = tuple(self.inherited_overrides)
        blocker_targets: list[str] = []
        for blocker in blockers:
            if not isinstance(blocker, str) or ":" not in blocker:
                raise GateEvaluationError("gate blockers must be '<gate_id>:<status>' strings")
            blocker_gate, blocker_status = blocker.rsplit(":", 1)
            _validate_identifier(blocker_gate, label="blocker gate id")
            if blocker_status not in {
                GateStatus.FAIL.value,
                GateStatus.NOT_EVALUABLE.value,
                "MISSING",
            }:
                raise GateEvaluationError("gate blocker has an invalid blocking status")
            blocker_targets.append(blocker_gate)
        if len(blockers) != len(set(blockers)):
            raise GateEvaluationError("gate blockers must be unique")
        for gate_id in inherited:
            _validate_identifier(gate_id, label="inherited override gate id")
        if len(inherited) != len(set(inherited)):
            raise GateEvaluationError("inherited override gate ids must be unique")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "inherited_overrides", inherited)
        reason = _normalize_override_reason(self.override_reason)
        dependency_reason = _normalize_override_reason(self.dependency_override_reason)
        object.__setattr__(self, "override_reason", reason)
        object.__setattr__(self, "dependency_override_reason", dependency_reason)
        if not isinstance(self.override_lineage, Sequence) or isinstance(
            self.override_lineage, (str, bytes)
        ):
            raise GateEvaluationError("override_lineage must be a sequence")
        lineage = tuple(self.override_lineage)
        if not all(isinstance(item, GateOverride) for item in lineage):
            raise GateEvaluationError("override_lineage must contain GateOverride objects")
        if len(lineage) != len(set(lineage)):
            raise GateEvaluationError("override_lineage must not contain duplicate decisions")
        object.__setattr__(self, "override_lineage", lineage)
        expected_inherited = tuple(
            dict.fromkeys(item.gate_id for item in lineage if item.gate_id != self.gate_id)
        )
        if inherited != expected_inherited:
            raise GateEvaluationError("inherited_overrides does not match override_lineage")

        own_gate_overrides = tuple(
            item
            for item in lineage
            if item.gate_id == self.gate_id and item.scope is OverrideScope.GATE
        )
        own_dependency_overrides = tuple(
            item
            for item in lineage
            if item.gate_id == self.gate_id and item.scope is OverrideScope.DEPENDENCIES
        )
        if reason is None:
            if own_gate_overrides:
                raise GateEvaluationError(
                    "override_lineage contains an unreported direct gate override"
                )
        elif (
            len(own_gate_overrides) != 1
            or own_gate_overrides[0].reason != reason
            or natural_status is GateStatus.PASS
        ):
            raise GateEvaluationError(
                "override_reason does not match one valid direct override decision"
            )
        if dependency_reason is not None and not blockers:
            raise GateEvaluationError("dependency override reason requires blocked dependencies")
        if dependency_reason is None:
            if own_dependency_overrides:
                raise GateEvaluationError(
                    "override_lineage contains an unreported dependency override"
                )
        elif (
            len(own_dependency_overrides) != 1
            or own_dependency_overrides[0].reason != dependency_reason
            or own_dependency_overrides[0].targets != tuple(blocker_targets)
        ):
            raise GateEvaluationError(
                "dependency_override_reason does not match its override lineage and blockers"
            )
        if blockers and dependency_reason is None:
            if reason is not None:
                raise GateEvaluationError(
                    "a direct gate override cannot bypass blocked dependencies"
                )
            expected_status = GateStatus.NOT_EVALUABLE
        elif reason is not None or (lineage and natural_status is GateStatus.PASS):
            expected_status = GateStatus.OVERRIDDEN
        else:
            expected_status = natural_status
        if status is not expected_status:
            raise GateEvaluationError(
                "gate result status is inconsistent with evidence, blockers, and override lineage"
            )

    @property
    def can_proceed(self) -> bool:
        return self.status in {GateStatus.PASS, GateStatus.OVERRIDDEN}

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "blockers": list(self.blockers),
            "checks": [check.to_dict() for check in self.checks],
            "gate_id": self.gate_id,
            "inherited_overrides": list(self.inherited_overrides),
            "natural_status": self.natural_status.value,
            "dependency_override_reason": self.dependency_override_reason,
            "override_lineage": [item.to_dict() for item in self.override_lineage],
            "override_reason": self.override_reason,
            "passed_count": self.passed_count,
            "required_passes": self.required_passes,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> GateResult:
        raw = _require_exact_object(
            value,
            keys={
                "blockers",
                "checks",
                "dependency_override_reason",
                "gate_id",
                "inherited_overrides",
                "natural_status",
                "override_lineage",
                "override_reason",
                "passed_count",
                "required_passes",
                "status",
            },
            label="gate result",
        )
        checks = raw["checks"]
        blockers = raw["blockers"]
        inherited = raw["inherited_overrides"]
        lineage = raw["override_lineage"]
        if not isinstance(checks, list):
            raise GateEvaluationError("gate result checks must be a list")
        if not isinstance(blockers, list):
            raise GateEvaluationError("gate result blockers must be a list")
        if not isinstance(inherited, list):
            raise GateEvaluationError("gate result inherited_overrides must be a list")
        if not isinstance(lineage, list):
            raise GateEvaluationError("gate result override_lineage must be a list")
        return cls(
            gate_id=raw["gate_id"],  # type: ignore[arg-type]
            status=raw["status"],  # type: ignore[arg-type]
            natural_status=raw["natural_status"],  # type: ignore[arg-type]
            checks=tuple(GateCheckResult.from_dict(check) for check in checks),
            passed_count=raw["passed_count"],  # type: ignore[arg-type]
            required_passes=raw["required_passes"],  # type: ignore[arg-type]
            blockers=tuple(blockers),  # type: ignore[arg-type]
            inherited_overrides=tuple(inherited),  # type: ignore[arg-type]
            override_reason=raw["override_reason"],  # type: ignore[arg-type]
            dependency_override_reason=raw["dependency_override_reason"],  # type: ignore[arg-type]
            override_lineage=tuple(GateOverride.from_dict(item) for item in lineage),
        )


def evaluate_dependencies(
    required: Sequence[str],
    results: Mapping[str, GateResult | GateStatus | str],
    *,
    override_reason: str | None = None,
) -> DependencyDecision:
    """Decide whether prerequisite gates allow a downstream stage to run."""

    reason = _normalize_override_reason(override_reason)
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        raise GateEvaluationError("required dependencies must be a sequence of gate ids")
    for gate_id in required:
        _validate_identifier(gate_id, label="dependency gate id")
    if len(required) != len(set(required)):
        raise GateEvaluationError("dependency gate ids must be unique")
    blockers: list[str] = []
    inherited_overrides: list[str] = []
    override_lineage: list[GateOverride] = []
    for gate_id in required:
        if gate_id not in results:
            blockers.append(f"{gate_id}:MISSING")
            continue
        result = results[gate_id]
        if isinstance(result, GateResult):
            if result.gate_id != gate_id:
                raise GateEvaluationError(
                    f"dependency mapping key {gate_id!r} contains result for {result.gate_id!r}"
                )
            status = result.status
            # A descendant can naturally fail while still carrying an ancestral
            # override. Preserve that taint even when this immediate result is not
            # itself OVERRIDDEN, so later dependency overrides cannot erase history.
            override_lineage.extend(result.override_lineage)
            inherited_overrides.extend(item.gate_id for item in result.override_lineage)
        else:
            try:
                status = GateStatus(result)
            except ValueError as exc:
                raise GateEvaluationError(
                    f"invalid dependency status for {gate_id}: {result!r}"
                ) from exc
            if status is GateStatus.OVERRIDDEN:
                raise GateEvaluationError(
                    f"dependency {gate_id!r} is OVERRIDDEN but has no GateResult reason"
                )
        if status not in {GateStatus.PASS, GateStatus.OVERRIDDEN}:
            blockers.append(f"{gate_id}:{status.value}")

    if blockers and reason is None:
        return DependencyDecision(
            allowed=False,
            blockers=tuple(blockers),
            inherited_overrides=tuple(dict.fromkeys(inherited_overrides)),
            override_lineage=tuple(dict.fromkeys(override_lineage)),
            override_reason=None,
        )
    if reason is not None and not blockers:
        raise GateEvaluationError("dependency override supplied but no dependency is blocked")
    return DependencyDecision(
        allowed=True,
        blockers=tuple(blockers),
        inherited_overrides=tuple(dict.fromkeys(inherited_overrides)),
        override_lineage=tuple(dict.fromkeys(override_lineage)),
        override_reason=reason,
    )


class GateEvaluator:
    """Evaluate strict declarative rules against a metric mapping."""

    def evaluate(
        self,
        rule: GateRule | Mapping[str, object],
        observations: Mapping[str, object],
        *,
        dependencies: Mapping[str, GateResult | GateStatus | str] | None = None,
        override_reason: str | None = None,
        dependency_override_reason: str | None = None,
    ) -> GateResult:
        normalized_rule = rule if isinstance(rule, GateRule) else GateRule.from_dict(rule)
        gate_reason = _normalize_override_reason(override_reason)
        dependency_reason = _normalize_override_reason(dependency_override_reason)
        check_results = tuple(
            self._evaluate_check(check, observations) for check in normalized_rule.checks
        )
        counted = [result for result in check_results if result.counts_toward_gate]
        passed_count = sum(result.status is CheckStatus.PASS for result in counted)
        missing_count = sum(result.status is CheckStatus.NOT_EVALUABLE for result in counted)
        required_passes = normalized_rule.min_passes
        assert required_passes is not None
        if passed_count >= required_passes:
            natural_status = GateStatus.PASS
        elif passed_count + missing_count < required_passes:
            natural_status = GateStatus.FAIL
        else:
            natural_status = GateStatus.NOT_EVALUABLE
        if gate_reason is not None and natural_status is GateStatus.PASS:
            raise GateEvaluationError("gate override supplied but the gate naturally passes")

        dependency_decision = evaluate_dependencies(
            normalized_rule.dependencies,
            {} if dependencies is None else dependencies,
            override_reason=dependency_reason,
        )
        lineage = list(dependency_decision.override_lineage)
        if dependency_decision.blockers and dependency_reason is not None:
            targets = tuple(blocker.rsplit(":", 1)[0] for blocker in dependency_decision.blockers)
            lineage.append(
                GateOverride(
                    gate_id=normalized_rule.gate_id,
                    scope=OverrideScope.DEPENDENCIES,
                    reason=dependency_reason,
                    targets=targets,
                )
            )
        if gate_reason is not None:
            lineage.append(
                GateOverride(
                    gate_id=normalized_rule.gate_id,
                    scope=OverrideScope.GATE,
                    reason=gate_reason,
                )
            )
        if not dependency_decision.allowed:
            status = GateStatus.NOT_EVALUABLE
        elif gate_reason is not None or (lineage and natural_status is GateStatus.PASS):
            status = GateStatus.OVERRIDDEN
        else:
            status = natural_status

        return GateResult(
            gate_id=normalized_rule.gate_id,
            status=status,
            natural_status=natural_status,
            checks=check_results,
            passed_count=passed_count,
            required_passes=required_passes,
            blockers=dependency_decision.blockers,
            inherited_overrides=dependency_decision.inherited_overrides,
            override_reason=gate_reason,
            dependency_override_reason=dependency_reason,
            override_lineage=tuple(lineage),
        )

    @staticmethod
    def _evaluate_check(check: GateCheck, observations: Mapping[str, object]) -> GateCheckResult:
        found, observed = _resolve_observation(observations, check.metric)
        if not found:
            return GateCheckResult(
                name=check.name,
                metric=check.metric,
                status=CheckStatus.NOT_EVALUABLE,
                observed=None,
                operator=check.operator,
                threshold=check.threshold,
                counts_toward_gate=check.counts_toward_gate,
                detail="metric is missing",
            )
        try:
            normalized_observed = _json_normalize(observed)
            passed = _compare(check.operator, normalized_observed, check.threshold)
        except (TypeError, ValueError) as exc:
            return GateCheckResult(
                name=check.name,
                metric=check.metric,
                status=CheckStatus.NOT_EVALUABLE,
                observed=None,
                operator=check.operator,
                threshold=check.threshold,
                counts_toward_gate=check.counts_toward_gate,
                detail=f"metric is invalid: {exc}",
            )
        return GateCheckResult(
            name=check.name,
            metric=check.metric,
            status=CheckStatus.PASS if passed else CheckStatus.FAIL,
            observed=normalized_observed,
            operator=check.operator,
            threshold=check.threshold,
            counts_toward_gate=check.counts_toward_gate,
            detail="comparison passed" if passed else "comparison failed",
        )


def _resolve_observation(observations: Mapping[str, object], metric: str) -> tuple[bool, object]:
    if metric in observations:
        return True, observations[metric]
    current: object = observations
    for part in metric.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _compare(operator: ComparisonOperator, observed: JSONValue, threshold: JSONValue) -> bool:
    if operator in _NUMERIC_OPERATORS:
        if not _is_finite_number(observed):
            raise TypeError("numeric comparison requires a finite numeric observation")
        assert isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
        if operator is ComparisonOperator.LT:
            return observed < threshold
        if operator is ComparisonOperator.LE:
            return observed <= threshold
        if operator is ComparisonOperator.GT:
            return observed > threshold
        return observed >= threshold
    if operator is ComparisonOperator.EQ:
        return observed == threshold and type(observed) is type(threshold)
    if operator is ComparisonOperator.NE:
        return observed != threshold or type(observed) is not type(threshold)
    if operator is ComparisonOperator.BETWEEN_CLOSED:
        if not _is_finite_number(observed):
            raise TypeError("between_closed requires a finite numeric observation")
        assert isinstance(threshold, Sequence)
        lower, upper = threshold
        assert isinstance(lower, (int, float)) and isinstance(upper, (int, float))
        return lower <= observed <= upper
    if operator is ComparisonOperator.IS_TRUE:
        if not isinstance(observed, bool):
            raise TypeError("is_true requires a boolean observation")
        return observed
    if operator is ComparisonOperator.IS_FALSE:
        if not isinstance(observed, bool):
            raise TypeError("is_false requires a boolean observation")
        return not observed
    raise GateEvaluationError(f"unsupported comparison operator: {operator}")


def _normalize_override_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    if not isinstance(reason, str) or not reason.strip():
        raise GateEvaluationError("override_reason must be a non-empty string")
    return reason.strip()
