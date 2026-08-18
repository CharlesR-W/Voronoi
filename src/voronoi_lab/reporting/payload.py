"""Strict saved-payload contract for mock and measured research reports.

The report builder consumes only this schema and the embedded README. It does not
import model, training, checkpoint, or experiment-runner modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from .io import atomic_write_text

OKABE_ITO: dict[str, str] = {
    "black": "#000000",
    "blue": "#0072B2",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "yellow": "#F0E442",
}

ColorKey = Literal["black", "blue", "green", "orange", "purple", "sky", "vermillion", "yellow"]
DashStyle = Literal["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
EvidenceStatus = Literal["MOCKUP", "PLANNED", "UNRUN", "DIAGNOSTIC", "MEASURED"]
ExperimentKey = Literal["formation", "snapping"]


def _require_true(value: bool) -> bool:
    if value is not True:
        raise ValueError("value must be true")
    return value


StrictTrue = Annotated[bool, Field(strict=True), AfterValidator(_require_true)]
VersionOne = Annotated[int, Field(ge=1, le=1, strict=True)]


class PayloadModel(BaseModel):
    """Strict finite-data base model suitable for direct JSON embedding."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class VisualGrammar(PayloadModel):
    sequential_scale: Literal["Cividis"] = "Cividis"
    categorical_palette: Literal["Okabe-Ito"] = "Okabe-Ito"
    line_styles: tuple[DashStyle, ...] = ("solid", "dash", "dot", "dashdot")
    comparable_scales_fixed: StrictTrue = True


class LineSeries(PayloadModel):
    name: str = Field(min_length=1)
    color_key: ColorKey
    dash: DashStyle
    y: tuple[float | None, ...]


class LinePlot(PayloadModel):
    kind: Literal["line"] = "line"
    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    x_label: str = Field(min_length=1)
    y_label: str = Field(min_length=1)
    x: tuple[float, ...]
    series: tuple[LineSeries, ...]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    x_reference: float | None = None
    x_reference_label: str | None = None
    y_reference: float | None = None
    y_reference_label: str | None = None
    takeaway: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shape_and_scales(self) -> LinePlot:
        if len(self.x) < 2:
            raise ValueError("line plots require at least two x values")
        if not self.series:
            raise ValueError("line plots require at least one series")
        if any(len(series.y) != len(self.x) for series in self.series):
            raise ValueError("every line series must match the x-axis length")
        if self.x_range[0] >= self.x_range[1] or self.y_range[0] >= self.y_range[1]:
            raise ValueError("line plot ranges must be strictly increasing")
        if (self.x_reference is None) != (self.x_reference_label is None):
            raise ValueError("x reference value and label must be provided together")
        if (self.y_reference is None) != (self.y_reference_label is None):
            raise ValueError("y reference value and label must be provided together")
        if self.x_reference is not None and not (
            self.x_range[0] <= self.x_reference <= self.x_range[1]
        ):
            raise ValueError("x reference must lie inside x_range")
        if self.y_reference is not None and not (
            self.y_range[0] <= self.y_reference <= self.y_range[1]
        ):
            raise ValueError("y reference must lie inside y_range")
        return self


class HeatmapPlot(PayloadModel):
    kind: Literal["heatmap"] = "heatmap"
    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    x_label: str = Field(min_length=1)
    y_label: str = Field(min_length=1)
    colorbar_label: str = Field(min_length=1)
    x_labels: tuple[str, ...]
    y_labels: tuple[str, ...]
    z: tuple[tuple[float | None, ...], ...]
    z_range: tuple[float, float]
    colorscale: Literal["Cividis"] = "Cividis"
    takeaway: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_grid_and_scale(self) -> HeatmapPlot:
        if not self.x_labels or not self.y_labels:
            raise ValueError("heatmaps require non-empty axes")
        if len(self.z) != len(self.y_labels) or any(
            len(row) != len(self.x_labels) for row in self.z
        ):
            raise ValueError("heatmap z dimensions must match its labeled axes")
        if self.z_range[0] >= self.z_range[1]:
            raise ValueError("heatmap z_range must be strictly increasing")
        return self


PlotPayload = Annotated[LinePlot | HeatmapPlot, Field(discriminator="kind")]


class InterpretationGuide(PayloadModel):
    supports: str = Field(min_length=1)
    weakens: str = Field(min_length=1)


class ExperimentSection(PayloadModel):
    key: ExperimentKey
    title: str = Field(min_length=1)
    question: str = Field(min_length=1)
    status: EvidenceStatus
    status_detail: str = Field(min_length=1)
    motivation: str = Field(min_length=1)
    equation: str = Field(min_length=1)
    equation_where: str = Field(min_length=1)
    methodology: tuple[str, ...] = Field(min_length=1)
    guide: InterpretationGuide
    plots: tuple[PlotPayload, ...] = Field(min_length=1)
    takeaway: str = Field(min_length=1)
    caveat: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_plot_keys(self) -> ExperimentSection:
        keys = [plot.key for plot in self.plots]
        if len(keys) != len(set(keys)):
            raise ValueError("plot keys must be unique within an experiment")
        return self


class OverviewPayload(PayloadModel):
    question: str = Field(min_length=1)
    current_answer: str = Field(min_length=1)
    status: EvidenceStatus
    central_equation: str = Field(min_length=1)
    equation_where: str = Field(min_length=1)
    experiment_map: tuple[str, ...] = Field(min_length=1)
    caveats: tuple[str, ...] = Field(min_length=1)


class ProvenancePayload(PayloadModel):
    payload_source: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    seeds: tuple[int, ...]
    artifact_ids: tuple[str, ...]
    warnings: tuple[str, ...]


class ReportPayload(PayloadModel):
    schema_version: VersionOne = 1
    mode: Literal["mockup", "real"]
    overview: OverviewPayload
    experiments: tuple[ExperimentSection, ...]
    visual_grammar: VisualGrammar = Field(default_factory=VisualGrammar)
    provenance: ProvenancePayload

    @model_validator(mode="after")
    def enforce_complete_report_and_evidence_labels(self) -> ReportPayload:
        expected = ("formation", "snapping")
        actual = tuple(experiment.key for experiment in self.experiments)
        if actual != expected:
            raise ValueError(f"experiments must appear exactly in overview order {expected}")
        plot_keys = [plot.key for experiment in self.experiments for plot in experiment.plots]
        if len(plot_keys) != len(set(plot_keys)):
            raise ValueError("plot keys must be unique across the report")
        if self.mode == "mockup":
            if self.overview.status != "MOCKUP" or any(
                experiment.status != "MOCKUP" for experiment in self.experiments
            ):
                raise ValueError("mock payloads must label every scientific section MOCKUP")
            titles = [experiment.title for experiment in self.experiments]
            titles.extend(
                plot.title for experiment in self.experiments for plot in experiment.plots
            )
            if any("MOCKUP" not in title for title in titles):
                raise ValueError("every mock experiment and plot title must contain MOCKUP")
        elif self.overview.status == "MOCKUP" or any(
            experiment.status == "MOCKUP" for experiment in self.experiments
        ):
            raise ValueError("real payloads cannot carry MOCKUP evidence labels")
        elif any(
            "MOCKUP" in title
            for experiment in self.experiments
            for title in (experiment.title, *(plot.title for plot in experiment.plots))
        ):
            raise ValueError("real payload experiment and plot titles cannot contain MOCKUP")
        return self


def _round(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.round(values, 5))


def make_mock_payload(seed: int = 20260816) -> ReportPayload:
    """Create deterministic schematic data using the real report schema."""

    rng = np.random.default_rng(seed)
    checkpoints = ("0", "1", "5", "20", "100")
    cuts = ("stage 1", "stage 2", "stage 3", "stage 4")
    base_distortion = np.array(
        [
            [0.86, 0.81, 0.69, 0.52, 0.38],
            [0.88, 0.82, 0.66, 0.47, 0.32],
            [0.90, 0.84, 0.70, 0.50, 0.35],
            [0.91, 0.87, 0.76, 0.59, 0.43],
        ]
    )
    distortion = np.clip(base_distortion + rng.normal(0.0, 0.006, base_distortion.shape), 0, 1)

    epochs = np.array([0.0, 1.0, 5.0, 20.0, 100.0])
    real_distortion = np.array([0.86, 0.81, 0.69, 0.52, 0.38])
    global_null_distortion = np.array([0.85, 0.82, 0.78, 0.76, 0.74])
    class_null_distortion = np.array([0.84, 0.80, 0.75, 0.72, 0.69])

    boundary_coordinate = np.linspace(0.0, 2.0, 21)
    empirical_boundary = 0.12 + 0.82 * np.exp(-(((boundary_coordinate - 1.0) / 0.16) ** 2))
    off_cloud_boundary = 0.19 + 0.58 * np.exp(-(((boundary_coordinate - 1.04) / 0.25) ** 2))
    shifted_boundary_null = np.full_like(boundary_coordinate, 0.25)

    path_coordinate = np.linspace(0.0, 1.0, 21)
    linear_support = 1.0 + 1.2 * np.sin(np.pi * path_coordinate) ** 2
    spherical_support = 1.0 + 0.28 * np.sin(np.pi * path_coordinate) ** 2
    graph_support = 1.0 + 0.12 * np.sin(np.pi * path_coordinate) ** 2

    alpha = np.array([0.0, 0.25, 0.5, 1.0])
    snap = np.clip(np.array([0.0, 0.012, 0.025, 0.055]) + rng.normal(0, 0.001, 4), 0, None)
    random_control = np.clip(np.array([0.0, 0.035, 0.09, 0.24]) + rng.normal(0, 0.002, 4), 0, None)
    away_control = np.clip(np.array([0.0, 0.055, 0.15, 0.39]) + rng.normal(0, 0.002, 4), 0, None)

    boundary_fractions = np.array([0.5, 0.9, 1.1])
    gain_sparse = np.array([0.62, 0.78, 0.91])
    gain_coherent = np.array([0.72, 0.88, 1.04])
    gain_off_cloud = np.array([0.96, 1.12, 1.29])
    recovery_sparse = np.array([0.92, 0.83, 0.69])
    recovery_coherent = np.array([0.87, 0.75, 0.58])
    recovery_off_cloud = np.array([0.73, 0.54, 0.31])

    blocks = np.arange(1.0, 9.0)
    transplant_z = np.array([-0.8, -0.5, 0.1, 0.7, 1.1, 0.5, -0.2, -0.9])
    boundary_z = np.array([0.6, 0.2, -0.3, -0.7, -0.8, -0.1, 0.5, 0.9])
    contraction_z = np.array([0.7, 0.4, -0.1, -0.6, -0.9, -0.2, 0.4, 0.8])

    experiments = (
        ExperimentSection(
            key="formation",
            title="Experiment 1A-1C: formation and boundary alignment — MOCKUP",
            question=(
                "Do held-out residual states become more compact than matched nulls, and does "
                "functional sensitivity align with candidate-cell crossings on supported paths?"
            ),
            status="MOCKUP",
            status_detail=(
                "Synthetic schematic values only; no checkpoint geometry, path, or intervention "
                "output is represented."
            ),
            motivation=(
                "A fitted Voronoi codebook is automatic. The scientific burden is to show held-out "
                "compression and boundary-aligned sensitivity beyond matched Gaussian controls."
            ),
            equation=(
                r"D_{t,\ell,K}="
                r"\frac{\mathbb E\|z-\mu_{q(z)}\|_2^2}"
                r"{\mathbb E\|z-\bar z\|_2^2}"
            ),
            equation_where=(
                "$z$ is a standardized sitewise residual state and $q(z)$ is its "
                "nearest fitted centroid."
            ),
            methodology=(
                "Fit codebooks only on the declared training bank and score geometry "
                "on held-out images.",
                "Compare fixed K values and raw versus standardized metrics against "
                "moment-matched nulls.",
                "Resample images, never spatial tokens; keep plot scales fixed across "
                "checkpoints and cuts.",
                "Locate the first fitted crossing at r=1, compare with a within-path shifted-null, "
                "and display path-support diagnostics beside response curves.",
            ),
            guide=InterpretationGuide(
                supports=(
                    "Lower held-out distortion together with stable assignments and "
                    "boundary enrichment "
                    "supports a useful finite-state description."
                ),
                weakens=(
                    "No advantage over Gaussian nulls, or sensitivity confined to "
                    "off-cloud directions, "
                    "weakens the cell interpretation."
                ),
            ),
            plots=(
                HeatmapPlot(
                    key="formation_distortion",
                    title="Held-out normalized distortion across training — MOCKUP",
                    x_label="checkpoint epoch",
                    y_label="sentinel residual cut",
                    colorbar_label="$D_{t,\\ell,K}$",
                    x_labels=checkpoints,
                    y_labels=cuts,
                    z=tuple(tuple(float(value) for value in row) for row in distortion),
                    z_range=(0.0, 1.0),
                    takeaway=(
                        "MOCKUP reading: darker late cells would indicate lower normalized "
                        "distortion; "
                        "the real plot must include null comparisons before supporting a claim."
                    ),
                ),
                LinePlot(
                    key="formation_nulls",
                    title="Held-out distortion versus matched nulls at a sentinel cut — MOCKUP",
                    x_label="checkpoint epoch",
                    y_label="normalized distortion (lower is better)",
                    x=_round(epochs),
                    series=(
                        LineSeries(
                            name="real residual states",
                            color_key="blue",
                            dash="solid",
                            y=_round(real_distortion),
                        ),
                        LineSeries(
                            name="global Gaussian null",
                            color_key="orange",
                            dash="dash",
                            y=_round(global_null_distortion),
                        ),
                        LineSeries(
                            name="class-conditional Gaussian null",
                            color_key="purple",
                            dash="dot",
                            y=_round(class_null_distortion),
                        ),
                    ),
                    x_range=(0.0, 100.0),
                    y_range=(0.25, 0.95),
                    takeaway=(
                        "MOCKUP reading: a growing late-training gap below both nulls would be "
                        "necessary but not sufficient; stability and functional alignment must "
                        "agree."
                    ),
                ),
                LinePlot(
                    key="boundary_alignment",
                    title="Predictive sensitivity around the first fitted boundary — MOCKUP",
                    x_label="boundary-normalized displacement r = s / s*",
                    y_label="predictive sensitivity g_pred² (schematic units)",
                    x=_round(boundary_coordinate),
                    series=(
                        LineSeries(
                            name="empirical chord",
                            color_key="blue",
                            dash="solid",
                            y=_round(empirical_boundary),
                        ),
                        LineSeries(
                            name="off-cloud direction",
                            color_key="vermillion",
                            dash="dot",
                            y=_round(off_cloud_boundary),
                        ),
                        LineSeries(
                            name="within-path shifted-boundary null",
                            color_key="black",
                            dash="dash",
                            y=_round(shifted_boundary_null),
                        ),
                    ),
                    x_range=(0.0, 2.0),
                    y_range=(0.0, 1.05),
                    x_reference=1.0,
                    x_reference_label="first fitted boundary",
                    takeaway=(
                        "MOCKUP reading: a narrow empirical-chord peak centered at r=1 and above "
                        "the shifted null would support boundary alignment; off-cloud-only peaks "
                        "would not."
                    ),
                ),
                LinePlot(
                    key="path_support",
                    title="Path support diagnostic for matched endpoints — MOCKUP",
                    x_label="normalized path arc length",
                    y_label="nearest-neighbor distance / endpoint baseline",
                    x=_round(path_coordinate),
                    series=(
                        LineSeries(
                            name="linear chord",
                            color_key="vermillion",
                            dash="dot",
                            y=_round(linear_support),
                        ),
                        LineSeries(
                            name="radius-preserving path",
                            color_key="orange",
                            dash="dash",
                            y=_round(spherical_support),
                        ),
                        LineSeries(
                            name="neighbor-graph path",
                            color_key="green",
                            dash="solid",
                            y=_round(graph_support),
                        ),
                    ),
                    x_range=(0.0, 1.0),
                    y_range=(0.8, 2.4),
                    y_reference=1.0,
                    y_reference_label="endpoint support baseline",
                    takeaway=(
                        "MOCKUP reading: boundary effects confined to the elevated red segment "
                        "would "
                        "indicate off-support fragility, not data-relevant discreteness."
                    ),
                ),
            ),
            takeaway=(
                "No result yet. This panel specifies the held-out geometry comparison "
                "required by the coarse gate."
            ),
            caveat=(
                "Compact clusters alone do not establish plateaus, contraction, or "
                "discrete computation."
            ),
        ),
        ExperimentSection(
            key="snapping",
            title="Experiment 1D-1E: snapping, recovery, and module comparison — MOCKUP",
            question=(
                "Does centroid-directed motion preserve computation better than equal-norm "
                "controls, "
                "and do finite perturbations contract and recover downstream state?"
            ),
            status="MOCKUP",
            status_detail="Schematic dose-response curves only; the functional gate remains unrun.",
            motivation=(
                "Snapping is a causal sufficiency test: tolerance is informative only "
                "when the same displacement "
                "in generic or adverse directions causes more damage."
            ),
            equation=r"z^{(\alpha)}=z+\alpha\bigl(\mu_{q(z)}-z\bigr)",
            equation_where=(
                r"$\alpha$ is the snap dose and the clean assignment $q(z)$ is held "
                "fixed throughout the intervention."
            ),
            methodology=(
                "Insert displacements in native activation tensors after defining norms "
                "in standardized coordinates.",
                "Compare snap, random, away-from-centroid, other-centroid, and identity "
                "interventions at equal norm.",
                "Use paired image bootstrap intervals and test both sparse-token and "
                "spatially coherent support.",
                "Keep the eight-block transplant comparison descriptive: no association sign is "
                "preregistered, and one training seed cannot establish replication.",
            ),
            guide=InterpretationGuide(
                supports=(
                    "A snap curve below every equal-norm control, together with gain below "
                    "one and downstream-cell "
                    "recovery, supports finite-state sufficiency."
                ),
                weakens=(
                    "Overlapping curves suggest generic smoothness; damage from snapping "
                    "rejects centroid sufficiency."
                ),
            ),
            plots=(
                LinePlot(
                    key="snapping_damage",
                    title="Predictive damage versus intervention dose — MOCKUP",
                    x_label="snap/control dose $\\alpha$",
                    y_label="predictive KL from clean output",
                    x=_round(alpha),
                    series=(
                        LineSeries(
                            name="toward centroid", color_key="blue", dash="solid", y=_round(snap)
                        ),
                        LineSeries(
                            name="random equal norm",
                            color_key="orange",
                            dash="dash",
                            y=_round(random_control),
                        ),
                        LineSeries(
                            name="away from centroid",
                            color_key="vermillion",
                            dash="dot",
                            y=_round(away_control),
                        ),
                    ),
                    x_range=(0.0, 1.0),
                    y_range=(0.0, 0.45),
                    takeaway=(
                        "MOCKUP reading: a lower solid blue curve would favor snapping "
                        "over norm-matched controls."
                    ),
                ),
                LinePlot(
                    key="finite_gain",
                    title="Next-block RMS gain after finite perturbations — MOCKUP",
                    x_label="perturbation / directional boundary distance",
                    y_label="RMS gain κ",
                    x=_round(boundary_fractions),
                    series=(
                        LineSeries(
                            name="empirical chord · sparse token",
                            color_key="blue",
                            dash="solid",
                            y=_round(gain_sparse),
                        ),
                        LineSeries(
                            name="empirical chord · coherent support",
                            color_key="sky",
                            dash="dash",
                            y=_round(gain_coherent),
                        ),
                        LineSeries(
                            name="off-cloud · coherent support",
                            color_key="vermillion",
                            dash="dot",
                            y=_round(gain_off_cloud),
                        ),
                    ),
                    x_range=(0.45, 1.15),
                    y_range=(0.4, 1.4),
                    y_reference=1.0,
                    y_reference_label="no contraction (κ = 1)",
                    takeaway=(
                        "MOCKUP reading: κ below one for both sparse and coherent empirical "
                        "perturbations would be one required correction signature, not an "
                        "attractor claim."
                    ),
                ),
                LinePlot(
                    key="cell_recovery",
                    title="Recovery of the clean downstream cell — MOCKUP",
                    x_label="perturbation / directional boundary distance",
                    y_label="clean downstream-cell recovery probability",
                    x=_round(boundary_fractions),
                    series=(
                        LineSeries(
                            name="empirical chord · sparse token",
                            color_key="blue",
                            dash="solid",
                            y=_round(recovery_sparse),
                        ),
                        LineSeries(
                            name="empirical chord · coherent support",
                            color_key="sky",
                            dash="dash",
                            y=_round(recovery_coherent),
                        ),
                        LineSeries(
                            name="off-cloud · coherent support",
                            color_key="vermillion",
                            dash="dot",
                            y=_round(recovery_off_cloud),
                        ),
                    ),
                    x_range=(0.45, 1.15),
                    y_range=(0.0, 1.0),
                    takeaway=(
                        "MOCKUP reading: preferential clean-cell recovery must accompany κ < 1; "
                        "either signature alone is insufficient for the functional gate."
                    ),
                ),
                LinePlot(
                    key="module_comparison",
                    title="Blockwise functional signatures beside transplant damage — MOCKUP",
                    x_label="residual block",
                    y_label="within-metric standardized score (descriptive)",
                    x=_round(blocks),
                    series=(
                        LineSeries(
                            name="transplant damage",
                            color_key="black",
                            dash="dash",
                            y=_round(transplant_z),
                        ),
                        LineSeries(
                            name="boundary enrichment",
                            color_key="orange",
                            dash="dot",
                            y=_round(boundary_z),
                        ),
                        LineSeries(
                            name="contraction strength",
                            color_key="blue",
                            dash="solid",
                            y=_round(contraction_z),
                        ),
                    ),
                    x_range=(1.0, 8.0),
                    y_range=(-1.3, 1.3),
                    y_reference=0.0,
                    y_reference_label="within-metric mean",
                    takeaway=(
                        "MOCKUP reading: either association direction is scientifically live; with "
                        "eight blocks and one seed, the real panel would remain descriptive."
                    ),
                ),
            ),
            takeaway=(
                "No result yet. The separation shown here is an interpretation "
                "schematic, not evidence."
            ),
            caveat=(
                "Tolerance to an intervention does not show that the unperturbed network "
                "performs snapping."
            ),
        ),
    )

    return ReportPayload(
        mode="mockup",
        overview=OverviewPayload(
            question=(
                "During ResNet training, do sitewise residual states form stable candidate cells "
                "whose interiors suppress perturbations and whose crossings concentrate "
                "sensitivity?"
            ),
            current_answer=(
                "No empirical answer yet. This MOCKUP encodes the gated evidence "
                "structure and expected readings "
                "without presenting schematic values as results."
            ),
            status="MOCKUP",
            central_equation=(
                r"\begin{aligned}"
                r"&\text{null-relative geometry}\;\land\;\text{boundary alignment}\\"
                r"&\land\;\text{snapping advantage}\;\land\;\kappa<1"
                r"\;\Longrightarrow\;\text{candidate finite-state sufficiency}"
                r"\end{aligned}"
            ),
            equation_where=(
                "Every conjunction is required. Passing only one diagnostic narrows the claim "
                "rather "
                "than establishing discrete computation."
            ),
            experiment_map=(
                "Experiment 1A: held-out formation geometry versus matched nulls",
                "Experiment 1B-1C: boundary response plus path-support diagnostics",
                "Experiment 1D: snapping, contraction, and clean-cell recovery",
                "Experiment 1E: descriptive module-transplant comparison",
            ),
            caveats=(
                "A fitted Voronoi partition is a definition, not evidence for discrete "
                "computation.",
                "Sharpness only in off-cloud or unsupported path segments indicates fragility, not "
                "data-relevant cell boundaries.",
                "Every schematic trace is watermarked and uses the same schema intended "
                "for measured data.",
            ),
        ),
        experiments=experiments,
        provenance=ProvenancePayload(
            payload_source="deterministic built-in interpretation mock",
            run_id="MOCKUP-no-run",
            config_hash="MOCKUP-no-config-hash",
            seeds=(seed,),
            artifact_ids=(),
            warnings=(
                "MOCKUP values are schematic and must not be cited as experiment output.",
                "The motivating plateau/boundary literature audit remains incomplete.",
            ),
        ),
    )


def _reject_nonstandard_constant(token: str) -> None:
    raise ValueError(f"saved report payload contains invalid JSON constant {token}")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"saved report payload contains duplicate key {key!r}")
        result[key] = value
    return result


def load_payload(path: str | Path) -> ReportPayload:
    """Load and strictly validate a finite saved JSON payload."""

    raw = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_constant,
        object_pairs_hook=_reject_duplicate_pairs,
    )
    return ReportPayload.model_validate(raw)


def write_payload(payload: ReportPayload, path: str | Path) -> Path:
    """Write stable standards-compliant JSON for artifact-based report rebuilding."""

    return atomic_write_text(
        path,
        json.dumps(
            payload.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )
