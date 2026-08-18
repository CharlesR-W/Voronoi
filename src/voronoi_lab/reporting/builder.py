"""Artifact-only builder for the self-contained Voronoi research report."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar, Literal
from urllib.parse import unquote, urlsplit

from voronoi_lab.core import canonical_hash

from .htmlreport import Report
from .io import atomic_write_text
from .payload import (
    OKABE_ITO,
    ExperimentSection,
    HeatmapPlot,
    LinePlot,
    ReportPayload,
    make_mock_payload,
)

TAB_ORDER = (
    ("overview", "Overview"),
    ("formation", "Exp 1A-1C · formation / boundaries"),
    ("snapping", "Exp 1D-1E · snapping / recovery"),
    ("synthetic", "Later gate · synthetic recovery"),
    ("real_algebra", "Blocked · conditional algebra"),
    ("spec", "Spec"),
    ("provenance", "Provenance"),
)

REPORT_CSS = r"""
<style>
  html { max-width: 100%; overflow-x: hidden; }
  body { max-width: 1500px; margin: 28px auto; padding: 0 28px 48px; overflow-x: hidden; }
  .report-banner { border-left: 5px solid #0072B2; background: #f6f4ef; padding: 12px 16px;
    border-radius: 6px; margin: 12px 0 18px; max-width: 1100px; }
  .report-banner.mock { border-left-color: #D55E00; }
  .report-banner strong { letter-spacing: .04em; }
  .tab-panel { max-width: 1180px; min-width: 0; }
  .experiment { position: relative; }
  .experiment h2 { margin-top: 18px; }
  .eyebrow { margin: 0 0 4px; color: #6d645b; font-weight: 750; font-size: 12px;
    letter-spacing: .08em; text-transform: uppercase; }
  .question { font-size: 18px; line-height: 1.45; max-width: 980px; margin: 0 0 14px; }
  .question, .status-box, .experiment-map li, .methodology li, .guide > div {
    overflow-wrap: anywhere; }
  .status-box { border: 1px solid #d8d0c6; border-left: 5px solid #0072B2; padding: 10px 14px;
    border-radius: 6px; background: white; max-width: 980px; margin: 0 0 18px; }
  .status-box.mock { border-left-color: #D55E00; background: #fff8ef; }
  .status-label { font-weight: 850; letter-spacing: .08em; margin-right: 8px; }
  .methodology { max-width: 980px; }
  .methodology li { margin-bottom: 6px; }
  .equation-box { margin: 1rem 0; padding: .9rem 1.1rem; border: 1px solid #d8d0c6;
    border-left: 4px solid #0072B2; border-radius: .5rem; background: #f8fafb;
    overflow-x: auto; max-width: 980px; text-align: center; font-size: 16px; }
  .where-line { color: #6d645b; font-size: 13px; max-width: 980px; }
  .guide { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px;
    max-width: 980px; margin: 8px 0 18px; }
  .guide > div { border: 1px solid #d8d0c6; border-radius: 6px; padding: 12px 14px;
    background: #faf8f5; }
  .guide strong { display: block; margin-bottom: 5px; }
  .guide .supports { border-top: 4px solid #0072B2; }
  .guide .weakens { border-top: 4px dashed #D55E00; }
  .plot-shell { position: relative; margin: 16px 0 24px; }
  .plot-takeaway { max-width: 980px; margin: 6px 0 0; font-size: 13px; }
  .takeaway { border-left: 4px solid #0072B2; padding: 9px 13px; background: #f8fafb;
    max-width: 980px; margin-top: 18px; }
  .caveat { color: #6d645b; max-width: 980px; font-size: 13px; }
  .overview-grid { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(280px, .65fr);
    gap: 24px; align-items: start; }
  .overview-grid > *, .tab-panel > * { min-width: 0; }
  .experiment-map { counter-reset: experiment; list-style: none; padding: 0; }
  .experiment-map li { counter-increment: experiment; border-top: 1px solid #e5e1dc;
    padding: 10px 0 10px 36px; position: relative; }
  .experiment-map li::before { content: counter(experiment); position: absolute; left: 0; top: 8px;
    background: #0072B2; color: white; width: 23px; height: 23px; border-radius: 50%;
    text-align: center; line-height: 23px; font-size: 12px; font-weight: 800; }
  .spec-body { max-width: 980px; }
  .spec-body h1 { font-size: 25px; margin-top: 18px; }
  .spec-body h2 { font-size: 21px; }
  .spec-body table { border-collapse: collapse; width: 100%; font-size: 13px; }
  .spec-body th, .spec-body td { border: 1px solid #d8d0c6; padding: 7px 9px; vertical-align: top; }
  .provenance-list dt { font-weight: 750; margin-top: 10px; }
  .provenance-list dd { margin-left: 0; color: #6d645b; }
  @media (max-width: 760px) {
    body { margin: 14px auto; padding: 0 14px 36px; }
    .overview-grid, .guide { grid-template-columns: 1fr; }
    .question, .status-box, .equation-box, .where-line { width: 100%; max-width: 100%; }
    .top-tabs { display: flex; overflow-x: auto; width: 100%; max-width: 100%; }
    .tab-btn { flex: 0 0 auto; }
  }
</style>
"""


class _ResourceInspector(HTMLParser):
    _FETCH_ATTRIBUTES: ClassVar[dict[str, set[str]]] = {
        "audio": {"src"},
        "base": {"href"},
        "embed": {"src"},
        "iframe": {"src"},
        "image": {"href", "xlink:href"},
        "img": {"src", "srcset"},
        "input": {"src"},
        "link": {"href"},
        "object": {"data"},
        "script": {"src"},
        "source": {"src", "srcset"},
        "track": {"src"},
        "use": {"href", "xlink:href"},
        "video": {"poster", "src"},
    }

    def __init__(self) -> None:
        super().__init__()
        self.external: list[str] = []
        self._inside_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "meta" and (attributes.get("http-equiv") or "").lower() == "refresh":
            self.external.append("meta[http-equiv]=refresh")
        if tag == "style":
            self._inside_style = True
        fetch_attributes = self._FETCH_ATTRIBUTES.get(tag, set())
        for name, value in attrs:
            if value is None:
                continue
            if tag == "a" and name == "href" and not self._is_safe_anchor_reference(value):
                self.external.append(f"{tag}[{name}]={value}")
            elif name == "srcset":
                self.external.append(f"{tag}[srcset]={value}")
            elif name in fetch_attributes and not self._is_embedded_reference(value):
                self.external.append(f"{tag}[{name}]={value}")
            if name in {"action", "formaction"}:
                self.external.append(f"{tag}[{name}]={value}")
            if name == "style" and self._style_fetches_externally(value):
                self.external.append(f"{tag}[style]={value}")

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self._inside_style = False

    def handle_data(self, data: str) -> None:
        if self._inside_style and self._style_fetches_externally(data):
            self.external.append("style contains an external URL")

    @classmethod
    def _is_embedded_reference(cls, value: str) -> bool:
        normalized = value.strip().lower()
        return normalized.startswith("data:") or normalized.startswith("#")

    @classmethod
    def _style_fetches_externally(cls, value: str) -> bool:
        if re.search(r"@import\b", value, flags=re.IGNORECASE):
            return True
        references = re.findall(r"url\(\s*['\"]?([^)'\"\s]+)", value, flags=re.IGNORECASE)
        return any(not cls._is_embedded_reference(reference) for reference in references)

    @staticmethod
    def _is_safe_anchor_reference(value: str) -> bool:
        normalized = unquote(value.strip())
        if not normalized or any(ord(character) < 32 for character in normalized):
            return False
        if normalized.startswith("#"):
            return True
        parsed = urlsplit(normalized)
        if parsed.scheme:
            return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)
        if parsed.netloc or normalized.startswith(("/", "\\")):
            return False
        path_segments = parsed.path.replace("\\", "/").split("/")
        return bool(parsed.path) and ".." not in path_segments


def _assert_offline(html_document: str) -> None:
    inspector = _ResourceInspector()
    inspector.feed(html_document)
    if inspector.external:
        raise ValueError(f"report contains external resources: {inspector.external}")


def _markdown_to_html(markdown_text: str) -> str:
    try:
        import markdown
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("building a report requires the 'report' optional dependency") from error
    return markdown.markdown(
        # Python-Markdown deliberately preserves raw HTML.  The embedded spec
        # is data, not trusted report code, so escape it before Markdown adds
        # its own small, predictable tag vocabulary.
        html.escape(markdown_text, quote=False),
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )


def _plot_html(plot: LinePlot | HeatmapPlot) -> str:
    plot_id = f"plot-{plot.key}"
    return (
        '<div class="plot-shell">'
        f'<div class="plot" id="{html.escape(plot_id)}" role="img" '
        f'aria-label="{html.escape(plot.title)}" style="min-height:480px"></div>'
        '<p class="plot-takeaway"><strong>Figure reading.</strong> '
        f"{html.escape(plot.takeaway)}</p>"
        "</div>"
    )


def _experiment_html(experiment: ExperimentSection, *, mockup: bool) -> str:
    status_class = "status-box mock" if mockup else "status-box"
    methods = "".join(f"<li>{html.escape(item)}</li>" for item in experiment.methodology)
    plots = "".join(_plot_html(plot) for plot in experiment.plots)
    return f"""
<article class="experiment">
  <h2>{html.escape(experiment.title)}</h2>
  <p class="eyebrow">Question</p>
  <p class="question">{html.escape(experiment.question)}</p>
  <div class="{status_class}"><span class="status-label">
    STATUS · {html.escape(experiment.status)}</span>
    {html.escape(experiment.status_detail)}</div>
  <h3>Why this test matters</h3>
  <p>{html.escape(experiment.motivation)}</p>
  <div class="equation-box">$${html.escape(experiment.equation)}$$</div>
  <p class="where-line"><strong>Where:</strong> {html.escape(experiment.equation_where)}</p>
  <h3>Methodology</h3>
  <ul class="methodology">{methods}</ul>
  <h3>Interpretation guide</h3>
  <div class="guide">
    <div class="supports"><strong>Pattern that supports</strong>
      {html.escape(experiment.guide.supports)}</div>
    <div class="weakens"><strong>Pattern that weakens</strong>
      {html.escape(experiment.guide.weakens)}</div>
  </div>
  {plots}
  <p class="takeaway"><strong>Takeaway.</strong> {html.escape(experiment.takeaway)}</p>
  <p class="caveat"><strong>Caveat.</strong> {html.escape(experiment.caveat)}</p>
</article>
"""


def _overview_html(payload: ReportPayload) -> str:
    overview = payload.overview
    status_class = "status-box mock" if payload.mode == "mockup" else "status-box"
    experiment_map = "".join(f"<li>{html.escape(item)}</li>" for item in overview.experiment_map)
    caveats = "".join(f"<li>{html.escape(item)}</li>" for item in overview.caveats)
    return f"""
<div class="overview-grid">
  <div>
    <p class="eyebrow">Central question</p>
    <p class="question">{html.escape(overview.question)}</p>
    <div class="{status_class}"><span class="status-label">STATUS · {overview.status}</span>
      {html.escape(overview.current_answer)}</div>
    <div class="equation-box">$${html.escape(overview.central_equation)}$$</div>
    <p class="where-line">{html.escape(overview.equation_where)}</p>
  </div>
  <div>
    <h3>Experiment map</h3>
    <ol class="experiment-map">{experiment_map}</ol>
  </div>
</div>
<h3>Interpretive guardrails</h3>
<ul>{caveats}</ul>
"""


def _provenance_html(payload: ReportPayload) -> str:
    provenance = payload.provenance
    seeds = ", ".join(str(seed) for seed in provenance.seeds) or "none declared"
    artifacts = "".join(f"<li>{html.escape(value)}</li>" for value in provenance.artifact_ids)
    warnings = "".join(f"<li>{html.escape(value)}</li>" for value in provenance.warnings)
    return f"""
<h2>Provenance and negative-result surface</h2>
<p>This audit view records the exact saved payload consumed by the report builder.
It is secondary to the scientific questions but remains embedded for reproducibility.</p>
<dl class="provenance-list">
  <dt>Payload source</dt><dd>{html.escape(provenance.payload_source)}</dd>
  <dt>Run ID</dt><dd>{html.escape(provenance.run_id)}</dd>
  <dt>Configuration hash</dt><dd>{html.escape(provenance.config_hash)}</dd>
  <dt>Seeds</dt><dd>{html.escape(seeds)}</dd>
</dl>
<h3>Artifact identities</h3>
<ul>{artifacts or "<li>None: this is not a measured run.</li>"}</ul>
<h3>Warnings, exclusions, and failed gates</h3>
<ul>{warnings or "<li>No warnings recorded.</li>"}</ul>
"""


REPORT_JS = r"""
const REPORT_TABS = DATA.tabs;
const REPORT_PAYLOAD = DATA.report;
const OKABE_ITO = DATA.okabe_ito;
const renderedPlots = new Set();

function mockWatermark() {
  if (REPORT_PAYLOAD.mode !== 'mockup') return [];
  return [{xref:'paper', yref:'paper', x:0.5, y:0.5,
    text:'MOCKUP — SCHEMATIC, NOT DATA', showarrow:false, textangle:-20,
    font:{size:30, color:'rgba(213,94,0,0.19)'}}];
}

function renderLine(plot) {
  const traces = plot.series.map(series => ({
    type:'scatter', mode:'lines+markers', name:series.name, x:plot.x, y:series.y,
    line:{color:OKABE_ITO[series.color_key], dash:series.dash, width:2.4},
    marker:{color:OKABE_ITO[series.color_key], size:6}, connectgaps:false,
    hovertemplate:esc(series.name) + '<br>x=%{x}<br>y=%{y:.4f}<extra></extra>'
  }));
  const shapes = [];
  const referenceAnnotations = [];
  if (plot.x_reference != null) {
    shapes.push({type:'line', x0:plot.x_reference, x1:plot.x_reference,
      yref:'paper', y0:0, y1:1, line:{color:'#6d645b', width:1.4, dash:'dash'}});
    referenceAnnotations.push({xref:'x', yref:'paper', x:plot.x_reference, y:1.03,
      text:plot.x_reference_label, showarrow:false, font:{size:11, color:'#6d645b'}});
  }
  if (plot.y_reference != null) {
    shapes.push({type:'line', y0:plot.y_reference, y1:plot.y_reference,
      xref:'paper', x0:0, x1:1, line:{color:'#6d645b', width:1.4, dash:'dash'}});
    referenceAnnotations.push({xref:'paper', yref:'y', x:1, y:plot.y_reference,
      text:plot.y_reference_label, xanchor:'right', yanchor:'bottom', showarrow:false,
      font:{size:11, color:'#6d645b'}});
  }
  const layout = merge(LAYOUT_BASE, {
    height:500, margin:{l:64, r:18, t:92, b:60},
    title:{text:plot.title, font:{size:15}, x:0.5, y:0.98},
    annotations:mockWatermark().concat(referenceAnnotations), shapes:shapes,
    xaxis:merge(LAYOUT_BASE.xaxis, {title:plot.x_label, range:plot.x_range, autorange:false}),
    yaxis:merge(LAYOUT_BASE.yaxis, {title:plot.y_label, range:plot.y_range, autorange:false}),
    legend:{orientation:'h', x:0, y:1.14, font:{size:11}}
  });
  return Plotly.react('plot-' + plot.key, traces, layout, PLOT_CONFIG);
}

function renderHeatmap(plot) {
  const trace = {type:'heatmap', x:plot.x_labels, y:plot.y_labels, z:plot.z,
    colorscale:'Cividis', zmin:plot.z_range[0], zmax:plot.z_range[1], zauto:false,
    colorbar:{title:{text:plot.colorbar_label}},
    hovertemplate:esc(plot.y_label) + ' %{y}<br>' + esc(plot.x_label) +
      ' %{x}<br>%{z:.4f}<extra></extra>'};
  const layout = merge(LAYOUT_BASE, {
    height:480, title:{text:plot.title, font:{size:15}}, annotations:mockWatermark(),
    xaxis:merge(LAYOUT_BASE.xaxis, {title:plot.x_label, type:'category'}),
    yaxis:merge(LAYOUT_BASE.yaxis, {title:plot.y_label, type:'category'})
  });
  return Plotly.react('plot-' + plot.key, [trace], layout, PLOT_CONFIG);
}

function renderTab(key) {
  const experiment = REPORT_PAYLOAD.experiments.find(item => item.key === key);
  if (!experiment) return;
  experiment.plots.forEach(plot => {
    if (renderedPlots.has(plot.key)) {
      Plotly.Plots.resize('plot-' + plot.key);
      return;
    }
    const pending = plot.kind === 'line' ? renderLine(plot) : renderHeatmap(plot);
    Promise.resolve(pending).then(() => renderedPlots.add(plot.key));
  });
}

function setTab(key) {
  REPORT_TABS.forEach(tab => {
    const panel = document.getElementById('panel-' + tab.key);
    const button = document.querySelector('[data-tab="' + tab.key + '"]');
    if (panel) panel.style.display = tab.key === key ? '' : 'none';
    if (button) button.classList.toggle('active', tab.key === key);
  });
  renderTab(key);
  const panel = document.getElementById('panel-' + key);
  if (panel && window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise([panel]);
  }
}

const tabsElement = document.getElementById('tabs');
REPORT_TABS.forEach(tab => {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'tab-btn';
  button.dataset.tab = tab.key;
  button.textContent = tab.label;
  button.onclick = () => setTab(tab.key);
  tabsElement.appendChild(button);
});
setTab('overview');
"""


def render_report(payload: ReportPayload, readme_markdown: str) -> str:
    """Render the deterministic MOCKUP payload and embedded specification.

    Measured rendering is deliberately unavailable until a receipt-linked
    assembler can construct the payload from verified stage artifacts.
    """

    if payload.mode != "mockup":
        raise ValueError(
            "real report publication is disabled until payloads are assembled from "
            "verified receipt-linked artifacts"
        )

    mockup = True
    title = "Voronoi Experiment 1 Dashboard — MOCKUP"
    subtitle = (
        "Synthetic plot patterns for the planned candidate-cell formation, boundary, snapping, "
        "recovery, and module-comparison tests."
    )
    report = Report(title, subtitle)
    report.html(REPORT_CSS)
    banner_class = "report-banner mock" if mockup else "report-banner"
    banner_label = (
        "MOCKUP · SCHEMATIC VALUES · NOT EMPIRICAL EVIDENCE" if mockup else "SAVED RESULT PAYLOAD"
    )
    report.html(
        f'<div class="{banner_class}"><strong>{html.escape(banner_label)}</strong><br>'
        "Every experiment states its question, evidence status, methodology, interpretation rule, "
        "takeaway, and caveat.</div>"
    )
    report.html('<nav id="tabs" class="top-tabs" aria-label="Report sections"></nav>')

    report.html(
        f'<section id="panel-overview" class="tab-panel">{_overview_html(payload)}</section>'
    )
    for experiment in payload.experiments:
        report.html(
            f'<section id="panel-{experiment.key}" class="tab-panel" style="display:none">'
            f"{_experiment_html(experiment, mockup=mockup)}</section>"
        )
    spec_html = _markdown_to_html(readme_markdown)
    report.html(
        '<section id="panel-spec" class="tab-panel spec-body" style="display:none">'
        "<h2>Embedded experiment specification</h2>"
        '<p class="note">Built from the repository README at report-generation time.</p>'
        f"{spec_html}</section>"
    )
    report.html(
        '<section id="panel-provenance" class="tab-panel" style="display:none">'
        f"{_provenance_html(payload)}</section>"
    )

    report.data("tabs", [{"key": key, "label": label} for key, label in TAB_ORDER])
    report.data("report", payload.model_dump(mode="json"))
    report.data("okabe_ito", OKABE_ITO)
    report.script(REPORT_JS)
    document = report.render()
    payload_digest = canonical_hash(payload.model_dump(mode="json"))
    document = document.replace(
        "<head>",
        '<head>\n<meta http-equiv="Content-Security-Policy" '
        "content=\"default-src 'none'; script-src 'unsafe-inline'; "
        "style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; "
        "frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'\">\n"
        '<meta name="voronoi-report-payload-sha256" '
        f'content="{payload_digest}">',
        1,
    )
    _assert_offline(document)
    return document


def build_report(
    output_path: str | Path,
    *,
    mode: Literal["mockup", "real"] = "mockup",
    payload_path: str | Path | None = None,
    readme_path: str | Path = "README.md",
) -> Path:
    """Build the self-contained MOCKUP report.

    Measured publication stays disabled until a verified receipt-linked payload
    assembler exists; ``payload_path`` is retained only to reject ambiguous calls.
    """

    if mode not in {"mockup", "real"}:
        raise ValueError("report mode must be 'mockup' or 'real'")
    if mode == "real":
        raise ValueError(
            "real report publication is disabled until payloads are assembled from "
            "verified receipt-linked artifacts"
        )
    elif payload_path is not None:
        raise ValueError("mockup mode uses its deterministic built-in payload")
    else:
        payload = make_mock_payload()

    readme = Path(readme_path).read_text(encoding="utf-8")
    document = render_report(payload, readme)
    return atomic_write_text(output_path, document)
