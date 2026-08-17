# ruff: noqa: E501
"""Reusable, offline self-contained HTML report generator (warm paper theme).

A small builder for presentation-quality, single-file HTML reports: summary-stat
cards, sticky-header tables, base64-embedded static figures, and interactive
Plotly charts and MathJax -- all rendered to one portable ``.html`` you can open
offline or hand to a collaborator. The implementation originated in the shared
interactive-report skill and is vendored here so report regeneration is stable.

The theme (palette, typography, components) is baked in so every report looks
consistent. Typical use::

    from htmlreport import Report

    r = Report("Weight statistics", "ResNet-50 across training")
    r.stat_cards([("Checkpoints", 144), ("Layers", 50)])
    r.section("Trajectories")
    r.plot([{"x": xs, "y": ys, "type": "scatter", "mode": "lines", "name": "loss"}])
    r.table(["layer", "alpha"], [["conv1", "2.91"]])
    r.write("report.html")

For richer interactivity (dropdowns/radios/sliders that re-draw a chart, animated
frames over a sweep dimension), stash a JSON payload with :meth:`data` and emit
custom JavaScript with :meth:`script`; both share the ``DATA`` object and the
``LAYOUT_BASE`` / ``PALETTE`` / ``PLOT_CONFIG`` / ``merge`` / ``esc`` / ``fmt``
helpers defined in the prelude. See patterns.md for the canonical JS recipes.
"""

from __future__ import annotations

import base64
import html
import json
from collections.abc import Iterable, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

from .io import atomic_write_text

# Warm "paper" theme -- one palette so every report looks the same.
STYLE = """
  * { box-sizing: border-box; }
  :root {
    --text:#3d3832; --muted:#6d645b; --rule:#e5e1dc; --border:#d8d0c6;
    --soft:#faf8f5; --bg:#fffdf9; --plot-bg:#faf8f5;
    --blue:#6b7fb5; --red:#b56b6b; --purple:#8f74ad; --orange:#b07d5c;
    --green:#5d8f75; --teal:#7aa6a1;
  }
  body { margin:40px; background:var(--bg); color:var(--text); line-height:1.45;
    font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
  h1 { margin:0 0 6px; font-size:28px; line-height:1.2; }
  h2 { margin-top:34px; border-top:1px solid var(--rule); padding-top:22px; line-height:1.2; }
  h3 { margin:24px 0 10px; }
  a { color:var(--blue); }
  .subtitle { max-width:980px; color:var(--muted); margin-bottom:18px; font-size:13px; }
  .note { font-size:12px; color:var(--muted); margin:8px 0 16px; max-width:980px; }
  .controls { display:flex; gap:18px; align-items:center; flex-wrap:wrap; padding:12px 16px;
    background:white; border:1px solid var(--border); border-radius:6px; margin-bottom:14px; }
  .controls label { font-weight:650; font-size:13px; color:var(--muted); }
  .controls .ctrl { display:inline-flex; align-items:center; gap:6px; }
  .controls .radio { font-weight:normal; margin-right:10px; color:var(--text); }
  select { padding:5px 10px; font-size:14px; border-radius:5px; border:1px solid #d0c8bd; background:white; }
  .top-tabs { display:inline-flex; gap:4px; background:var(--soft); border:1px solid var(--border);
    border-radius:6px; padding:3px; margin:6px 0 16px; }
  .tab-btn { border:0; border-radius:4px; padding:7px 12px; cursor:pointer; background:transparent;
    color:var(--muted); font-weight:650; font-size:13px; }
  .tab-btn.active { background:white; color:var(--text); box-shadow:0 1px 2px rgba(0,0,0,.08); }
  .summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; margin:14px 0; }
  .summary-stat { border:1px solid var(--border); border-radius:6px; padding:10px 12px; background:var(--soft); }
  .summary-stat .label { font-size:12px; color:var(--muted); margin-bottom:4px; }
  .summary-stat .value { font-size:20px; font-weight:750; color:var(--text); }
  .summary-stat.clickable { cursor:pointer; }
  .summary-stat.clickable:hover { border-color:var(--blue); }
  .summary-stat.clickable .label::after { content:" \\2304"; color:var(--blue); }
  .summary-stat.active { border-color:var(--blue); box-shadow:0 1px 2px rgba(0,0,0,.08); }
  .stat-detail { background:white; border:1px solid var(--border); border-radius:6px;
    padding:2px 16px 14px; margin:0 0 16px; }
  .doc-table { border-collapse:collapse; width:100%; margin:8px 0 18px; font-size:12px; }
  .doc-table th, .doc-table td { border:1px solid var(--border); padding:6px 10px; text-align:left; vertical-align:top; }
  .doc-table td.num, .doc-table th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .doc-table th { background:#e5e1dc; font-weight:650; position:sticky; top:0; z-index:2; }
  .table-shell { max-height:620px; overflow:auto; }
  .plot { background:white; border:1px solid var(--border); border-radius:6px; margin:6px 0 4px; }
  img.figure { max-width:100%; border:1px solid var(--border); border-radius:6px; background:white; }
  pre { background:var(--soft); border:1px solid var(--border); border-radius:6px; padding:12px; overflow:auto; font-size:12px; }
  .third-party-licenses { margin-top:36px; max-width:980px; color:var(--muted); font-size:12px; }
  .third-party-licenses summary { cursor:pointer; font-weight:650; }
  .third-party-licenses pre { max-height:360px; white-space:pre-wrap; }
"""

# Plotly defaults + tiny helpers, available to any custom script() block.
JS_PRELUDE = """
  const PLOT_BG = '#faf8f5';
  const PALETTE = {blue:'#6b7fb5', red:'#b56b6b', purple:'#8f74ad', orange:'#b07d5c',
    green:'#5d8f75', teal:'#7aa6a1', muted:'#7d736a'};
  const PALETTE_CYCLE = ['#6b7fb5','#b07d5c','#5d8f75','#8f74ad','#b56b6b','#7aa6a1','#9a8c5c','#7d736a'];
  const PLOT_CONFIG = {responsive:true, displaylogo:false};
  const LAYOUT_BASE = {
    margin:{l:64, r:18, t:40, b:60}, paper_bgcolor:'#fffdf9', plot_bgcolor:PLOT_BG,
    font:{family:'system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif', size:12, color:'#3d3832'},
    xaxis:{showgrid:true, gridcolor:'#e5e1dc', zeroline:false},
    yaxis:{showgrid:true, gridcolor:'#e5e1dc', zeroline:false},
    legend:{font:{size:11}}
  };
  const merge = (a, b) => Object.assign({}, a, b || {});
  const fmt = (x, d=3) => Number.isFinite(Number(x)) ? Number(x).toFixed(d) : '';
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
"""


def _json(obj: Any) -> str:
    """Compact JSON, safe to embed inside a <script> tag.

    NOTE: this uses json.dumps, which emits the bare tokens ``NaN`` / ``Infinity``
    for non-finite floats -- and ``JSON.parse`` rejects those, but a JS *literal*
    (which is how DATA is embedded) tolerates them inconsistently across plots.
    Always convert non-finite values to ``None`` (-> ``null``) BEFORE calling
    :meth:`data` / :meth:`plot`. Plotly treats ``null`` as a gap / transparent cell.
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace(
        "</", "<\\/"
    )


def _inline_libraries() -> str:
    """Load pinned installed Plotly and vendored MathJax into the output file."""

    try:
        from plotly.offline import get_plotlyjs
    except ImportError as error:  # pragma: no cover - exercised by optional-dep users
        raise RuntimeError("building a report requires the 'report' optional dependency") from error
    plotly_js = get_plotlyjs().replace("</script", "<\\/script")
    if "plotly.js v3.7.0" not in plotly_js[:256]:
        raise RuntimeError("the report bundle requires Plotly.js 3.7.0 for its pinned notice")
    mathjax_path = files("voronoi_lab.reporting").joinpath("assets/mathjax-tex-svg.js")
    mathjax_js = mathjax_path.read_text(encoding="utf-8").replace("</script", "<\\/script")
    mathjax_config = """
<script>
window.MathJax = {
  tex: {inlineMath: [['$', '$']], displayMath: [['$$', '$$']]},
  svg: {fontCache: 'global'},
  options: {skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']}
};
</script>
"""
    return (
        mathjax_config
        + f'<script id="mathjax-bundle">{mathjax_js}</script>\n'
        + f'<script id="plotly-bundle">{plotly_js}</script>'
    )


def _third_party_licenses_html() -> str:
    """Embed the notices required to redistribute the inlined JS bundles."""

    assets = files("voronoi_lab.reporting").joinpath("assets")
    mathjax_license = assets.joinpath("LICENSE-MathJax.txt").read_text(encoding="utf-8")
    plotly_license = assets.joinpath("LICENSE-Plotly.txt").read_text(encoding="utf-8")
    return (
        '<details class="third-party-licenses" id="third-party-licenses">'
        "<summary>Third-party licenses for embedded JavaScript</summary>"
        "<h3>MathJax 3.2.2</h3>"
        "<p>Copyright (c) 2009-2022 The MathJax Consortium.</p>"
        f"<pre>{html.escape(mathjax_license)}</pre>"
        "<h3>Plotly.js 3.7.0</h3>"
        "<p>Bundle header: Copyright 2012-2026 Plotly, Inc.</p>"
        f"<pre>{html.escape(plotly_license)}</pre>"
        "</details>"
    )


class Report:
    """Accumulate content blocks, then :meth:`render` / :meth:`write` one HTML file."""

    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle  # may contain inline HTML (links etc.)
        self._body: list[str] = []
        self._scripts: list[str] = []
        self._data: dict[str, Any] = {}
        self._plot_n = 0

    # -- escape hatches -----------------------------------------------------
    def html(self, raw: str) -> Report:
        """Append raw HTML (layout containers, custom controls)."""
        self._body.append(raw)
        return self

    def script(self, js: str) -> Report:
        """Append JavaScript that runs after Plotly + the DATA object are ready.

        Top-level ``function foo(){}`` declarations here are global, so multiple
        script() blocks can call each other and external drivers can reach them.
        """
        self._scripts.append(js)
        return self

    def data(self, name: str, obj: Any) -> Report:
        """Expose a JSON payload to scripts as ``DATA.<name>``."""
        self._data[name] = obj
        return self

    # -- content blocks -----------------------------------------------------
    def section(self, title: str, *, note: str | None = None) -> Report:
        self._body.append(f"<h2>{html.escape(title)}</h2>")
        if note:
            self._body.append(f'<p class="note">{html.escape(note)}</p>')
        return self

    def note(self, text: str) -> Report:
        self._body.append(f'<p class="note">{html.escape(text)}</p>')
        return self

    def stat_cards(self, pairs: Iterable[tuple[str, Any]]) -> Report:
        cards = "".join(
            f'<div class="summary-stat"><div class="label">{html.escape(str(label))}</div>'
            f'<div class="value">{html.escape(str(value))}</div></div>'
            for label, value in pairs
        )
        self._body.append(f'<div class="summary-grid">{cards}</div>')
        return self

    def table(
        self,
        headers: Sequence[Any],
        rows: Iterable[Sequence[Any]],
        *,
        numeric_from: int = 1,
        note: str | None = None,
    ) -> Report:
        """A sticky-header table; columns from ``numeric_from`` onward are right-aligned."""

        def cell(tag: str, i: int, value: Any) -> str:
            cls = ' class="num"' if i >= numeric_from else ""
            return f"<{tag}{cls}>{html.escape(str(value))}</{tag}>"

        head = "".join(cell("th", i, h) for i, h in enumerate(headers))
        body = "".join(
            "<tr>" + "".join(cell("td", i, c) for i, c in enumerate(row)) + "</tr>" for row in rows
        )
        out = (
            f'<div class="table-shell"><table class="doc-table"><thead><tr>{head}</tr>'
            f"</thead><tbody>{body}</tbody></table></div>"
        )
        if note:
            out += f'<p class="note">{html.escape(note)}</p>'
        self._body.append(out)
        return self

    def image(self, src: str | Path | bytes, *, caption: str | None = None) -> Report:
        """Embed a PNG (path or raw bytes) base64 -- always works offline."""
        raw = Path(src).read_bytes() if isinstance(src, (str, Path)) else src
        b64 = base64.b64encode(raw).decode()
        cap = f'<p class="note">{html.escape(caption)}</p>' if caption else ""
        self._body.append(f'<img class="figure" src="data:image/png;base64,{b64}">{cap}')
        return self

    def plot(
        self,
        traces: list[dict],
        layout: dict | None = None,
        *,
        height: int = 480,
        note: str | None = None,
    ) -> Report:
        """A one-shot interactive Plotly chart with the shared theme baked in."""
        self._plot_n += 1
        div = f"plot{self._plot_n}"
        lay = dict(layout or {})
        lay.setdefault("height", height)
        self._body.append(f'<div class="plot" id="{div}" style="min-height:{height}px"></div>')
        if note:
            self._body.append(f'<p class="note">{html.escape(note)}</p>')
        self._scripts.append(
            f"Plotly.newPlot('{div}', {_json(traces)}, merge(LAYOUT_BASE, {_json(lay)}), PLOT_CONFIG);"
        )
        return self

    def raw_config(self, cfg: dict) -> Report:
        self._body.append(f"<pre>{html.escape(json.dumps(cfg, indent=2))}</pre>")
        return self

    # -- output -------------------------------------------------------------
    def render(self) -> str:
        data_blob = f"  const DATA = {_json(self._data)};\n" if self._data else ""
        subtitle = f'<div class="subtitle">{self.subtitle}</div>' if self.subtitle else ""
        body = "\n".join(self._body)
        scripts = "\n".join(self._scripts)
        libraries = _inline_libraries()
        third_party_licenses = _third_party_licenses_html()
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(self.title)}</title>
{libraries}
<style>{STYLE}</style>
</head>
<body>
<h1>{html.escape(self.title)}</h1>
{subtitle}
{body}
{third_party_licenses}
<script>
{JS_PRELUDE}
{data_blob}{scripts}
</script>
</body>
</html>
"""

    def write(self, path: str | Path) -> Path:
        return atomic_write_text(path, self.render())
