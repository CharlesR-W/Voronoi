from __future__ import annotations

import copy
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
from pydantic import ValidationError

from voronoi_lab.core import ArtifactStore, canonical_json_bytes
from voronoi_lab.pipeline import DEFAULT_STAGES, PipelineError, validate_stage_output
from voronoi_lab.reporting.builder import TAB_ORDER, _assert_offline, build_report, render_report
from voronoi_lab.reporting.payload import (
    ReportPayload,
    load_payload,
    make_mock_payload,
    write_payload,
)


class _TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.script_sources: list[str] = []
        self.external_resources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "script" and attributes.get("src"):
            self.script_sources.append(attributes["src"] or "")
        for name in ("src", "href"):
            value = attributes.get(name)
            if value and re.match(r"^(?:https?:)?//", value):
                self.external_resources.append(value)


def test_mock_payload_is_deterministic_and_conspicuously_labeled() -> None:
    first = make_mock_payload()
    second = make_mock_payload()
    assert first.model_dump_json() == second.model_dump_json()
    assert first.mode == "mockup"
    assert first.overview.status == "MOCKUP"
    assert tuple(experiment.key for experiment in first.experiments) == (
        "formation",
        "snapping",
        "synthetic",
        "real_algebra",
    )
    for experiment in first.experiments:
        assert experiment.status == "MOCKUP"
        assert "MOCKUP" in experiment.title
        assert all("MOCKUP" in plot.title for plot in experiment.plots)
    assert {plot.key for plot in first.experiments[0].plots} == {
        "formation_distortion",
        "formation_nulls",
        "boundary_alignment",
        "path_support",
    }
    assert {plot.key for plot in first.experiments[1].plots} == {
        "snapping_damage",
        "finite_gain",
        "cell_recovery",
        "module_comparison",
    }


def test_payload_rejects_unknown_fields_nonfinite_values_and_bad_mock_labels() -> None:
    valid = make_mock_payload().model_dump(mode="python")

    unknown = copy.deepcopy(valid)
    unknown["unexpected"] = True
    with pytest.raises(ValidationError, match="unexpected"):
        ReportPayload.model_validate(unknown)

    nonfinite = copy.deepcopy(valid)
    nonfinite["experiments"][1]["plots"][0]["series"][0]["y"] = (
        0.0,
        float("nan"),
        0.1,
        0.2,
    )
    with pytest.raises(ValidationError):
        ReportPayload.model_validate(nonfinite)

    mislabeled = copy.deepcopy(valid)
    mislabeled["experiments"][0]["title"] = "Unlabeled schematic"
    with pytest.raises(ValidationError, match="MOCKUP"):
        ReportPayload.model_validate(mislabeled)

    boolean_version = copy.deepcopy(valid)
    boolean_version["schema_version"] = True
    with pytest.raises(ValidationError):
        ReportPayload.model_validate(boolean_version)

    integer_true = copy.deepcopy(valid)
    integer_true["visual_grammar"]["comparable_scales_fixed"] = 1
    with pytest.raises(ValidationError):
        ReportPayload.model_validate(integer_true)


def test_real_report_publication_is_disabled_until_receipt_linkage_exists(tmp_path) -> None:
    with pytest.raises(ValueError, match="receipt-linked"):
        build_report(tmp_path / "real.html", mode="real")

    mock_path = write_payload(make_mock_payload(), tmp_path / "mock.json")
    with pytest.raises(ValueError, match="receipt-linked"):
        build_report(tmp_path / "real.html", mode="real", payload_path=mock_path)

    untrusted_real = make_mock_payload().model_copy(update={"mode": "real"})
    with pytest.raises(ValueError, match="receipt-linked"):
        render_report(untrusted_real, "# Untrusted result")


def test_mock_report_embeds_spec_tabs_and_inline_libraries(tmp_path) -> None:
    output = build_report(tmp_path / "MOCKUP.html", mode="mockup", readme_path="README.md")
    document = output.read_text(encoding="utf-8")

    assert "Voronoi Experiment 1 Dashboard — MOCKUP" in document
    assert "The project deliberately separates claims that are easy to conflate." in document
    assert "Embedded experiment specification" in document
    assert 'id="plotly-bundle"' in document
    assert 'id="mathjax-bundle"' in document
    assert 'id="third-party-licenses"' in document
    assert "MathJax 3.2.2" in document
    assert "Apache License" in document
    assert "Plotly.js 3.7.0" in document
    assert "Permission is hereby granted" in document
    assert "colorscale:'Cividis'" in document
    assert "OKABE_ITO" in document
    assert "MOCKUP — SCHEMATIC, NOT DATA" in document
    assert "first fitted boundary" in document
    assert "no contraction (κ = 1)" in document

    tab_positions = [document.index(f'"key":"{key}"') for key, _ in TAB_ORDER]
    assert tab_positions == sorted(tab_positions)

    collector = _TagCollector()
    collector.feed(document)
    assert collector.script_sources == []
    assert collector.external_resources == []


def test_offline_report_allows_https_citations_but_rejects_external_fetches(tmp_path) -> None:
    readme = tmp_path / "linked-readme.md"
    readme.write_text(
        "# Linked specification\n\nSee [the source paper](https://example.org/paper).\n",
        encoding="utf-8",
    )

    output = build_report(tmp_path / "MOCKUP.html", mode="mockup", readme_path=readme)
    document = output.read_text(encoding="utf-8")
    assert '<a href="https://example.org/paper">the source paper</a>' in document

    for external_resource in (
        '<script src="https://example.org/code.js"></script>',
        '<img src="https://example.org/figure.png">',
        '<img src="figures/local-but-not-embedded.png">',
        '<img srcset="data:image/png;base64,AAAA 1x, https://example.org/leak.png 2x">',
        '<link rel="stylesheet" href="https://example.org/style.css">',
        '<link rel="stylesheet" href="styles/local.css">',
        '<iframe src="https://example.org/embed"></iframe>',
        "<style>body { background: url(https://example.org/bg.png); }</style>",
        "<style>body { background: url(../figures/bg.png); }</style>",
        "<style>@import 'local.css';</style>",
        '<meta http-equiv="refresh" content="0;url=https://example.org">',
        '<form action="https://example.org/leak"></form>',
    ):
        with pytest.raises(ValueError, match="external resources"):
            _assert_offline(external_resource)


def test_embedded_spec_escapes_raw_active_html() -> None:
    malicious = """# Spec
<script>fetch("https://example.org/leak")</script>
<meta http-equiv="refresh" content="0;url=https://example.org">
<form action="https://example.org/leak"><button>send</button></form>
"""
    document = render_report(make_mock_payload(), malicious)

    assert "&lt;script&gt;fetch" in document
    assert '<meta http-equiv="refresh" content="0;url=https://example.org">' not in document
    assert '<form action="https://example.org/leak">' not in document
    assert "Content-Security-Policy" in document
    _assert_offline(document)


@pytest.mark.parametrize(
    "target",
    (
        "javascript:alert(1)",
        "javas%63ript:alert(1)",
        "data:text/html,boom",
        "file:///etc/passwd",
        "//example.org/leak",
        "/etc/passwd",
        "../outside.html",
    ),
)
def test_embedded_spec_rejects_unsafe_markdown_link_targets(target: str) -> None:
    with pytest.raises(ValueError, match="external resources"):
        render_report(make_mock_payload(), f"[unsafe]({target})")


def test_mock_report_stage_contract_binds_html_payload_and_exact_inventory(tmp_path) -> None:
    output = build_report(tmp_path / "MOCKUP.html", mode="mockup", readme_path="README.md")
    payload = make_mock_payload()
    store = ArtifactStore(tmp_path / "artifacts")
    stage = DEFAULT_STAGES.get("report.build")
    base_files = {
        "report.html": output.read_bytes(),
        "report_payload.json": canonical_json_bytes(payload.model_dump(mode="json")),
        "spec.md": Path("README.md").read_bytes(),
    }
    media_types = {
        "report.html": "text/html; charset=utf-8",
        "report_payload.json": "application/json",
        "spec.md": "text/markdown; charset=utf-8",
    }
    valid = store.put_files(
        base_files,
        kind="report/mockup",
        metadata={"result_schema_version": 1},
        media_types=media_types,
    )
    assert validate_stage_output(valid, stage, store) is None

    extra = store.put_files(
        {**base_files, "extra.bin": b"unexpected"},
        kind="report/mockup",
        metadata={"result_schema_version": 1},
        media_types=media_types,
    )
    with pytest.raises(PipelineError, match="exactly HTML"):
        validate_stage_output(extra, stage, store)

    unbound = store.put_files(
        {**base_files, "report.html": b"<!doctype html><html>MOCKUP</html>"},
        kind="report/mockup",
        metadata={"result_schema_version": 1},
        media_types=media_types,
    )
    with pytest.raises(PipelineError, match=r"visible MOCKUP warning|bound|exactly match"):
        validate_stage_output(unbound, stage, store)


def test_load_payload_rejects_duplicate_json_keys(tmp_path) -> None:
    payload_path = tmp_path / "duplicate.json"
    payload_path.write_text('{"mode":"mockup","mode":"real"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key 'mode'"):
        load_payload(payload_path)
