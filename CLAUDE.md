# Voronoi lab project context

`README.md` is the authoritative scientific specification and claim ledger.
`docs/HANDOFF.md` is the implementation-status and continuation document; update it
after material milestones instead of putting mutable status here.

## Durable boundaries

- Treat the configured sibling `../Experiments/Tracking2` as a read-only external input.
  Enter only through `configs/inputs/tracking2_seed0.yaml` and
  `voronoi_lab.exp1.tracking2`; never import its experiment runners or validators.
- Scientific stage outputs are immutable, content-addressed artifacts. Mutable run
  state and cross-run cache consumption live only in the SQLite run index.
- Scientific artifact identity is cache-invariant: stage signature, selected config,
  source/environment identity, and upstream artifact IDs. Per-run production or cache
  consumption provenance belongs in the SQLite stage record, never in the artifact ID.
- Randomness is derived from the configured root seed with semantic namespaces.
  Bootstrap and probe units are images, not spatial sites.
- Experiment 2 matrices use destination rows, source columns, C-order product-state
  indexing, and float64 oracle calculations.
- Unresolved scientific choices must remain explicit configuration or strategy
  parameters. Do not silently turn scaffold defaults into confirmatory protocols.
- MOCKUP and future measured report payloads share one strict schema. Every schematic
  title and plot must retain a visible `MOCKUP` label and watermark. Real report
  generation is disabled until payloads are assembled from verified, receipt-linked
  stage artifacts; arbitrary local payloads are not evidence.

## Routine checks

```bash
uv sync --all-extras
.venv/bin/python -m pytest
.venv/bin/python -m ruff check src tests
.venv/bin/python -m ruff format --check src tests
```

No Experiment 1 phenomenon claim is warranted until the corresponding configured
gate is measured and passes. The legacy Tracking2 weights have exploratory lineage.
