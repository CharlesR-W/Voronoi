# Implementation handoff

Last updated: 2026-08-17 (US/Pacific)

## Scientific status

The infrastructure-first milestone is implemented. Two bounded validation
targets are runnable:

- `gate.mechanical`: seeded bank materialization, distinct hash-pinned
  train/test source files, geometry round trips, exact split-model identity,
  per-cut JVP checks, and synthetic generator/twirl invariants on the pinned
  seed-0 substrate. This is not yet a row-level dataset-overlap audit.
- `gate.synthetic_exact`: a narrow noiseless exhaustive $(2,3)$ recovery
  subgate with independent held-out primitives and one resumable shard per
  instance.

Neither target is a phenomenon result. No five-checkpoint formation geometry,
coarse boundary enrichment, snapping/recovery, three-seed confirmation,
sampled synthetic sweep, or real-algebra evidence has been run. Those stages
are `PLANNED`, and the runner refuses a target whose dependency closure includes
one of them. Real-report CLI mode is also disabled; only the watermarked
`MOCKUP` is publishable.

## Implemented infrastructure

- Python 3.12 `uv` project with a checked-in lockfile and strict Pydantic/YAML
  configuration. Duplicate keys, coercive scalar types, non-finite numbers,
  unsupported strategies, and inconsistent axes are rejected.
- Immutable SHA-256-addressed artifacts with canonical JSON, atomic
  publication, checksum verification, safe payload paths, and stage-specific
  semantic output contracts.
- SQLite execution index with concurrency-safe initialization, atomic claims,
  cross-run cache lineage, generation-checked cache invalidation/re-election,
  explicit reclaim, interruption cleanup, append-only attempt/cache-election
  histories, and immutable successful-run receipts. Scientific artifacts
  remain cache-invariant across consumer runs.
- Immutable run provenance and database-independent receipts. Receipts bind the
  complete config, source identity, ordered stage closure, artifacts, cache
  producers, attempts, gate rules/results/authorizations, and exact-instance
  shard/reducer evidence.
- Source-drift guards before resume/cache/handler boundaries and after handlers.
  A long run aborts rather than publishing under a stale initial source hash.
  Git identity is also bound to the bytes of the actually imported package.
- Versioned semantic `SeedDeriver` namespaces for probes, sites, bootstrap
  plans, mechanics, and every exact synthetic instance.
- A hash-pinned, read-only Tracking2 adapter. It verifies the model source,
  five weight-only checkpoints, separate CIFAR train/test Parquet files, and
  legacy transplant JSON before use; it never imports Tracking2 runners or its
  currently dirty validators. The immutable input artifact preserves the exact
  manifest and raw transplant bytes, then reconstructs its normalized summary
  from them during verification.
- Reusable shard executor/reducer machinery. `exp2.exact` is the first default
  proof of use: `runtime.workers=0` is sequential, positive values use bounded
  threads, and scheduling does not change shard scientific identity.
- Model-independent Experiment 1 primitives for coordinate metrics,
  codebooks, nulls, interventions, boundary energy, image bootstrap summaries,
  activation shard serialization, and descriptive transplant joins.
- Synthetic generator, symmetry/twirl, support decomposition, alignment,
  exhaustive search, sampling, and per-instance evidence persistence.
- Caller-scoped semantic-validation memoization. A successful expensive
  producer replay is reused within one runner/gate/receipt chain only after
  rechecking its artifact, direct upstreams, and exact reducer/shard closure;
  failures and changed config/contracts/source never reuse it.
- Self-contained, offline Plotly/MathJax report builder. The current report is
  interpretation-first and visibly `MOCKUP`; ordinary citation hyperlinks are
  allowed, while fetch-bearing external resources and raw active Markdown HTML
  are rejected. The artifact embeds both third-party license texts, preserves
  the spec snapshot, and must exactly rerender from that snapshot plus its typed
  payload.
- A Python 3.11/3.12 CI matrix covering Ruff, formatting, the full core
  suite, and true multiprocess artifact/index initialization contention, plus
  a Python 3.12 CPU job that installs the optional ResNet dependencies and
  exercises the adapter/mechanical path.

`runtime.deterministic: true` currently means seeded, versioned construction
for runnable CPU paths, not a cross-hardware bitwise guarantee. Accelerator and
BLAS/thread determinism must be resolved and recorded before planned training
or activation stages become runnable.

## Runnable DAG

```text
inputs.tracking2 -> exp1.probe_banks -> exp1.mechanical -> gate.mechanical

exp2.exact (per-instance shards + reducer) -> gate.synthetic_exact

report.build (zero-dependency MOCKUP only)
```

`gate.mechanical` also directly binds the saved probe-bank artifact. Gate
outcomes are reconstructed from strict payloads and compared with the exact
configured rule; a self-asserted `PASS`, mismatched rule, or unauthorized
override is rejected on fresh output, cache hit, resume, CLI inspection, and
receipt verification.

## External substrate and caveats

The sibling root is configured, never hard-coded, as
`../Experiments/Tracking2`. Exact hashes live in
`configs/inputs/tracking2_seed0.yaml`.

The five checkpoints contain only 42 fp32 tensors / 11,169,162 parameters—no
optimizer, scheduler, or RNG state. The legacy transplant file also lacks
modern provenance fields. Consequently the input manifest is labeled
`lineage_quality: exploratory_legacy`. The sibling working tree contains a
documented ResNet/VGG regression; do not execute or import its criticality
runners or validators. Only the verified `tracking2.models` bytes are loaded.

The external model/checkpoint/dataset bytes are hash-pinned but not vendored,
so this repository alone is not clone-complete. The two mathematical design
notes are archived under `references/notes/`; the motivating primary
plateau/boundary literature is still unidentified.

Receipts prove content integrity, declared lineage, and conformance to the
versioned result contracts. They are not cryptographic attestations that a
trusted machine ran the code, and they do not independently recompute every
scientific quantity from raw tensors. In particular, the mechanical validator
reconstructs every reported identity/JVP statistic from saved logits and JVP
vectors, but source provenance—not an independent model execution—binds those
vectors to the pinned model. Raw provenance also retains local absolute paths
(`cwd`, Git root, Python executable, and input locations); redact or restrict
the receipt before publishing it outside the trusted research environment.

## Commands

```bash
uv sync --all-extras
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m ruff format --check src tests

.venv/bin/voronoi-lab validate --inputs --json
.venv/bin/voronoi-lab plan --json
.venv/bin/voronoi-lab run --until gate.mechanical --json
.venv/bin/voronoi-lab run --until gate.synthetic_exact --json
.venv/bin/voronoi-lab run --stage report.build --json

# Verify a successful run without consulting SQLite:
.venv/bin/voronoi-lab run receipt RECEIPT_ARTIFACT_ID --json
```

The standalone receipt command verifies immutable integrity and historical
contracts. It deliberately reports current-source compatibility as
`NOT_CHECKED` and current semantic replay as `SKIPPED`; receipt publication in
the runner is the path that requires matching current source and strict
semantic replay.

For a crashed `RUNNING` record, inspect it and reclaim only with an explicit
reason:

```bash
.venv/bin/voronoi-lab run inspect RUN_ID --json
.venv/bin/voronoi-lab run reclaim RUN_ID STAGE --reason "worker host terminated" --json
```

## Provisional choices intentionally left replaceable

- The exact motivating plateau/boundary definition and primary sources.
- Codebook assignment-stability/refit details and scale normalization.
- Empirical-chord/off-cloud construction and the nonuniform boundary path grid.
- One-seed coarse and functional gate statistics.
- The sampled synthetic “easy regime,” baselines, and calibration details.
- The confirmation training recipe is explicit but marked `unfrozen`.

`protocol.mode: confirmatory` is rejected until a future schema supports a
registered full-protocol hash. Diagnostic overrides are scoped per gate and
carry target, reason, actor, and timestamp; they never count as literal `PASS`.

## Recommended next work

1. Archive and audit the motivating primary papers; freeze the operational
   plateau/boundary estimand against those sources.
2. Implement `exp1.activations` using the existing shard executor and typed
   activation payloads. Materialize the three configured input recipes once.
3. Implement codebook/null/static-geometry/boundary handlers and the coarse
   exploratory gate. Keep image-level bootstrap units and continuous
   trapezoidal path-energy weighting explicit.
4. Only if the coarse gate passes, implement snapping/recovery and freeze the
   three-seed protocol. Do not enable nominal confirmatory mode first.
5. In parallel, implement sampled synthetic recovery/null calibration. Keep it
   separate from the already passing tiny exact optimization subgate.
6. Enable measured report assembly only from verified receipt-linked typed
   artifacts, with negative results and gate failures preserved.

The canonical scientific specification remains `README.md`; update this file
after every material infrastructure or evidence milestone.
