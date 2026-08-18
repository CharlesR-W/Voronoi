# Experiment 1 continuation handoff

Last updated: 2026-08-18 (US/Pacific)

## Read order

1. [`../STATUS.md`](../STATUS.md) -- current milestone and claim boundary.
2. [`../RESULTS.md`](../RESULTS.md) -- measured pilot and animation links.
3. [`../artifacts/EXPERIMENT_1_DATA.md`](../artifacts/EXPERIMENT_1_DATA.md) -- exact
   artifact IDs, child shards, arrays, and verification commands.
4. [`../references/activation-plateau-source-audit.md`](../references/activation-plateau-source-audit.md)
   -- which requested element came from which source.
5. Relevant Experiment 1 sections in [`../README.md`](../README.md) for the broader
   candidate-cell gates that remain planned.

## Completed in this milestone

- Identified Stefan Heimersheim and separated four distinct motivating source methods.
  Two arXiv PDFs were read from hash-pinned local audit copies; the public repository
  records official links and hashes but deliberately does not redistribute the PDFs.
- Audited and strict-loaded complete seed-0 epoch 0/1/5/20/100 trajectories for the
  Tracking2 ResNet-18 and VGG-19+BN models. Added a read-only, hash-pinned VGG input
  stage without importing the sibling project's runners.
- Added a deterministic three-class Gaussian-mixture task and normalization-free,
  four-block residual MLP with numeric NPZ checkpoints.
- Added collection for:

    - real versus empirical-covariance-Gaussian centers;
    - paired covariance-Gaussian perturbation targets and 37-point response paths;
    - complete frozen CNN host contexts and exact site intervention vectors;
    - $17\times17$ local response/Jacobian planes with raw $DT$ fields everywhere and
      residual-update $D(T-I)$ fields for residual transitions;
    - Hutchinson next-transition site Jacobians and ResNet $J-I$ estimates; and
    - three-real-anchor, three-context $21\times21$ RGB/output-distance fields.
- Added optional deterministic CUDA execution for `exp1.plateau.cifar`; saved arrays
  remain CPU NumPy float32 with exact runtime backend provenance.
- Added four fixed-layout, synchronized GIF bundles with static fallbacks and strict
  media/readback validation.
- Ran and integrity-verified the canonical protocol-v2 CPU pilot
  `exp1-cpu-seed0-20260817-v4`.

The key implementation files are:

- `src/voronoi_lab/exp1/synthetic_task.py`
- `src/voronoi_lab/exp1/tracking2_vgg.py`
- `src/voronoi_lab/exp1/surface_geometry.py`
- `src/voronoi_lab/exp1/plateau_collection.py`
- `src/voronoi_lab/exp1/animation.py`
- stage integration in `src/voronoi_lab/{config,pipeline,stage_handlers,cli}.py`
- focused tests under `tests/test_exp1_*` and `tests/test_tracking2_vgg_adapter.py`

## Scientific semantics to preserve

Do not collapse these quantities under one name:

1. `path_response_l2` and `path_response_kl` are the small-model analogue of the
   Heimersheim--Mendel real-versus-covariance-Gaussian response experiment.
2. `anchor_output_distances` and the derived RGB fields are a Janiak et al. stable-region
   analogue. They are not Voronoi assignments and are not Jacobian fields.
3. Plane finite differences and Hutchinson estimates are new hybrid diagnostics inspired
   by the later Shinkle--Heimersheim Jacobian work.
4. A fake center is a marginal site vector inserted into the same real frozen host as
   its paired real center. It is intentionally not described as a complete fake CNN
   activation state.
5. Preserve raw transition fields for every architecture. For residual transitions,
   inspect $J-I$ beside raw $J$ and display plane-restricted $D(T-I)$ by default because
   the identity skip dominates raw same-site and plane norms. VGG has no residual-update
   field and displays plane-restricted $DT$.

The canonical collection has four centers per kind and one legacy training seed. It is a
visualization/data-contract pilot, not a powered plateau test. The ResNet/VGG comparison
is descriptive and multiply confounded. Do not infer skip-connection causality.

## Canonical data and immutability

Use only the v4 IDs recorded in `artifacts/EXPERIMENT_1_DATA.md`. Earlier v1/v2 objects
remain in the local store. The immutable v3 predecessor is also retained, but it
predates the protocol-v2 residual-adjusted plane-display contract and is historical.
Never edit any object in place. A methodological change requires a new protocol
version, run ID, and content-addressed output.

The sibling Tracking2 root is `../Experiments/Tracking2`. It is read-only for this
project. ResNet and VGG checkpoint lineage is explicitly `exploratory_legacy`; do not
retroactively claim missing optimizer, RNG, package, Git, or dataset provenance.

## Public release

The source repository is public at <https://github.com/CharlesR-W/Voronoi>. Curated
GIF/PNG/metadata presentation copies live under `docs/assets/experiment1/`; large
checkpoints, datasets, immutable artifact stores, run databases, generated reports, and
the local environment remain ignored. ArXiv PDF audit copies are also local-only because
their distribution license does not grant this repository a separate redistribution
right. The repository currently has no project-wide open-source license: public
visibility permits viewing, but does not itself grant general reuse rights.

## RunPod note

Two temporary A40 pod setup attempts were deleted without uploading project data. The
workspace upload was blocked by the environment's external-disclosure policy and was not
circumvented. As of 2026-08-17 19:51 UTC there were no running or stopped recent pods,
the current spend rate was `$0/hour`, and the billing endpoint had posted
`$0.0154466592` total for the two attempts. This is safely below the authorized `$10`
cap; later billing reconciliation could still adjust the final amount.

## Recommended next analysis

1. Choose a scalar plateau statistic and matched null before expanding the run. The
   source papers use several different operationalizations; none should be silently
   substituted for another.
2. Use the stored raw response paths to estimate variability over centers/directions and
   decide whether the current site/transition is worth scaling.
3. If proceeding, add model seeds and more center hosts. Reuse fixed image identities and
   global animation normalization, but publish under a new protocol/run.
4. For an architecture claim, train matched ResNet and VGG-family models under a shared
   recipe. The current legacy trajectories are insufficient.
5. Keep the broader candidate-cell, snapping, commutant, and factorization work gated.

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m ruff format --check src tests

.venv/bin/voronoi-lab artifact verify \
  --artifact-root artifacts/exp1-measured --json \
  a5afce7cdeb6d5d91e6408b5d7990c10cd2210906467ebcc6ba0aeb579ff2521 \
  6a9e8cd36b73cc5d3434f43b3420167078a7176826c1b8c6dd39c61fd03d321b \
  e0090abd43cfd95ae7def9204424dffb71e474d23bf78c91afcd252b8fadf8bf \
  89a9f61b7d5c15a9dfa216075c3a2e9cd57db582e59f9b7c6d0b5555a7074930
```

The concurrent dashboard/mock-up work belongs to another local agent. Preserve its
changes in `README.md`, `reports/`, `src/voronoi_lab/reporting/`, and reporting tests.
