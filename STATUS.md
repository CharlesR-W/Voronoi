# Project status

Last verified: 2026-08-18 (US/Pacific)

## Current milestone

Experiment 1 activation-geometry collection is implemented and has one verified local
measured run under protocol v2. The run covers a normalization-free synthetic residual
MLP and CIFAR-10 with matched ResNet-18 and VGG-19+BN transition shapes at epochs 0, 1,
5, 20, and 100. It publishes raw intervention data, source-analogue response fields,
explicitly new Jacobian diagnostics, and four synchronized GIFs.

Canonical run: `exp1-cpu-seed0-20260817-v4`

Public repository: <https://github.com/CharlesR-W/Voronoi>

- receipt: `89a9f61b7d5c15a9dfa216075c3a2e9cd57db582e59f9b7c6d0b5555a7074930`
- CIFAR collection: `a5afce7cdeb6d5d91e6408b5d7990c10cd2210906467ebcc6ba0aeb579ff2521`
- animations: `e0090abd43cfd95ae7def9204424dffb71e474d23bf78c91afcd252b8fadf8bf`
- integrity: 20 objects, 117 payload files, 15 checkpoint inventories, and 790 NPZ
  arrays passed hash, inventory, finite-value, and semantic replay checks; the receipt
  status is `INTEGRITY_VERIFIED` with registry/config/provenance compatibility matched

The immutable v3 run remains in the store as a historical predecessor. It is
superseded because its animations displayed identity-dominated raw transition fields
for residual models; it must not be relabelled as satisfying the protocol-v2 display
contract.

Read [RESULTS.md](RESULTS.md) for the measured summary and
[artifacts/EXPERIMENT_1_DATA.md](artifacts/EXPERIMENT_1_DATA.md) for the exact object and
array inventory. The literature-to-estimand mapping is in
[references/activation-plateau-source-audit.md](references/activation-plateau-source-audit.md).

## Runnable Experiment 1 stages

```text
inputs.tracking2 --------+---------------------> exp1.probe_banks --+
                         |                                         |
inputs.tracking2_vgg ----+-----------------------------------------+--> exp1.plateau.cifar --+
                                                                                          |
exp1.synthetic_task --------------------> exp1.plateau.synthetic -------------------------+--> exp1.plateau.animations
```

`runtime.device: cuda` is supported only by `exp1.plateau.cifar`; persisted floating
arrays remain CPU NumPy float32. The canonical run used CPU. The sibling Tracking2
inputs are hash-pinned and read-only but not vendored, so a fresh clone is not
self-contained.

## Scientific boundary

This is a data-collection milestone, not evidence that stable cells, activation
plateaus, or residual-stream causality have been established. Specifically:

- real-versus-covariance-Gaussian response curves are an architecture-adapted analogue
  of the Heimersheim--Mendel method;
- three-anchor RGB fields are an architecture-adapted Janiak et al. construction and
  are neither literal Voronoi assignments nor Jacobian fields;
- two-direction and Hutchinson Jacobian quantities are new hybrid CNN diagnostics;
  protocol v2 preserves raw $DT$ everywhere, displays plane-restricted $D(T-I)$ for
  synthetic/ResNet residual transitions, and displays plane-restricted $DT$ for VGG;
- ResNet versus VGG is a single-seed descriptive comparison confounded by batch
  normalization, parameter count, regularization, and training history; and
- both checkpoint trajectories have `exploratory_legacy` lineage. They lack optimizer,
  scheduler, and RNG state and are unsuitable for a confirmatory architecture claim.

The older candidate-cell formation, boundary, snapping/recovery, three-seed, and real
algebra gates remain planned. The new activation-geometry collection does not make those
gates pass.

## Next useful work

1. Choose a quantitative plateau statistic and matched null before increasing scale.
2. Increase centers/directions and add seeds only after freezing that statistic and
   null. The present four centers per kind are a visualization pilot.
3. If the ResNet/VGG contrast remains scientifically important, train matched models
   under one controlled recipe; do not treat the legacy pair as a causal skip-connection
   ablation.
4. Keep the commutant, factorization, and full candidate-cell program gated as specified
   in the README.

The active continuation notes are in [docs/HANDOFF.md](docs/HANDOFF.md).
