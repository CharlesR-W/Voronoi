# Experiment 1 local data inventory

Last verified: 2026-08-17 (US/Pacific)

This file is the durable pointer to the local, ignored, content-addressed Experiment 1
store. Do not edit an object in place. A revised collection must use a new protocol,
run ID, and artifact ID.

## Canonical run

- run ID: `exp1-cpu-seed0-20260817-v4`
- artifact root: `artifacts/exp1-measured`
- run index: `runs/exp1-measured.sqlite`
- receipt: `89a9f61b7d5c15a9dfa216075c3a2e9cd57db582e59f9b7c6d0b5555a7074930`
- provenance: `4ad6699fb0de79b548625cb5ed6b654e65ffb9b5564e5f4e3be5a36783eab30c`
- config: `configs/pilot.yaml`
- execution: CPU, PyTorch `2.13.0+cu130`, deterministic algorithms enabled,
  persisted floating arrays CPU NumPy float32

The standalone receipt verifier reports `INTEGRITY_VERIFIED`; config, provenance, and
registry compatibility match the current run record. Standalone verification does not
rerun the models and reports semantic replay as skipped by design.

The production run used a frozen `/tmp` source snapshot to avoid concurrent dashboard
edits. Its exact 55-file source inventory matches the current worktree byte-for-byte,
but the provenance correctly records no Git commit. A read-only audit verified 20
unique objects, 117 payload files, 15 checkpoint inventories, and 790 NPZ arrays. All
hashes, sizes, names, dtypes, and shapes match; every NPZ loads with
`allow_pickle=False`; every numeric array is finite; and the saved plane fields replay
exactly from their stored vectors and transition outputs.

## Parent artifact IDs

| Stage | Artifact ID |
|---|---|
| `inputs.tracking2` | `d466e232a2b98c17dccf4c9c69a61202b6dd513e1885bf7c58cd6f9a56e9f3dc` |
| `inputs.tracking2_vgg` | `1c8c45097d05625ff8383c6822c96560097055a2dc72af99518e7866a3716274` |
| `exp1.synthetic_task` | `348eb3cc2d9b899d3df9c1f7a19820f51cd49a1f3f04d3abe6f4ed50c0e55e4c` |
| `exp1.plateau.synthetic` | `6a9e8cd36b73cc5d3434f43b3420167078a7176826c1b8c6dd39c61fd03d321b` |
| `exp1.probe_banks` | `1da31e3d2a6d0d114f32d84f794c36ae16643af7673fd5f1889bceff2a5afe3d` |
| `exp1.plateau.cifar` | `a5afce7cdeb6d5d91e6408b5d7990c10cd2210906467ebcc6ba0aeb579ff2521` |
| `exp1.plateau.animations` | `e0090abd43cfd95ae7def9204424dffb71e474d23bf78c91afcd252b8fadf8bf` |
| run receipt | `89a9f61b7d5c15a9dfa216075c3a2e9cd57db582e59f9b7c6d0b5555a7074930` |

The CIFAR parent `summary.json` resolves the ten immutable child shards:

| Architecture | Epoch | Child artifact ID |
|---|---:|---|
| ResNet | 0 | `74dd6682157722ad78609f1fcf2a981db13453878f6a3daf099234ec79a18a7f` |
| ResNet | 1 | `3ca5236386a6f909fabf1ef79f14c72e11bee3f8d843f9221a8157973aec156a` |
| ResNet | 5 | `aed4965b425322a2c09598f7ef88a416462f63faaaa91ff139a3f48b8d56cf0e` |
| ResNet | 20 | `3628027c00ea59ce33940873b85a465daf66eb22cfc3872f0bdbc7b87cb2e3d7` |
| ResNet | 100 | `c259fc8335f7c19f76bc018ae1b61f4d882f2a4430e4f95a5f231894b5649718` |
| VGG | 0 | `accf5582a065b4d630e8b710bbfc059c2a5c19261d6ff9d6c9a82b2b5febb763` |
| VGG | 1 | `a153ef46fcd493fcc6991e9df50cc44296c2ba42828d92b2b371a88d19d4948c` |
| VGG | 5 | `39f7a4ef332eb95e950fd5a1f103d6f6e3f1a9b7ae4ada4ecc080573cd5d2ba4` |
| VGG | 20 | `a8c71f7c1a464a46abc1d0a15a31a88b7020eb175cbce1feea37125c2a482915` |
| VGG | 100 | `0cca7315428fb97f71c97997fadcdd613b83810eb343e2c6e738845d52a7bd3b` |

These ten children occupy about 52 MiB. The whole local store is larger because it
retains superseded v1/v2 attempts and the immutable v3 predecessor; use only the v4 IDs
above for current analysis. V3 predates the residual-adjusted plane-display contract
and is historical, not corrupt.

## What each checkpoint shard contains

Each child has `arrays.npz`, `metadata.json`, and `inventory.json`. NumPy files must be
loaded with `allow_pickle=False`. The exact inventory is in metadata; the main groups
are:

- fixed provenance: train/test image IDs, labels, cut, spatial site, full checkpoint
  identity, and runtime backend;
- covariance fit: `train_site_vectors` with shape `(256, 128)`;
- centers and hosts: four real and four empirical-covariance-Gaussian site vectors,
  matched full `(4, 128, 16, 16)` frozen host activations, Gaussian coefficients, and
  realized Gaussian direction targets;
- paths: coefficients `(37,)`, exact interventions `(2, 4, 8, 37, 128)`, logits,
  next-transition sites, logit $L_2$/KL response, and logit/transition directional
  Jacobians;
- local planes: exact grids `(2, 4, 17, 17, 128)`, output response, and the raw
  `local_transition_plane_jacobian`; residual shards additionally store
  `local_residual_update_plane_jacobian`, computed by differentiating $T(z)-z$ before
  taking a norm;
- full same-site estimates: eight saved Rademacher probes and Hutchinson next-transition
  Frobenius estimates for real/fake centers; ResNet shards additionally store the
  residual-adjusted $J-I$ probes and estimates; and
- three-anchor fields: three distinct image/label/host contexts, exact
  `(21, 21, 128)` plane vectors, per-context logits and transition sites, raw output
  distances, source-style per-frame RGB, and raw
  `anchor_transition_plane_jacobian_by_context`; residual shards additionally store
  `anchor_residual_update_plane_jacobian_by_context`.

The paired Gaussian coefficient streams and held-out image identities are fixed across
architectures and checkpoints. The full host activations make later site interventions
replayable without pretending that a marginal site vector is a complete on-data CNN
state.

## Synthetic data

The task artifact stores the generated train/test points, full training progress, and
five numeric NPZ checkpoints for a width-32 four-block residual MLP. The plateau artifact
contains the same response/Jacobian/anchor field families at every checkpoint under
`files/checkpoints/epoch_*/`. Test accuracy is 2/3 at initialization and 1.0 from epoch
1 onward.

## Animation data

The animation artifact contains four bundles under `files/animations/`. Each bundle has
a GIF, final PNG, and metadata JSON. `files/summary.json` records the raw source artifact
IDs, estimand classifications, timing, and the fixed cross-checkpoint normalization.
The five checkpoint data frames are followed by one presentation-only conclusion hold;
metadata distinguishes those six encoded frames.

The display policy is architecture-specific and explicit: synthetic and ResNet panels
use the plane-restricted residual-update fields $D(T-I)$, while VGG panels use raw $DT$.
The combined figure shares one numerical Jacobian color scale for descriptive viewing,
but the row operators differ; RGB channels are normalized separately per architecture
and anchor channel. Neither treatment makes the rows causally comparable.

Small byte-identical presentation copies of the four GIF/PNG/metadata bundles are
tracked under `docs/assets/experiment1/` so a public clone can display the results.
Those copies are conveniences, not canonical scientific artifacts; provenance and
verification continue to resolve through the content-addressed IDs above.

## Verify or reproduce

```bash
.venv/bin/voronoi-lab artifact verify \
  --artifact-root artifacts/exp1-measured --json \
  a5afce7cdeb6d5d91e6408b5d7990c10cd2210906467ebcc6ba0aeb579ff2521 \
  6a9e8cd36b73cc5d3434f43b3420167078a7176826c1b8c6dd39c61fd03d321b \
  e0090abd43cfd95ae7def9204424dffb71e474d23bf78c91afcd252b8fadf8bf \
  89a9f61b7d5c15a9dfa216075c3a2e9cd57db582e59f9b7c6d0b5555a7074930

.venv/bin/voronoi-lab run receipt \
  89a9f61b7d5c15a9dfa216075c3a2e9cd57db582e59f9b7c6d0b5555a7074930 \
  --artifact-root artifacts/exp1-measured \
  --run-index runs/exp1-measured.sqlite --json

# New runs must use a new run ID. CPU takes roughly a few minutes on this host.
.venv/bin/voronoi-lab run -c configs/pilot.yaml \
  --until exp1.plateau.animations \
  --run-id NEW_RUN_ID \
  --artifact-root artifacts/exp1-measured \
  --run-index runs/exp1-measured.sqlite --json
```

The real-data stages require the hash-pinned sibling
`../Experiments/Tracking2`. Both input lineages are exploratory legacy data, and the
VGG manifest intentionally records weaker historical provenance rather than inventing
missing source or dataset attestations.
