# Voronoi Residual Computation Lab

> **Status: activation-geometry collection implemented and one local exploratory
> pilot integrity-verified; candidate-cell phenomenon gates remain unrun.** The
> repository contains a strict stage DAG, immutable artifacts and receipts,
> hash-pinned ResNet/VGG checkpoint adapters, a trained synthetic residual task,
> real/fake response and Jacobian fields, synchronized checkpoint GIFs, mechanical
> checks, and a self-contained `MOCKUP` report. The measured pilot is not evidence
> that residual candidate cells form, snap, or recover.

This project asks whether continuous residual computation admits a useful
finite-state description: whether learned representations develop functionally
flat regions separated by narrow sensitive fissures, and whether states become
increasingly committed to those regions over training.

The order matters. Fitting centroids always creates a Voronoi partition. The
scientific question is whether that partition is stable on held-out data,
predictively adequate, and causally useful. The geometry is useful only if it
tracks something the network actually does.

## Motivation: flats, fissures, and commitment

The motivating whiteboard picture is a continuous state cloud that may become
discrete-like without acquiring a literal hard quantizer. A learned map could have
broad interiors in which moving an activation changes the downstream computation
very little, separated by narrow **fissures** in which a small displacement changes
the computation sharply. Candidate centroids and their Voronoi boundaries are probes
for that picture, not evidence that the picture is true.

This suggests several connected questions:

- **Functional flatness.** Are downstream predictions unusually insensitive to
  within-region activation perturbations, relative to matched directions and null
  representations?
- **Fissures.** Is sensitivity concentrated in narrow regions, and do those regions
  align with independently fitted candidate-cell boundaries rather than arbitrary
  locations along the same paths?
- **Basin commitment.** Do representations become more stably assigned to candidate
  regions over training, and, separately, do perturbed states recover toward the clean
  downstream region? The first is assignment commitment; the second is dynamical
  recovery. Neither alone licenses attractor language.
- **Noise across scales.** If residual noise is swept over logarithmic or power-law
  scales, is the response scale-free and smooth, or is there a characteristic regime
  in which cell crossings and functional damage concentrate? Either pattern is a
  diagnostic; a kink, threshold, or power law does not by itself prove discrete
  computation.
- **Flatness and plasticity.** Does activation-space functional flatness covary with
  Fisher- or parameter-space flatness and with independent measures of module
  plasticity? These live in different spaces and must be measured separately; the
  whiteboard connection is a comparison to test, not an equivalence.
- **Where and when.** Are these effects strongest in middle layers, do they emerge
  from early to late training, and do they differ between residual and non-residual
  architectures, real and matched fake states, or different classes?

The immediate experiments therefore combine geometry, interventions, and training
time. They ask whether apparent cells are stable, whether their interiors are
functionally flat, whether their boundaries behave like fissures, whether snapping
or finite downstream recovery is exceptional, and how those quantities relate to
module criticality. The current pilot collects some of the required activation data;
it does not yet answer those questions.

## Start here

Install the locked environment and inspect the dependency-closed plan:

```bash
uv sync --all-extras
.venv/bin/voronoi-lab validate --inputs --json
.venv/bin/voronoi-lab plan --json
```

The currently runnable scientific-validation target is `gate.mechanical`. The
exploratory collection stages `exp1.synthetic_task`, `exp1.plateau.synthetic`,
`exp1.plateau.cifar`, and `exp1.plateau.animations` are also runnable;
`report.build` still creates only the
visibly watermarked schematic. Every successful `run` publishes an immutable,
database-independent receipt and prints its artifact ID. For example:

```bash
.venv/bin/voronoi-lab run --until gate.mechanical --json
.venv/bin/voronoi-lab run --until exp1.plateau.animations --json
.venv/bin/voronoi-lab run --stage report.build --json
```

The motivating activation-plateau request was traced to several distinct methods,
not one published estimand. The source audit, exact code revisions, official paper
links, and local audit-copy hashes are recorded in
[`references/activation-plateau-source-audit.md`](references/activation-plateau-source-audit.md).
The first CNN collection keeps source analogues separate from explicitly new hybrid
Jacobians and makes no exact-replication claim.

Current inputs are local-only and are **not clone-safe**:

- the ResNet definition, configuration, checkpoints, and transplant artifact
  are in the sibling `../Experiments/Tracking2/` project; their presence was
  verified locally on 2026-08-16, but they are not vendored here;
- the VGG-19+BN definition and five seed-0 checkpoints are also hash-pinned in
  that sibling project under an explicitly weaker `exploratory_legacy` lineage.

The canonical local pilot, raw array inventory, and animation links are in
[`RESULTS.md`](RESULTS.md) and
[`artifacts/EXPERIMENT_1_DATA.md`](artifacts/EXPERIMENT_1_DATA.md). Current project
state is summarized in [`STATUS.md`](STATUS.md).

The infrastructure boundary is deliberate. Runnable stages use strict,
versioned payload schemas, semantic seed namespaces, content-addressed objects,
cross-run caching, explicit gate rules and authorizations, source-drift checks,
and append-only run receipts. The formation, boundary, snapping, and three-seed
confirmation stages remain `PLANNED` and cannot be accidentally executed.

The intended dependency is:

```text
reported activation plateaus / sharp residual boundaries
                         |
       define the residual unit, metric, probes, and codebook
                         |
       held-out finite-state fidelity and stability
                  +------+------------------+
                  |                         |
       recovery, boundary,         noise-scale and
       and snapping tests          commitment diagnostics
                  |                         |
                  +------------+------------+
                               |
                exploratory comparison with
             module plasticity and transplant damage
```

## Claim ledger

The project deliberately separates claims that are easy to conflate.

| Statement | Current status |
|---|---|
| A fitted codebook induces Voronoi cells | Definition; true by construction |
| Residual states develop a stable, useful finite-state description during training | Primary empirical hypothesis |
| Within-region functional response becomes unusually flat | Primary functional hypothesis |
| Narrow sensitivity fissures align with independently fitted cell boundaries | Stronger, separate hypothesis |
| Assignment and recovery commitment increase through training | Two related but distinct hypotheses |
| Noise-scale response distinguishes scale-free sensitivity from a characteristic crossing regime | Exploratory diagnostic; not yet a gate |
| Activation flatness is related to Fisher flatness or module plasticity | Exploratory comparison; no equivalence assumed |
| Perturbed states are corrected toward the interiors of those cells | Stronger, separate hypothesis |
| Functional sensitivity is concentrated near cell boundaries | Stronger, separate hypothesis |
| Moving a residual to its nearest centroid preserves or improves computation | Planned causal test |
| Recovery is related to module-transplant sensitivity | Exploratory comparison; direction unspecified |

Until the corresponding gates pass, this README uses **candidate cell** rather
than basin, **recovery** rather than attraction, and **boundary sensitivity** or
candidate fissure rather than a Jacobian wall.

Several distinctions are load-bearing:

- A Voronoi partition is not evidence for naturally discrete computation.
- Cluster compactness, a piecewise-flat response, contraction, and a
  high-sensitivity boundary are four different properties.
- A depth-indexed sequence of different feed-forward residual blocks is not an
  autonomous dynamical system. “Attractor” language requires more than finite
  recovery through later blocks.
- A fitted centroid boundary is not automatically a ReLU boundary or a
  boundary of the neural function.
- A finite dataset enumerates observed residuals, not every occupied point or
  ambient-space boundary.
- Tolerance to centroid snapping does not show that the unperturbed network
  itself performs snapping.
- Module-transplant tolerance does not by itself measure activation error
  correction.
- Activation-space response flatness, Fisher or parameter-space flatness, and
  continued trainability are different quantities. Any relationship among them
  must be measured rather than inferred from shared vocabulary.
- A power law, threshold, or other shape in a noise-response curve is not by
  itself evidence for discrete computation; alignment with held-out cells and
  matched nulls remains necessary.

## Experiment 1: do candidate cells form during ResNet training?

### Question

Do trained residual maps become well approximated, on the states they actually
visit, by stable finite codebooks whose within-cell perturbations are
functionally suppressed and whose cell crossings concentrate sensitivity?

The proposed first substrate is a real, finite dataset rather than grokking:
CIFAR-10 and the normalization-free preactivation ResNet-18 V2 already used in
the sibling Tracking2 project. That model has eight addressable residual blocks
and clean post-addition cuts. Read-only seed-0 checkpoints already exist at
epochs 0, 1, 5, 20, and 100, together with same-model final-target module
transplant measurements. Reusing them makes a cheap first phenomenon gate
possible without treating the sibling results as evidence produced by this
project.

The proposed training configuration is the existing one: width 64, SGD with
learning rate 0.05 and momentum 0.9, no weight decay, learning-rate milestones
at epochs 30, 60, and 90, and 100 epochs. A smaller four-block model may be used
only for mechanical smoke tests.

### What counts as one residual state?

At checkpoint $t$ and residual block $\ell$, write the post-addition activation
for image $x$ as

$$
H_{t,\ell}(x)\in\mathbb R^{C_\ell\times S_\ell\times S_\ell}.
$$

The pilot treats the channel vector at spatial site $p$ as the CNN analogue of
one token residual in a language model:

$$
h_{t,\ell}(x,p)\in\mathbb R^{C_\ell}.
$$

This choice is useful because the state can be quantized and intervened on in
the model's native activation tensor without learning a decoder. It also limits
the claim: the pilot concerns sitewise channel states, not a demonstrated
factorization of the entire image activation.

For each checkpoint and cut, standardize channels using means and RMS values
estimated on the codebook-fit bank, then fit centroids
$\{\mu_{t,\ell,c}\}_{c=1}^K$. The candidate cell assignment is

$$
q_{t,\ell}(z)
=
\arg\min_c \|z-\mu_{t,\ell,c}\|_2.
$$

More explicitly, if $\bar h_{t,\ell}$ is the fit-bank channel mean and
$D_{t,\ell}$ is the diagonal matrix of channel RMS values, then

$$
z=D_{t,\ell}^{-1}(h-\bar h_{t,\ell}),
\qquad
h=\bar h_{t,\ell}+D_{t,\ell}z.
$$

All primary directions, boundary distances, and matched intervention norms are
defined in standardized Euclidean coordinates. A standardized displacement
$sv$ is inserted natively as $sD_{t,\ell}v$; the corresponding native-coordinate
normal to a standardized boundary is proportional to
$D_{t,\ell}^{-\top}(\mu_j-\mu_a)$. The raw-Euclidean sensitivity analysis
refits and reevaluates its own codebook in native coordinates rather than
mixing the two metrics.

The primary proposal is $K=32$, with $K\in\{16,64\}$ as required sensitivity
checks. Raw Euclidean distances must also be reported because every Voronoi
claim is metric-dependent. Geometry in residual space, empirical occupancy of
the cells, and the metric later used for operator errors remain separate
objects; a possibly singular centroid Gram matrix is not silently reused as a
positive-definite state-space metric.

Proposed fixed banks are:

- 2,000 training images for codebook fitting;
- 2,000 held-out test images for geometry;
- 256 held-out images for path and Jacobian interventions; and
- at most 32 sampled sites per image, with equal total weight per image.

All uncertainty resampling is by image, never by spatial token. Labels do not
enter codebook fitting. Label and coarse-position associations are reported
only to diagnose whether a cell system is mostly rediscovering class, padding,
or spatial location.

### 1A. Static formation measurements

At every checkpoint and cut, measure held-out normalized distortion

$$
D_{t,\ell,K}
=
\frac{
\mathbb E\|z-\mu_{q(z)}\|_2^2
}{
\mathbb E\|z-\bar z\|_2^2
},
$$

bootstrap stability of held-out assignments, the effective occupied-cell count
$\exp H(q)$, and distance to the nearest candidate-cell boundary. If
$a=q(z)$, the latter is

$$
m(z)
=
\min_{j\ne a}
\frac{
\|z-\mu_j\|_2^2-\|z-\mu_a\|_2^2
}{
2\|\mu_j-\mu_a\|_2
}.
$$

This is the Euclidean distance from $z$ to the complement of its fitted cell in
standardized coordinates. The second-closest centroid by point distance is not
necessarily the closest bisector. The margin should be normalized by a reported
codebook scale before comparison across layers.

The identical pipeline is applied to an epoch-0 network, a global
covariance-matched Gaussian, and a class-conditional Gaussian preserving the
first two moments. If position dependence is large, add a position-conditioned
null or separate border and interior sites. Cell-label and cell-position mutual
information are descriptive diagnostics, not names for what a cell “means.”

A finite-state account gains support only if its held-out compactness and
stability improve over training, survive the metric and $K$ checks, and exceed
the matched nulls. A visually attractive clustering is not a pass.

### 1B. Plateau and boundary-sensitivity measurements

The word *plateau* needs an operational test. The pilot uses two direction
families without claiming that either exactly recovers a data manifold:

- **Empirical chord:** a direction from an anchor token toward a nearby
  observed token assigned to another cell.
- **Off-cloud direction:** a norm-matched random direction orthogonal to the
  leading local-neighbor PCA span.

For a standardized unit direction $v$ from an anchor $z$ in cell $a$, the first
positive crossing of a fitted Voronoi boundary is known exactly:

$$
s^*
=
\min_{j:v^\top(\mu_j-\mu_a)>0}
\frac{
\|z-\mu_j\|_2^2-\|z-\mu_a\|_2^2
}{
2v^\top(\mu_j-\mu_a)
}.
$$

Sweep the dimensionless coordinate $r=s/s^*$ so that $r=1$ is the first fitted
boundary. Insert the native displacement $sD_{t,\ell}v$ into one token of an
otherwise real activation map and measure the next-block directional derivative
and suffix predictive sensitivity. A
finite-difference form convenient for class distributions is

$$
g_{\mathrm{pred}}^2(r)
\approx
\frac{
2D_{\mathrm{KL}}(p_r\,\|\,p_{r+\Delta r})
}{
(\Delta r)^2
}.
$$

Report the fraction of derivative energy near $r=1$, the interval containing
80% of that local energy, and the offset between the derivative peak and the
codebook boundary. Validate automatic differentiation or JVP calculations by
finite differences. A within-path circular-shift null preserves each path's
autocorrelation and derivative distribution while breaking alignment to the
fitted boundary.

A plateau-like result requires low functional response in high-margin
interiors and response concentrated near crossings. If sharpness appears only
in off-cloud directions, the conservative interpretation is off-support
fragility rather than data-relevant discrete computation. The hard quantizer's
own discontinuity is never counted as evidence about the network Jacobian.

#### Exploratory path geometry and embedding diagnostics

The empirical chord is convenient, but it may pass through activation-space
regions that are atypical even when both endpoints are observed states. This is
especially plausible for high-dimensional distributions concentrated in a thin
radial shell: a Euclidean chord can move inward through a low-density region.
That concentration-of-measure concern does not by itself establish that the
activations lie on a smooth manifold, or that any proposed alternative remains
on one.

Before interpreting a plateau measurement, repeat it with the same endpoint
pairs under several declared path constructions:

- the current linear chord;
- a centered, radius-preserving spherical interpolation;
- a piecewise-linear path through a held-out local-neighbor graph; and
- if a separately validated activation generative model becomes available, a
  path constrained by that model.

Spherical interpolation controls the radial-shell failure mode but is not an
on-manifold guarantee. A neighbor-graph path stays close to observed states by
construction but changes path length and direction and can introduce corners.
Compare schemes at matched endpoint pairs and normalized arc length, and report
pathwise nearest-neighbor distance, local-density score, activation norm,
suffix predictive entropy, and the number and location of fitted-cell
crossings. The plateau conclusion is robust only if the qualitative
interior-versus-boundary contrast survives among paths with support diagnostics
comparable to observed activations. Strong effects confined to unsupported
segments are evidence about off-support behavior, not meaningful interpolation
between data states.

As a descriptive visualization, construct two precomputed-distance embeddings
at each selected checkpoint and cut. The first uses pairwise cosine distance
between standardized activations. The second uses the **integrated Jacobian
barrier** between each pair: a symmetrized path cost obtained by integrating, or
discretely averaging, a declared next-block or suffix Jacobian response along the
chosen interpolation path. This is the intended meaning of the whiteboard circles
containing dots: the dots are individual examples, and their plotted geometry is
induced by the accumulated functional barrier between examples.

Run this as a dedicated MNIST experiment using every test digit, not merely a
small visual sample. Each dot represents one MNIST image. For convolutional layers,
freeze an explicit image-level state construction before looking at the plot; do not
silently turn spatial sites into additional dots or change pooling across layers. At
every selected layer and training checkpoint, fit the same
standardized candidate-centroid analysis on a disjoint training bank, assign every
test point, and retain its margin and centroid distance. Retain the complete pairwise
barrier matrix and derive one fixed-identity point layout from it; show fitted centroids
and assignments without presenting projected boundaries as the measured cells. Color
by digit only after computing the unsupervised geometry. The motivating
hypothesis is that, as the network identifies classes, integrated barriers among
same-class examples decrease relative to barriers between classes, producing
increasingly class-aligned neighborhoods. Test that hypothesis on the original
pairwise costs with within-class and between-class summaries and label-permutation
nulls; do not infer it from the two-dimensional circles alone. Because all-pairs
path integration over the full MNIST test set is quadratic and potentially
expensive, any approximation must be declared, convergence-checked against an
exact subset, and kept separate from the stated full-matrix target.

The Jacobian-derived cost must be named precisely and checked for nonnegativity,
symmetry, and triangle-inequality violations rather than assumed to be a metric.
For the present CIFAR-10 substrate, retain the smaller fixed held-out bank and use
class coloring only as a diagnostic.

Fit one aligned layout per distance definition across all checkpoints, keep the
point identities and camera fixed, and animate the two layouts side by side.
Coordinates are comparable through time within a panel, but not geometrically
across the cosine and Jacobian panels. Accompany the animation with the complete
distance or similarity matrices as checkpoint heatmaps using one fixed sample
order and color scale. Quantitative conclusions must come from the original
high-dimensional distances and their association with cells, labels, margins,
and training time, not from apparent clusters in UMAP.

### 1C. On-data and off-data perturbations

The current operational interpretation of “on versus off dataset
perturbations” has two explicit axes:

1. **Input-derived states:** clean held-out images and states induced by ordinary
   crop, flip, or mild color perturbations.
2. **Residual interventions:** empirical-chord, local-covariance, boundary-normal,
   and off-cloud directions applied directly at a cut.

Magnitudes are matched in the declared residual metric. Results are always
split by direction family. Calling all input augmentations “on-manifold” would
make a stronger claim than the experiment earns.

### 1D. Centroid snapping and finite recovery

For a clean standardized token $z$, hold its clean assignment fixed and apply
partial or full snapping

$$
z^{(\alpha)}
=
z+\alpha\bigl(\mu_{q(z)}-z\bigr),
\qquad
\alpha\in\{0,0.25,0.5,1\}.
$$

Invert the channel standardization and insert the result into the native
activation tensor. Test both sparse token subsets and all-token snapping.
Measure predictive KL from the clean output, held-out cross-entropy and
accuracy change, later-cut agreement with the clean cell sequence, and
normalized distance from the clean residual trajectory.

Required matched controls are:

- a same-norm random direction;
- the same-norm direction away from the assigned centroid;
- the same-norm direction toward a different centroid;
- clean identity intervention; and
- centroids fit on an independent training split.

Preservation after snapping is interesting only relative to these controls. It
could otherwise be explained by a small displacement or an insensitive
direction.

Separately inject perturbations at $0.5$, $0.9$, and $1.1$ times the directional
boundary distance. Let $\delta$ denote the complete native activation-tensor
perturbation, including its spatial support. For the next block $F_\ell$, define
the directly measured RMS gain

$$
\kappa
=
\frac{
\operatorname{RMS}\!\left(F_\ell(H+\delta)-F_\ell(H)\right)
/\operatorname{RMS}(H_{\ell+1})
}{
\operatorname{RMS}(\delta)/\operatorname{RMS}(H_\ell)
}.
$$

Values below one, together with recovery of the clean downstream assignment,
are evidence for finite basin-like correction. They do not establish an
attractor. Single-token and spatially coherent perturbations must both be
tested so that global average pooling cannot manufacture apparent robustness.

### 1E. Comparison with critical modules

The existing Tracking2 intervention replaces one block in the epoch-100 model
with its same-trajectory epoch-$s$ version or a fresh random block. Its damage is

$$
T_{\ell,s\to100}
=
\mathcal L_{\mathrm{test}}\!\left(f_{100}^{[\ell\leftarrow s]}\right)
-\mathcal L_{\mathrm{test}}(f_{100}).
$$

This measures historical compatibility or dependence on a learned parameter
block in an otherwise final, co-adapted network. It does **not** measure
activation contraction, centroid snapping, capacity, activation-noise
robustness, higher-moment dependence, or a critical period.

Place each block's transplant-damage curve beside its distortion, boundary
enrichment, finite-recovery, and snapping curves. Do not preregister a sign.
Two opposing readings are both live:

- robust or “non-critical” blocks could be correction modules, giving low
  transplant damage and strong contraction; or
- critical blocks could implement the discretizing map, giving high transplant
  damage and strong boundary concentration.

With eight blocks and one seed, this is descriptive. Any association must
repeat across independently trained seeds before it receives much weight.

### 1F. Whiteboard-derived secondary diagnostics

Three comparisons motivate the project but are not yet part of a frozen gate.

First, sweep direct residual perturbations over a predeclared logarithmic range of
scales, optionally sampling magnitudes from a declared power-law family. Plot
functional damage, cell-crossing probability, and boundary-normalized displacement
on the same scale axis. Compare against covariance-matched and direction-shuffled
nulls. The purpose is to distinguish broadly scale-free sensitivity from damage
concentrated near a characteristic crossing scale, not to declare either curve
shape intrinsically discrete.

Second, track **assignment commitment**—held-out assignment stability, normalized
margin, and persistence under codebook refits—from early through late training.
Keep it separate from **recovery commitment**, which asks whether a perturbed state
returns to the clean downstream assignment. Report both across depth so a middle-layer
effect is visible rather than averaged away.

Third, place the activation-space flatness and fissure statistics beside independently
defined Fisher- or parameter-space flatness and module-plasticity measurements. This
is an exploratory association with no preregistered sign. Shared words such as
“flatness,” “plasticity,” or “critical” must not collapse the distinct interventions
or uncertainty units.

### Experiment 1 gates

1. **Mechanical gate.** Verify distinct hash-pinned train/test source files,
   deterministic probe banks, zero-intervention parity, native-space centroid
   reconstruction, synthetic mixture/Gaussian behavior, and JVP versus
   finite-difference agreement. Distinct paths and whole-file hashes do not by
   themselves prove absence of row-level overlap, so this bounded gate does not
   claim a sample-level leakage audit.
2. **Coarse phenomenon gate.** Reuse the read-only seed-0 checkpoints at epochs
   0, 1, 5, 20, and 100. Measure static geometry at all eight cuts and functional
   paths at four sentinel cuts. Produce only a five-frame diagnostic animation.
3. **Functional gate.** Proceed to the full snapping and correction battery only
   if held-out clustering is stable beyond matched Gaussian controls and
   boundary sensitivity beats the shifted-boundary null.
4. **Confirmation gate.** Freeze $K$, metrics, path construction, thresholds,
   and probe banks, then train at least three independent seeds. Save dense
   early-batch and per-epoch checkpoints only if the coarse gate warrants a
   smooth formation GIF.
5. **Transplant comparison.** Join confirmed geometry tables to same-seed
   transplant tables. Continue to label the association exploratory.

The account is weakened if effects disappear on held-out images, under the
$K$/metric/codebook-initialization checks, or relative to moment-matched nulls;
if sensitivity peaks fail to align with candidate boundaries; if sharpness
exists only off-cloud; if snapping performs no better than matched
displacements; if cells mostly encode class or position; or if effects coincide
only with learning-rate drops and confidence changes.

Useful negative outcomes remain distinct:

- static compression without functional alignment is a codebook description,
  not a discrete computation;
- snapping tolerance without exceptional clustering suggests generic
  smoothness;
- boundary sensitivity without snapping tolerance suggests piecewise-sensitive
  computation without cell sufficiency; and
- strong geometry with indispensable within-cell information rejects the
  proposed coarse state at that resolution.

## Algebraic factor recovery moved to a separate project

The former Experiment 2—synthetic factor recovery, commutant spectra, operator
decomposition, real-transition intertwiners, and multiscale quotients—now lives in the
standalone local sibling project `../FactorizedDynamics`. Those questions require their
own synthetic benchmarks, validation ladder, and claim ledger. They are no longer a
downstream stage or scientific gate for this activation-geometry project.

## Dashboard and animation specification

The implemented report builder produces one self-contained HTML document that
embeds this specification. Its current payload is deterministic schematic data:
every scientific section and plot is visibly marked `MOCKUP`, includes an
interpretation guide, and states what would weaken the hypothesis. The CLI
intentionally rejects `--mode real`; measured reporting will be enabled only
after a typed report-payload artifact is assembled from verified run receipts
and gate outputs. Thus an arbitrary local JSON file cannot self-assert evidence.

Planned views are:

1. **Overview.** The dependency graph above, claim ledger, gate status, and a
   prominent distinction between definitions, planned tests, and results.
2. **Formation through training.** Depth-by-checkpoint heatmaps for distortion,
   stability, boundary margin, occupancy, and boundary-sensitivity enrichment.
3. **Fixed-layout animation.** The same probe tokens and aligned centroids at
   every checkpoint, beside empirical-chord and off-cloud sensitivity strips
   and a moving marker on the quantitative time curves. One joint projection is
   fixed across frames; per-frame embeddings are prohibited because their
   motion can manufacture apparent formation. Any two-dimensional Voronoi
   overlay is labeled as a projection, not the measured high-dimensional cell.
4. **MNIST centroid and Jacobian-barrier geometry.** For every MNIST test point,
   show candidate-centroid assignments, margins, and a fixed-identity layout induced
   by the integrated Jacobian barrier, synchronized across selected layers and
   training checkpoints. Pair it with the complete pairwise barrier heatmap and
   quantitative within-class versus between-class summaries under label-permutation
   nulls. A cosine-distance layout is a companion baseline. The two-dimensional
   circles are descriptive; the class-alignment hypothesis is tested on the original
   pairwise costs.
5. **Interpolation sensitivity.** Plateau and boundary-response curves under
   linear, radius-preserving, and neighbor-graph paths, paired with path-support
   diagnostics so off-support segments remain visible.
6. **Snapping and recovery.** Dose-response curves for $\alpha$, separated from
   equal-norm controls, plus clean-versus-perturbed trajectories through depth.
7. **Module comparison.** Transplant damage beside contraction and boundary
   measurements, labeled descriptive until independently replicated.
8. **Commitment, noise scale, and flatness comparisons.** Assignment and recovery
   commitment through depth and training; noise-response and crossing curves over
   scale; and activation flatness beside separately defined Fisher flatness and
   module plasticity.
9. **Provenance and negative results.** Configurations, hashes, probe banks,
   metrics, $K$, occupancy, seeds, uncertainty units, nulls, failed gates, and
   explicit non-claims.

The GIF is a view of measured dynamics, not the evidence by itself. Stable
layout, synchronized clean/perturbed panels, and a displayed checkpoint clock
are required.

## Stopping rules and project order

The following decision rules are provisional specifications, not results. The
mechanical tolerances are fixed before code is run. The one-seed phenomenon
rule is explicitly exploratory; it selects whether to spend compute, not
whether to accept a scientific claim. All other thresholds are frozen before
their held-out or confirmatory data are inspected.

| Gate | Primary statistic and comparator | Uncertainty unit and provisional pass rule | Action |
|---|---|---|---|
| Mechanical | Identity intervention, standardized/native round trip, and JVP versus centered finite difference | Exact identity parity; relative RMS round-trip error $<10^{-6}$ in fp32; median JVP relative error $<10^{-2}$ and 95th percentile $<5\times10^{-2}$ | Fix implementation before any phenomenon plot if one check fails |
| Coarse ResNet phenomenon | $D_K$ and bootstrap assignment stability versus both Gaussian nulls; empirical-chord boundary enrichment versus the path-shift null | Image is the resampling unit. At least two of the four predeclared stage-end cuts, including one nonfinal cut, must have 95% bootstrap intervals favoring the real representation on both geometry measures and enrichment above the 95th null percentile | If passed, run the functional battery; otherwise report the coarse negative result and stop the dense GIF plan |
| Snapping and recovery | Hard-snap predictive KL versus every same-norm control; $\kappa$ and clean downstream-cell recovery versus matched perturbations | Paired image bootstrap. At least two predeclared cuts must have a 95% interval showing lower hard-snap damage than the controls, $\kappa<1$, and higher cell recovery on empirical-chord perturbations | If passed, freeze the full protocol and train confirmation seeds; otherwise do not infer discrete-state sufficiency |
| Three-seed confirmation | The predeclared geometry, boundary, snapping, and recovery contrasts above | Training seed is the replication unit; the same two or more cuts must pass with the same effect direction in each of three seeds. Image bootstraps remain within-seed uncertainty and are not counted as model replicates | Only then describe the phenomenon as replicated; module-transplant association remains exploratory |

The minimum program is:

1. Preserve the completed motivating-literature audit and keep its distinct response,
   RGB stable-region, and Jacobian estimands separate.
2. Use the completed small-model activation-geometry collection to choose and freeze a
   quantitative plateau statistic; then run the mechanical and five-checkpoint ResNet
   candidate-cell gates.
3. Specify and build the all-MNIST integrated-Jacobian-barrier plot across layers and
   training checkpoints, retaining the original pairwise matrices and class-alignment
   tests behind the embedding.
4. Run snapping and recovery only if the candidate cells have held-out and
   functional support.
5. Add the noise-scale, commitment, and Fisher-flatness/plasticity comparisons only
   after their estimands, controls, and uncertainty units are frozen.
6. Train confirmation seeds only after the exploratory gates justify the cost.

Stop or narrow the language at each failure:

- no held-out stability advantage: do not call the clusters computational
  cells;
- no contraction: do not use attractor language;
- no benefit over matched snapping controls: do not infer quantized
  computation; and
- no within-class reduction in integrated MNIST barrier relative to the
  label-permutation null: do not describe the barrier geometry as class-aligned.

## Implemented scope and unresolved choices

The implementation now includes infrastructure, input/probe materialization, bounded
mechanics, and a five-checkpoint synthetic/CIFAR activation-geometry collection with
synchronized animations. It does **not** run the candidate-cell formation analysis,
the coarse or functional phenomenon gates, the all-MNIST barrier geometry,
noise-scale/Fisher comparisons, or confirmation training. The specified candidate-cell
stages remain visible in the DAG as `PLANNED`, with typed configuration placeholders
where choices are sufficiently specified.
`protocol.mode: confirmatory` is rejected until a future schema can bind it to a
preregistered full-protocol hash.

Several author choices remain consequential, although the exploratory pilot
configuration uses explicit provisional defaults so infrastructure work can
proceed:

- the audited literature uses several different operationalizations of activation
  plateaus and stable regions, so a future powered run still needs one preregistered
  primary statistic and null;
- “on/off dataset perturbations” is currently interpreted as input-derived
  states versus direct, norm-matched residual interventions;
- the integrated Jacobian barrier still needs a frozen path, response functional,
  symmetrization rule, numerical quadrature, and feasible all-pairs computation plan;
  and
- Fisher flatness, module plasticity, and activation flatness still need separate
  estimands before any association is tested.

## Source status

This write-up was developed from the project description, the whiteboard discussion,
and the activation-plateau source audit:

- [`references/activation-plateau-source-audit.md`](references/activation-plateau-source-audit.md)

The source audit separates the Heimersheim--Mendel real/fake response curves, the SAE
study of four activation types, the Janiak et al. three-anchor RGB construction, and the
Shinkle--Heimersheim Jacobian work. The local CNN Jacobian surfaces and checkpoint-time
GIFs are labeled as new hybrid diagnostics rather than source replications.

Initial specification drafted collaboratively by CRW and Codex on 2026-08-16;
the infrastructure implementation, source audit, and first Experiment 1 collection
pilot were completed on 2026-08-17.
