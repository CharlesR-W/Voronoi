# Voronoi Residual Computation Lab

> **Status: infrastructure and bounded validation implemented; scientific
> phenomenon runs remain unrun.** The repository now contains a strict stage
> DAG, immutable artifacts and run receipts, a hash-pinned Tracking2 adapter,
> deterministic probe banks, mechanical checks, a tiny exact synthetic subgate,
> and a self-contained `MOCKUP` report. These implementation validations are not
> evidence that residual candidate cells form, snap, recover, or factor.

This project asks whether continuous residual computation admits a useful
finite-state description and, if it does, whether the resulting transition
maps have approximate symmetries or low-interaction factor coordinates.

The order matters. Fitting centroids always creates a Voronoi partition. The
scientific question is whether that partition is stable on held-out data,
predictively adequate, and causally useful. Only after those tests pass does it
make sense to treat the occupied cells as states of a coarse computation and
ask whether that computation factors.

## Start here

Install the locked environment and inspect the dependency-closed plan:

```bash
uv sync --all-extras
.venv/bin/voronoi-lab validate --inputs --json
.venv/bin/voronoi-lab plan --json
```

The currently runnable scientific-validation targets are
`gate.mechanical` and `gate.synthetic_exact`; `report.build` creates only the
visibly watermarked schematic. Every successful `run` publishes an immutable,
database-independent receipt and prints its artifact ID. For example:

```bash
.venv/bin/voronoi-lab run --until gate.mechanical --json
.venv/bin/voronoi-lab run --until gate.synthetic_exact --json
.venv/bin/voronoi-lab run --stage report.build --json
```

The next scientific prerequisite is still to identify and archive the primary
papers behind the motivating “activation plateau” and sharp-boundary reports.
That audit must finish before the project freezes a literature-matched plateau
definition or makes any replication claim.

Current inputs are local-only and are **not clone-safe**:

- the ResNet definition, configuration, checkpoints, and transplant artifact
  are in the sibling `../Experiments/Tracking2/` project; their presence was
  verified locally on 2026-08-16, but they are not vendored here; and
- the two mathematical design notes are archived under `references/notes/`
  with hashes listed in `references/README.md`; they are proposals, not
  empirical sources.

The infrastructure boundary is deliberate. Runnable stages use strict,
versioned payload schemas, semantic seed namespaces, content-addressed objects,
cross-run caching, explicit gate rules and authorizations, source-drift checks,
shard-level recovery for the exact synthetic gate, and append-only run receipts.
The formation, boundary, snapping, three-seed confirmation, sampled recovery,
and real-algebra stages remain `PLANNED` and cannot be accidentally executed.

The intended dependency is:

```text
reported activation plateaus / sharp residual boundaries
                         |
       define the residual unit, metric, probes, and codebook
                         |
       held-out finite-state fidelity and stability
                  +------+-------------------+
                  |                          |
       recovery, boundary,          validated coarse
       and snapping tests           transition model --------+
                  |                                           |
       exploratory comparison                                 |
       with transplant damage                                 |
                                                              v
synthetic known-factor benchmark --> algebraic methods pass -->+
                                                              |
                                         +--------------------+----------+
                                         |                               |
                              real symmetry screen             real factor search
                                                                         |
                                                              multiscale quotients
                                                                   (deferred)
```

## Claim ledger

The project deliberately separates claims that are easy to conflate.

| Statement | Current status |
|---|---|
| A fitted codebook induces Voronoi cells | Definition; true by construction |
| Residual states develop a stable, useful finite-state description during training | Primary empirical hypothesis |
| Perturbed states are corrected toward the interiors of those cells | Stronger, separate hypothesis |
| Functional sensitivity is concentrated near cell boundaries | Stronger, separate hypothesis |
| Moving a residual to its nearest centroid preserves or improves computation | Planned causal test |
| Recovery is related to module-transplant sensitivity | Exploratory comparison; direction unspecified |
| Coarse transition families have nontrivial approximate symmetries | Conditional empirical hypothesis |
| Several transitions simplify in one shared product coordinate system | Conditional and stronger hypothesis |
| Nested quotients expose rules with localized exceptions | Deferred direction |

Until the corresponding gates pass, this README uses **candidate cell** rather
than basin, **recovery** rather than attraction, **boundary sensitivity** rather
than a Jacobian wall, and **low-commutator direction** rather than learned
symmetry algebra.

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
- A nontrivial commutant need not imply a tensor product, and a useful tensor
  product need not leave a nontrivial global commutant.

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

## Experiment 2: recover hidden factors and operators synthetically

### Question

Given only a family of transition operators on an arbitrarily relabeled finite
state set, can an algorithm recover both a product labeling of the states and a
low-order decomposition of the transitions? Separately, can the low spectrum of
a commutator map recover deliberately planted approximate symmetries?

The synthetic benchmark is not optional validation. Any later claim about
factors in neural candidate cells is gated on recovery where the latent answer
is known.

### Generative model

Let the latent state set be

$$
\mathcal C_*
=
[n_1]\times\cdots\times[n_k],
\qquad
N=\prod_{r=1}^k n_r,
$$

but expose it only through a random bijection
$\pi:\mathcal C\to\mathcal C_*$. The first stages supply $k$ and the factor
sizes; discovering them is deferred.

Matrices use the column convention $P_{ji}=P(j\mid i)$, so valid generators
have nonnegative off-diagonal entries and zero column sums.

Use continuous-time Markov generators because independent asynchronous factor
dynamics add at generator level. For primitive $\alpha$ and support
$S\subseteq[k]$, draw a sparse valid generator $R_{\alpha,S}$ on the factors in
$S$ and lift it by the identity on the remaining factors.

Interaction strength and symmetry breaking should be independent experimental
knobs. Choose fixed finite permutation groups $G_r$ acting on the factor labels
and their compatible product $G=\prod_r G_r$. For each support use the
restriction $G_S=\prod_{r\in S}G_r$, represent every $g$ by its permutation
matrix $U_S(g)$, and twirl a generator,

$$
\bar R_{\alpha,S}
=
\frac{1}{|G_S|}
\sum_{g\in G_S}
U_S(g)R_{\alpha,S}U_S(g)^{-1},
$$

then set

$$
R_{\alpha,S}(\delta)
=
(1-\delta)\bar R_{\alpha,S}+\delta R_{\alpha,S}.
$$

Take $0\leq\delta\leq1$. Permutation conjugation, group averaging, and this
convex interpolation then preserve the generator cone.

The full latent generator is

$$
L_\alpha^*
=
a_\alpha\left[
\sum_{|S|=1}w_{\alpha,S}R_{\alpha,S}(\delta)\otimes I_{\bar S}
+
\rho\sum_{|S|=2}w_{\alpha,S}R_{\alpha,S}(\delta)\otimes I_{\bar S}
\right],
$$

where $a_\alpha$ normalizes the exit rate. Require $a_\alpha>0$,
$w_{\alpha,S}\geq0$, and $\rho\geq0$, so the sum remains a valid generator. Thus $\rho$ controls
genuine pairwise interaction while $\delta$ controls departure from one
compatible planted global symmetry. Twirling the pair terms too prevents
interaction and symmetry breaking from becoming interchangeable. The input knob
$\rho$ is not itself the recovered Hilbert--Schmidt pair energy, so every
generated instance also records its realized normalized support-order spectrum.

The observed generator is only a conjugated version,

$$
L_\alpha
=
\Pi_\pi^\top L_\alpha^*\Pi_\pi.
$$

The noiseless task supplies the generators exactly. The sampled task takes a
known sufficiently small $\tau$, forms $P_\alpha=I+\tau L_\alpha$, and supplies
multinomial transition counts from every state and primitive. This avoids an
unqualified matrix logarithm, whose branches and sampling instability would
otherwise muddy the benchmark.

### Joint factor and operator objective

For each proposed factor, decompose its operator space orthogonally as

$$
\operatorname{End}(\mathcal H_r)
=
\operatorname{span}\{I_r\}
\oplus
\operatorname{span}\{I_r\}^{\perp}.
$$

Tensoring these decompositions gives a unique support decomposition

$$
L_\alpha
=
\sum_{S\subseteq[k]}L_{\alpha,S},
$$

where a support-$S$ term is non-identity on exactly the factors in $S$. This
prevents a nominal pair interaction from quietly retaining unary components.
Let $\mathsf P_S^\pi$ be the corresponding projection after relabeling states by
$\pi$. A direct training objective is

$$
J(\pi)
=
\sum_{\alpha\in\mathrm{train}}
\sum_{S\subseteq[k]}
\lambda_{|S|}
\left\|\mathsf P_S^\pi(L_\alpha)\right\|_F^2,
\qquad
\lambda_0=\lambda_1=0<\lambda_2<\lambda_3<\cdots.
$$

The outer search finds a shared state relabeling; the inner orthogonal
projections simultaneously give the best operators at each interaction order.
After selecting a labeling, refit the retained components **jointly**, requiring
the reconstructed sum to have nonnegative off-diagonal entries and zero column
sums. Individual orthogonal support components need not themselves be Markov
generators. With sampled counts, use held-out likelihood plus the same
order-weighted penalty. Held-out primitives are essential: they test whether a
coordinate system was recovered, not merely whether one family of coefficients
was fit.

For nonuniform visitation, replace Frobenius error with an explicitly declared
data-weighted operator metric. Uniform state enumeration is primary in the
first benchmark because it makes the score invariant to the hidden
permutation.

### Symmetry diagnostic

Given a family of recovered or observed operators, define

$$
\mathcal C(X)
=
\bigl([X,L_1],\ldots,[X,L_m]\bigr).
$$

Zero singular modes give the exact commutant; low singular modes are
low-commutator directions. The scalar identity must be projected out, candidate
operators normalized, and the low spectrum compared with sampling-aware nulls
and held-out primitives. A low mode is not automatically a group element: it
may be a projection or observable. Only after checking the relevant
invertibility, unitary, or permutation constraints should one round it to a
candidate symmetry transformation.

Symmetry and factorization are parallel tests. A family of local operations can
generate the full matrix algebra and have only scalar global commutants while
still possessing an excellent product description. Conversely, a large
commutant can arise from duplicated or insufficiently sampled states without
revealing factors.

### Recovery ladder

1. **Oracle-coordinate check.** Given the true labels, recover the
   Hilbert--Schmidt support components computed from the exact full generator,
   verify the common global twirl defect is zero at $\delta=0$, verify the full
   reconstructed sum remains a generator, and verify convergence under finite
   sampling. Do not demand that each raw generative summand equal one orthogonal
   support component.
2. **Tiny exhaustive check.** Use $k=2$ and $(n_1,n_2)=(2,3)$ so exhaustive
   relabeling supplies a gold-standard objective result.
3. **Cartesian graph baseline.** At $\rho=0$, factor the colored aggregate
   transition graph. Under noise and interaction, use the result only as an
   initializer.
4. **Commuting-subalgebra route.** Cluster primitives whose cross-group
   commutators are small, infer multiplicity structure from the associated
   algebras, and separately round the abstract basis information to a discrete
   state labeling using positivity, sparsity, and the transition graph.
5. **Joint refinement.** Alternate closed-form support projection/operator
   fitting with swaps of state assignments that lower $J$. Compare graph,
   algebraic, random, and oracle initializations on a $(4,5)$ state product.
6. **Noise and null sweep.** Vary $(\rho,\delta)$ independently, then repeat a
   subset with finite transition counts and held-out primitives.
7. **Later only:** enumerate divisor tuples of $N$ to choose factor sizes using
   held-out likelihood plus a complexity penalty; then consider three factors,
   conditional updates, and centroid geometry.

Required baselines include random balanced factor labels, no relabeling,
Cartesian factorization alone, commutant methods alone, and oracle labels.
Required nulls include dense random Markov generators matched in size, a block
cluster model with no Cartesian product, and a genuinely independent generator
incorrectly scored at finite-step $P$ level to demonstrate why generator-level
additivity matters.

Report factor recovery only up to independent within-factor relabeling and
permutation of equal-sized factors. Metrics include coordinate adjusted mutual
information, exact tuple recovery after optimal alignment, held-out transition
negative log likelihood, aligned support-term error, recovered interaction
order, principal angles between planted and recovered commutant subspaces,
false-positive rate on unfactored nulls, and bootstrap stability. The excess
held-out high-order energy above the oracle-label energy is more informative
than raw residual when interactions are deliberately planted.

Failure can be correct. The labeling is not identifiable if the primitives do
not distinguish coordinates, if factor action families are indistinguishable,
or if the aggregate graph admits multiple product decompositions. Abstract
matrix algebras determine factors only up to local changes of basis; recovering
a product labeling of one-hot cells requires the additional Markov positivity
or graph structure. Exact symmetry can itself make orbit members
indistinguishable. Large interactions should eventually destroy stable
recovery rather than force the algorithm to return a preferred ontology.

## Conditional bridge to real cell transitions

If Experiment 1 supplies stable candidate states, empirical transitions may be
estimated as

$$
P_{t,\ell,\alpha}(j\mid i)
=
\Pr\!\left(
q_{t,\ell+1}(h_{t,\ell+1})=j
\mid
q_{t,\ell}(h_{t,\ell})=i,\alpha
\right).
$$

The conditioning variable $\alpha$ must be explicit: for example, perturbation
family, intervention, or another declared context. Such an empirical operator
depends on the context distribution and cell occupancy; it is not
automatically a distribution-free property of the network.

Sitewise CNN states also require an explicit pairing rule across depth. The
first real transition pass therefore uses only the four within-stage
first-block-to-second-block maps, whose spatial grids have the same resolution,
and pairs matching spatial sites. Transitions across stride-2 blocks are
excluded until a receptive-field or transport rule is specified and tested.

Layer-specific cell sets usually differ, so adjacent-cut maps are rectangular.
The natural approximate relation is then an intertwining condition

$$
X_{\ell+1}P_{\ell,\alpha}
\approx
P_{\ell,\alpha}X_\ell,
$$

not a commutator. A common $[X,P_\alpha]$ objective is justified only after
constructing and validating a shared state space on which every $P_\alpha$ is
square. The synthetic experiment deliberately starts with that easier square
case. Even in the rectangular case the identity pair is a trivial exact
intertwiner, and rank deficiency creates additional exact modes supported only
in common kernels or cokernels. Before interpreting a low mode, restrict to the
empirically occupied input/output subspaces, quotient the identity and common
kernel/cokernel solutions, normalize the admissible $X_\ell$, and require the
remaining structure to survive held-out contexts. A raw rectangular
intertwiner spectrum is not a symmetry spectrum.

For a square family and a declared positive-definite state metric $G$, the
coordinate-aware operator norm is

$$
\|A\|_{\mathrm{HS},G}^2
=
\operatorname{Tr}\!\left(G^{-1}A^\dagger G A\right).
$$

This keeps the algebraic statement “what commutes?” distinct from the metric
statement “how badly is commutation broken?” Low-commutator modes must survive
resampling, held-out contexts, and occupancy-matched nulls. A proposed factor
coordinate system must compress held-out transitions; operator-Schmidt spectra
can score a proposed factorization but do not discover one by themselves.

## Deferred direction: multiscale quotients

The eventual “resolution dial” should start from nested partitions of a
validated fine cell set,

$$
\Pi_0\prec\Pi_1\prec\cdots\prec\Pi_L,
$$

which induce nested spaces of block-constant observables

$$
V_0\subseteq V_1\subseteq\cdots\subseteq\mathbb C^{\mathcal C}.
$$

For a transition operator acting on distributions, exact lumpability is
invariance of $V_\ell$ under its corresponding observable operator. One useful
approximate defect in the shared square-state case is

$$
E_\ell
=
\sum_\alpha
\left\|
(I-\Pi_{V_\ell})P_\alpha^\top\Pi_{V_\ell}
\right\|^2.
$$

Here every projection uses the declared occupancy-weighted inner product. For a
rectangular transition $P:\mathbb C^{\mathcal C_{\mathrm{in}}}\to
\mathbb C^{\mathcal C_{\mathrm{out}}}$ with separate block-constant observable
spaces, the compatible condition and defect are instead

$$
P^\top V_{\mathrm{out}}\subseteq V_{\mathrm{in}},
\qquad
E_{\mathrm{in},\mathrm{out}}
=
\left\|
(I-\Pi_{V_{\mathrm{in}}})P^\top\Pi_{V_{\mathrm{out}}}
\right\|^2.
$$

Large local contributions say that a coarse block hides distinctions needed to
predict its outgoing transitions, which provides a direct trigger for local
refinement. This is the probabilistic version of “$A$ usually maps to $B$, but
the refined subcase $A_1$ maps to $C$.”

The coarse operator algebras act on different quotient spaces, so they are not
automatically a canonical chain
$\mathcal A^{(0)}\subseteq\mathcal A^{(1)}\subseteq\cdots$. Any such algebra
filtration needs explicit restriction and lift maps. This mathematical and
experimental direction is recorded here, but it is not part of the first two
experiments.

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
4. **Snapping and recovery.** Dose-response curves for $\alpha$, separated from
   equal-norm controls, plus clean-versus-perturbed trajectories through depth.
5. **Module comparison.** Transplant damage beside contraction and boundary
   measurements, labeled descriptive until independently replicated.
6. **Synthetic recovery.** Planted versus recovered product coordinates,
   interaction-order spectra, commutator spectra, held-out reconstruction, and
   recovery curves across $(\rho,\delta,\text{sample count})$.
7. **Conditional real algebra.** Only after both gates pass: low-commutator or
   intertwiner spectra, factor-coordinate compression, null comparisons, and
   stability.
8. **Provenance and negative results.** Configurations, hashes, probe banks,
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
| Mechanical | Identity intervention, standardized/native round trip, JVP versus centered finite difference, generator and twirl invariants | Exact identity parity; relative RMS round-trip error $<10^{-6}$ in fp32; median JVP relative error $<10^{-2}$ and 95th percentile $<5\times10^{-2}$; generator/commutator checks at dtype-scaled test tolerance | Fix implementation before any phenomenon plot if one check fails |
| Coarse ResNet phenomenon | $D_K$ and bootstrap assignment stability versus both Gaussian nulls; empirical-chord boundary enrichment versus the path-shift null | Image is the resampling unit. At least two of the four predeclared stage-end cuts, including one nonfinal cut, must have 95% bootstrap intervals favoring the real representation on both geometry measures and enrichment above the 95th null percentile | If passed, run the functional battery; otherwise report the coarse negative result and stop the dense GIF plan |
| Snapping and recovery | Hard-snap predictive KL versus every same-norm control; $\kappa$ and clean downstream-cell recovery versus matched perturbations | Paired image bootstrap. At least two predeclared cuts must have a 95% interval showing lower hard-snap damage than the controls, $\kappa<1$, and higher cell recovery on empirical-chord perturbations | If passed, freeze the full protocol and train confirmation seeds; otherwise do not infer discrete-state sufficiency |
| Synthetic recovery | Exact tuple labels up to gauge, support-component error, held-out likelihood, and false positives on unfactored nulls | Across 20 noiseless $(2,3)$ instances: 100% aligned tuple recovery and relative support error $<10^{-8}$ in float64. On the predeclared easy sampled regime: median coordinate AMI $\geq0.9$, held-out likelihood better than every non-oracle baseline, and at most 5 positives among 100 null instances | Real factor discovery remains blocked unless both exact and sampled gates pass |
| Three-seed confirmation | The predeclared geometry, boundary, snapping, and recovery contrasts above | Training seed is the replication unit; the same two or more cuts must pass with the same effect direction in each of three seeds. Image bootstraps remain within-seed uncertainty and are not counted as model replicates | Only then describe the phenomenon as replicated; module-transplant association remains exploratory |
| Real algebra | Low-mode stability and held-out operator compression versus occupancy/sampling nulls | Calibrate a threshold with at most 5% false positives on the synthetic/null suite before unblinding real transitions; require a 95% transition-bootstrap interval for positive held-out compression | Otherwise report no useful real symmetry or factor evidence |

The minimum program is:

1. Audit the motivating literature and pin down exactly what its authors call
   an activation plateau or residual boundary. Primary-source PDFs should be
   stored locally before any replication claim.
2. Run the mechanical and five-checkpoint ResNet gates.
3. Run snapping and recovery only if the candidate cells have held-out and
   functional support.
4. In parallel, validate symmetry and factor recovery on the known synthetic
   benchmark.
5. Apply the algebraic methods to real transitions only if both the geometry
   and synthetic gates pass.
6. Consider multiscale quotients only after a useful finest-scale state model
   exists.

Stop or narrow the language at each failure:

- no held-out stability advantage: do not call the clusters computational
  cells;
- no contraction: do not use attractor language;
- no benefit over matched snapping controls: do not infer quantized
  computation;
- no synthetic recovery: do not interpret real-data factor outputs;
- low commutator modes that vanish under resampling or nulls: report no
  symmetry evidence; and
- no compression of held-out transitions: report no useful factorization.

## Implemented scope and unresolved choices

The implementation currently stops at infrastructure, input/probe
materialization, bounded Experiment 1 mechanics, and the tiny-state exact
Experiment 2 subgate. It does **not** run the five-checkpoint formation analysis,
the coarse or functional phenomenon gates, confirmation training, sampled
synthetic recovery, or real algebra. Those stages remain visible in the DAG as
`PLANNED`, with typed configuration placeholders where choices are sufficiently
specified. `protocol.mode: confirmatory` is rejected until a future schema can
bind it to a preregistered full-protocol hash.

Several author choices remain consequential, although the exploratory pilot
configuration uses explicit provisional defaults so infrastructure work can
proceed:

- “activation plateau” may refer in the motivating literature to a different
  object than sitewise residual states; the exact papers and definition still
  need to be pinned down;
- “on/off dataset perturbations” is currently interpreted as input-derived
  states versus direct, norm-matched residual interventions; and
- the real transition analysis currently assumes layer-specific codebooks and
  intertwiners unless a shared cell space earns separate validation.

## Source status

This write-up was developed from the project description and two local notes,
read in full and archived here:

- [`references/notes/approximate-symmetry-operator-algebra.md`](references/notes/approximate-symmetry-operator-algebra.md)
- [`references/notes/approximate-factorization-voronoi-residual.md`](references/notes/approximate-factorization-voronoi-residual.md)

Those notes provide mathematical proposals, not empirical findings. The
motivating plateau/boundary literature has not yet been audited in this
repository, so this README deliberately makes no citation-dependent literature
claims.

Initial specification drafted collaboratively by CRW and Codex on 2026-08-16;
the infrastructure implementation and status audit were completed on
2026-08-17.
