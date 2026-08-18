# Activation-plateau source audit

Last checked: 2026-08-17.

The requested plots combine methods from four sources. They must remain separate in
artifact metadata and figure labels. The relevant author is Stefan Heimersheim.

## 1. Real versus matched-Gaussian response curves

- Stefan Heimersheim and Jake Mendel, "[Interim research report] Activation plateaus &
  sensitive directions in GPT2" (2024):
  <https://www.lesswrong.com/posts/LajDyGyiyX8DNNsuF/interim-research-report-activation-plateaus-and-sensitive-1>
- Access: full web post. There is no formal PDF.
- Approximate released reproduction inspected at commit
  `9ef7d42b8bd911d88c978b13f4db2d506fcde2a7`:
  <https://gist.github.com/Stefan-Heimersheim/85c1091408e113e2ef9ca2a798ec6553>
- Source estimand: downstream response along perturbation paths, comparing real base
  activations with bases drawn from a Gaussian having the empirical activation mean
  and covariance. This is not a Jacobian-norm surface.

The Experiment 1 path payload is an architecture-adapted analogue: it stores logit
$L_2$ and KL response along straight paths toward independently drawn
covariance-Gaussian targets. It also stores the exact coefficients, realized targets,
hosts, and intervention vectors.

## 2. Four activation-center types

- Giorgi Giglemiani et al., "Evaluating Synthetic Activations composed of SAE Latents
  in GPT-2" (2024): <https://arxiv.org/abs/2409.15019>
- Access: official full 18-page arXiv PDF. The audited local copy is ignored by Git;
  its path and hash are recorded in `references/README.md`.
- Source estimand: response curves for four center types: real/model-generated,
  covariance-matched Gaussian, SAE synthetic-baseline, and SAE
  synthetic-structured. This is likely the origin of the request's "four fake
  activations" wording; the paper has four types total, not four fake points.

The first implementation intentionally includes only real and covariance-Gaussian
centers. SAE center types are out of scope for the small CNN pilot.

## 3. Three-anchor stable-region RGB fields

- Jett Janiak et al., "Characterizing stable regions in the residual stream of LLMs"
  (2024): <https://arxiv.org/abs/2409.17113>
- Access: official full 17-page arXiv PDF. The audited local copy is ignored by Git;
  its path and hash are recorded in `references/README.md`.
- Source construction: form the affine plane through three real activations. For each
  grid point, intervene in each anchor's separately frozen prompt context, compute the
  downstream distance from that context's anchor output, normalize the three distance
  fields, and encode them as RGB complements.

This construction is neither a literal Voronoi tessellation nor a Jacobian-norm field.
Experiment 1 saves raw per-context distances and uses a fixed normalization over all
rendered checkpoints so animation colors remain temporally comparable. It also saves
the source-style per-frame RGB field for audit.

## 4. Jacobian norms and the moving-circle animation

- Matthew Shinkle and Stefan Heimersheim, "Activation Plateaus: Where and How They
  Emerge" (2025):
  <https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge>
- Access: full web post. There is no formal PDF.
- Released MIT-licensed code inspected at commit
  `4222c3efb77fe7983676cf96b3254dc000e8a0bc`:
  <https://github.com/MShinkle/activation_plateau_mechanisms>
- Source estimand: exact local MLP activation-Jacobian Frobenius norms along selected
  GPT-2 Large interpolation paths. The published moving-circle GIF is an explanatory
  animation over depth/interpolation, not a checkpoint-time measurement.

Experiment 1 adds two explicitly new CNN diagnostics: finite-difference Jacobian norms
restricted to saved two-direction activation planes, and Hutchinson estimates of the
full same-site next-transition Jacobian Frobenius norm. Raw $DT$ fields are preserved
for every architecture. Residual transitions additionally store fields obtained by
differentiating $T(z)-z$, plus full same-site $J-I$ estimates, because the identity skip
otherwise dominates the raw norm. Protocol-v2 animations display plane-restricted
$D(T-I)$ for synthetic/ResNet transitions and plane-restricted $DT$ for VGG; these are
different operators and remain a new hybrid diagnostic rather than a source
replication.

## Interpretation boundary

The implementation is a small-model, CIFAR-10/synthetic analogue, not an exact
Transformer replication. The checkpoint animation is also new. ResNet-versus-VGG is a
descriptive comparison confounded by normalization, nonlinearities, parameter count,
regularization, and training history; it is not a causal ablation of residual streams.
