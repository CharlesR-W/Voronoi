# Activation-geometry source archive

The two mathematical notes about factor recovery and commutants moved to the standalone
[Factorized Dynamics Lab](https://github.com/CharlesR-W/FactorizedDynamics), with local
sibling `../../FactorizedDynamics`, together with their hashes and extraction provenance.

The activation-plateau request was traced to several distinct sources rather than one
published estimand. The source audit and implementation mapping are recorded in
`activation-plateau-source-audit.md`. Two arXiv papers were read from local audit
copies with the following identities:

| File | Access | SHA-256 |
|---|---|---|
| `papers/characterizing-stable-regions-2409.17113.pdf` | local-only arXiv audit copy, 17 pages | `98bf84088b12cdb1a85352d911dd9eb5d3f2c311df42b6e631c844d1141a18cd` |
| `papers/evaluating-synthetic-activations-2409.15019.pdf` | local-only arXiv audit copy, 18 pages | `62684843a314a95c6c31c6054ec930ff09295bcb2d78b1a84be54490c156f8ac` |

The PDFs are deliberately ignored by Git rather than redistributed. Public clones
should download them from the official arXiv links in the source audit and may compare
the bytes against the hashes above.

The 2024 real-versus-Gaussian activation-plateau report and the 2025 Jacobian report
are web publications rather than formal PDFs. Their URLs, released-code revisions,
and access limitations are therefore pinned in the audit instead of being represented
as local papers.
