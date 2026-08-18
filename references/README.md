# Local source archive

The files under `notes/` are the two mathematical notes used to design the project.
They are proposals, not empirical sources. Both were read in full before implementation.

| File | SHA-256 |
|---|---|
| `notes/approximate-factorization-voronoi-residual.md` | `2a20886c4d1d56ecd89603ad120841ad490c0d5048db715934fbceb076750db8` |
| `notes/approximate-symmetry-operator-algebra.md` | `09d2e2cf59abf2d4901e10ba4761ccd1da181815a1b35baf75fa02a013948fe8` |

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
