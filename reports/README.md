# Reports

The currently supported report target is a self-contained schematic built without
importing model-training code. It lives under `reports/MOCKUP/`, includes `MOCKUP` in
its title and filename, and is ignored by Git apart from this directory scaffold. Its
immutable stage artifact contains the rendered HTML, typed payload, and exact Markdown
specification snapshot; verification rerenders them and requires byte-for-byte equality.

The landing page and first two scientific tabs form the Experiment 1 mock dashboard:
null-relative formation, boundary alignment, path-support diagnostics, snapping against
matched controls, finite recovery, and the descriptive module comparison. Later synthetic
and algebra tabs are retained only to show the downstream gates. Build the local viewer with
`voronoi-lab report build --mode mockup`; the configured output is
`reports/MOCKUP/voronoi_lab_MOCKUP.html`.

Measured report publication is intentionally disabled until a builder can assemble its
payload exclusively from verified, receipt-linked stage artifacts. The reporting library
already defines the presentation schema, but an arbitrary local JSON payload is not an
authenticated scientific result.
