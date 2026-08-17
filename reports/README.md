# Reports

The currently supported report target is a self-contained schematic built without
importing model-training code. It lives under `reports/MOCKUP/`, includes `MOCKUP` in
its title and filename, and is ignored by Git apart from this directory scaffold. Its
immutable stage artifact contains the rendered HTML, typed payload, and exact Markdown
specification snapshot; verification rerenders them and requires byte-for-byte equality.

Measured report publication is intentionally disabled until a builder can assemble its
payload exclusively from verified, receipt-linked stage artifacts. The reporting library
already defines the presentation schema, but an arbitrary local JSON payload is not an
authenticated scientific result.
