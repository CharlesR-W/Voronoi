# Artifact storage

Runtime artifacts are immutable and content-addressed under `artifacts/objects/`.
That directory is intentionally ignored by Git. Every object carries a schema-versioned
manifest and payload checksums. Stage outputs additionally bind their stage signature,
resolved signed configuration, source identity, and upstream artifact hashes; individual
payload schemas record seeds and array/table inventories where those concepts apply.
The Tracking2 input-stage object also preserves the exact signed YAML manifest and raw
legacy transplant JSON; large checkpoints and Parquet files remain external hash-pinned
inputs rather than duplicated artifact payloads.

Mutable run indexes live under `runs/` and point to immutable objects. They are also
ignored because they are machine- and execution-specific.

The canonical local Experiment 1 collection uses a separate ignored store at
`artifacts/exp1-measured/`. Its exact parent/child IDs, array inventory, verification
commands, and immutability rules are recorded in
[`EXPERIMENT_1_DATA.md`](EXPERIMENT_1_DATA.md).
