# Local Reference Notes

Local references are existing `entropy_exp` outputs used for context. They are
not official SparseVLM reproductions.

The most important naming distinction:

- `local_reference`: produced by this workspace, useful for local comparisons.
- `official`: produced by the official method checkout under `third_party/`.

Known local-reference caveats:

- The local S-S-S baseline is implemented via `sparsevlm_boost_hybrid` with
  `layer_modes=["S","S","S"]`.
- It uses the local pruning engine, local cache/position-id handling, and local
  result aggregation.
- It must not be cited as an official SparseVLM-v2 reproduction.
