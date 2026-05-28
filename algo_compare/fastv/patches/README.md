# FastV Patches

Do not edit official source in place for normal runs. If a compatibility patch
is needed, document it here and keep the patch separate from the fetched source
tree.

## `fastv_token_mask_bounds_guard.patch`

Minimal local compatibility patch for the official token-mask path. It sizes
the RoPE cache from the actual KV length instead of the official hard-coded
`1000`, then clamps FastV's fixed image-token window and attention-rank indices
to the current attention/mask sequence bounds. This avoids CUDA device-side
asserts on local TextVQA OCR prompts and ScienceQA long-context prompts, while
preserving the same K/R token-mask behavior when the official fixed window is
valid.

The runtime wrapper also keeps a conservative per-sample fallback knob:
`--fastv-max-expanded-tokens`; samples above that estimated expanded prompt
length are decoded with FastV disabled and are marked in `answers.jsonl`
metadata. The method default is `0`, so fallback is disabled unless explicitly
requested.
