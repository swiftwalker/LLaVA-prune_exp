# DivPrune LLaVA-NeXT Compatibility

The pinned official checkout remains unmodified. Its LLaVA-NeXT pruning block
uses `SYS_TOKEN_LEN=35`, which is not valid for the Mistral prompt template.

`algo_compare.llava_next_official.install_divprune_next_adapter` suppresses
that block at runtime, infers the visual span from `IMAGE_TOKEN_INDEX`, and
calls the unchanged official `DivPrune()` selector on the merged anyres visual
sequence. This also produces per-sample dynamic token-count metadata.
