# CDPruner LLaVA-NeXT Compatibility

No source patch is applied to the pinned checkout. The Mistral compatibility
bridge lives in `algo_compare.llava_next_official.generate_with_cdpruner`.

It loads the Mistral checkpoint without the unsupported constructor keyword,
sets `visual_token_num` after loading, calls the official shared multimodal DPP
prepare path, and then invokes parent Mistral generation with the resulting
embeddings.

The same helper replaces the fork's hard-coded `672x672` canvas choice at
runtime with canonical LLaVA-NeXT anyres resolution selection. The official
conditional-DPP implementation itself remains unchanged.
