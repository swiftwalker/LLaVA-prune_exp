# Official Repo Notes

- Official repository: `https://github.com/Cooperx521/PyramidDrop`
- Pinned commit: `6444f304aee4b7edcb022839fb9e8cd07859704a`
- Main PDROP implementation: `llava/model/modeling_llama_pdrop.py`
- Official LLaVA-1.5 eval examples: `scripts/v1_5/pdrop_eval/`

The official examples pass:

```bash
--layer_list '[8,16,24]'
--image_token_ratio_list '[0.5,0.25,0.125]'
--conv-mode vicuna_v1
```

The local wrapper preserves those arguments and swaps only paths, output
locations, and local evaluator commands.
