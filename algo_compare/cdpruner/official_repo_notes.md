# CDPruner Official Repo Notes

- Repository: `https://github.com/Theia-4869/CDPruner`
- Target commit: `9541616c40fcd5625de1cdb8ea6c33c129eb7864`
- Official control: `--visual_token_num`, interpreted per 576-token crop on NeXT.
- Selection path: CLIP instruction relevance, conditional similarity kernel,
  and official fast MAP-DPP subset selection.
- Official NeXT scripts target Vicuna. The local wrapper supplies only the
  missing Mistral model interface and preserves the official selector.
- The official fork hard-codes the anyres canvas to `672x672`. The local wrapper
  restores the target NeXT model's canonical resolution choice so comparisons
  use the same image crops and input compute.
