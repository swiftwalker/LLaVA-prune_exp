"""
Masking attention-score pruning strategy.

Uses the same importance aggregation as AttnScoreStrategy, but applies the
decision as a logits mask inside the target layer instead of physically
removing visual tokens.
"""

from .attn_score import AttnScoreStrategy


class MaskingAttnScoreStrategy(AttnScoreStrategy):
    """Attention-score pruning applied as an intra-layer attention mask."""

    def prune_stage(self) -> str:
        return "masking"
