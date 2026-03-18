"""
Pre-prune attention-score strategy.

Uses the same aggregation as AttnScoreStrategy, but the actual score is computed
from the current layer input before the layer forward runs.
"""

from .attn_score import AttnScoreStrategy


class PreAttnScoreStrategy(AttnScoreStrategy):
    """Attention-score pruning applied before the target layer forward."""

    def prune_stage(self) -> str:
        return "pre"
