"""
Tail masking attention-score pruning strategy.

Uses the same masking implementation as MaskingAttnScoreStrategy, but is
intended to be paired with an internally expanded prune_layers list so that
every layer from a configured start layer onward applies masking.
"""

from .masking_attn_score import MaskingAttnScoreStrategy


class TailMaskingAttnScoreStrategy(MaskingAttnScoreStrategy):
    """Attention-score masking applied on every layer in the configured tail."""

