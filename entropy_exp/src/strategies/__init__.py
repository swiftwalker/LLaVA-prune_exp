"""Pruning strategy registry."""

from .attn_score import AttnScoreStrategy
from .entropy import EntropyStrategy
from .masking_attn_score import MaskingAttnScoreStrategy
from .pre_attn_score import PreAttnScoreStrategy
from .random import RandomStrategy
from .sparsevlm import SparseVLMStrategy
from .tail_masking_attn_score import TailMaskingAttnScoreStrategy

STRATEGY_REGISTRY = {
    "attn_score": AttnScoreStrategy,
    "entropy": EntropyStrategy,
    "masking_attn_score": MaskingAttnScoreStrategy,
    "pre_attn_score": PreAttnScoreStrategy,
    "random": RandomStrategy,
    "sparsevlm": SparseVLMStrategy,
    "tail_masking_attn_score": TailMaskingAttnScoreStrategy,
}


def get_strategy(name: str, config: dict):
    """Instantiate a pruning strategy by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](config)
