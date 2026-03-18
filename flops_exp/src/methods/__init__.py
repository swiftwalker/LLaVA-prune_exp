from methods.attn_score import AttnScoreEstimator
from methods.baseline import BaselineEstimator
from methods.entropy import EntropyEstimator


def get_method_estimator(strategy_name: str):
    if strategy_name == "baseline":
        return BaselineEstimator()
    if strategy_name == "attn_score":
        return AttnScoreEstimator()
    if strategy_name == "entropy":
        return EntropyEstimator()
    raise ValueError(f"Unsupported strategy for FLOPs estimation: {strategy_name}")

