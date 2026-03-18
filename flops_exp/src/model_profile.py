from dataclasses import asdict, dataclass
from pathlib import Path

import json


@dataclass
class ModelProfile:
    model_name: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_model_profile(model_path: str) -> ModelProfile:
    config_path = Path(model_path) / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return ModelProfile(
        model_name=config.get("_name_or_path", Path(model_path).name),
        num_hidden_layers=int(config["num_hidden_layers"]),
        hidden_size=int(config["hidden_size"]),
        intermediate_size=int(config["intermediate_size"]),
        num_attention_heads=int(config["num_attention_heads"]),
    )

