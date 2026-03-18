from dataclasses import asdict, dataclass
from pathlib import Path

from utils.io import load_yaml
from utils.overrides import apply_overrides


@dataclass
class ResolvedPruneSpec:
    strategy_name: str
    layer_selection: str
    prune_layers: list[int]
    prune_ratio_map: dict[int, float]
    ratio_mode: str
    scoring_mode: str
    run_mode: str | None
    config_source: str
    source_path: str
    source_dataset: str | None
    source_run_name: str | None
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_layers(raw_layers) -> list[int]:
    if raw_layers is None:
        return []
    if isinstance(raw_layers, int):
        return [int(raw_layers)]
    return [int(layer) for layer in raw_layers]


def _normalize_ratio_map(prune_layers: list[int], raw_ratio) -> dict[int, float]:
    if raw_ratio is None:
        return {layer: 0.0 for layer in prune_layers}
    if isinstance(raw_ratio, (int, float)):
        return {layer: float(raw_ratio) for layer in prune_layers}
    ratios = [float(value) for value in raw_ratio]
    if len(ratios) != len(prune_layers):
        raise ValueError(
            f"prune_layers ({prune_layers}) and prune_ratio ({ratios}) must match in length"
        )
    return dict(zip(prune_layers, ratios))


def _resolve_run_config_path(
    run_config_path: str | None = None,
    run_dir: str | None = None,
) -> Path:
    if run_config_path:
        return Path(run_config_path)
    if run_dir:
        return Path(run_dir) / "config.yaml"
    raise ValueError("FLOPs estimation requires either run_config_path or run_dir")


def _build_baseline_spec(
    source_path: Path,
    run_mode: str | None,
    source_dataset: str | None,
    notes: list[str],
) -> ResolvedPruneSpec:
    source_run_name = source_path.parent.name if source_path.name == "config.yaml" else source_path.stem
    return ResolvedPruneSpec(
        strategy_name="baseline",
        layer_selection="fixed",
        prune_layers=[],
        prune_ratio_map={},
        ratio_mode="static",
        scoring_mode="none",
        run_mode=run_mode,
        config_source="run_config",
        source_path=str(source_path),
        source_dataset=source_dataset,
        source_run_name=source_run_name,
        notes=notes,
    )


def resolve_prune_spec(
    run_config_path: str | None = None,
    run_dir: str | None = None,
    overrides: list[str] | dict | None = None,
    strategy_override: str | None = None,
) -> tuple[ResolvedPruneSpec, dict]:
    source_path = _resolve_run_config_path(run_config_path=run_config_path, run_dir=run_dir)
    config = load_yaml(source_path)
    resolved_config = apply_overrides(config, overrides)

    pruning = resolved_config.get("pruning", {})
    run_meta = resolved_config.get("_run_meta", {})
    run_mode = run_meta.get("run_mode")
    source_dataset = run_meta.get("dataset")
    source_run_name = source_path.parent.name if source_path.name == "config.yaml" else source_path.stem
    notes: list[str] = []

    if overrides:
        notes.append("Explicit overrides were applied on top of the saved run config")

    if strategy_override is not None:
        notes.append(f"Method override applied: {strategy_override}")

    if run_mode == "baseline" or strategy_override == "baseline":
        notes.append("Baseline mode resolved from run metadata")
        return _build_baseline_spec(source_path, run_mode, source_dataset, notes), resolved_config

    strategy_name = strategy_override or pruning.get("strategy", "baseline")
    if strategy_name == "baseline":
        notes.append("Baseline mode resolved from strategy")
        return _build_baseline_spec(source_path, run_mode, source_dataset, notes), resolved_config

    if run_mode is None:
        notes.append("Run metadata does not include _run_meta.run_mode; using pruning config only")

    layer_selection = pruning.get("layer_selection", "fixed")
    if layer_selection == "dynamic":
        raise ValueError(
            "Dynamic layer_selection is not fully specified by the saved run config. "
            "Provide an explicit observed prune plan before estimating FLOPs."
        )

    prune_layers = _normalize_layers(pruning.get("prune_layers"))
    prune_ratio_map = _normalize_ratio_map(prune_layers, pruning.get("prune_ratio", 0.0))
    ratio_mode = "static"
    scoring_mode = "attn_score"

    if strategy_name == "entropy":
        scoring_mode = "entropy_weighted"
        entropy_cfg = pruning.get("entropy", {})
        if entropy_cfg.get("dynamic_ratio", False):
            ratio_mode = "dynamic_estimated"
            notes.append(
                "entropy.dynamic_ratio=true: FLOPs length reduction uses configured base ratios as an estimate"
            )
    elif strategy_name == "attn_score":
        scoring_mode = "attn_score"
    else:
        scoring_mode = strategy_name
        notes.append(f"Custom strategy {strategy_name!r} treated as a generic method profile")

    return (
        ResolvedPruneSpec(
            strategy_name=strategy_name,
            layer_selection=layer_selection,
            prune_layers=prune_layers,
            prune_ratio_map=prune_ratio_map,
            ratio_mode=ratio_mode,
            scoring_mode=scoring_mode,
            run_mode=run_mode,
            config_source="run_config",
            source_path=str(source_path),
            source_dataset=source_dataset,
            source_run_name=source_run_name,
            notes=notes,
        ),
        resolved_config,
    )
