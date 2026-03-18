import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).absolute().parent
FLOPS_ROOT = THIS_DIR.parent
REPO_ROOT = FLOPS_ROOT.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from batch_flops import compute_batch_flops
from llm_flops import compute_flops_report
from model_profile import load_model_profile
from prune_config_resolver import resolve_prune_spec
from report import save_flops_bundle, save_text_stats_bundle
from text_stats.compute_stats import compute_text_length_stats
from text_stats.prompt_builder import compute_canonical_template_overhead
from text_stats.tokenizer_utils import load_model_flags, load_tokenizer
from utils.io import load_yaml, make_run_dir, read_json


def compute_text_stats(
    config_path: str,
    dataset_name: str,
    dataset_config_path: str | None = None,
    sample_limit: int | None = None,
):
    config = load_yaml(config_path)
    model_cfg = config["model"]
    input_cfg = config["input"]
    output_cfg = config["output"]
    stats, records, metadata = compute_text_length_stats(
        model_path=str(REPO_ROOT / model_cfg["model_path"]),
        tokenizer_path=(str(REPO_ROOT / model_cfg["tokenizer_path"]) if model_cfg.get("tokenizer_path") else None),
        conv_mode=model_cfg["conv_mode"],
        image_token_len=int(input_cfg["image_token_len"]),
        dataset_name=dataset_name,
        dataset_config_path=dataset_config_path,
        sample_limit=sample_limit,
    )
    run_dir = make_run_dir(
        REPO_ROOT / output_cfg["base_dir"],
        "text_stats",
        dataset_name,
    )
    summary = stats.to_dict()
    summary["metadata"] = metadata
    save_text_stats_bundle(
        run_dir=run_dir,
        config_snapshot={
            "config_path": config_path,
            "dataset_name": dataset_name,
            "dataset_config_path": dataset_config_path,
            "sample_limit": sample_limit,
        },
        summary=summary,
        per_sample=[record.to_dict() for record in records],
    )
    return run_dir, summary


def _resolve_text_length_source(
    config: dict,
    fixed_text_len: float | None,
    stats_json: str | None,
    length_stat: str,
) -> tuple[float, int, int, dict]:
    model_cfg = config["model"]
    input_cfg = config["input"]
    if stats_json:
        summary = read_json(stats_json)
        text_len = float(summary[f"raw_text_token_len_{length_stat}"])
        template_overhead_tokens = int(summary["template_overhead_tokens"])
        image_token_len = int(summary["image_token_len"])
        source = {
            "mode": "stats_based",
            "stats_json": stats_json,
            "length_stat": length_stat,
            "text_token_len": text_len,
        }
        return text_len, template_overhead_tokens, image_token_len, source

    if fixed_text_len is None:
        raise ValueError("Either --fixed-text-len or --stats-json must be provided")

    model_path = str(REPO_ROOT / model_cfg["model_path"])
    tokenizer_path = (
        str(REPO_ROOT / model_cfg["tokenizer_path"])
        if model_cfg.get("tokenizer_path")
        else model_path
    )
    tokenizer = load_tokenizer(tokenizer_path)
    flags = load_model_flags(model_path)
    template_overhead_tokens = compute_canonical_template_overhead(
        tokenizer=tokenizer,
        conv_mode=model_cfg["conv_mode"],
        mm_use_im_start_end=flags["mm_use_im_start_end"],
    )
    image_token_len = int(input_cfg["image_token_len"])
    source = {
        "mode": "fixed_text_length",
        "text_token_len": float(fixed_text_len),
        "template_overhead_tokens": template_overhead_tokens,
        "image_token_len": image_token_len,
    }
    return float(fixed_text_len), template_overhead_tokens, image_token_len, source


def compute_llm_flops(
    config_path: str,
    run_config_path: str | None = None,
    run_dir: str | None = None,
    fixed_text_len: float | None = None,
    stats_json: str | None = None,
    length_stat: str = "mean",
    overrides: list[str] | None = None,
    strategy_override: str | None = None,
):
    if not run_config_path and not run_dir:
        raise ValueError("FLOPs estimation requires --run-config or --run-dir")

    config = load_yaml(config_path)
    model_cfg = config["model"]
    output_cfg = config["output"]

    model_profile = load_model_profile(str(REPO_ROOT / model_cfg["model_path"]))
    prune_spec, resolved_prune_config = resolve_prune_spec(
        run_config_path=(str(REPO_ROOT / run_config_path) if run_config_path and not Path(run_config_path).is_absolute() else run_config_path),
        run_dir=(str(REPO_ROOT / run_dir) if run_dir and not Path(run_dir).is_absolute() else run_dir),
        overrides=overrides,
        strategy_override=strategy_override,
    )
    text_len, template_overhead_tokens, image_token_len, text_length_source = _resolve_text_length_source(
        config=config,
        fixed_text_len=fixed_text_len,
        stats_json=stats_json,
        length_stat=length_stat,
    )

    flops = compute_flops_report(
        model_profile=model_profile,
        prune_spec=prune_spec,
        text_token_len=text_len,
        template_overhead_tokens=template_overhead_tokens,
        image_token_len=image_token_len,
    )
    run_name = prune_spec.source_run_name or "run"
    run_dir_out = make_run_dir(
        REPO_ROOT / output_cfg["base_dir"],
        "flops",
        run_name,
    )
    report = {
        "model_profile": model_profile.to_dict(),
        "text_length_source": text_length_source,
        "resolved_prune_spec": prune_spec.to_dict(),
        "baseline": flops["baseline"],
        "pruned": flops["pruned"],
        "reduction_ratio": flops["reduction_ratio"],
        "prefill_seq_len": flops["prefill_seq_len"],
        "baseline_schedule": flops["baseline_schedule"],
        "pruned_schedule": flops["pruned_schedule"],
        "assumptions": [
            "Counts only LLM prefill FLOPs and prune method overhead",
            "Vision tower and mm projector are excluded",
            "Sorting/top-k/cache movement are treated as non-FLOP overhead",
        ] + prune_spec.notes,
    }
    save_flops_bundle(
        run_dir=run_dir_out,
        config_snapshot={
            "config_path": config_path,
            "run_config_path": run_config_path,
            "run_dir": run_dir,
            "length_stat": length_stat,
            "fixed_text_len": fixed_text_len,
            "stats_json": stats_json,
            "overrides": overrides or [],
        },
        resolved_method={
            "prune_spec": prune_spec.to_dict(),
            "resolved_prune_config": resolved_prune_config,
        },
        report=report,
        notes=report["assumptions"],
    )
    return run_dir_out, report


def compute_batch_llm_flops(
    config_path: str,
    runs_root: str,
    scope: str = "runs",
    dataset: str | None = None,
    stats_json: str | None = None,
    fixed_text_len: float | None = None,
    length_stat: str = "mean",
    overrides: list[str] | None = None,
):
    return compute_batch_flops(
        config_path=config_path,
        runs_root=str(REPO_ROOT / runs_root) if not Path(runs_root).is_absolute() else runs_root,
        compute_llm_flops_fn=compute_llm_flops,
        output_base_dir=str(REPO_ROOT / load_yaml(config_path)["output"]["base_dir"]),
        scope=scope,
        dataset=dataset,
        stats_json=stats_json,
        fixed_text_len=fixed_text_len,
        length_stat=length_stat,
        overrides=overrides,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="flops_exp workspace entrypoint")
    parser.add_argument(
        "--config",
        default=str(FLOPS_ROOT / "configs" / "default.yaml"),
        help="Path to the flops_exp default config",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    text_parser = subparsers.add_parser("text-stats", help="Compute dataset text token statistics")
    text_parser.add_argument("--dataset", required=True, choices=["mme", "gqa", "pope"])
    text_parser.add_argument("--dataset-config", default=None)
    text_parser.add_argument("--sample-limit", type=int, default=None)

    flops_parser = subparsers.add_parser("flops", help="Estimate LLM FLOPs")
    flops_parser.add_argument("--run-config", default=None)
    flops_parser.add_argument("--run-dir", default=None)
    flops_parser.add_argument("--method", default=None)
    flops_parser.add_argument("--fixed-text-len", type=float, default=None)
    flops_parser.add_argument("--stats-json", default=None)
    flops_parser.add_argument("--length-stat", default="mean", choices=["mean", "median", "p25", "p75"])
    flops_parser.add_argument("--set", dest="overrides", action="append", default=[])

    batch_parser = subparsers.add_parser("batch-flops", help="Estimate FLOPs for multiple runs")
    batch_parser.add_argument("--runs-root", default="entropy_exp/outputs/runs")
    batch_parser.add_argument("--scope", default="runs", choices=["runs"])
    batch_parser.add_argument("--dataset", default=None, choices=["mme", "gqa", "pope"])
    batch_parser.add_argument("--stats-json", default=None)
    batch_parser.add_argument("--fixed-text-len", type=float, default=None)
    batch_parser.add_argument("--length-stat", default="mean", choices=["mean", "median", "p25", "p75"])
    batch_parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.command == "text-stats":
        run_dir, _ = compute_text_stats(
            config_path=args.config,
            dataset_name=args.dataset,
            dataset_config_path=args.dataset_config,
            sample_limit=args.sample_limit,
        )
        print(run_dir)
        return

    if args.command == "flops":
        run_dir, _ = compute_llm_flops(
            config_path=args.config,
            run_config_path=args.run_config,
            run_dir=args.run_dir,
            fixed_text_len=args.fixed_text_len,
            stats_json=args.stats_json,
            length_stat=args.length_stat,
            overrides=args.overrides,
            strategy_override=args.method,
        )
        print(run_dir)
        return

    if args.command == "batch-flops":
        run_dir, _, _ = compute_batch_llm_flops(
            config_path=args.config,
            runs_root=args.runs_root,
            scope=args.scope,
            dataset=args.dataset,
            stats_json=args.stats_json,
            fixed_text_len=args.fixed_text_len,
            length_stat=args.length_stat,
            overrides=args.overrides,
        )
        print(run_dir)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
