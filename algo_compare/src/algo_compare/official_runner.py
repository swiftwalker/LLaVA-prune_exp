"""Resolver and command builder for official method wrappers."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .manifest import load_yaml
from .paths import ALGO_COMPARE_ROOT, REPO_ROOT, resolve_repo_path


SUPPORTED_DATASETS = ("gqa", "mme", "pope", "textvqa", "scienceqa", "mmbench", "mmvet", "ai2d")


@dataclass
class OfficialRun:
    method: str
    dataset: str
    variant: str
    retain_token: int
    use_version: str
    model_path: Path
    model_name: str
    question_file: Path
    image_folder: Path | None
    output_dir: Path
    answers_file: Path
    official_repo: Path
    inference_module: str
    python_bin: str
    command: list[str]
    env: dict[str, str]
    cwd: Path
    eval_command: list[str] | None = None
    run_config: dict[str, Any] = field(default_factory=dict)
    method_params: dict[str, Any] = field(default_factory=dict)


def sanitize_label(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    label = "".join(cleaned).strip("._")
    return label or "unknown"


def shell_join(command: list[str]) -> str:
    return shlex.join([str(part) for part in command])


def method_dir(method: str) -> Path:
    path = ALGO_COMPARE_ROOT / method
    if not (path / "method.yaml").is_file():
        raise FileNotFoundError(f"Unknown method or missing method.yaml: {method}")
    return path


def load_method_config(method: str) -> dict[str, Any]:
    return load_yaml(method_dir(method) / "method.yaml")


def load_method_env(method: str) -> dict[str, Any]:
    env_path = method_dir(method) / "env.yaml"
    return load_yaml(env_path) if env_path.is_file() else {}


def load_prune_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = resolve_repo_path(path or "entropy_exp/configs/prune.yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing local prune config: {cfg_path}")
    return load_yaml(cfg_path)


def resolve_path(value: str | Path | None) -> Path | None:
    if value in {None, ""}:
        return None
    return resolve_repo_path(value)


def variant_use_version(method_cfg: dict[str, Any], variant: str | None) -> tuple[str, str]:
    defaults = method_cfg.get("defaults", {}) or {}
    variants = method_cfg.get("variants", []) or []
    first_variant = variants[0].get("name") if variants else None
    selected = variant or str(defaults.get("variant") or first_variant or method_cfg.get("name") or "official")
    for item in method_cfg.get("variants", []) or []:
        if item.get("name") == selected:
            return selected, str(item.get("use_version") or defaults.get("use_version") or "")
    return selected, str(defaults.get("use_version") or "")


def _prefer(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _dataset_defaults(dataset: str, method_env: dict[str, Any], prune_cfg: dict[str, Any]) -> dict[str, Any]:
    datasets = method_env.get("datasets", {}) or {}
    local = prune_cfg.get("datasets", {}) or {}
    method_ds = datasets.get(dataset, {}) or {}
    prune_ds = local.get(dataset, {}) or {}
    return {
        "question_file": _prefer(method_ds.get("question_file"), prune_ds.get("question_file")),
        "image_folder": _prefer(method_ds.get("image_folder"), prune_ds.get("image_folder")),
        "supports_local_eval": bool(method_ds.get("supports_local_eval", True)),
        "max_new_tokens": _prefer(method_ds.get("max_new_tokens"), prune_ds.get("max_new_tokens")),
        "notes": method_ds.get("notes"),
    }


def _model_defaults(method_cfg: dict[str, Any], method_env: dict[str, Any], prune_cfg: dict[str, Any]) -> tuple[str, str]:
    method_paths = method_env.get("paths", {}) or {}
    model_cfg = prune_cfg.get("model", {}) or {}
    defaults = method_cfg.get("defaults", {}) or {}
    model_path = _prefer(method_paths.get("default_model_path"), model_cfg.get("path"))
    model_name = _prefer(defaults.get("model_name"), model_cfg.get("name"))
    if not model_path:
        raise ValueError("Cannot resolve model path from env.yaml or entropy_exp/configs/prune.yaml")
    return str(model_path), str(model_name or Path(str(model_path)).name)


def default_python_bin(method_env: dict[str, Any]) -> str:
    python_cfg = method_env.get("python", {}) or {}
    preferred = python_cfg.get("preferred")
    fallback = python_cfg.get("fallback") or "python3"
    if preferred and Path(str(preferred)).exists():
        return str(preferred)
    return str(fallback)


def fastv_params(method_cfg: dict[str, Any], args: Any) -> dict[str, Any]:
    defaults = (method_cfg.get("defaults", {}) or {}).get("fastv", {}) or {}
    image_token_length = int(args.fastv_image_token_length or defaults.get("image_token_length") or 576)
    ratio = float(args.fastv_r if args.fastv_r is not None else defaults.get("r", 0.5))
    attention_rank = args.fastv_attention_rank
    if attention_rank is None:
        attention_rank = int(round((1.0 - ratio) * image_token_length))
    return {
        "k": int(args.fastv_k if args.fastv_k is not None else defaults.get("k") or 2),
        "r": ratio,
        "attention_rank": int(attention_rank),
        "image_token_length": image_token_length,
        "sys_length": int(args.fastv_sys_length or defaults.get("sys_length") or 35),
        "mode": str(args.fastv_mode or defaults.get("mode") or "token_mask"),
        "max_expanded_tokens": int(
            args.fastv_max_expanded_tokens
            if args.fastv_max_expanded_tokens is not None
            else defaults.get("max_expanded_tokens", 900)
        ),
    }


def pdrop_params(method_cfg: dict[str, Any], args: Any) -> dict[str, Any]:
    defaults = (method_cfg.get("defaults", {}) or {}).get("pdrop", {}) or {}
    layer_list = getattr(args, "pdrop_layer_list", None) or defaults.get("layer_list") or "[8,16,24]"
    ratio_list = (
        getattr(args, "pdrop_image_token_ratio_list", None)
        or defaults.get("image_token_ratio_list")
        or "[0.5,0.25,0.125]"
    )
    return {
        "layer_list": str(layer_list),
        "image_token_ratio_list": str(ratio_list),
    }


def visionzip_params(method_cfg: dict[str, Any], args: Any) -> dict[str, Any]:
    defaults = (method_cfg.get("defaults", {}) or {}).get("visionzip", {}) or {}
    dominant = int(getattr(args, "visionzip_dominant", None) or defaults.get("dominant") or 54)
    contextual = int(getattr(args, "visionzip_contextual", None) or defaults.get("contextual") or 10)
    return {
        "dominant": dominant,
        "contextual": contextual,
        "retained_visual_tokens": dominant + contextual,
    }


def divprune_params(method_cfg: dict[str, Any], args: Any) -> dict[str, Any]:
    defaults = (method_cfg.get("defaults", {}) or {}).get("divprune", {}) or {}
    subset_ratio = float(
        getattr(args, "divprune_subset_ratio", None)
        if getattr(args, "divprune_subset_ratio", None) is not None
        else defaults.get("subset_ratio", 0.098)
    )
    visual_token_count = int(
        getattr(args, "divprune_visual_token_count", None) or defaults.get("visual_token_count") or 576
    )
    return {
        "baseline": str(getattr(args, "divprune_baseline", None) or defaults.get("baseline") or "OURS"),
        "layer_index": int(getattr(args, "divprune_layer_index", None) or defaults.get("layer_index") or 0),
        "subset_ratio": subset_ratio,
        "visual_token_count": visual_token_count,
        "retained_visual_tokens": int(round(subset_ratio * visual_token_count)),
    }


def _format_float_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _compact_list_label(value: str) -> str:
    return (
        value.replace("[", "")
        .replace("]", "")
        .replace(" ", "")
        .replace(",", "-")
        .replace(".", "p")
        .replace("'", "")
        .replace('"', "")
    )


def default_output_dir(
    method: str,
    dataset: str,
    model_name: str,
    variant: str,
    retain_token: int,
    method_params: dict[str, Any] | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [sanitize_label(dataset), sanitize_label(model_name), sanitize_label(variant)]
    if method == "fastv" and method_params:
        parts.append(f"k{method_params['k']}")
        parts.append(f"r{_format_float_label(float(method_params['r']))}")
    elif method == "pdrop" and method_params:
        parts.append(f"layers{_compact_list_label(str(method_params['layer_list']))}")
        parts.append(f"ratios{_compact_list_label(str(method_params['image_token_ratio_list']))}")
    elif method == "visionzip" and method_params:
        parts.append(f"dom{method_params['dominant']}")
        parts.append(f"ctx{method_params['contextual']}")
    elif method == "divprune" and method_params:
        parts.append(f"ratio{_format_float_label(float(method_params['subset_ratio']))}")
        parts.append(f"layer{method_params['layer_index']}")
    else:
        parts.append(f"retain{retain_token}")
    parts.append(timestamp)
    label = "_".join(parts)
    return ALGO_COMPARE_ROOT / method / "outputs" / label


def build_official_run(args: Any) -> OfficialRun:
    method = args.method
    dataset = args.dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; expected one of {', '.join(SUPPORTED_DATASETS)}")

    method_cfg = load_method_config(method)
    method_env = load_method_env(method)
    prune_cfg = load_prune_config(getattr(args, "prune_config", None))
    variant, use_version = variant_use_version(method_cfg, getattr(args, "variant", None))
    if method == "fastv":
        method_params = fastv_params(method_cfg, args)
    elif method == "pdrop":
        method_params = pdrop_params(method_cfg, args)
    elif method == "visionzip":
        method_params = visionzip_params(method_cfg, args)
    elif method == "divprune":
        method_params = divprune_params(method_cfg, args)
    else:
        method_params = {}

    default_model_path, default_model_name = _model_defaults(method_cfg, method_env, prune_cfg)
    model_path = resolve_path(args.model_path or default_model_path)
    if model_path is None:
        raise ValueError("Model path resolved to None")
    model_name = args.model_name or default_model_name or model_path.name

    ds_defaults = _dataset_defaults(dataset, method_env, prune_cfg)
    question_file = resolve_path(args.question_file or ds_defaults.get("question_file"))
    if question_file is None:
        raise ValueError(f"Cannot resolve question file for dataset {dataset}")
    image_folder = resolve_path(args.image_folder or ds_defaults.get("image_folder"))

    defaults = method_cfg.get("defaults", {}) or {}
    inference_defaults = prune_cfg.get("inference", {}) or {}
    retain_token = int(args.retain_token or defaults.get("retain_token") or 192)
    if method == "cdpruner":
        method_params = {"visual_token_num_per_crop": retain_token}
    conv_mode = args.conv_mode or defaults.get("conv_mode") or inference_defaults.get("conv_mode") or "vicuna_v1"
    temperature = _prefer(args.temperature, defaults.get("temperature"), inference_defaults.get("temperature"), 0)
    top_p = _prefer(args.top_p, defaults.get("top_p"), inference_defaults.get("top_p"))
    num_beams = int(_prefer(args.num_beams, defaults.get("num_beams"), inference_defaults.get("num_beams"), 1))
    max_new_tokens = int(
        _prefer(
            args.max_new_tokens,
            ds_defaults.get("max_new_tokens"),
            defaults.get("max_new_tokens"),
            inference_defaults.get("max_new_tokens"),
            128,
        )
    )
    num_chunks = int(_prefer(args.num_chunks, 1))
    chunk_idx = int(_prefer(args.chunk_idx, 0))

    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else default_output_dir(method, dataset, model_name, variant, retain_token, method_params)
    )
    if output_dir is None:
        raise ValueError("Output dir resolved to None")
    answers_file = output_dir / "answers.jsonl"

    official = method_cfg.get("official", {}) or {}
    official_repo = resolve_path(args.official_repo or official.get("local_checkout"))
    if official_repo is None:
        raise ValueError("Cannot resolve official repo checkout path")
    entrypoints = method_cfg.get("entrypoints", {}) or {}
    dataset_modules = entrypoints.get("dataset_inference_modules", {}) or {}
    dataset_scripts = entrypoints.get("dataset_inference_scripts", {}) or {}
    inference_module = dataset_modules.get(dataset) or entrypoints.get("inference_module")
    inference_script_value = dataset_scripts.get(dataset) or entrypoints.get("inference_script")
    inference_script = resolve_path(inference_script_value) if inference_script_value else None
    if not inference_module and inference_script is None:
        raise ValueError(f"Missing entrypoints.inference_module in {method}/method.yaml")
    python_bin = args.python_bin or default_python_bin(method_env)

    command = [python_bin]
    if inference_script is not None:
        command.append(str(inference_script))
    else:
        command.extend(["-m", str(inference_module)])
    command.extend(
        [
            "--model-path",
            str(model_path),
            "--question-file",
            str(question_file),
            "--image-folder",
            str(image_folder or ""),
            "--answers-file",
            str(answers_file),
            "--temperature",
            str(temperature),
            "--conv-mode",
            str(conv_mode),
            "--num-chunks",
            str(num_chunks),
            "--chunk-idx",
            str(chunk_idx),
        ]
    )
    if method == "fastv":
        command.extend(
            [
                "--dataset",
                dataset,
                "--num_beams",
                str(num_beams),
                "--max_new_tokens",
                str(max_new_tokens),
                "--fastv-k",
                str(method_params["k"]),
                "--fastv-r",
                str(method_params["r"]),
                "--fastv-attention-rank",
                str(method_params["attention_rank"]),
                "--fastv-image-token-length",
                str(method_params["image_token_length"]),
                "--fastv-sys-length",
                str(method_params["sys_length"]),
                "--fastv-mode",
                str(method_params["mode"]),
                "--fastv-max-expanded-tokens",
                str(method_params["max_expanded_tokens"]),
            ]
        )
    elif method == "pdrop":
        command.extend(
            [
                "--layer_list",
                str(method_params["layer_list"]),
                "--image_token_ratio_list",
                str(method_params["image_token_ratio_list"]),
            ]
        )
        if dataset == "scienceqa":
            command.extend(["--single-pred-prompt"])
        elif dataset == "mmbench":
            command.extend(["--single-pred-prompt", "--lang", "en", "--pdrop_infer"])
        else:
            command.extend(["--num_beams", str(num_beams), "--max_new_tokens", str(max_new_tokens), "--pdrop_infer"])
    elif method == "visionzip":
        command.extend(
            [
                "--dataset",
                dataset,
                "--num_beams",
                str(num_beams),
                "--max_new_tokens",
                str(max_new_tokens),
                "--visionzip-dominant",
                str(method_params["dominant"]),
                "--visionzip-contextual",
                str(method_params["contextual"]),
            ]
        )
    elif method == "divprune":
        command.extend(
            [
                "--dataset",
                dataset,
                "--num_beams",
                str(num_beams),
                "--max_new_tokens",
                str(max_new_tokens),
                "--divprune-baseline",
                str(method_params["baseline"]),
                "--divprune-layer-index",
                str(method_params["layer_index"]),
                "--divprune-subset-ratio",
                str(method_params["subset_ratio"]),
                "--divprune-visual-token-count",
                str(method_params["visual_token_count"]),
            ]
        )
    elif method == "cdpruner":
        command.extend(
            [
                "--dataset",
                dataset,
                "--num-beams",
                str(num_beams),
                "--max-new-tokens",
                str(max_new_tokens),
                "--visual-token-num",
                str(retain_token),
            ]
        )
    elif dataset == "scienceqa":
        command.extend(["--single-pred-prompt", "--retained_tokens", str(retain_token), "--max-new-tokens", str(max_new_tokens)])
    elif dataset == "mmbench":
        command.extend(["--single-pred-prompt", "--lang", "en", "--retained_tokens", str(retain_token)])
    else:
        command.extend(["--num_beams", str(num_beams), "--max_new_tokens", str(max_new_tokens)])
    if args.model_base:
        command.extend(["--model-base", str(resolve_path(args.model_base) or args.model_base)])
    if top_p is not None and dataset != "scienceqa":
        command.extend(["--top_p", str(top_p)])
    if method in {"divprune", "cdpruner"} and getattr(args, "max_samples", None) is not None:
        command.extend(["--max-samples", str(args.max_samples)])

    env = {"RETAIN_TOKN": str(retain_token)}
    if args.use_version or use_version:
        env["USE_VERSION"] = str(args.use_version or use_version)
    if method == "fastv":
        env.update(
            {
                "FASTV_K": str(method_params["k"]),
                "FASTV_R": str(method_params["r"]),
                "FASTV_ATTENTION_RANK": str(method_params["attention_rank"]),
                "FASTV_IMAGE_TOKEN_LENGTH": str(method_params["image_token_length"]),
                "FASTV_SYS_LENGTH": str(method_params["sys_length"]),
                "FASTV_MODE": str(method_params["mode"]),
                "FASTV_MAX_EXPANDED_TOKENS": str(method_params["max_expanded_tokens"]),
                "FASTV_OFFICIAL_REPO": str(official_repo),
                "FASTV_ALLOW_TOKENIZERS_015": "1",
            }
        )
        fastv_python_paths = [
            method_dir(method) / "scripts" / "python_compat",
            official_repo / "src" / "transformers" / "src",
            official_repo / "src" / "FastV",
            official_repo / "src" / "LLaVA",
            official_repo / "src" / "FastV" / "llava-hf" / "transformers" / "src",
            official_repo,
        ]
        existing_pythonpath = os.environ.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(path) for path in fastv_python_paths] + ([existing_pythonpath] if existing_pythonpath else [])
        )
    elif method == "pdrop":
        env.update(
            {
                "PDROP_LAYER_LIST": str(method_params["layer_list"]),
                "PDROP_IMAGE_TOKEN_RATIO_LIST": str(method_params["image_token_ratio_list"]),
                "PDROP_OFFICIAL_REPO": str(official_repo),
            }
        )
        existing_pythonpath = os.environ.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join([str(official_repo)] + ([existing_pythonpath] if existing_pythonpath else []))
    elif method == "visionzip":
        env.update(
            {
                "VISIONZIP_DOMINANT": str(method_params["dominant"]),
                "VISIONZIP_CONTEXTUAL": str(method_params["contextual"]),
                "VISIONZIP_OFFICIAL_REPO": str(official_repo),
                "VISIONZIP_LLAVA_ROOT": str(REPO_ROOT),
            }
        )
        existing_pythonpath = os.environ.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(official_repo), str(REPO_ROOT)] + ([existing_pythonpath] if existing_pythonpath else [])
        )
    elif method == "divprune":
        llava_root = official_repo / "LLaVA"
        env.update(
            {
                "BASELINE": str(method_params["baseline"]),
                "LAYER_INDEX": str(method_params["layer_index"]),
                "SUBSET_RATIO": str(method_params["subset_ratio"]),
                "DIVPRUNE_VISUAL_TOKEN_COUNT": str(method_params["visual_token_count"]),
                "DIVPRUNE_OFFICIAL_REPO": str(official_repo),
                "DIVPRUNE_LLAVA_ROOT": str(llava_root),
            }
        )
        existing_pythonpath = os.environ.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(llava_root), str(official_repo)] + ([existing_pythonpath] if existing_pythonpath else [])
        )
    elif method == "cdpruner":
        env.update(
            {
                "CDPRUNER_OFFICIAL_REPO": str(official_repo),
                "CDPRUNER_VISUAL_TOKEN_NUM": str(retain_token),
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
            }
        )
        existing_pythonpath = os.environ.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(official_repo)] + ([existing_pythonpath] if existing_pythonpath else [])
        )

    eval_command = None
    if args.eval:
        eval_script = REPO_ROOT / "entropy_exp" / "src" / "eval_datasets.py"
        eval_command = [
            python_bin,
            str(eval_script),
            "--dataset",
            dataset,
            "--answers-file",
            str(answers_file),
            "--output-dir",
            str(output_dir / "eval"),
            "--question-file",
            str(question_file),
        ]
        if dataset == "mme" and image_folder is not None:
            eval_command.extend(["--mme-data-path", str(image_folder)])

    run_config = {
        "method": method,
        "dataset": dataset,
        "variant": variant,
        "source_type": "official",
        "official_repo": str(official_repo),
        "official_repo_target_commit": official.get("target_commit"),
        "inference_module": inference_module,
        "inference_script": str(inference_script) if inference_script else None,
        "model": {
            "path": str(model_path),
            "name": model_name,
            "base": str(resolve_path(args.model_base)) if args.model_base else None,
        },
        "dataset_paths": {
            "question_file": str(question_file),
            "image_folder": str(image_folder) if image_folder else None,
            "supports_local_eval": bool(ds_defaults.get("supports_local_eval", True)),
            "max_new_tokens": ds_defaults.get("max_new_tokens"),
            "notes": ds_defaults.get("notes"),
        },
        "official_env": env,
        "retain_token": retain_token,
        "use_version": env.get("USE_VERSION"),
        "method_params": method_params,
        "inference": {
            "conv_mode": conv_mode,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
            "num_chunks": num_chunks,
            "chunk_idx": chunk_idx,
            "max_samples": getattr(args, "max_samples", None),
        },
        "paths": {
            "output_dir": str(output_dir),
            "answers_file": str(answers_file),
            "eval_summary": str(output_dir / "eval" / "summary.json") if args.eval else None,
        },
        "command": command,
        "command_shell": shell_join(command),
        "eval_command": eval_command,
        "eval_command_shell": shell_join(eval_command) if eval_command else None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    return OfficialRun(
        method=method,
        dataset=dataset,
        variant=variant,
        retain_token=retain_token,
        use_version=env.get("USE_VERSION", ""),
        model_path=model_path,
        model_name=model_name,
        question_file=question_file,
        image_folder=image_folder,
        output_dir=output_dir,
        answers_file=answers_file,
        official_repo=official_repo,
        inference_module=inference_module,
        python_bin=python_bin,
        command=command,
        env=env,
        cwd=official_repo,
        eval_command=eval_command,
        run_config=run_config,
        method_params=method_params,
    )


def require_runnable(run: OfficialRun) -> None:
    commit_marker = run.official_repo.parent / f"{run.official_repo.name}_FETCHED_COMMIT.txt"
    has_fetched_source = (run.official_repo / ".git").is_dir() or commit_marker.is_file()
    if not has_fetched_source:
        raise FileNotFoundError(
            f"Official repo is missing at {run.official_repo}. Run the method fetch_official.sh first."
        )
    if not run.question_file.is_file():
        raise FileNotFoundError(f"Question file does not exist: {run.question_file}")
    if run.image_folder is not None and not run.image_folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {run.image_folder}")
    if not run.model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {run.model_path}")


def write_run_files(run: OfficialRun) -> None:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    with (run.output_dir / "run_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run.run_config, f, sort_keys=False, allow_unicode=True)

    command_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(run.cwd))}",
    ]
    for key, value in run.env.items():
        command_lines.append(f"export {key}={shlex.quote(str(value))}")
    command_lines.append(shell_join(run.command))
    if run.eval_command:
        command_lines.append(shell_join(run.eval_command))
    command_lines.append("")
    command_file = run.output_dir / "command.sh"
    command_file.write_text("\n".join(command_lines), encoding="utf-8")
    command_file.chmod(0o755)


def execute_command(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=merged_env, check=True)


def collect_command(run: OfficialRun) -> list[str]:
    return [
        run.python_bin,
        str(ALGO_COMPARE_ROOT / run.method / "scripts" / "collect_results.py"),
        "--run-dir",
        str(run.output_dir),
        "--dataset",
        run.dataset,
        "--method",
        run.method,
        "--run-config",
        str(run.output_dir / "run_config.yaml"),
        "--official-repo",
        str(run.official_repo),
    ]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def manifest_entry(run: OfficialRun, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = summary or read_json(run.output_dir / "official_summary.json")
    return {
        "label": run.output_dir.name,
        "source_type": "official",
        "method": run.method,
        "variant": run.variant,
        "dataset": run.dataset,
        "model": run.model_name,
        "retain_token": run.retain_token,
        "use_version": run.use_version,
        "method_params": run.method_params,
        "official_repo_commit": summary.get("official_repo_commit") or run.run_config.get("official_repo_target_commit"),
        "command": summary.get("command") or shell_join(run.command),
        "output_root": str(run.output_dir),
        "answers_file": str(run.answers_file),
        "eval_summary": str(run.output_dir / "eval" / "summary.json") if run.eval_command else None,
        "summary_json": str(run.output_dir / "official_summary.json"),
        "metric_name": summary.get("metric_name"),
        "metric_value": summary.get("metric_value"),
    }


def print_dry_run(run: OfficialRun) -> None:
    payload = {
        "method": run.method,
        "dataset": run.dataset,
        "variant": run.variant,
        "official_repo": str(run.official_repo),
        "output_dir": str(run.output_dir),
        "answers_file": str(run.answers_file),
        "model_path": str(run.model_path),
        "question_file": str(run.question_file),
        "image_folder": str(run.image_folder) if run.image_folder else None,
        "env": run.env,
        "method_params": run.method_params,
        "command": shell_join(run.command),
        "eval_command": shell_join(run.eval_command) if run.eval_command else None,
    }
    print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).rstrip())
