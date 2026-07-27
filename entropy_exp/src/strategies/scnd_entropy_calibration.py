"""Architecture-level entropy calibration for SCND control parameters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ENTROPY_DEFINITION = "raw_positive_shannon_div_log_n_v1"
VALID_CALIBRATION_MODES = {
    "identity",
    "quantile_affine",
    "budget_adaptive_quantile",
    "budget_adaptive_tail_quantile",
}


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def model_config_fingerprint(metadata: Mapping[str, Any]) -> str:
    """Return a stable fingerprint for calibration-critical model metadata."""
    keys = (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "mm_vision_tower",
        "mm_vision_select_layer",
        "mm_vision_select_feature",
        "image_aspect_ratio",
        "image_grid_pinpoints",
        "mm_patch_merge_type",
    )
    payload = {key: metadata.get(key) for key in keys}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_calibration_artifact(
    path: str | Path,
    *,
    expected_model_name: str | None = None,
    expected_model_fingerprint: str | None = None,
    expected_layer: int | None = None,
) -> dict[str, Any]:
    artifact_path = Path(path).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = Path.cwd() / artifact_path
    if not artifact_path.is_file():
        raise FileNotFoundError(f"SCND entropy calibration artifact not found: {artifact_path}")

    with artifact_path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, dict):
        raise ValueError("SCND entropy calibration artifact must be a JSON object")
    if int(artifact.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported SCND entropy calibration schema: {artifact.get('schema_version')!r}")
    if artifact.get("mode") != "quantile_affine":
        raise ValueError(f"Calibration artifact mode must be 'quantile_affine', got {artifact.get('mode')!r}")
    if artifact.get("entropy_definition") != ENTROPY_DEFINITION:
        raise ValueError(
            "Calibration artifact entropy definition mismatch: "
            f"{artifact.get('entropy_definition')!r} != {ENTROPY_DEFINITION!r}"
        )

    q_low = float(artifact["quantiles"]["low_value"])
    q_high = float(artifact["quantiles"]["high_value"])
    if not 0.0 <= q_low < q_high <= 1.0 or q_high - q_low < 1e-4:
        raise ValueError(f"Invalid calibration quantile interval: [{q_low}, {q_high}]")

    if expected_model_name and artifact.get("model_name") != expected_model_name:
        raise ValueError(
            f"Calibration model mismatch: {artifact.get('model_name')!r} != {expected_model_name!r}"
        )
    if expected_model_fingerprint and artifact.get("model_config_fingerprint") != expected_model_fingerprint:
        raise ValueError("Calibration model-config fingerprint does not match the loaded model")
    if expected_layer is not None and int(artifact.get("capture_layer", -1)) != int(expected_layer):
        raise ValueError(
            f"Calibration layer mismatch: {artifact.get('capture_layer')!r} != {int(expected_layer)}"
        )

    artifact["_resolved_path"] = str(artifact_path.resolve())
    return artifact


def apply_entropy_calibration(
    raw_entropy_norm: float,
    mode: str,
    artifact: Mapping[str, Any] | None,
    *,
    keep_fraction: float | None = None,
    budget_keep_fraction_low: float = 0.125,
    budget_keep_fraction_high: float = 0.5,
    diversity_tail_gain: float = 1.0,
) -> dict[str, Any]:
    mode = str(mode).lower()
    if mode not in VALID_CALIBRATION_MODES:
        raise ValueError(f"entropy_calibration.mode only supports {sorted(VALID_CALIBRATION_MODES)}, got {mode!r}")

    raw_value = _clamp01(raw_entropy_norm)
    if mode == "identity":
        return {
            "control_value": raw_value,
            "quantile_value": None,
            "mode": mode,
            "q_low": None,
            "q_high": None,
            "clipped_low": False,
            "clipped_high": False,
            "artifact_path": None,
            "budget_keep_fraction": keep_fraction,
            "budget_pressure": 0.0,
            "budget_keep_fraction_low": budget_keep_fraction_low,
            "budget_keep_fraction_high": budget_keep_fraction_high,
            "diversity_tail_gain": diversity_tail_gain,
            "diversity_tail_delta": 0.0,
        }
    if artifact is None:
        raise ValueError(f"{mode} entropy calibration requires an artifact")

    q_low = float(artifact["quantiles"]["low_value"])
    q_high = float(artifact["quantiles"]["high_value"])
    quantile_value = _clamp01((raw_value - q_low) / (q_high - q_low))
    budget_pressure = 0.0
    diversity_tail_delta = 0.0
    control_value = quantile_value
    if mode in {"budget_adaptive_quantile", "budget_adaptive_tail_quantile"}:
        if keep_fraction is None:
            raise ValueError(f"{mode} requires keep_fraction")
        keep_fraction = float(keep_fraction)
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
        if not 0.0 < budget_keep_fraction_low < budget_keep_fraction_high <= 1.0:
            raise ValueError(
                "budget keep-fraction bounds must satisfy 0 < low < high <= 1, got "
                f"{budget_keep_fraction_low}, {budget_keep_fraction_high}"
            )
        if diversity_tail_gain < 0.0:
            raise ValueError(f"diversity_tail_gain must be non-negative, got {diversity_tail_gain}")

        budget_pressure = _clamp01(
            (budget_keep_fraction_high - keep_fraction)
            / (budget_keep_fraction_high - budget_keep_fraction_low)
        )
        if mode == "budget_adaptive_quantile":
            diversity_tail_delta = max(quantile_value - raw_value, 0.0) * float(diversity_tail_gain)
            diversity_control = _clamp01(raw_value + diversity_tail_delta)
        else:
            # Preserve the confident q=0 endpoint and expand only the uncertain
            # interior/tail of the architecture-calibrated quantile range.
            diversity_tail_delta = (
                quantile_value * (1.0 - quantile_value) * float(diversity_tail_gain)
            )
            diversity_control = _clamp01(quantile_value + diversity_tail_delta)
        control_value = (1.0 - budget_pressure) * quantile_value + budget_pressure * diversity_control

    return {
        "control_value": _clamp01(control_value),
        "quantile_value": quantile_value,
        "mode": mode,
        "q_low": q_low,
        "q_high": q_high,
        "clipped_low": raw_value <= q_low,
        "clipped_high": raw_value >= q_high,
        "artifact_path": artifact.get("_resolved_path"),
        "budget_keep_fraction": keep_fraction,
        "budget_pressure": budget_pressure,
        "budget_keep_fraction_low": float(budget_keep_fraction_low),
        "budget_keep_fraction_high": float(budget_keep_fraction_high),
        "diversity_tail_gain": float(diversity_tail_gain),
        "diversity_tail_delta": float(diversity_tail_delta),
    }
