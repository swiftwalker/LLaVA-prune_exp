"""Architecture-level entropy calibration for SCND control parameters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ENTROPY_DEFINITION = "raw_positive_shannon_div_log_n_v1"
VALID_CALIBRATION_MODES = {"identity", "quantile_affine"}


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


def apply_entropy_calibration(raw_entropy_norm: float, mode: str, artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    mode = str(mode).lower()
    if mode not in VALID_CALIBRATION_MODES:
        raise ValueError(f"entropy_calibration.mode only supports {sorted(VALID_CALIBRATION_MODES)}, got {mode!r}")

    raw_value = min(max(float(raw_entropy_norm), 0.0), 1.0)
    if mode == "identity":
        return {
            "control_value": raw_value,
            "mode": mode,
            "q_low": None,
            "q_high": None,
            "clipped_low": False,
            "clipped_high": False,
            "artifact_path": None,
        }
    if artifact is None:
        raise ValueError("quantile_affine entropy calibration requires an artifact")

    q_low = float(artifact["quantiles"]["low_value"])
    q_high = float(artifact["quantiles"]["high_value"])
    scaled = (raw_value - q_low) / (q_high - q_low)
    return {
        "control_value": min(max(scaled, 0.0), 1.0),
        "mode": mode,
        "q_low": q_low,
        "q_high": q_high,
        "clipped_low": raw_value <= q_low,
        "clipped_high": raw_value >= q_high,
        "artifact_path": artifact.get("_resolved_path"),
    }
