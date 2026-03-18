import json
from copy import deepcopy


def parse_override_value(raw: str):
    stripped = raw.strip()
    lowered = stripped.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if stripped.startswith("[") or stripped.startswith("{"):
        return json.loads(stripped)
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return stripped


def apply_overrides(config: dict, overrides: list[str] | dict | None) -> dict:
    if not overrides:
        return deepcopy(config)

    if isinstance(overrides, dict):
        items = overrides.items()
    else:
        pairs = []
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Invalid override format: {item!r}")
            key, raw_value = item.split("=", 1)
            pairs.append((key, parse_override_value(raw_value)))
        items = pairs

    merged = deepcopy(config)
    for dotted_key, value in items:
        node = merged
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return merged

