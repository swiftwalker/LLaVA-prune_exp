#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$METHOD_DIR/../.." && pwd)"
METHOD_YAML="$METHOD_DIR/method.yaml"
TARGET_DIR="$METHOD_DIR/third_party/SparseVLMs"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

read_yaml_field() {
  local key="$1"
  awk -F': ' -v key="$key" '$1 ~ "^[[:space:]]*" key "$" {gsub(/"/, "", $2); print $2; exit}' "$METHOD_YAML"
}

REPO_URL="$(read_yaml_field "repo_url")"
TARGET_COMMIT="$(read_yaml_field "target_commit")"

if [[ -z "$REPO_URL" || -z "$TARGET_COMMIT" ]]; then
  echo "Failed to read repo_url or target_commit from $METHOD_YAML" >&2
  exit 1
fi

echo "Official SparseVLM repo: $REPO_URL"
echo "Target commit: $TARGET_COMMIT"
echo "Checkout dir: $TARGET_DIR"

if [[ "$DRY_RUN" -eq 1 ]]; then
  if [[ -d "$TARGET_DIR/.git" ]]; then
    current="$(git -C "$TARGET_DIR" rev-parse HEAD 2>/dev/null || true)"
    echo "Existing checkout commit: ${current:-unknown}"
  else
    echo "Would clone and checkout target commit."
  fi
  exit 0
fi

mkdir -p "$METHOD_DIR/third_party"

if [[ -d "$TARGET_DIR/.git" ]]; then
  current="$(git -C "$TARGET_DIR" rev-parse HEAD)"
  if [[ "$current" == "$TARGET_COMMIT" ]]; then
    echo "Official checkout already at target commit."
    exit 0
  fi
  echo "Existing checkout is at $current, expected $TARGET_COMMIT." >&2
  echo "Not modifying existing checkout. Move it aside or checkout manually after preserving changes." >&2
  exit 1
fi

if [[ -e "$TARGET_DIR" ]]; then
  echo "$TARGET_DIR exists but is not a git checkout. Not overwriting it." >&2
  exit 1
fi

git clone "$REPO_URL" "$TARGET_DIR"
git -C "$TARGET_DIR" checkout "$TARGET_COMMIT"
echo "Fetched official SparseVLM at $(git -C "$TARGET_DIR" rev-parse HEAD)"
