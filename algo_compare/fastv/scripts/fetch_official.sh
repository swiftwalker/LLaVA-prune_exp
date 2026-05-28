#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
METHOD_YAML="$METHOD_DIR/method.yaml"
TARGET_DIR="$METHOD_DIR/third_party/FastV"
COMMIT_FILE="$METHOD_DIR/third_party/FastV_FETCHED_COMMIT.txt"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

read_yaml_field() {
  local key="$1"
  awk -F': ' -v key="$key" '$1 ~ "^[[:space:]]*" key "$" {gsub(/"/, "", $2); print $2; exit}' "$METHOD_YAML"
}

REPO_URL="$(read_yaml_field "repo_url")"
TARGET_REF="$(read_yaml_field "target_ref")"
TARGET_COMMIT="$(read_yaml_field "target_commit")"
TARGET_REF="${TARGET_REF:-main}"
TARGET_COMMIT="${TARGET_COMMIT:-}"
if [[ "$TARGET_COMMIT" == "null" || "$TARGET_COMMIT" == "~" ]]; then
  TARGET_COMMIT=""
fi

echo "Official FastV repo: $REPO_URL"
echo "Target ref: $TARGET_REF"
echo "Target commit: ${TARGET_COMMIT:-<floating $TARGET_REF>}"
echo "Checkout dir: $TARGET_DIR"

if [[ "$DRY_RUN" -eq 1 ]]; then
  if [[ -d "$TARGET_DIR/.git" ]]; then
    current="$(git -C "$TARGET_DIR" rev-parse HEAD 2>/dev/null || true)"
    echo "Existing checkout commit: ${current:-unknown}"
  else
    echo "Would clone and checkout target."
  fi
  exit 0
fi

mkdir -p "$METHOD_DIR/third_party"

if [[ -d "$TARGET_DIR/.git" ]]; then
  current="$(git -C "$TARGET_DIR" rev-parse HEAD)"
  if [[ -n "$TARGET_COMMIT" && "$current" != "$TARGET_COMMIT" ]]; then
    echo "Existing checkout is at $current, expected $TARGET_COMMIT." >&2
    echo "Not modifying existing checkout. Move it aside or checkout manually after preserving changes." >&2
    exit 1
  fi
  echo "$current" > "$COMMIT_FILE"
  echo "Official checkout already present at $current."
  exit 0
fi

if [[ -e "$TARGET_DIR" ]]; then
  echo "$TARGET_DIR exists but is not a git checkout. Not overwriting it." >&2
  exit 1
fi

git_cmd=(git)
if [[ "${NO_GIT_PROXY:-0}" == "1" ]]; then
  git_cmd=(git -c http.proxy= -c https.proxy=)
fi

if [[ -n "$TARGET_COMMIT" ]]; then
  "${git_cmd[@]}" clone "$REPO_URL" "$TARGET_DIR"
  "${git_cmd[@]}" -C "$TARGET_DIR" checkout "$TARGET_COMMIT"
else
  "${git_cmd[@]}" clone --depth 1 --branch "$TARGET_REF" "$REPO_URL" "$TARGET_DIR"
fi

current="$(git -C "$TARGET_DIR" rev-parse HEAD)"
echo "$current" > "$COMMIT_FILE"
echo "Fetched official FastV at $current"
