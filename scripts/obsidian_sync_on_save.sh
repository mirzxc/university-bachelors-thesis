#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vault_dir="$repo_root/obsidian"
lock_file="/tmp/thesis-obsidian-sync.lock"
log_file="/tmp/thesis-obsidian-sync.log"

if ! command -v inotifywait >/dev/null 2>&1; then
  echo "inotifywait is required but not installed." >&2
  exit 1
fi

cd "$repo_root"

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "Another Obsidian sync watcher is already running." >&2
  exit 1
fi

sync_once() {
  local branch
  branch="$(git branch --show-current)"

  if ! git status --porcelain -- obsidian | grep -q .; then
    return 0
  fi

  git add obsidian

  if git diff --cached --quiet -- obsidian; then
    return 0
  fi

  git commit -m "obsidian: sync vault $(date -Iseconds)"

  if ! git pull --rebase --autostash origin "$branch"; then
    echo "$(date -Iseconds) pull --rebase failed; manual intervention required" >>"$log_file"
    return 1
  fi

  if ! git push origin "$branch"; then
    echo "$(date -Iseconds) push failed; manual intervention required" >>"$log_file"
    return 1
  fi
}

echo "$(date -Iseconds) watcher started for $vault_dir" >>"$log_file"

while inotifywait \
  --quiet \
  --recursive \
  --event close_write,create,delete,move \
  --exclude '(^|/)\.obsidian/workspace.*\.json$|(^|/)\.obsidian/cache($|/)|(^|/)\.trash($|/)' \
  "$vault_dir"
do
  sleep 1
  sync_once || true
done
