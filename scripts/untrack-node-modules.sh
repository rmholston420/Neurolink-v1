#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# untrack-ignored-paths.sh
#
# Removes paths from git's index that are listed in .gitignore but were
# committed before the ignore rules were added.  Safe to run multiple times.
# Files on disk are NOT deleted.
#
# Usage (from repo root):
#   chmod +x scripts/untrack-node-modules.sh
#   ./scripts/untrack-node-modules.sh
#   git commit -m "chore: untrack ignored paths from index"
#   git push
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

untrack() {
  local path="$1"
  if git ls-files "$path" | grep -q .; then
    echo "  removing: $path"
    git rm -r --cached "$path" 2>/dev/null || true
  else
    echo "  already untracked: $path"
  fi
}

echo "-> Untracking ignored paths from git index..."

# node_modules
untrack frontend/node_modules/

# Python bytecode cache -- find all committed __pycache__ dirs
git ls-files | grep '/__pycache__/' | sed 's|/__pycache__/.*|/__pycache__/|' | sort -u | while read -r dir; do
  echo "  removing: $dir"
  git rm -r --cached "$dir" 2>/dev/null || true
done

# Compiled .pyc files at any depth
git ls-files | grep '\.pyc$' | while read -r f; do
  echo "  removing: $f"
  git rm --cached "$f" 2>/dev/null || true
done

# pytest cache bytecode (.pyc files named *-pytest-*.pyc)
git ls-files | grep 'pytest' | grep '\.pyc$' | while read -r f; do
  echo "  removing: $f"
  git rm --cached "$f" 2>/dev/null || true
done

# Runtime data and coverage artifacts
untrack data/
untrack backend/data/
untrack frontend/coverage/

# vitest internal cache
git ls-files | grep 'node_modules/.vite/' | while read -r f; do
  git rm --cached "$f" 2>/dev/null || true
done

# uv.lock
git ls-files | grep 'uv\.lock' | while read -r f; do
  echo "  removing: $f"
  git rm --cached "$f" 2>/dev/null || true
done

echo ""
echo "Done. Review with: git status"
echo ""
echo "Then commit and push:"
echo "  git commit -m 'chore: untrack ignored paths from index'"
echo "  git push"
