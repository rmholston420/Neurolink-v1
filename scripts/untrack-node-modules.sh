#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# untrack-node-modules.sh
#
# Removes frontend/node_modules/ from git's index (cached tracking) without
# deleting the files on disk.  Run this once from the repo root after cloning
# if `git status` shows node_modules files as tracked.
#
# Why this is needed:
#   frontend/node_modules/ is listed in .gitignore, but the directory was
#   committed to the index before the ignore rule was added.  Git does NOT
#   retroactively untrack committed paths when they appear in .gitignore —
#   the ignore rule only prevents *new* additions.  This script fixes it.
#
# Usage:
#   chmod +x scripts/untrack-node-modules.sh
#   ./scripts/untrack-node-modules.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "→ Removing frontend/node_modules from git index (files on disk untouched)..."
git rm -r --cached frontend/node_modules/ 2>/dev/null && \
  echo "✓ Done. Commit the result:" && \
  echo "  git commit -m 'chore: untrack frontend/node_modules from index'" || \
  echo "ℹ  frontend/node_modules was not tracked — nothing to do."
