#!/usr/bin/env bash
# Safe, human-reviewed publication preparation. No automatic upload or repository creation.
set -euo pipefail
cd "$(dirname "$0")/.."
echo 'Git branch:'
git status --short --branch
# These forbidden patterns must never be tracked in any branch submitted to GitHub.
if git ls-files | grep -E -i '(^|/)(\.env$|sharew[789].*\.(dta|zip)$|outputs/private/|data/private/|results_private/)|\.(dta|parquet|pkl|pickle)$' ; then
  echo 'STOP: forbidden licensed/credential/data artifact is tracked' >&2
  exit 12
fi
python scripts/check_protocol_freeze.py
if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Local repository passed the tracked-file and frozen-specification checks.
Create a **private** GitHub repository named share-causal-uncertainty in your own GitHub account, without auto-generating a README/LICENSE.
After verifying the destination, run:
  bash scripts/github_prepare.sh https://github.com/OWNER/share-causal-uncertainty.git
This script never pushes without explicit --push.
EOF
  exit 0
fi
url="$1"
if [[ "$url" != https://github.com/*/share-causal-uncertainty.git ]]; then
  echo 'Expected an HTTPS URL ending OWNER/share-causal-uncertainty.git; no changes made' >&2
  exit 13
fi
if [[ "${2:-}" != '--push' ]]; then
  printf 'Remote URL format checked: %s\nNo push performed. After verifying owner and privacy, rerun with --push.\n' "$url"
  exit 0
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'STOP: Commit and review all intended changes first' >&2
  exit 14
fi
if git remote get-url origin &>/dev/null; then
  actual="$(git remote get-url origin)"
  if [[ "$actual" != "$url" ]]; then
    echo "STOP: origin already points elsewhere ($actual)" >&2
    exit 15
  fi
else
  git remote add origin "$url"
fi
git push -u origin HEAD:refs/heads/main
echo 'Push attempted. Inspect the actual GitHub Actions run before calling CI passed.'
