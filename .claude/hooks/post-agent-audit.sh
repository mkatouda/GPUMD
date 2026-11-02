#!/usr/bin/env bash
# PostToolUse hook (matcher: "Agent"): after every subagent call, report git
# status/diffstat and warn about changes that need human approval per AGENTS.md
# (specs/, fixtures/golden/) or that look like generated outputs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 0

status=$(git status --porcelain=v1 2>&1 || true)
diffstat=$(git diff --stat 2>&1 || true)

if [ -z "$status" ]; then
  context="[subagent audit] working tree is clean (no changes since this subagent call)."
else
  warn=""
  if echo "$status" | grep -qE '(^|[ /])specs/'; then
    warn="${warn}
WARNING: specs/ changed - spec changes require human approval."
  fi
  if echo "$status" | grep -qE 'fixtures/golden/'; then
    warn="${warn}
WARNING: fixtures/golden/ changed - golden regeneration requires human approval and a stated reason."
  fi
  if echo "$status" | grep -qE '\.(out|xyz|h5|npy)$|nep\.txt|regression_runs|run_regression'; then
    warn="${warn}
NOTE: possible generated outputs in the tree - these must not be committed."
  fi
  context="[subagent audit] working tree changed after this subagent call.
git status --porcelain:
${status}

git diff --stat:
${diffstat}${warn}"
fi

python3 - "$context" <<'PY'
import json, sys
print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": sys.argv[1]}}))
PY
exit 0
