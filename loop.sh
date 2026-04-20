#!/usr/bin/env bash
# Autonomous loop driver — runs ONE agent iteration per claude invocation,
# then restarts forever. Ctrl-C to stop.
#
# Prereqs:
#   1. pip install -r requirements.txt
#   2. python prepare.py --fetch         # first-time data download
#   3. python prepare.py                 # baseline score
#      echo <that score> > best_score.txt
#      git add strategy.py best_score.txt && git commit -m "baseline"
#
# Then: ./loop.sh

set -u
cd "$(dirname "$0")"

MAX_ITERS="${MAX_ITERS:-1000}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-5}"

# Keep main clean: every run lives on its own experiments branch.
# Override with LOOP_BRANCH=experiments/foo to resume an existing run.
BRANCH="${LOOP_BRANCH:-experiments/autoloop-$(date -u +%Y%m%d-%H%M%S)}"
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  echo "[loop.sh] resuming existing branch: $BRANCH"
  git checkout "$BRANCH" || exit 1
else
  echo "[loop.sh] creating new branch: $BRANCH (off $(git rev-parse --abbrev-ref HEAD))"
  git checkout -b "$BRANCH" || exit 1
fi

i=0
while (( i < MAX_ITERS )); do
  i=$((i + 1))
  echo ""
  echo "=============================================================="
  echo "  iteration $i  |  $(date -u +%FT%TZ)  |  branch: $BRANCH  |  best: $(cat best_score.txt 2>/dev/null || echo '?')"
  echo "=============================================================="

  claude -p "$(cat program.md)" \
    --dangerously-skip-permissions \
    || echo "[loop.sh] claude exited non-zero on iter $i (continuing)"

  sleep "$SLEEP_BETWEEN"
done
