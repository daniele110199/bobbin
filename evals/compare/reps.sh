#!/bin/zsh
# Run one repetition of the full real-repo matrix (36 runs, roughly 4 hours).
#
#   evals/compare/reps.sh 2
#
# Launch it detached, or a harness that stops background tasks will take the
# run with it. That is not hypothetical: the first attempt at these reps died
# at 33 of 36 cells that way.
#
#   setsid nohup evals/compare/reps.sh 2 > rep2.log 2>&1 < /dev/null &
#
# Rows are written after every run, so an interrupted rep keeps what it had,
# and REALREPO_ONLY / REALREPO_MODELS can fill in whatever is missing.
set -e
REP=${1:?usage: reps.sh <rep-number>}
cd "$(dirname "$0")/../.."
C=evals/compare
echo "=== rep $REP starting $(date +%H:%M:%S)"
REALREPO_REP=$REP python3 $C/realrepo.py $C/results/realrepo_rep$REP.json
echo "=== rep $REP done $(date +%H:%M:%S)"
