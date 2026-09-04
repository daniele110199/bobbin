#!/bin/zsh
# Two more reps of the fixture matrix, 114 runs each, about 2.8 hours per rep.
# Detached on purpose: the real-repo reps were killed at 33 of 36 cells when a
# harness stopped their background task.
cd /home/daniele/llm-agent-project
C=evals/compare
for rep in 2 3; do
    echo "=== fixture rep $rep starting $(date +%H:%M:%S)"
    VERSUS_REP=$rep python3 $C/versus.py $C/results/versus_rep$rep.json
    echo "=== fixture rep $rep done $(date +%H:%M:%S)"
done
echo "ALL FIXTURE REPS COMPLETE"
