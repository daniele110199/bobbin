#!/usr/bin/env python3
"""Is 40 steps the binding constraint on a 12k-line repo, or is it variance?

The first probe re-ran qwen's two failed renames at 80 steps and came back
8/80 and 13/80 — nowhere near the 40-step ceiling those same tasks had hit on
the original run (40/40 and 44/40). One rep cannot tell "80 steps fixed it"
from "these runs vary a lot", and the two samples we have disagree about the
failure mode as well as the outcome: at 40 the agent edited, broke 1991 tests
and claimed success; at 80 it spent 8 steps describing the function and never
edited.

So: both budgets, three reps, everything else fixed. `steps_used` is the
column that matters — a budget can only be the constraint in runs that reach
it.
"""
import json, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repo root, wherever it was cloned
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent.sandbox import Workspace
from evals.run import build_agent
from realrepo import TASKS, PRISTINE, fresh, hits, tests_pass, dirty

WANTED = ["real-rename-across-files", "real-rename-internal"]
REPS = 3
out = Path(sys.argv[1])
rows = []

for rep in range(1, REPS + 1):
    for budget in (40, 80):
        for task in [t for t in TASKS if t["id"] in WANTED]:
            root = fresh(git=False)
            opts = SimpleNamespace(host="http://127.0.0.1:11434", num_ctx=None,
                                   allow_edits=True, mode="direct",
                                   max_steps=budget, playbook="default",
                                   subtasks=2, gather_steps=6)
            agent = build_agent("qwen3-coder:30b", Workspace(root), opts)
            started = time.monotonic()
            try:
                answer = agent.ask(task["prompt"])
            except Exception as exc:
                answer = f"[crashed: {exc}]"
            used = agent.stats.steps
            made = bool(task["check"](root))
            touched = dirty(root)
            green = tests_pass(root)
            rows.append({"task": task["id"], "rep": rep, "steps_budget": budget,
                         "steps_used": used, "hit_ceiling": used >= budget,
                         "change_made": made, "touched_src": touched,
                         "tests_pass": green, "passed": made and green,
                         "seconds": round(time.monotonic() - started, 1),
                         "answer": answer[:400]})
            print(f"{'PASS' if rows[-1]['passed'] else 'fail'}  rep{rep} "
                  f"budget={budget:3} used={used:3}{'*' if used >= budget else ' '} "
                  f"{task['id']:26} change={made!s:5} touched={touched!s:5} "
                  f"tests={green!s:5} {rows[-1]['seconds']:6.0f}s", flush=True)
            out.write_text(json.dumps(rows, indent=2))
            subprocess.run(["rm", "-rf", str(root.parent)])
print("MATCHED PROBE COMPLETE")
