#!/usr/bin/env python3
"""Was the real-repo cascade failure the budget, or the task?

qwen hit `steps=40` on both renames and then reported success it had not
achieved; on one of them it said outright that an import was still wrong. That is
a budget ceiling, not a misunderstanding — so the same two tasks are re-run at 80
steps, changing nothing else. If they pass, "cascades do not survive at 12k lines"
is the wrong conclusion and "our budgets were sized on 24-file fixtures" is the
right one.
"""
import json, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repo root, wherever it was cloned
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent.sandbox import Workspace
from evals.run import build_agent
from realrepo import TASKS, fresh, hits, tests_pass

WANTED = {"real-rename-across-files", "real-rename-internal"}
rows = []
for steps in (80,):
    for task in [t for t in TASKS if t["id"] in WANTED]:
        for model in ("qwen3-coder:30b",):
            root = fresh(git=False)
            opts = SimpleNamespace(host="http://127.0.0.1:11434", num_ctx=None,
                                   allow_edits=True, mode="direct", max_steps=steps,
                                   playbook="default", subtasks=2, gather_steps=6)
            agent = build_agent(model, Workspace(root), opts)
            started = time.monotonic()
            answer = agent.ask(task["prompt"])
            made = bool(task["check"](root))
            green = tests_pass(root)
            rows.append({"task": task["id"], "model": model, "steps_budget": steps,
                         "steps_used": agent.stats.steps, "change_made": made,
                         "tests_pass": green, "passed": made and green,
                         "seconds": round(time.monotonic() - started, 1),
                         "answer": answer[:300]})
            print(f"{'PASS' if rows[-1]['passed'] else 'fail'}  budget={steps} "
                  f"used={agent.stats.steps:3} {task['id']:26} "
                  f"change={made!s:5} tests={green!s:5} {rows[-1]['seconds']:.0f}s",
                  flush=True)
            Path(sys.argv[1]).write_text(json.dumps(rows, indent=2))
            subprocess.run(["rm", "-rf", str(root.parent)])
