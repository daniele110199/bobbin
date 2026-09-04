#!/usr/bin/env python3
"""One table over every comparison cell, recovered and freshly run alike."""
import json, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
ARMS = ("ours", "aider-told", "aider-find")
MODELS = ("qwen3-coder:30b", "nemotron-3.5-lightning")


def load(*names, key):
    rows = []
    for name in names:
        path = R / name
        if path.is_file():
            for r in json.load(open(path)):
                r["_key"] = r.get(key) or r.get("case") or r.get("task")
                rows.append(r)
    return rows


def table(title, rows, order=None):
    """One line per cell, showing passes over reps rather than a single verdict.

    A cell that reads 2/3 is the honest rendering of what these runs are: one
    sample per rep, and this project has already been misled once by treating a
    single sample as an answer.
    """
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["_key"], r["arm"])].append(r)
    keys = order or sorted({r["_key"] for r in rows})
    tally = defaultdict(lambda: [0, 0])
    reps = sorted({r.get("rep", 1) for r in rows})
    print(f"\n########## {title}")
    if len(reps) > 1:
        print(f"  {len(reps)} reps per cell; a cell shows passes/reps")
    for model in MODELS:
        print(f"\n=== {model}")
        print(f"  {'':26} {'ours':>7} {'told':>7} {'find':>7}   median seconds")
        for k in keys:
            marks, secs = [], []
            for arm in ARMS:
                got = cells.get((model, k, arm), [])
                passes = sum(bool(r["passed"]) for r in got)
                if not got:
                    marks.append("   --  "); secs.append("-")
                elif len(got) == 1:
                    marks.append("  ok   " if passes else " fail  ")
                else:
                    marks.append(f" {passes}/{len(got)}   ")
                if got:
                    secs.append(f"{sorted(r['seconds'] for r in got)[len(got)//2]:.0f}")
                    tally[(model, arm)][0] += passes; tally[(model, arm)][1] += len(got)
                    tally[("BOTH", arm)][0] += passes; tally[("BOTH", arm)][1] += len(got)
            print(f"  {k:26} {''.join(marks)}   {'/'.join(secs)}")
        print(f"  {'TOTAL':26} " + "".join(
            f"{tally[(model, a)][0]:>4}/{tally[(model, a)][1]:<3}" for a in ARMS))
    print("\n  both models: " + "   ".join(
        f"{a}: {tally[('BOTH', a)][0]}/{tally[('BOTH', a)][1]}"
        f" ({100*tally[('BOTH', a)][0]/max(tally[('BOTH', a)][1],1):.0f}%)" for a in ARMS))
    return {a: tuple(tally[("BOTH", a)]) for a in ARMS}


REAL_ORDER = ["real-rename-across-files", "real-rename-internal", "real-move-function",
              "real-signature", "real-single-file", "real-nonexistent"]

fix = table("fixtures — 19 cases",
            load("fixtures_recovered.json", "versus_rep2.json",
                 "versus_rep3.json", key="case"))
real = table("real repo — pallets/click @36baa15, judged by its own 1991 tests",
             load("realrepo_recovered.json", "realrepo_last.json",
                  "realrepo_rep2.json", "realrepo_rep2_tail.json",
                  "realrepo_rep3.json", key="task"),
             REAL_ORDER)

matched = R / "budget_matched.json"
if matched.is_file():
    rows = json.load(open(matched))
    print("\n########## budget: was 40 steps the constraint? (qwen, 3 reps, matched)")
    tally = defaultdict(lambda: [0, 0])
    for r in rows:
        tally[(r["task"], r["steps_budget"])][0] += r["passed"]
        tally[(r["task"], r["steps_budget"])][1] += 1
    for task in ("real-rename-across-files", "real-rename-internal"):
        print(f"  {task:26} 40 steps: {tally[(task,40)][0]}/{tally[(task,40)][1]}"
              f"    80 steps: {tally[(task,80)][0]}/{tally[(task,80)][1]}")
    for b in (40, 80):
        s = [r for r in rows if r["steps_budget"] == b]
        print(f"  budget {b}: reached the ceiling in {sum(r['hit_ceiling'] for r in s)}/{len(s)}"
              f" runs; steps used {sorted(r['steps_used'] for r in s)}")
    print(f"  click's suite went red in {sum(1 for r in rows if not r['tests_pass'])}/{len(rows)}"
          f" runs; src touched in {sum(1 for r in rows if r['touched_src'])}/{len(rows)}")
    print("  -> doubling the budget changed neither pass rate; at 80 no run ever")
    print("     reached its ceiling. The budget was not the binding constraint.")
else:
    print("\n  (matched budget probe not yet run)")
