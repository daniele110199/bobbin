#!/usr/bin/env python3
"""Which cases are bistable — read off the runs already on disk.

    python3 -m evals.stability                  # cases that flip under one config
    python3 -m evals.stability --diverge        # where their trajectories split
    python3 -m evals.stability --case <id>      # one case in detail

`web-search-then-fetch` failed **1/15** across two days on qwen3-coder:30b with a
byte-identical trajectory every time, which read as a near-deterministic defect
and justified building a fix for it. Then, with no code change touching it, no
ollama restart and an identical recorded configuration, it went **11/11 passing**.
Four distinct trajectories are now on record at temperature **0.0**, and the
failing and passing runs share their first six tool calls before diverging at the
seventh — a near-tie, and near-ties are where floating-point non-determinism in a
GPU runtime decides the outcome. Greedy decoding is not reproducible decoding.

That case was caught by hand, because a fix was being built for it. This module
is the question that follows: **how much of the suite behaves that way?**

## What it measures

Runs are grouped by everything that is recorded about their configuration — case,
model, switches, `num_ctx`, budget, mode, edit permission and the exact tool set.
Within a group the configuration is fixed, so anything that varies is either the
runtime's non-determinism or a code change (see the caveat below).

Two signals, and the second is the sharper one:

* **Outcome flips.** A group that is neither all-pass nor all-fail. Cheap to
  compute and easy to over-read: some cases are *designed* to sit near a
  threshold, and a model at 50% is not the same thing as a model that is bistable.
* **Trajectory splits.** The same configuration producing two or more distinct
  tool-call sequences, and *where* they diverge. This is the near-tie signature:
  a long shared prefix followed by a fork. A case whose runs differ from the first
  call is merely varied; a case whose runs agree for six calls and then split is
  sitting on a coin.

## The caveat that cannot be engineered away

Result files record switches, not code versions. A group spanning weeks spans
edits to the loop, the prompts and the tools, and this module cannot tell that
from runtime noise. `--since` narrows the window; read wide groups as
*candidates*, not findings — which is what the whole module is for.

**The single-sitting filter narrows that and does not close it, and the first
audit that used these numbers proved so.** `cascade-move` reads 0.39 within a
sitting, which was quoted as "cannot be code drift". It is code drift:
`repeat_blocks >= 5` predicts failure in 41 of its 46 nemotron runs, the repeat
guard fix landed on 2026-08-16, and it landed *mid-day* — so the before and the
after sit in the same sitting and are pooled into one group. A day is not an
atomic unit of code in a repository under active development.

So every rate here is an **upper bound on runtime non-determinism**, never an
estimate of it. What separates the two is a trajectory signature: `cascade-move`
splits cleanly on `repeat_blocks`, and once split, each side is nearly
deterministic. `web-search-then-fetch` has no such split — its runs share their
first six calls and then diverge — which is what makes it noise rather than a
recorded change.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "evals"

RESULTS = Path(__file__).resolve().parent / "results"


def config_key(header: dict, row: dict) -> tuple:
    """Everything recorded that could change behaviour, as one hashable key.

    The tool set is included because a registry change is a prompt change: the
    schemas are text the model reads on every request, which this project has
    measured as costing four cases.
    """
    return (
        row.get("case"),
        (row.get("model") or "").replace(":latest", ""),
        json.dumps(header.get("switches") or {}, sort_keys=True),
        row.get("num_ctx"),
        row.get("budget"),
        header.get("mode"),
        bool(header.get("allow_edits")),
        header.get("playbook"),
        ",".join(row.get("tools_available") or []),
    )


def load(results: Path, since: str | None = None) -> dict:
    groups: dict = collections.defaultdict(list)
    for path in sorted(results.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        stamp = data.get("timestamp") or path.stem
        if since and stamp < since:
            continue
        for row in data.get("rows", []):
            groups[config_key(data, row)].append((stamp, row))
    return groups


def divergence(seqs: list[tuple]) -> int | None:
    """Index where a set of tool-call sequences stops agreeing.

    `None` when they agree entirely (one sequence, or identical ones). 0 means
    they differ from the very first call, which is ordinary variety rather than a
    near-tie; a large value is the interesting case — a run that was reproducible
    right up to the point where it was not.
    """
    if len(set(seqs)) < 2:
        return None
    shortest = min(len(s) for s in seqs)
    for i in range(shortest):
        if len({s[i] for s in seqs}) > 1:
            return i
    return shortest


def noise_floor(groups: dict, min_pairs: int = 3,
                same_sitting: bool = True) -> dict:
    """Per (case, model): the chance two runs of one configuration disagree.

    Counted over pairs rather than over groups, because that is the number a
    reader of a table actually needs. A group of `n` runs with `k` passes
    contains `k * (n - k)` disagreeing pairs out of `n * (n - 1) / 2`, and
    summing both over every fixed-configuration group gives
    **P(two runs under identical settings disagree)** directly.

    Pairs, not a variance: a pass rate is a proportion of Bernoulli trials whose
    success probability is exactly what is in doubt here, and quoting a standard
    error computed *from* that proportion would assume the stability the number
    is meant to question. "One run in five disagrees with its twin" needs no
    distributional assumption at all.

    `same_sitting` is the honest default and the reason the pooled number is not
    quoted. A group spanning weeks of tuning carries code changes as well as
    runtime noise, and pooling those puts several cases at 0.5 — which is not a
    noise floor, it is a record of the case being actively changed. Restricting
    to groups whose runs all fall on one day cannot remove that entirely, but it
    removes the part that matters, and it is the difference between a number a
    table can carry and one that would just frighten a reader.
    """
    tally: dict = collections.defaultdict(lambda: [0, 0, 0])
    for key, runs in groups.items():
        n = len(runs)
        if n < 2:
            continue
        if same_sitting and len({stamp[:8] for stamp, _ in runs}) > 1:
            continue
        k = sum(bool(r.get("passed")) for _, r in runs)
        cell = tally[(key[0], key[1])]
        cell[0] += k * (n - k)
        cell[1] += n * (n - 1) // 2
        cell[2] += n
    return {key: {"disagree": d, "pairs": p, "runs": r,
                  "rate": (d / p) if p else 0.0}
            for key, (d, p, r) in tally.items() if p >= min_pairs}


def survey(groups: dict, min_runs: int = 3) -> list[dict]:
    out = []
    for key, runs in groups.items():
        if len(runs) < min_runs:
            continue
        passed = [bool(r.get("passed")) for _, r in runs]
        seqs = [tuple(r.get("tool_calls") or []) for _, r in runs]
        flips = 0 < sum(passed) < len(passed)
        if not flips and len(set(seqs)) < 2:
            continue
        stamps = sorted(s for s, _ in runs)
        out.append({
            "case": key[0], "model": key[1], "runs": len(runs),
            "passes": sum(passed), "flips": flips,
            "shapes": len(set(seqs)), "diverge": divergence(seqs),
            "span": (stamps[0][:13], stamps[-1][:13]),
            "one_sitting": stamps[0][:8] == stamps[-1][:8],
        })
    # A long agreed prefix is the near-tie signature, so rank by it, then by how
    # badly the outcome flips.
    out.sort(key=lambda r: (-(r["diverge"] or 0), -r["shapes"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-runs", type=int, default=3)
    ap.add_argument("--since", help="only results at or after this stamp, e.g. 20260901")
    ap.add_argument("--case", help="restrict to one case id")
    ap.add_argument("--diverge", action="store_true",
                    help="show only groups whose runs agree for a while, then split")
    ap.add_argument("--noise", action="store_true",
                    help="print the per-case noise floor instead of the survey")
    ap.add_argument("--pooled", action="store_true",
                    help="with --noise: include groups spanning sittings, which "
                         "mixes code changes in with runtime noise")
    ap.add_argument("--results", default=str(RESULTS))
    opts = ap.parse_args()

    groups = load(Path(opts.results), opts.since)

    if opts.noise:
        floor = noise_floor(groups, same_sitting=not opts.pooled)
        if opts.case:
            floor = {k: v for k, v in floor.items() if k[0] == opts.case}
        print(f"{'case':26} {'model':22} {'runs':6} {'pairs':7} {'disagree':9} rate")
        for (case, model), v in sorted(floor.items(), key=lambda kv: -kv[1]["rate"]):
            mark = "  <-- a single rep says little" if v["rate"] >= 0.2 else ""
            print(f"{case:26} {model:22} {v['runs']:<6} {v['pairs']:<7} "
                  f"{v['disagree']:<9} {v['rate']:.2f}{mark}")
        return 0

    rows = survey(groups, opts.min_runs)
    if opts.case:
        rows = [r for r in rows if r["case"] == opts.case]
    if opts.diverge:
        rows = [r for r in rows if (r["diverge"] or 0) > 0]

    total = sum(len(v) for v in groups.values())
    print(f"{total} runs in {len(groups)} fixed-configuration groups; "
          f"{len(rows)} of them vary (>= {opts.min_runs} runs)\n")

    print(f"{'case':26} {'model':22} {'runs':5} {'pass':6} {'shapes':7} "
          f"{'split at':9} {'span'}")
    for r in rows[:30]:
        split = "same" if r["diverge"] is None else str(r["diverge"])
        mark = "  <-- flips" if r["flips"] else ""
        sitting = "" if r["one_sitting"] else " (multi-sitting)"
        print(f"{r['case']:26} {r['model']:22} {r['runs']:<5} "
              f"{r['passes']}/{r['runs']:<4} {r['shapes']:<7} {split:9} "
              f"{r['span'][0]}..{r['span'][1]}{mark}{sitting}")

    flipping = [r for r in rows if r["flips"]]
    if flipping:
        print(f"\n{len(flipping)} group(s) flip outcome under a fixed configuration:")
        for r in flipping:
            where = ("from the first call" if r["diverge"] == 0
                     else f"after {r['diverge']} identical calls"
                     if r["diverge"] else "with identical trajectories")
            print(f"  {r['case']} / {r['model']}: {r['passes']}/{r['runs']}, "
                  f"diverging {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
