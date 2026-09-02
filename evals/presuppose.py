#!/usr/bin/env python3
"""Price the presupposition guard over stored runs, before it sees a GPU.

    python3 -m evals.presuppose            # every file in results/
    python3 -m evals.presuppose --show     # print every fire

The failure it targets is `multi-refuse-followup`, the session suite's own
finding and the only failure in it with no mechanism behind it: cold, the model
refuses to change a constant that does not exist (3/3); with one successful turn
in front of it, the same model greps twice, reads the file, sees no such
constant, writes it, and reports success (3/3).

The detection is mechanical. A request that says *change* X presupposes X, so a
turn that ends with X bound where nothing bound it before has answered a question
about the world by changing the world. `presupposed_names()` and
`invented_bindings()` live in `agent/loop.py`; this module runs **those**
functions — not a copy of the rule — over every stored run whose edits can be
replayed, the way `evals/replay.py` does.

Two things this measurement had to learn the hard way, both worth keeping:

- **The first version was blind to its own true positives.** It asked
  `definition_pattern()` whether a name was defined, and that pattern matches
  `def`/`class` only — so `RETRY_BACKOFF = 2`, the exact string the failure
  writes, was not a "definition" and the price came back a clean 0 fires out of
  85. A detector that cannot see the case it was built for reports good news.
- **Retrospective precision is not proof.** The create guard scored 188/23 on
  stored artifacts and then failed on the first real request, because it keyed on
  the *shape of the request* and eval prompts are the weakest possible sample of
  that. This guard keys on request phrasing too, which is why it ships **off by
  default** behind `AGENT_PRESUPPOSITION_GUARD=1`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):  # allow `python3 evals/presuppose.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "evals"

from agent.loop import invented_bindings, presupposed_names
from agent.sandbox import Workspace

from .cases import ALL_CASES

EVALS = Path(__file__).resolve().parent
RESULTS = EVALS / "results"
BY_ID = {case.id: case for case in ALL_CASES}

DIM, BOLD, RED, GREEN, RESET = (
    "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[0m")


def apply_edits(calls: list[dict], root: Path) -> dict[str, str | None]:
    """Replay this turn's edits onto `root`; return each file as it began."""
    before: dict[str, str | None] = {}
    for call in calls:
        if call.get("name") not in ("edit_file", "write_file"):
            continue
        args = call.get("args") or {}
        rel = args.get("path")
        if not isinstance(rel, str):
            continue
        rel = rel.split("/fixture/")[-1] if rel.startswith("/") else rel
        target = root / rel
        current = target.read_text() if target.is_file() else None
        before.setdefault(rel, current)          # earliest state wins
        if call["name"] == "write_file":
            after = args.get("content")
            if not isinstance(after, str):
                continue
        else:
            old, new = args.get("old_string"), args.get("new_string")
            if (not isinstance(old, str) or not isinstance(new, str)
                    or current is None or old not in current):
                continue                          # drifted replay, skip the edit
            after = current.replace(old, new, 1)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(after)
    return before


def units(row: dict) -> list[tuple[str, list[dict], bool]]:
    """(request, calls, passed) per turn for a session, or once for a one-shot."""
    turns = row.get("turns") or []
    if not turns:
        case = BY_ID.get(row.get("case"))
        return [((getattr(case, "prompt", "") or ""),
                 row.get("tool_calls_detail") or [], bool(row.get("passed")))]
    return [(t.get("prompt") or "", t.get("tool_calls_detail") or [],
             bool(t.get("passed"))) for t in turns]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="result JSONs (default: all of them)")
    ap.add_argument("--show", action="store_true", help="print every fire")
    opts = ap.parse_args()

    paths = [Path(f) for f in opts.files] or sorted(RESULTS.glob("*.json"))
    presupposing = fires = fires_failed = 0
    by_case: Counter = Counter()
    passed_fires = []
    by_arm: Counter = Counter()

    for path in paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # A run whose guard was **on** cannot price the guard. Its fires are not
        # counterfactual: the challenge went out, the model answered it, and a
        # turn that then passed may have passed *because* of the fire. Counting
        # those as "fired on a passing turn, therefore a false positive" is how
        # this instrument reported 24 of them the day the create-request case was
        # added — 15 of which were the guard doing exactly its job.
        #
        # Three states, not two. A file written before that field existed does
        # not record its arm, and the honest handling is a third bucket rather
        # than a default: back-filling arms into stored results from the script
        # that launched them would be putting a guess into the evidence.
        switches = data.get("switches")
        arm = ("unknown" if switches is None
               else "on" if switches.get("AGENT_PRESUPPOSITION_GUARD") else "off")
        for row in data.get("rows") or []:
            case = BY_ID.get(row.get("case"))
            if not case:
                continue
            source = EVALS / case.fixture
            if not source.is_dir():
                continue
            # A session's turns share one tree, so they are replayed in order and
            # each turn is judged against the tree its predecessors left behind.
            tmp = Path(tempfile.mkdtemp())
            try:
                root = tmp / "fixture"
                shutil.copytree(source, root)
                ws = Workspace(root)
                for request, calls, passed in units(row):
                    if not presupposed_names(request):
                        apply_edits(calls, root)
                        continue
                    presupposing += 1
                    before = apply_edits(calls, root)
                    if not before:
                        continue
                    invented = invented_bindings(request, ws, before)
                    if not invented:
                        continue
                    fires += 1
                    by_arm[arm] += 1
                    by_case[case.id] += 1
                    fires_failed += not passed
                    if passed and arm != "on":
                        passed_fires.append((case.id, row.get("model"), invented))
                    if opts.show:
                        mark = (f"{GREEN}on a run that PASSED{RESET}" if passed
                                else f"{RED}on a run that failed{RESET}")
                        print(f"{BOLD}{path.name}{RESET} {DIM}{case.id} "
                              f"({row.get('model')}){RESET} {mark}\n"
                              f"      invented {', '.join(invented)}   "
                              f"{DIM}{request[:60]}{RESET}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(paths)} file(s)")
    print(f"  turns whose request presupposes a name: {BOLD}{presupposing}{RESET}")
    print(f"  turns that then created a binding for it: {BOLD}{fires}{RESET}")
    print(f"    of those, the turn FAILED: {BOLD}{fires_failed}{RESET}"
          f"{DIM} — a fire on a passing turn is a false positive *if the guard"
          f" was off*, and may be the guard's own work if it was on{RESET}")
    print(f"    {DIM}by arm: " + ", ".join(f"{n} {name}" for name, n
                                             in sorted(by_arm.items())) +
          f"{RESET}")
    if by_arm["unknown"]:
        print(f"    {DIM}\"unknown\" is every file written before results recorded"
              f" their switches. A fire under a guard that was ON is not"
              f" counterfactual — the challenge went out, and a turn that passed"
              f" may have passed because of it — so those cannot price the guard,"
              f" and the unknown bucket cannot be split.{RESET}")
    if passed_fires:
        print(f"  {RED}fired on a passing turn (guard off or unknown):{RESET}")
        for cid, model, names in passed_fires:
            print(f"    {cid} ({model}): {', '.join(names)}")
    print(f"\n  {BOLD}by case{RESET}")
    for cid, n in by_case.most_common():
        print(f"    {cid:28} {n}")
    print(f"\n  {DIM}A replay says what the guard would have fired on, not what the "
          f"model does when challenged. That is the arm.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
