#!/usr/bin/env python3
"""Price a session-scoped step budget over stored runs, before it sees a GPU.

    python3 -m evals.budget            # every file in results/
    python3 -m evals.budget --show     # per-turn detail where the rule changes

`budget_for()` reads the *current request* and nothing else. In a session that is
the wrong unit, and the stored artifacts say so in one line —
`multi-cascade-turns`:

    turn 1  "In src/store/models.py, rename Order.total() to subtotal()"  -> 26
    turn 2  "Now update everything that calls it."                        -> 12

The turn with the most work gets the base budget, because its object is a
pronoun and `task_files()` finds no filenames in it. Single-turn-shaped in
exactly the way `MAX_STEPS = 12` was read-only-shaped.

Two candidate rules, priced side by side because they differ in the direction
this project has already been burned in:

- **union** — size on every request in the session so far. Simple, and it never
  shrinks a budget, but it hands extra steps to *every* later turn including the
  ones that were finishing comfortably.
- **inherit** — size on the current request when it names files of its own;
  fall back to the session's accumulated names only when it names none. Targets
  the pronoun case and leaves self-describing turns exactly as they are.

The second is the conservative one and the difference is not cosmetic:
[[agent-tuning-dead-ends]] records that blocking an outlet does not remove the
drive, and that **a model with room left over keeps editing** — so surplus budget
is not free, and a rule that inflates every late turn is buying its fix with the
currency this project already knows is expensive.

What a replay can and cannot say: it recomputes the *number*, exactly, on the
real prompts and the real fixture. It cannot say what a model does with the extra
steps. That is the arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):  # allow `python3 evals/budget.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "evals"

from agent.loop import budget_for, task_files
from agent.sandbox import Workspace

from .cases import ALL_CASES

EVALS = Path(__file__).resolve().parent
RESULTS = EVALS / "results"
BY_ID = {case.id: case for case in ALL_CASES}

DIM, BOLD, RED, GREEN, RESET = (
    "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[0m")


def rules(prompts: list[str], ws: Workspace, base: int) -> tuple[int, int, int]:
    """(today, union, inherit) for the *last* prompt in `prompts`.

    `base` is the case's own `max_steps` where it sets one, because a case that
    raised its ceiling has already made this decision for itself and comparing
    against the global default would invent a gap that never existed.
    """
    current = prompts[-1]
    today = budget_for(current, ws, base)
    union = budget_for(" ".join(prompts), ws, base)
    try:
        named_here = task_files(current, ws)
    except Exception:
        named_here = set()
    inherit = today if named_here else union
    return today, union, inherit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="result JSONs (default: all of them)")
    ap.add_argument("--show", action="store_true",
                    help="print every turn where a rule would differ")
    opts = ap.parse_args()

    paths = [Path(f) for f in opts.files] or sorted(RESULTS.glob("*.json"))
    seen: set[tuple] = set()
    turns = 0
    gained = Counter()
    extra = Counter()
    starved_helped = Counter()
    starved_total = 0

    for path in paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in data.get("rows") or []:
            case = BY_ID.get(row.get("case"))
            turn_rows = row.get("turns") or []
            if not case or len(turn_rows) < 2:
                continue
            source = EVALS / case.fixture
            if not source.is_dir():
                continue
            # One reading per (case, turn) — the corpus holds many runs of the
            # same session and the number is a function of the prompt, not the
            # run, so counting every replica would just scale the same finding.
            key = (case.id, len(turn_rows))
            if key in seen:
                continue
            seen.add(key)

            ws = Workspace(source)
            base = getattr(case, "max_steps", None) or 12
            prompts: list[str] = []
            for i, turn in enumerate(turn_rows, 1):
                prompt = turn.get("prompt") or ""
                if not prompt:
                    continue
                prompts.append(prompt)
                turns += 1
                today, union, inherit = rules(prompts, ws, base)
                exhausted = bool(turn.get("budget_exhausted"))
                starved_total += exhausted
                for name, value in (("union", union), ("inherit", inherit)):
                    if value > today:
                        gained[name] += 1
                        extra[name] += value - today
                        if exhausted:
                            starved_helped[name] += 1
                if opts.show and (union > today or inherit > today):
                    flag = f" {RED}[was starved]{RESET}" if exhausted else ""
                    print(f"{BOLD}{case.id}{RESET} turn {i}{flag}\n"
                          f"      today={today}  union={union}  inherit={inherit}"
                          f"   {DIM}{prompt[:66]}{RESET}")

    print(f"\n{len(seen)} distinct session(s), {turns} turn(s)")
    print(f"  turns that actually ran out of steps: {BOLD}{starved_total}{RESET}")
    for name in ("union", "inherit"):
        n = gained[name]
        print(f"\n  {BOLD}{name}{RESET}: raises {n} turn(s) "
              f"({100 * n / turns:.0f}% of all turns)"
              f", +{extra[name]} steps in total"
              f"{DIM} (avg +{extra[name] / n:.1f} where it applies){RESET}"
              if n else f"\n  {BOLD}{name}{RESET}: raises nothing")
        print(f"    of the {starved_total} starved turn(s) it helps: "
              f"{BOLD}{starved_helped[name]}{RESET}")
    print(f"\n  {DIM}A replay recomputes the number, not the behaviour. Extra steps "
          f"are not free — see the surplus-effort finding in the README.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
