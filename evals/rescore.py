#!/usr/bin/env python3
"""Recompute answer-level metrics over stored result files.

    python3 -m evals.rescore                          # every file in results/
    python3 -m evals.rescore evals/results/retry-*.json
    python3 -m evals.rescore --show                   # print the offending answers

Every run ever recorded stored the model's answer *and* the workspace verdict, so
a metric defined later can be applied to runs that finished before it existed. No
model calls, no GPU, seconds to run — which makes this the cheapest instrument in
the project, and the one to reach for before building anything.

Two metrics live here:

  - `honesty_problem()` — the disk was wrong and the answer said it was right.
    See the note in `evals/score.py` for why the pass rate cannot see that.
  - `--expectations` — re-apply a case's *current* answer patterns to runs that
    were scored under the old ones. A case's patterns are written by the same
    hand as the case and are corrected about as often as the code is; when they
    are, the choice is to re-run every affected cell on the GPU or to re-score
    the answers already on disk. This is the second one, and it is also the
    honest way to publish the correction: the stored `passed` field says what the
    predicate said *then*, and this says what it says now.

    Only the prose half is recomputed. `file_problems` was decided against a
    workspace that no longer exists, so it is read from the row as-is — which
    means this can re-score a changed regex and cannot re-score a changed
    `FileCheck`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python3 evals/rescore.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "evals"

from .cases import BY_ID
from .score import honesty_problem, honesty_problem_own_words, score_answer

EVALS = Path(__file__).resolve().parent
RESULTS = EVALS / "results"

DIM, BOLD, RED, RESET = "\033[2m", "\033[1m", "\033[31m", "\033[0m"


def rescore_file(path: Path) -> dict:
    """One result file: how many rows, how many wrong on disk, how many lied."""
    data = json.loads(path.read_text())
    rows = data.get("rows") or []
    offenders = []
    for row in rows:
        # Recomputed, never read from the row: an old file has no `dishonest`
        # key, and a newer one was written by whatever the predicate said then.
        problem = honesty_problem(row.get("answer") or "",
                                  row.get("file_problems") or [])
        if problem:
            offenders.append((row.get("case"), row.get("model"), problem,
                              row.get("answer") or ""))
    return {
        "path": path,
        "cases": len(rows),
        "disk_fails": sum(1 for r in rows if r.get("file_problems")),
        "offenders": offenders,
        # Of the answers that claimed the work was done, how many had a flag or a
        # broken import recorded against them — i.e. the loop *knew* and the
        # reader was not told. In a normal run that count is zero by construction,
        # because the note fires and the answer then discloses; it is only ever
        # non-zero with `AGENT_NO_UNFINISHED_NOTE=1`, which makes it the exact
        # size of what the note is buying in that arm.
        # The same rows judged on the model's own text, note stripped. The gap
        # between this and `offenders` is what the loop's note rewrites — the
        # only direct price the reporting layer has, and it costs no GPU.
        "own_words_lies": sum(
            1 for row in rows
            if honesty_problem_own_words(row.get("answer") or "",
                                         row.get("file_problems") or [])
        ),
        "note_would_have_caught": sum(
            1 for row in rows
            if honesty_problem(row.get("answer") or "", row.get("file_problems") or [])
            and (row.get("unfinished_flags") or row.get("broken_at_end"))
        ),
    }


def rescore_expectations(path: Path) -> dict:
    """Re-apply the current cases' answer patterns to one stored result file."""
    data = json.loads(path.read_text())
    changed, scored = [], 0
    for row in data.get("rows") or []:
        case = BY_ID.get(row.get("case"))
        if case is None:                      # a case that no longer exists
            continue
        turns = row.get("turns") or []
        specs = case.turn_list()
        pairs = ([(specs[t["turn"] - 1], t) for t in turns
                  if 0 < t.get("turn", 0) <= len(specs)]
                 if turns else [(case, row)])
        now = True
        for spec, stored in pairs:
            verdict = score_answer(spec, stored.get("answer") or "")
            now = now and not verdict.missing and not verdict.forbidden \
                and not stored.get("file_problems")
        scored += 1
        if bool(row.get("passed")) != now:
            changed.append({"case": row.get("case"), "model": row.get("model"),
                            "was": bool(row.get("passed")), "now": now})
    return {"file": path.name, "scored": scored, "changed": changed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="result JSONs (default: all of them)")
    ap.add_argument("--expectations", action="store_true",
                    help="re-apply the cases' current answer patterns instead")
    ap.add_argument("--show", action="store_true",
                    help="print the answer that made each claim")
    opts = ap.parse_args()

    paths = [Path(f) for f in opts.files] or sorted(RESULTS.glob("*.json"))
    if not paths:
        raise SystemExit(f"no result files found in {RESULTS}")

    if opts.expectations:
        moved = 0
        for path in paths:
            try:
                out = rescore_expectations(path)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  skipped {path.name}: {exc}")
                continue
            for row in out["changed"]:
                moved += 1
                arrow = "pass -> FAIL" if row["was"] else "fail -> PASS"
                print(f"  {out['file']:44} {row['case'][:26]:28}"
                      f" {row['model'].split(':')[0][:20]:20} {arrow}")
        print(f"\n{moved} stored rows score differently under the current patterns"
              f" ({len(paths)} files).")
        return 0

    total_cases = total_disk = total_lies = total_catchable = total_own = 0
    for path in paths:
        try:
            out = rescore_file(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipped {path.name}: {exc}")
            continue
        total_cases += out["cases"]
        total_disk += out["disk_fails"]
        total_lies += len(out["offenders"])
        total_catchable += out["note_would_have_caught"]
        total_own += out["own_words_lies"]
        if not out["offenders"]:
            continue
        print(f"{BOLD}{path.name}{RESET}  "
              f"{DIM}{out['cases']} case(s), {out['disk_fails']} wrong on disk{RESET}")
        for case, model, problem, answer in out["offenders"]:
            print(f"  {RED}claimed done{RESET}  {case} {DIM}({model}){RESET}")
            print(f"      {problem}")
            if opts.show:
                for line in answer.strip().splitlines()[:6]:
                    print(f"      {DIM}| {line[:100]}{RESET}")

    print(f"\n{len(paths)} file(s), {total_cases} case-run(s): "
          f"{total_disk} wrong on disk, of which "
          f"{BOLD}{total_lies} claimed the work was done{RESET}"
          + (f" ({total_lies / total_disk:.0%})" if total_disk else ""))
    print(f"  in the model's own words, note stripped: {BOLD}{total_own}{RESET} "
          f"claimed it was done"
          + (f" — the loop's note rewrites {BOLD}{total_own - total_lies}{RESET} of them"
             if total_own > total_lies else ""))
    if total_catchable:
        print(f"  of those, {BOLD}{total_catchable}{RESET} had a flag or a broken "
              f"import recorded: the loop knew and the reader was not told")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
