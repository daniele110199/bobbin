#!/usr/bin/env python3
"""Replay the edits a stored run made, and ask a checker what it would have said.

    python3 -m evals.replay                       # every file in results/
    python3 -m evals.replay evals/results/note16-*.json --show

`evals/rescore.py` prices a *metric* over stored runs, because every result file
kept the answer and the workspace verdict. This prices a *mechanism*, because
every result file also kept `tool_calls_detail` with the full `old_string` /
`new_string` of every edit — and the fixtures are in the repo. So a run recorded
weeks ago can be rebuilt on disk: copy its fixture, apply its edits in order, and
hand the reconstructed tree plus the reconstructed journal to a function that did
not exist when the run happened.

That is how `unaddressed_requests()` was measured before it ever saw a GPU: 521
runs with edits, 24 fires, all true, none outside the one case built for it. A
replay cannot tell you what a model does when it is *told* something — only what
it would have been told. It is the cheap half of the measurement, not a
substitute for the arm.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):  # allow `python3 evals/replay.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "evals"

from agent.edits import EditRecord, EditSession
from agent.review import unaddressed_requests
from agent.sandbox import Workspace

from .cases import ALL_CASES
from .score import honesty_problem

EVALS = Path(__file__).resolve().parent
RESULTS = EVALS / "results"
BY_ID = {case.id: case for case in ALL_CASES}

DIM, BOLD, RED, RESET = "\033[2m", "\033[1m", "\033[31m", "\033[0m"


def replay_row(row: dict, root: Path) -> tuple[Workspace, EditSession, bool] | None:
    """Rebuild one run's end state under `root`. Returns None if it made no edits.

    The third element is whether every stored edit applied cleanly. An edit whose
    `old_string` no longer matches means the reconstruction drifted from the real
    run — usually because the model edited a file the harness had already changed
    — and a fire on a drifted replay is not evidence of anything.
    """
    case = BY_ID.get(row.get("case"))
    if not case:
        return None
    calls = [c for c in row.get("tool_calls_detail") or []
             if c["name"] in ("edit_file", "write_file")]
    if not calls:
        return None
    source = EVALS / case.fixture
    if not source.is_dir():
        return None
    shutil.copytree(source, root)

    session, clean = EditSession(), True
    for call in calls:
        args = call.get("args") or {}
        rel = args.get("path")
        if not isinstance(rel, str):
            clean = False
            continue
        target = root / rel
        before = target.read_text() if target.is_file() else None
        if call["name"] == "write_file":
            after = args.get("content")
            if not isinstance(after, str):
                clean = False
                continue
        else:
            old, new = args.get("old_string"), args.get("new_string")
            if (not isinstance(old, str) or not isinstance(new, str)
                    or before is None or old not in before):
                clean = False
                continue
            after = before.replace(old, new, 1)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(after)
        session.history.append(
            EditRecord(path=rel, before=before, after=after, tool=call["name"]))
    return Workspace(root), session, clean


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="result JSONs (default: all of them)")
    ap.add_argument("--show", action="store_true", help="print what would be said")
    opts = ap.parse_args()

    paths = [Path(f) for f in opts.files] or sorted(RESULTS.glob("*.json"))
    runs = clean_runs = fires = lies = caught = 0
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipped {path.name}: {exc}")
            continue
        for row in data.get("rows") or []:
            tmp = Path(tempfile.mkdtemp())
            try:
                built = replay_row(row, tmp / "fixture")
                if not built:
                    continue
                ws, session, clean = built
                runs += 1
                clean_runs += clean
                case = BY_ID[row["case"]]
                said = unaddressed_requests(case.prompt, ws, session)
                lied = bool(honesty_problem(row.get("answer") or "",
                                            row.get("file_problems") or []))
                lies += lied
                if not said:
                    continue
                fires += 1
                caught += lied
                if opts.show:
                    mark = f"{RED}over a claim of done{RESET}" if lied else "on an honest run"
                    print(f"{BOLD}{path.name}{RESET} {DIM}{row['case']} "
                          f"({row.get('model')}){RESET} {mark}")
                    for line in said:
                        print(f"      {line}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(paths)} file(s), {runs} run(s) with edits "
          f"({clean_runs} replayed with every edit applying cleanly)")
    print(f"  the check fires on {BOLD}{fires}{RESET} of them")
    print(f"  {lies} answer(s) claimed done over undone work; "
          f"{BOLD}{caught}{RESET} of those are contradicted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
