#!/usr/bin/env python3
"""Price the absence guard's trigger over stored runs, before it sees a GPU.

    python3 -m evals.absence                     # every file in results/
    python3 -m evals.absence --show              # print what each verdict saw
    python3 -m evals.absence --show --only-quoted

`claims_absence()` matches substrings in the *answer* — "there is no", "not
found" — and cannot tell the agent's own failed search from a sentence it is
quoting back out of a file it just read. Measured 2026-08-23, the misfire is not
rare and not one model's quirk: on `multi-absence-subject` the challenge fires in
**every** run of both models, passes included, and the derailed turn is only
sometimes scored as a failure (see the README).

The proposed discriminator needs no model call. It took two forms, and the
corpus rejected the first one:

    a matched phrase is a *quoted* absence when it appears both in what this
    turn's tools returned **and** in the workspace itself; anything else is a
    *claimed* absence, and only that deserves the challenge.

The second clause is not decoration — see `tree_text()`. Scoring against tool
output alone classified `edit-nonexistent`'s honest refusal as a quotation,
because grep's own no-match diagnostic contains the phrase an honest refusal
uses. One run in the corpus caught it.

This module prices it the way `evals/replay.py` prices `unaddressed_requests()`:
every result file kept `tool_calls_detail`, the fixtures are in the repo, and the
tools are deterministic functions of the tree — so a run recorded weeks ago can
have its tool output *recomputed* exactly, by copying its fixture, replaying its
calls in order, and reading what they return.

**The load-bearing question is not how many fires it removes. It is whether it
removes any true one.** `edit-nonexistent` and the honesty suite are the cases
the guard exists for, and there the negative follows an *empty* search — the
phrase cannot be in the output, so the trigger must survive. A single honesty
case classified "quoted" would kill the change, and that is what the by-tag
breakdown at the bottom of the report is for.

Two limits, stated because they bound the claim:

- **The stored answer is the final one.** On a run that was challenged, the text
  that actually tripped the guard was the pre-challenge answer, which no artifact
  keeps. So this prices the discriminator over a large corpus of real answers,
  not over the exact strings that fired historically.
- A replay cannot say what a model *does* when the challenge is withheld. That is
  the arm, and it still has to be run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):  # allow `python3 evals/absence.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "evals"

from agent.edits import EditSession
from agent.loop import (NEGATIVE_PHRASES, _phrase_in_workspace, claims_absence,
                        quoted_absence)
from agent.sandbox import Workspace
from agent.tools import build_registry

from .cases import ALL_CASES

EVALS = Path(__file__).resolve().parent
RESULTS = EVALS / "results"
BY_ID = {case.id: case for case in ALL_CASES}

DIM, BOLD, RED, GREEN, RESET = (
    "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[0m")

# Tools that return workspace text the model could be quoting. An edit tool's
# receipt ("edited 1 occurrence") is not content, but the edit still has to be
# applied, because a later read has to see the tree the model saw.
READ_TOOLS = ("read_file", "grep", "list_files", "find_files", "check_imports",
              "review_changes")


def local_path(value: str) -> str:
    """Rewrite a stored sandbox path onto the fresh copy.

    Stored args carry both forms — 11444 relative and 4443 absolute across the
    corpus — and the absolute ones point at `/tmp/llm-agent-evals/<model>/<case>/
    fixture/...`, which no longer exists and which `Workspace.resolve()` would
    refuse anyway. Everything after the last `fixture/` segment is the part that
    means anything.
    """
    if not value.startswith("/"):
        return value
    marker = "/fixture"
    idx = value.rfind(marker)
    if idx == -1:
        return value
    rest = value[idx + len(marker):].lstrip("/")
    return rest or "."


def replay_output(calls: list[dict], reg, capture_from: int) -> tuple[str, int]:
    """Run stored calls in order; return the output of those at/after an index.

    Earlier turns are replayed for their *effect* on the tree, not their text —
    a turn quoting a file has to see that file as its own turn found it.
    """
    seen, errors = [], 0
    for i, call in enumerate(calls):
        name = call.get("name")
        args = dict(call.get("args") or {})
        for key in ("path", "file_glob"):
            if isinstance(args.get(key), str):
                args[key] = local_path(args[key])
        try:
            out = reg.get(name).call(args)
        except Exception as exc:  # a hallucinated tool, a bad arg, a bad path
            out, errors = f"{exc}", errors + 1
        if i >= capture_from and name in READ_TOOLS:
            seen.append(out)
    return "\n".join(seen), errors


def matched_phrases(answer: str) -> list[str]:
    lowered = answer.lower()
    return [p for p in NEGATIVE_PHRASES if p in lowered]


def sentence_around(answer: str, phrase: str) -> str:
    """The phrase in enough context for a human to judge the verdict."""
    lowered = answer.lower()
    i = lowered.find(phrase)
    start = max(0, i - 90)
    end = min(len(answer), i + len(phrase) + 90)
    return ("…" if start else "") + answer[start:end].replace("\n", " ") + \
           ("…" if end < len(answer) else "")


def tree_text(root: Path) -> str:
    """Kept for the `--show` breakdown only; the verdict comes from the loop.

    Every readable file in the workspace, concatenated and lowercased.

    This is the half of the test that the first draft got wrong, and the corpus
    priced it in one run. A tool's own diagnostic is not workspace text:

        No matches for 'ENABLE_TELEMETRY' in ./. Searched 12 text file(s) … so
        the word you searched for is not the word this code uses.

    An empty search is *exactly* the moment a truthful absence claim is made, and
    that message is what such a claim paraphrases — so scoring the answer against
    raw tool output classified `edit-nonexistent`'s honest refusal as a quotation
    and would have suppressed the guard on the one case it exists for. Requiring
    the phrase to be in the repo *as well as* in what the turn saw keeps the loop's
    own prose out of the corpus, whatever any tool's message format is later.
    """
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            out.append(path.read_text(errors="ignore"))
        except OSError:
            continue
    return "\n".join(out).lower()


def classify(answer: str, output: str, ws) -> tuple[list[str], list[str]]:
    """Per-phrase detail for `--show`, using the loop's own workspace scan.

    Quoted means both: the phrase is in the workspace (so it is not the loop's
    own prose) *and* in what this turn's tools returned (so the agent actually
    saw it, rather than it coinciding with a file it never opened).

    The pass/fail verdict is **not** taken from here — it comes from
    `quoted_absence()` in `agent/loop.py`, the shipped predicate. Pricing a
    reimplementation of a mechanism is how a measurement ends up describing code
    that was never run.
    """
    lowered_out = output.lower()
    quoted, claimed = [], []
    for phrase in matched_phrases(answer):
        echo = phrase in lowered_out and _phrase_in_workspace(ws, phrase)
        (quoted if echo else claimed).append(phrase)
    return quoted, claimed


def units(row: dict) -> list[tuple[dict, int]]:
    """One scoring unit per answer: a turn if the row is a session, else the row.

    A session row repeats its turns' answers in the aggregate, so counting both
    would double-count every multi-turn case.
    """
    turns = row.get("turns") or []
    if not turns:
        return [(row, 0)]
    out, before = [], 0
    for turn in turns:
        out.append((turn, before))
        before += len(turn.get("tool_calls_detail") or [])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="result JSONs (default: all of them)")
    ap.add_argument("--show", action="store_true", help="print each fire")
    ap.add_argument("--only-quoted", action="store_true",
                    help="with --show, print only the ones that would be suppressed")
    opts = ap.parse_args()

    paths = [Path(f) for f in opts.files] or sorted(RESULTS.glob("*.json"))
    answers = fires = quoted_runs = claimed_runs = unreplayable = 0
    by_case: Counter = Counter()
    quoted_by_case: Counter = Counter()
    quoted_tags: Counter = Counter()

    for path in paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipped {path.name}: {exc}")
            continue
        for row in data.get("rows") or []:
            case = BY_ID.get(row.get("case"))
            if not case:
                continue
            source = EVALS / case.fixture
            if not source.is_dir():
                continue
            for unit, before in units(row):
                answer = unit.get("answer") or ""
                if not answer:
                    continue
                answers += 1
                if not claims_absence(answer):
                    continue
                fires += 1
                by_case[case.id] += 1

                tmp = Path(tempfile.mkdtemp())
                try:
                    root = tmp / "fixture"
                    shutil.copytree(source, root)
                    reg = build_registry(Workspace(root), EditSession())
                    calls = []
                    for turn in (row.get("turns") or [row])[:]:
                        calls += turn.get("tool_calls_detail") or []
                    if not row.get("turns"):
                        calls = row.get("tool_calls_detail") or []
                    output, errors = replay_output(calls, reg, before)
                    ws = Workspace(root)
                    # The verdict is the shipped function's, on the reconstructed
                    # tree and this turn's recomputed output.
                    suppressed = quoted_absence(answer, output, ws)
                    quoted, claimed = classify(answer, output, ws)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)

                if not output:
                    unreplayable += 1
                if suppressed:
                    quoted_runs += 1
                    quoted_by_case[case.id] += 1
                    for tag in case.tags:
                        quoted_tags[tag] += 1
                else:
                    claimed_runs += 1

                if opts.show and (suppressed or not opts.only_quoted):
                    verdict = (f"{RED}QUOTED — would no longer fire{RESET}"
                               if suppressed else f"{GREEN}claimed — still fires{RESET}")
                    print(f"{BOLD}{path.name}{RESET} {DIM}{case.id} "
                          f"({row.get('model')}){RESET} {verdict}")
                    for phrase in quoted:
                        print(f"      {DIM}echo{RESET}  {phrase!r}: "
                              f"{sentence_around(answer, phrase)}")
                    for phrase in claimed[:2]:
                        print(f"      own   {phrase!r}: "
                              f"{sentence_around(answer, phrase)}")

    print(f"\n{len(paths)} file(s), {answers} stored answer(s)")
    print(f"  the trigger fires on {BOLD}{fires}{RESET} of them "
          f"({100 * fires / answers:.1f}%)" if answers else "  no answers")
    if fires:
        print(f"  every matched phrase came out of a tool: "
              f"{BOLD}{quoted_runs}{RESET} ({100 * quoted_runs / fires:.1f}%) "
              f"{DIM}— the discriminator suppresses these{RESET}")
        print(f"  at least one phrase the tools never produced: "
              f"{BOLD}{claimed_runs}{RESET} {DIM}— still challenged{RESET}")
        if unreplayable:
            print(f"  {DIM}{unreplayable} of the fires replayed to no tool output "
                  f"at all (counted as claimed){RESET}")
        print(f"\n  {BOLD}suppressed, by tag — the honesty tags must be absent{RESET}")
        for tag, n in sorted(quoted_tags.items(), key=lambda kv: -kv[1]):
            mark = f"  {RED}<-- the guard exists for this{RESET}" if tag in (
                "honesty", "edit") else ""
            print(f"    {tag:14} {n}{mark}")
        print(f"\n  {BOLD}top cases by fires{RESET} {DIM}(fires / suppressed){RESET}")
        for cid, n in by_case.most_common(12):
            print(f"    {cid:28} {n:4} / {quoted_by_case.get(cid, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
