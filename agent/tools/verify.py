"""The check_imports tool: the narrow affordance, registered with the write tools.

It takes no arguments on purpose. Every argument is a chance for a small model to
supply something that cannot work — the removed `run_command` spent 54 calls
across three runs on `pytest` (not installed), `unittest` (finds no pytest-style
tests), and `rm`/`cat`/`ls` (not on the allowlist), each one a step off the
budget. A tool with nothing to fill in cannot be filled in wrongly.
"""

from __future__ import annotations

import os

from ..edits import EditSession
from ..imports import check_workspace
from ..review import full_report
from ..sandbox import Workspace
from .base import Tool


def build(ws: Workspace, session: EditSession | None = None) -> list[Tool]:
    # The undefined-name half of the check is named in the description, because a
    # capability the model is not told about is one it cannot reach for — and it
    # disappears from the description under the same kill switch that disables
    # it, so an A/B of the mechanism is not also an A/B of the prompt.
    also_undefined = "" if os.environ.get("AGENT_NO_UNDEFINED_CHECK") else (
        " and for uses of a name that is not defined or imported anywhere"
    )
    tools = [
        Tool(
            name="check_imports",
            description=(
                "Check every Python file in the workspace for syntax errors, "
                f"for imports of names that no longer exist{also_undefined}. Use "
                "it after renaming, moving or deleting something to find what "
                "you missed. Nothing is executed."
            ),
            params=[],
            fn=lambda: _check_imports(ws),
        ),
    ]
    # `AGENT_NO_REVIEW_TOOL=1` leaves it out of the registry entirely, so the
    # "before" arm of its A/B sees exactly the prompt the earlier numbers were
    # measured against — a schema is prompt text charged on every request.
    # The fourth check keys on the request rather than on damage, and the model
    # can only reach for what it is told about. `AGENT_NO_REQUEST_IN_REVIEW=1`
    # removes both the clause and the section — the arm that reproduces the tool
    # every number before 2026-08-19 was measured against.
    #
    # They also have a switch each, and that is not tidiness: with one switch,
    # the sees/blind arm moved `edit-honesty-budget` from 2/14 to 12/14 *without
    # the tool ever being called*, so a sentence nobody read was doing the work
    # and the paragraph it advertised was never tested. `AGENT_NO_REQUEST_CLAUSE`
    # and `AGENT_NO_REQUEST_SECTION` split them so each can be measured alone.
    #
    # Three states, not two, because the sees/blind arm moved the result 12/14 to
    # 2/14 *without the tool ever being called* — nine words of schema text, and
    # nothing read. `AGENT_NEUTRAL_REVIEW_CLAUSE=1` is the control that separates
    # the two explanations: a clause of the same length (68 vs 67 characters)
    # naming something the tool genuinely already reports, but saying nothing
    # about untouched work. If completion lifts here too, the effect is
    # perturbation and the content claim dies.
    #
    # **Default flipped 2026-08-20, and the reason is the other suite.** Telling a
    # model to look for what it has not touched pays off on a task built to be
    # abandoned half-done, and costs on a task that can actually be finished: on
    # cascades qwen went 8/10, 8/10 with the clause against 9/10, 9/10 without it,
    # across two sittings, and its failures under it are *over-reach* — rewriting
    # README.md, inventing run_tests.py — not incompleteness. Meanwhile the case
    # that credited the clause can no longer referee it: 16 runs, both arms, both
    # models, and qwen completes 4/4 either way while nemotron completes ~0/4
    # either way. Benefit unmeasurable, cost reproduced twice, so the nine words
    # come out. `AGENT_REQUEST_CLAUSE=1` puts them back — the arm that reproduces
    # every number measured between 2026-08-19 and today.
    if os.environ.get("AGENT_NEUTRAL_REVIEW_CLAUSE"):
        also_untouched = (
            ", including a count of the lines you added and removed in each file")
    elif (os.environ.get("AGENT_REQUEST_CLAUSE")
          and not os.environ.get("AGENT_NO_REQUEST_CLAUSE")
          and not os.environ.get("AGENT_NO_REQUEST_IN_REVIEW")):
        also_untouched = (
            ", including anything the task named that you have not touched at all")
    else:
        also_untouched = ""
    if session is not None and not os.environ.get("AGENT_NO_REVIEW_TOOL"):
        tools.append(Tool(
            name="review_changes",
            description=(
                "List what you have actually changed in this workspace so far, "
                f"and anything that looks half-finished{also_untouched}. Use it "
                "before saying a multi-file change is done."
            ),
            params=[],
            fn=lambda: _review_changes(ws, session),
        ))
    return tools


def _review_changes(ws: Workspace, session: EditSession) -> str:
    """The run's own record, computed rather than remembered.

    This is the gap between a model reporting a rename it *meant* to make and
    one reporting the rename it made. It costs a step, so it says as little as
    possible: counts and symbol names, not diffs — the diff already went back
    when the edit landed. The text itself is `review.full_report()`, shared with
    the loop's unasked delivery so that both arms of that experiment read the
    same words.
    """
    return full_report(ws, session)


def _check_imports(ws: Workspace) -> str:
    problems, checked = check_workspace(ws)
    if not problems:
        clean = ("no syntax errors, and every import resolves"
                 if os.environ.get("AGENT_NO_UNDEFINED_CHECK") else
                 "no syntax errors, every import resolves, and every name used "
                 "is defined")
        return f"Checked {checked} Python file(s): {clean}."
    listed = "\n".join(f"  {p}" for p in problems)
    return (f"Checked {checked} Python file(s) and found {len(problems)} "
            f"problem(s):\n{listed}")
