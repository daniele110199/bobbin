"""Computing over the run's own record, rather than remembering it.

The measured failure this exists for: the agent renames a symbol in three files
of four and reports all four, or copies a function and reports a move. It is not
lying about the repo — it is reporting from memory of what it *meant* to do,
because nothing lets it look at what it actually did.

Two functions, both derived from the edit journal plus the current tree, and
both deterministic:

    change_summary()      what this run changed, per file
    unfinished_reasons()  ways those changes look half-done

The loop uses `unfinished_reasons()` to annotate an answer (see
`UNFINISHED_NOTE`), and `review_changes` exposes both to the model. That
symmetry is the point: the model can ask the same question the environment will
judge it by, instead of being told the answer afterwards.
"""

from __future__ import annotations

import os
import re

from .edits import (
    RESERVED, EditSession, added_definitions, defined_here, removed_identifiers,
)
from .imports import definition_sites, files_mentioning
from .sandbox import Workspace


def _tally(before: str, after: str) -> tuple[int, int]:
    """Lines added and removed. Counted, not diffed: the diff already went back
    to the model when the edit landed, and repeating it here would spend the
    context this tool exists to save."""
    old, new = before.splitlines(), after.splitlines()
    common = set(old) & set(new)
    return (sum(1 for line in new if line not in common),
            sum(1 for line in old if line not in common))


def _binds(ws: Workspace, path: str, name: str) -> bool:
    """Does this file define or assign `name` at the top level?"""
    try:
        return defined_here(name, (ws.root / path).read_text(
            encoding="utf-8", errors="replace"))
    except OSError:
        return False


def change_summary(ws: Workspace, session: EditSession) -> str:
    """Every file this run touched, with what happened to it."""
    if not session.history:
        return "You have not changed any file in this run."

    # One entry per path, oldest `before` against newest `after`, so three edits
    # to one file read as one change rather than three.
    first: dict[str, str | None] = {}
    last: dict[str, str] = {}
    order: list[str] = []
    for record in session.history:
        if record.path not in first:
            first[record.path] = record.before
            order.append(record.path)
        last[record.path] = record.after

    lines = [f"You changed {len(order)} file(s) in this run:"]
    for path in order:
        before, after = first[path], last[path]
        if before is None:
            lines.append(f"  created {path} ({len(after.splitlines())} lines)")
            continue
        plus, minus = _tally(before, after)
        gone = removed_identifiers(before, after, limit=3)
        new = added_definitions(before, after, limit=3)
        detail = []
        if new:
            detail.append("now defines " + ", ".join(repr(n) for n in new))
        if gone:
            detail.append("no longer mentions " + ", ".join(repr(n) for n in gone))
        suffix = f" — {'; '.join(detail)}" if detail else ""
        lines.append(f"  modified {path} (+{plus}/-{minus}){suffix}")
    return "\n".join(lines)


def unfinished_reasons(ws: Workspace, session: EditSession) -> list[str]:
    """Ways this run's own edits look half-done.

    Keyed on what the edits *did*, never on words from the prompt. A trigger
    keyed on the prompt's identifier fires on 23 of 44 passing runs, because
    changing `PAGE_SIZE`'s value or adding a parameter to `slugify` leaves the
    name legitimately everywhere. Two conditions, from the measured failure
    taxonomy (45 "old code still there", 43 "new code never written", 0 broken
    imports):

      renamed   a name this run deleted from a file it edited *and that the file
                actually defined*, still used in a file it never touched
      copied    a definition this run added that some other file still defines
    """
    if not session.history:
        return []

    changed = {record.path for record in session.history}
    removed: set[str] = set()
    added: set[str] = set()
    for record in session.history:
        before, after = record.before or "", record.after
        # Only names this file owned. A module path component that changed
        # inside an import line is not a symbol anyone lost — without this, a
        # *correct* move is flagged because README.md names the old file.
        removed.update(n for n in removed_identifiers(before, after, limit=4)
                       if defined_here(n, before))
        added.update(added_definitions(before, after, limit=4))

    reasons: list[str] = []
    # A name is only *lost* if nothing defines it any more. Both halves of that
    # were learned the hard way:
    #
    #   Exempting any name that was re-created elsewhere ("it moved, it is not
    #   missing") exempts the broken move — original deleted, importers never
    #   repointed — which is why that check is not on `added` but on the tree.
    #
    #   Not checking the tree at all reports a *correct* module split: moving
    #   `TAX_RATE` from `config.py` to `tax.py` leaves `README.md` naming it, and
    #   the README is right. Measured on `cascade-split-module`, which passed
    #   while this function warned about it.
    #
    # Stale references to a name that still exists are a different problem — an
    # importer pointing at the wrong module — and `check_imports` reports those
    # precisely, rather than by inference from a word appearing in prose.
    for name in sorted(removed):
        mentions = files_mentioning(ws, name)
        stale = [p for p in mentions if p not in changed]
        if not stale:
            continue
        still_bound = any(_binds(ws, path, name) for path in mentions)
        if not still_bound:
            reasons.append(
                f"{name!r} was removed from what you edited, and nothing defines "
                f"it any more, but {', '.join(stale[:4])} still refers to it")
    for name in sorted(added):
        sites = definition_sites(ws, name)
        if len(sites) > 1:
            reasons.append(
                f"{name!r} is now defined in {len(sites)} files: "
                f"{', '.join(sites[:4])}")
    return reasons


# --- Keyed on the request, not on the damage --------------------------------
#
# Everything above keys on an artifact the run *damaged*: a name it deleted, a
# definition it duplicated. Measured 2026-08-18, that leaves one shape entirely
# unseen — work that was never begun. Asked to rename `Order` *and*
# `apply_discount`, qwen renamed `Order` in six files, never touched
# `apply_discount`, and reported both done. Nothing dangles, every import
# resolves, every name is bound, `broken_at_end` is 0: the tree is a healthy repo
# that does half of what was asked, and every guard agrees it is fine.
#
# So this one asks the only question the others cannot: **did the run engage with
# each thing the request named at all?** It is the first check here that can be
# wrong about what the *user* meant rather than about what the code says, which
# is why it reports and does not gate, and why every rule below is a
# false-positive rule.

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# All-caps prose that would otherwise pass the shape filter below. Kept tiny on
# purpose: anything longer is a hand-maintained list waiting to go stale, and a
# missed candidate is only a false negative.
_PROSE = frozenset({"TODO", "FIXME", "NOTE", "README", "API", "URL", "CLI",
                    "YES", "NO", "OK", "AND", "OR", "NOT", "ALL", "ANY"})


def _code_shaped(name: str) -> bool:
    """Does this token look like an identifier rather than an English word?

    Three shapes, and nothing softer. A prompt is prose, and the cost of
    admitting prose here is an accusation about a word the user never meant as a
    symbol. `slugify` and `total` are indistinguishable from English on shape
    alone — they get in only via the syntax rules in `request_identifiers()`.
    """
    return ("_" in name.strip("_")                    # apply_discount
            or bool(re.search(r"[a-z][A-Z]", name))   # PurchaseOrder
            or (name.isupper() and name not in _PROSE))  # TAX_RATE, MAX_ITEMS


def request_identifiers(request: str) -> list[str]:
    """Names in the request that are plausibly symbols, in the order asked.

    Three ways in, all cheap and all syntactic — no part of speech, no verb
    list, nothing that has an opinion about what the sentence means:

        shape       `apply_promotion`, `PurchaseOrder`, `VAT_RATE`
        called      `subtotal()` — the parens are the user calling it code
        quoted      `` `slugify` `` — the backticks are the user saying so

    Prose words are the whole risk, so a bare lowercase word without parens or
    backticks never qualifies, however much the sentence is about it.
    """
    found: dict[str, int] = {}
    for match in _NAME.finditer(request):
        name = match.group()
        if len(name) < 3 or name in RESERVED:
            continue
        before = request[match.start() - 1: match.start()]
        after = request[match.end(): match.end() + 1]
        if not (_code_shaped(name) or after == "("
                or (before == "`" and after == "`")):
            continue
        found.setdefault(name, match.start())
    return sorted(found, key=found.get)


def path_requested(request: str, path: str) -> bool:
    """Did the request ask for *this file* by name?

    The narrow question, deliberately. `request_identifiers()` above answers
    "which symbols did the user name"; a path is not a symbol and must not be
    inferred from one — `edit-trap` says "the compute_tax function" and a run
    that creates `compute_tax.py` has invented a file, not honoured a request.
    So the match is on the path as written or on its basename **with the
    extension**, both of which the user has to have typed.

    Priced over every stored run that created a file: 188 creations where the
    request names the path (`src/util/slug.py`, `src/store/discounts.py`,
    `src/tax.py`, `src/billing/discount.py`) and 23 where it does not
    (`run_tests.py`, `test_runner.py`, `test_slugify.py`,
    `test_slugify_max_length.py`, `compute_tax.py`). Every one of the 23 is a
    file the case forbids; every one of the 188 is the task itself.
    """
    cleaned = path.replace("\\", "/").lstrip("./")
    if not cleaned:
        return False
    return cleaned in request or cleaned.split("/")[-1] in request


def _known_to_workspace(ws: Workspace, name: str) -> bool:
    """Does the tree mention this name at all — in a file, or as a path?

    Paths count because "move it into `src/util/slug_text.py`" names the new
    module and nothing else will: a file created with that name is the work
    being done, even if no line inside it repeats the word.
    """
    if files_mentioning(ws, name):
        return True
    return any(name in ws.display(file).replace("/", " ").replace(".", " ").split()
               for file in ws.walk(ws.root))


def unaddressed_requests(request: str, ws: Workspace, session: EditSession,
                         limit: int = 3) -> list[str]:
    """Names the request asked about that this run never went near.

    Deliberately the narrow half of the idea. A name that the request mentions
    and the tree still contains is *ambiguous* — `cascade-delete-symbol` says
    "remove the check in place_order", and a correct run leaves `place_order`
    untouched and present, so flagging "mentioned but untouched" accuses a
    passing case. What is not ambiguous is a name the request introduces that
    exists **nowhere**: no file mentions it, no path carries it, and nothing this
    run wrote or deleted ever contained it. Then the request named a thing and
    the run never engaged with it.

    Both halves of "never went near" matter, and each kills a false positive:

        still in the tree    a rename that has not happened yet leaves the old
                             name everywhere — that is the other guards' case,
                             not this one
        in the journal       a *correct* deletion ends with the name gone from
                             the tree, and a *correct* rename ends with the old
                             name gone; both appear in the `before` of an edit,
                             so both stay silent here

    Silent when the run edited nothing: a read-only answer, or a refusal to edit
    a file that does not exist, is not half-done work.
    """
    if not session.history:
        return []
    seen = "\n".join(
        text for record in session.history
        for text in (record.before or "", record.after)
    )
    reasons: list[str] = []
    for name in request_identifiers(request):
        if re.search(rf"\b{re.escape(name)}\b", seen):
            continue
        if _known_to_workspace(ws, name):
            continue
        reasons.append(
            f"the request names {name!r}, and nothing in the workspace mentions "
            f"it — none of your edits wrote it, so that part of the request "
            f"looks untouched")
    return reasons[:limit]


def full_report(ws: Workspace, session: EditSession) -> str:
    """What `review_changes` says: the record, then anything half-done.

    Composed here rather than in the tool because the loop needs the identical
    text when it delivers the report unasked — measuring whether the *content*
    changes behaviour is only possible if the asked-for and the delivered
    versions are the same string.

    `AGENT_NO_REQUEST_SECTION=1` drops the request-keyed part alone, leaving the
    tool description untouched. That separation is the whole experiment: the
    sentence in the schema and the paragraph in the result had one switch between
    them, and a clause nobody reads was moving the result on its own.
    """
    summary = change_summary(ws, session)
    reasons = unfinished_reasons(ws, session)
    hide_section = (os.environ.get("AGENT_NO_REQUEST_SECTION")
                    or os.environ.get("AGENT_NO_REQUEST_IN_REVIEW")
                    or os.environ.get("AGENT_NEUTRAL_REVIEW_CLAUSE"))
    if session.request and not hide_section:
        try:
            reasons = reasons + unaddressed_requests(session.request, ws, session)
        except Exception:
            pass
    if not reasons:
        return f"{summary}\n\nNothing looks half-finished."
    listed = "\n".join(f"  - {r}" for r in reasons)
    return f"{summary}\n\nThis looks unfinished:\n{listed}"
