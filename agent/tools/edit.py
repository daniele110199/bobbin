"""Write tools: edit_file, write_file, undo_edit.

These are opt-in (`build_registry(ws, session=...)`). The read-only suite must
keep scoring exactly what it scored before, so a registry without a session is
byte-identical to the one that produced the 22-case baseline.

## The contract with the model is one call

An edit is a single tool call that applies and returns its own diff. The
alternative — propose, read the diff back, then confirm — costs two of a 7B's
twelve steps per edit and adds a step it can fail to take. The confirmation
lives with the *user* instead (`EditSession.approve`), which is where a
judgement about whether an edit is wanted actually belongs.

## What can go wrong, and where it is caught

  anchor copied with "12| " prefixes   -> stripped in edits.py
  anchor does not match                -> error shows the nearest real lines
  anchor matches many places           -> error lists them, demands replace_all
  editing a file never read            -> refused, tells it to read first
  whole-file write with "... rest ..." -> refused as elided
  anything outside the workspace       -> refused by sandbox.resolve()
"""

from __future__ import annotations

from pathlib import Path

import os
import re

from ..edits import (
    EditRecord, EditSession, added_definitions, definition_pattern, looks_elided,
    nearest_lines, removed_identifiers, strip_line_numbers, unified_diff,
)
from ..output import cap
from ..sandbox import ToolError, Workspace, looks_binary, read_text
from .base import Param, Tool
from .search import _search

MAX_DIFF_LINES = 120
MAX_REFERENCE_LINES = 6


def build(ws: Workspace, session: EditSession) -> list[Tool]:
    return [
        Tool(
            name="edit_file",
            description=(
                "Replace an exact block of text in a file. old_string must match "
                "the file exactly, including indentation. Read the file first."
            ),
            params=[
                Param("path", "string", "File relative to the workspace root.",
                      required=True),
                Param("old_string", "string",
                      "The exact text to replace, copied from the file.",
                      required=True),
                Param("new_string", "string",
                      "The text to put in its place. Empty string deletes it.",
                      default=""),
                Param("replace_all", "boolean",
                      "Replace every occurrence instead of requiring exactly one.",
                      default=False),
            ],
            fn=lambda path, old_string, new_string, replace_all: _edit_file(
                ws, session, path, old_string, new_string, replace_all),
        ),
        Tool(
            name="write_file",
            description=(
                "Write a complete file, creating it or replacing all of its "
                "contents. To change part of a file that already exists, use "
                "edit_file instead."
            ),
            params=[
                Param("path", "string", "File relative to the workspace root.",
                      required=True),
                Param("content", "string", "The entire new contents of the file.",
                      required=True),
            ],
            fn=lambda path, content: _write_file(ws, session, path, content),
        ),
        Tool(
            name="undo_edit",
            description=(
                "Undo the most recent change you made to a file, restoring its "
                "previous contents. Use this if an edit turned out to be wrong."
            ),
            params=[
                Param("path", "string",
                      "File to restore. Defaults to the most recently edited file.",
                      default=""),
            ],
            fn=lambda path: _undo_edit(ws, session, path),
            # Measured 2026-08-16: called **zero** times across 12 edit and
            # cascade cases by two different 30B models. Its schema was pure
            # prompt cost on every request of every edit run, so it is no longer
            # advertised — still dispatchable, because the journal it reads is
            # what makes a REPL user's "undo that" work, and a JSON or XML prose
            # call naming it still dispatches. The one thing it loses is the
            # *recited* shape, whose name list comes from the schemas sent; an
            # unadvertised tool is by definition not one the model was invited to
            # narrate. `AGENT_ADVERTISE_UNDO=1` restores it, which is how the A/B
            # was run.
            advertised=bool(os.environ.get("AGENT_ADVERTISE_UNDO")),
        ),
    ]


def _commit(ws: Workspace, session: EditSession, target: Path, rel: str,
            before: str | None, after: str, tool: str) -> str:
    """Run the human gate, write the bytes, journal the change, return the diff."""
    if before == after:
        raise ToolError(
            f"that edit would leave {rel} exactly as it is. Check old_string and "
            "new_string are actually different."
        )

    diff = unified_diff(rel, before or "", after)
    if not session.approve(rel, diff):
        return (
            f"REJECTED: the user declined this change to {rel}. It has NOT been "
            "written. Do not retry the same edit — explain what you wanted to do "
            "and wait for instructions."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(after, encoding="utf-8")
    session.history.append(EditRecord(path=rel, before=before, after=after, tool=tool))
    # A written file counts as read: the agent now knows its contents exactly,
    # so a follow-up edit should not be blocked by the read gate.
    session.note_read(rel)

    verb = "Created" if before is None else "Updated"
    body = cap(diff.split("\n"), MAX_DIFF_LINES, "diff lines")
    trailer = (dangling_references(ws, rel, before or "", after)
               + duplicate_definitions(ws, rel, before or "", after))
    return f"{verb} {rel}.\n{body}{trailer}"


def dangling_references(ws: Workspace, rel: str, before: str, after: str) -> str:
    """What this edit left behind: references elsewhere to a name it removed.

    The cascade suite says the 30B renames a symbol in three files out of four
    and then reports all four as done. The failure is not that it cannot edit
    `feed.py` — it is that nothing ever told it `feed.py` was still there. So the
    edit that removes a name goes looking for the rest itself, the same way a
    failed `grep` widens instead of reporting "no matches": put the fact in front
    of the model rather than writing a prompt rule telling it to look.

    Deliberately a note on a successful edit, not an error. Leaving a reference
    behind is sometimes right — a deleted alias, a name that also means something
    else — so this reports and lets the model decide. `Workspace.walk` skips
    ignored directories, so a vendored copy under `node_modules/` is not counted,
    which is the correct answer for the decoy in the cascade fixture.
    """
    notes: list[str] = []
    for name in removed_identifiers(before, after):
        try:
            matches, files, _, _, _ = _search(
                ws, rf"\b{re.escape(name)}\b", ws.root, "",
                ignore_case=False, files_only=False,
            )
        except ToolError:
            continue
        elsewhere = [m for m in matches if not m.startswith(f"{rel}:")]
        other_files = sorted({m.split(":", 1)[0] for m in elsewhere})
        if not elsewhere:
            continue
        shown = "\n".join(f"  {m}" for m in elsewhere[:MAX_REFERENCE_LINES])
        more = len(elsewhere) - MAX_REFERENCE_LINES
        if more > 0:
            shown += f"\n  ...and {more} more"
        notes.append(
            f"NOTE: {name!r} is gone from {rel}, but {len(elsewhere)} reference(s) "
            f"remain in {len(other_files)} other file(s):\n{shown}\n"
            f"Update them too, or say why they should stay."
        )
    return ("\n" + "\n".join(notes)) if notes else ""


def duplicate_definitions(ws: Workspace, rel: str, before: str, after: str) -> str:
    """The mirror of `dangling_references`: a definition that now exists twice.

    A move is a copy plus a delete, and the 30B reliably does the first half —
    `cascade-move` ends with `def slugify` in both `src/util/text.py` and the new
    `src/util/slug.py`, reported as a completed move. Nothing is removed by that
    write, so the reference note cannot see it; this is what does.

    Same rules as the reference note, for the same reasons: it rides along on a
    successful write rather than blocking one (two same-named methods on
    different classes are ordinary code), it is silent when the name is unique,
    and it does not count copies in ignored directories.
    """
    notes: list[str] = []
    for name in added_definitions(before, after):
        try:
            _, files, _, _, _ = _search(
                ws, definition_pattern(name), ws.root, "",
                ignore_case=False, files_only=True,
            )
        except ToolError:
            continue
        others = [f for f in files if f != rel]
        if not others:
            continue
        listed = ", ".join(others[:4]) + (", ..." if len(others) > 4 else "")
        notes.append(
            f"NOTE: {name!r} is now defined in {len(others) + 1} files: {rel}, "
            f"{listed}. If you meant to move it, the original is still there — "
            f"delete it. If two copies are intended, say why."
        )
    return ("\n" + "\n".join(notes)) if notes else ""


def _require_read(session: EditSession, rel: str) -> None:
    if session.has_read(rel):
        return
    raise ToolError(
        f"you have not read {rel} yet, so you cannot know what to change. "
        f"Call read_file(path='{rel}') first, then edit it."
    )


def _load(ws: Workspace, path: str) -> tuple[Path, str, str]:
    target = ws.resolve(path)
    rel = ws.display(target)
    if target.is_dir():
        raise ToolError(f"{rel} is a directory, not a file.")
    if not target.exists():
        raise ToolError(
            f"file does not exist: {rel}. Use write_file to create it, or "
            "find_files to check the real path."
        )
    if looks_binary(target):
        raise ToolError(f"{rel} is a binary file and cannot be edited as text.")
    return target, rel, read_text(target)


def _edit_file(ws: Workspace, session: EditSession, path: str,
               old_string: str, new_string: str, replace_all: bool) -> str:
    target, rel, content = _load(ws, path)
    _require_read(session, rel)

    old = strip_line_numbers(old_string)
    new = strip_line_numbers(new_string or "")
    if not old:
        raise ToolError(
            "old_string is empty. To create or overwrite a whole file use "
            "write_file; edit_file needs the exact text to replace."
        )

    count = content.count(old)

    if count == 0:
        hints = nearest_lines(content, old)
        if hints:
            shown = "\n".join(f"  {n}| {line}" for n, line in hints)
            raise ToolError(
                f"old_string does not appear in {rel}. The closest lines in the "
                f"file are:\n{shown}\n"
                "Copy one of those exactly, with its original indentation and no "
                "line-number prefix."
            )
        raise ToolError(
            f"old_string does not appear in {rel}, and nothing in the file "
            "resembles it. Call read_file on it and copy the real text."
        )

    if count > 1 and not replace_all:
        where = [
            str(i) for i, line in enumerate(content.split("\n"), start=1)
            if old.split("\n")[0] in line
        ]
        raise ToolError(
            f"old_string appears {count} times in {rel} (near lines "
            f"{', '.join(where[:8])}), so it is ambiguous. Either include more "
            "surrounding text to make it unique, or pass replace_all=true if you "
            "really mean every occurrence."
        )

    updated = content.replace(old, new) if replace_all else content.replace(old, new, 1)
    return _commit(ws, session, target, rel, content, updated, "edit_file")


def _write_file(ws: Workspace, session: EditSession, path: str, content: str) -> str:
    target = ws.resolve(path)
    rel = ws.display(target)
    if target.is_dir():
        raise ToolError(f"{rel} is a directory, not a file.")

    elided = looks_elided(content)
    if elided:
        raise ToolError(
            f"that content is abbreviated — it contains {elided!r} — so writing "
            f"it would delete the rest of {rel}. write_file needs the complete "
            "file. To change one part of it, use edit_file instead."
        )

    before = None
    if target.exists():
        if looks_binary(target):
            raise ToolError(f"{rel} is a binary file and cannot be written as text.")
        # Overwriting destroys whatever is there; the read gate applies. Creating
        # a new file has nothing to have read, so it does not.
        _require_read(session, rel)
        before = read_text(target)

    if content and not content.endswith("\n"):
        content += "\n"
    return _commit(ws, session, target, rel, before, content, "write_file")


def _undo_edit(ws: Workspace, session: EditSession, path: str) -> str:
    if not session.history:
        raise ToolError("there is nothing to undo — you have not changed any file.")

    rel = (path or "").strip()
    if rel:
        rel = ws.display(ws.resolve(rel))
        index = next(
            (i for i in range(len(session.history) - 1, -1, -1)
             if session.history[i].path == rel),
            None,
        )
        if index is None:
            changed = ", ".join(sorted({r.path for r in session.history}))
            raise ToolError(f"you have not edited {rel}. Files you changed: {changed}.")
    else:
        index = len(session.history) - 1

    record = session.history.pop(index)
    target = ws.resolve(record.path)

    if record.before is None:
        target.unlink(missing_ok=True)
        return f"Undid the creation of {record.path}; the file no longer exists."

    target.write_text(record.before, encoding="utf-8")
    diff = unified_diff(record.path, record.after, record.before)
    body = cap(diff.split("\n"), MAX_DIFF_LINES, "diff lines")
    return f"Reverted {record.path} to its previous contents.\n{body}"
