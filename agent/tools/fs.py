"""Filesystem tools: list_files, read_file, find_files."""

from __future__ import annotations

import fnmatch
from pathlib import Path

from ..edits import EditSession
from ..output import cap, human_size
from ..sandbox import ToolError, Workspace, looks_binary, read_text
from .base import Param, Tool

MAX_LIST_ENTRIES = 300
MAX_READ_LINES = 400
MAX_FIND_RESULTS = 200


def build(ws: Workspace, session: EditSession | None = None) -> list[Tool]:
    return [
        Tool(
            name="list_files",
            description=(
                "List files and directories at a path in the workspace. "
                "Use this first to find out what exists. Directories end with '/'."
            ),
            params=[
                Param("path", "string",
                      "Directory relative to the workspace root. Defaults to '.'",
                      default="."),
                Param("depth", "integer",
                      "How many directory levels to descend. 1 = this directory only. Max 4.",
                      default=1),
            ],
            fn=lambda path, depth: _list_files(ws, path, depth),
        ),
        Tool(
            name="read_file",
            description=(
                "Read a text file's contents with line numbers. "
                "Use offset/limit to page through a long file."
            ),
            params=[
                Param("path", "string", "File relative to the workspace root.",
                      required=True),
                Param("offset", "integer", "First line to read, 1-based. Defaults to 1.",
                      default=1),
                Param("limit", "integer",
                      f"How many lines to read. Defaults to {MAX_READ_LINES}.",
                      default=MAX_READ_LINES),
            ],
            fn=lambda path, offset, limit: _read_file(ws, path, offset, limit, session),
        ),
        Tool(
            name="find_files",
            description=(
                "Find files anywhere in the workspace whose name matches a glob "
                "pattern, e.g. '*.py' or 'test_*'. Searches recursively. "
                "Use this when you know part of a filename but not where it lives."
            ),
            params=[
                Param("pattern", "string",
                      "Glob matched against the file name, e.g. '*.py'.",
                      required=True),
                Param("path", "string",
                      "Directory to search under. Defaults to the whole workspace.",
                      default="."),
            ],
            fn=lambda pattern, path: _find_files(ws, pattern, path),
        ),
    ]


def _list_files(ws: Workspace, path: str, depth: int) -> str:
    root = ws.resolve(path)
    if not root.exists():
        raise ToolError(f"path does not exist: {ws.display(root)}")
    if root.is_file():
        return f"{ws.display(root)} is a file, not a directory ({human_size(root.stat().st_size)}). Use read_file."

    depth = max(1, min(int(depth or 1), 4))
    lines: list[str] = []
    _walk_depth(ws, root, root, depth, lines)

    if not lines:
        return f"{ws.display(root)}/ is empty."
    header = f"{ws.display(root)}/ ({len(lines)} entries, depth {depth}):"
    return header + "\n" + cap(lines, MAX_LIST_ENTRIES, "entries")


def _walk_depth(ws: Workspace, root: Path, current: Path, depth: int, out: list[str]) -> None:
    try:
        entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        out.append(f"{ws.display(current)}/  (permission denied)")
        return

    for entry in entries:
        if entry.is_dir():
            if ws.is_ignored_dir(entry.name):
                continue
            out.append(f"{ws.display(entry)}/")
            if depth > 1:
                _walk_depth(ws, root, entry, depth - 1, out)
        else:
            try:
                size = human_size(entry.stat().st_size)
            except OSError:
                size = "?"
            out.append(f"{ws.display(entry)}  {size}")


def _read_file(ws: Workspace, path: str, offset: int, limit: int,
               session: EditSession | None = None) -> str:
    target = ws.resolve(path)
    if not target.exists():
        raise ToolError(f"file does not exist: {ws.display(target)}")
    if target.is_dir():
        raise ToolError(f"{ws.display(target)} is a directory. Use list_files.")
    if looks_binary(target):
        raise ToolError(
            f"{ws.display(target)} looks like a binary file and cannot be read as text."
        )

    offset = max(1, int(offset or 1))
    limit = max(1, min(int(limit or MAX_READ_LINES), MAX_READ_LINES))

    all_lines = read_text(target).splitlines()
    total = len(all_lines)
    if offset > total:
        raise ToolError(
            f"{ws.display(target)} has only {total} lines; offset {offset} is past the end."
        )

    # Opening the file unlocks the write tools' read gate. A partial read counts:
    # the anchor still has to match the file exactly, so the gate is about having
    # looked at all, not about having seen every line.
    if session is not None:
        session.note_read(ws.display(target))

    chunk = all_lines[offset - 1: offset - 1 + limit]
    width = len(str(offset + len(chunk) - 1))
    body = [f"{offset + i:>{width}}| {line}" for i, line in enumerate(chunk)]

    last = offset + len(chunk) - 1
    header = f"{ws.display(target)} lines {offset}-{last} of {total}:"
    footer = ""
    if last < total:
        footer = f"\n... {total - last} more lines. Call read_file again with offset={last + 1}."
    return header + "\n" + cap(body, MAX_READ_LINES, "lines") + footer


def _find_files(ws: Workspace, pattern: str, path: str) -> str:
    root = ws.resolve(path)
    if not root.is_dir():
        raise ToolError(f"{ws.display(root)} is not a directory.")

    pattern = pattern.strip()
    # Models often write '**/*.py' out of habit; match on the basename instead.
    name_pattern = pattern.rsplit("/", 1)[-1] or pattern

    hits = [
        ws.display(p)
        for p in ws.walk(root)
        if fnmatch.fnmatch(p.name, name_pattern)
    ]
    if not hits:
        return (
            f"No files matching {pattern!r} under {ws.display(root)}/. "
            "Try a broader pattern such as '*.py'."
        )
    return f"{len(hits)} file(s) matching {pattern!r}:\n" + cap(hits, MAX_FIND_RESULTS, "files")
