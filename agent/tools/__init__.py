"""Tool registry assembly."""

from __future__ import annotations

from ..edits import EditSession
from ..sandbox import Workspace
from . import edit, fs, search, verify, web
from .base import Param, Registry, Tool

__all__ = ["Param", "Registry", "Tool", "EditSession", "build_registry"]


def build_registry(ws: Workspace, session: EditSession | None = None,
                   allow_web: bool = False) -> Registry:
    """The tool set. Write tools appear only when a session is supplied.

    Editing is opt-in so that the read-only registry stays byte-identical to the
    one the 22-case baseline was measured against — an extra tool schema is
    prompt text, charged at the same rate, and advertising one has already cost
    this project four cases once (see the grep-description result). Any change to
    the read-only numbers must come from a deliberate experiment, not from
    quietly adding three tools nobody asked for.

    `check_imports` rides with the write tools rather than the read-only set for
    the same reason: it is read-only in effect, but it exists to check an edit,
    and adding a schema to the 22-case prompt would invalidate every number this
    project has quoted.

    `fetch_url` is the third opt-in for the same arithmetic, and the starkest
    case of it: **no eval case can use it.** The fixtures are offline,
    self-contained repos, so adding its schema to the default set would be pure
    prompt cost against zero benefit, charged on all 40-odd cases — the exact
    shape that cost four cases the last time a description grew. It appears only
    when the caller asks for it with `--allow-web`, and `evals/run.py` never
    does, so no number on record moves.
    """
    tools = [*fs.build(ws, session), *search.build(ws)]
    if session is not None:
        tools += edit.build(ws, session) + verify.build(ws, session)
    if allow_web:
        tools += web.build()

    reg = Registry()
    for tool in tools:
        reg.add(tool)
    return reg
