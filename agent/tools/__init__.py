"""Tool registry assembly."""

from __future__ import annotations

from typing import Callable

from ..edits import EditSession
from ..sandbox import Workspace
from . import edit, fs, search, verify, web
from .base import Param, Registry, Tool

__all__ = ["Param", "Registry", "Tool", "EditSession", "build_registry"]


def build_registry(ws: Workspace, session: EditSession | None = None,
                   allow_web: bool = False,
                   post_approve: Callable[[str, str, str], bool] | None = None
                   ) -> Registry:
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

    The web tools — `web_search` and `fetch_url` — are the third opt-in for the
    same arithmetic, and the starkest case of it: **no eval case can use them.**
    The fixtures are offline, self-contained repos, so adding their schemas to
    the default set would be pure prompt cost against zero benefit, charged on
    all 40-odd cases — the exact shape that cost four cases the last time a
    description grew. They appear only when the caller asks with `--allow-web`,
    and `evals/run.py` never does, so no number on record moves.

    `http_post` is a fourth opt-in *inside* the third, because being allowed to
    read the web and being allowed to act on it are different grants. It needs an
    approver rather than a flag: there is no unattended POST, the same way there
    is no unattended write.
    """
    tools = [*fs.build(ws, session), *search.build(ws)]
    if session is not None:
        tools += edit.build(ws, session) + verify.build(ws, session)
    if allow_web or post_approve is not None:
        tools += web.build(post_approve)

    reg = Registry()
    for tool in tools:
        reg.add(tool)
    return reg
