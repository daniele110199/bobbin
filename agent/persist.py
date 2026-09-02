"""Save a session to disk and pick it up in a new process.

    save_session(agent, path)                     # after every turn
    restore_session(agent, load_session(path))    # into a freshly built Agent

A session is not the message list alone. Three other things have to survive or
the resumed agent is a different agent wearing the same history:

  * **the edit journal** — `undo_edit` reads it, and "undo that" is the first
    thing a user says after resuming a session they interrupted;
  * **the turn boundaries** — compaction cuts on them, and they are held by
    object *identity* in a running loop, which no file can store. They go out as
    indices and come back as identities;
  * **the workspace root** — resuming a session against a different tree would
    silently answer questions about files the history never saw, so it is
    checked rather than trusted.

What is deliberately *not* stored: the client, the registry and the playbook.
Those are how the next process is configured, and a session file that could
override them would be a config file with a conversation attached.
"""

from __future__ import annotations

import json
from pathlib import Path

from .edits import EditRecord

FORMAT = 1


def session_state(agent) -> dict:
    """Everything about a live agent that a later process cannot reconstruct."""
    index = {id(message): position for position, message
             in enumerate(agent.messages)}
    return {
        "format": FORMAT,
        "root": str(agent.workspace.root),
        "model": getattr(agent.client, "model", None),
        "messages": agent.messages,
        # Identity is what the loop uses; position is what a file can hold.
        "turn_marks": [index[id(mark)] for mark in agent.turn_marks
                       if id(mark) in index],
        "edits": [{"path": record.path, "before": record.before,
                   "after": record.after, "tool": record.tool}
                  for record in (agent.session.history if agent.session else [])],
    }


def save_session(agent, path: str | Path) -> Path:
    """Write the session beside whatever else the caller keeps. Atomic-ish."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(session_state(agent), indent=2))
    temporary.replace(path)
    return path


def load_session(path: str | Path) -> dict:
    state = json.loads(Path(path).read_text())
    if state.get("format") != FORMAT:
        raise ValueError(
            f"session file is format {state.get('format')}, this build reads "
            f"{FORMAT}")
    return state


def restore_session(agent, state: dict, *, force: bool = False) -> None:
    """Put a stored session back into a freshly constructed agent.

    The root check is not paranoia: the history is full of file contents and
    line numbers, and resuming it over a different tree produces answers that
    are wrong in the most convincing way available — confidently, with citations.
    """
    stored_root = state.get("root")
    if not force and stored_root and str(agent.workspace.root) != stored_root:
        raise ValueError(
            f"this session was recorded in {stored_root}, and the agent is "
            f"rooted at {agent.workspace.root}. Re-run from there, or pass "
            f"force=True if you know the tree moved.")

    agent.messages = list(state.get("messages") or [])
    agent.turn_marks = [agent.messages[position]
                        for position in state.get("turn_marks") or []
                        if 0 <= position < len(agent.messages)]
    if agent.session is not None:
        agent.session.history = [
            EditRecord(path=record["path"], before=record["before"],
                       after=record["after"], tool=record.get("tool", "edit_file"))
            for record in state.get("edits") or []
        ]
