"""Budgeting what a tool is allowed to put into the context window.

A 7B model with 32k context dies if `grep` dumps 4000 hits. Every tool result
goes through here, and truncation is always *announced* so the model knows the
view was partial and can narrow its query instead of assuming it saw everything.
"""

from __future__ import annotations

MAX_RESULT_CHARS = 12_000


def cap(lines: list[str], max_lines: int, note: str = "results") -> str:
    total = len(lines)
    if total > max_lines:
        shown = lines[:max_lines]
        shown.append(
            f"... truncated: showing {max_lines} of {total} {note}. "
            "Narrow your query to see the rest."
        )
        lines = shown

    text = "\n".join(lines)
    if len(text) > MAX_RESULT_CHARS:
        text = (
            text[:MAX_RESULT_CHARS]
            + f"\n... truncated at {MAX_RESULT_CHARS} characters."
        )
    return text


def human_size(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}G"
