"""Session state shared by the read and write tools.

The write tools need to know things `read_file` learned. Rather than let each
tool keep its own state, one `EditSession` is threaded through the registry:

  * `read_paths` — what the agent has actually opened. `edit_file` refuses to
    touch a file that is not in here.
  * `history`   — the previous contents of every file written, so `undo_edit`
    can put it back.
  * `approve`   — the human gate. The model never sees a confirmation step; the
    *user* does, in the REPL. Evals auto-approve.

## Why the read gate exists

"Plan before you edit" is a prompt rule, and prompt rules are the one thing this
project has repeatedly measured as worthless (see the playbook v2/v3 results).
The read gate is the same idea expressed as an environment: an edit to a file
the model has not opened is not discouraged, it is impossible. The model cannot
ignore it, and the failure is a one-step, self-correcting error rather than a
wrong edit.
"""

from __future__ import annotations

import difflib
import keyword
import re
from dataclasses import dataclass, field
from typing import Callable

# `read_file` renders lines as "  12| code". A model copying an anchor out of a
# tool result will copy those prefixes too — it is the single most predictable
# way an edit anchor fails, so the write tools strip them rather than making the
# model get it right.
LINE_PREFIX = re.compile(r"^\s*\d+\|\s?")

# Small models "helpfully" abbreviate when asked for a whole file, and a
# `write_file` that lands one of these silently deletes the rest of the file.
# Truncation that looks like content is worse than an error, so it is refused.
ELISION_PHRASES = (
    "rest of the file", "rest of file", "remainder of the file",
    "unchanged", "as before", "same as above", "existing code",
    "previous content", "no changes", "omitted", "truncated",
    "keep the rest", "etc.",
)


@dataclass
class EditRecord:
    path: str            # workspace-relative
    before: str | None   # None = the file did not exist (so undo deletes it)
    after: str
    tool: str


@dataclass
class EditSession:
    """Per-run mutable state. One per Agent, never global."""
    read_paths: set[str] = field(default_factory=set)
    history: list[EditRecord] = field(default_factory=list)
    # What the user asked for this turn, set by the loop at the top of `ask()`.
    # It lives here rather than in the tool closure because the registry is built
    # once, before any request exists — and `review_changes` takes no parameters
    # on purpose, so it cannot ask the model to restate the task it was given.
    request: str = ""
    # Returns True to let a write through. Default auto-approves, which is what
    # the eval harness wants; the REPL substitutes a y/N prompt.
    approve: Callable[[str, str], bool] = lambda path, diff: True

    def note_read(self, path: str) -> None:
        self.read_paths.add(path)

    def has_read(self, path: str) -> bool:
        return path in self.read_paths


def strip_line_numbers(text: str) -> str:
    """Remove "  12| " prefixes, but only if *every* non-blank line has one.

    The all-or-nothing rule is what makes this safe: a genuine code block with
    one line that happens to look like "3| x" is left alone, so this can never
    quietly mangle real content.
    """
    lines = text.split("\n")
    meaningful = [ln for ln in lines if ln.strip()]
    if not meaningful or not all(LINE_PREFIX.match(ln) for ln in meaningful):
        return text
    return "\n".join(LINE_PREFIX.sub("", ln) if ln.strip() else ln for ln in lines)


def looks_elided(content: str) -> str | None:
    """Detect "... rest of the file unchanged ...". Returns the offending line.

    Only comment-ish or ellipsis-ish lines count, so prose in a .md file that
    happens to say "unchanged" is not a false positive.
    """
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        is_marker = (
            stripped.startswith(("#", "//", "/*", "<!--", "...", "[", "(", "*"))
            or stripped.endswith(("...", "-->", "*/", "]", ")"))
        )
        if not is_marker:
            continue
        lowered = stripped.lower()
        if "..." in stripped and any(p in lowered for p in ELISION_PHRASES):
            return stripped
        if any(f"{p}" in lowered for p in ("rest of the file", "rest of file",
                                           "remainder of the file")):
            return stripped
    return None


# An identifier the edit deleted from a file is the signal that a change may
# have consequences elsewhere. Two kinds of noise have to stay out, or the note
# becomes one the model learns to skip:
#
#   prose      a rewritten docstring "removes" words like `title` and `Turn`
#   borrowed   deleting a body "removes" `lower`, `strip`, `sub` and `value` —
#              methods called on something else, and locals that were never
#              this file's to own
#
# So the file has to have *owned* the name: defined it, imported it, or bound it
# at the start of a line. A name that only ever appears after a dot belongs to
# whatever it was called on, and a parameter belongs to its function.
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# `from x import y` matches the import rule for the word "import" itself, which
# is how "you removed `import`, 40 references remain" gets reported. The
# keyword list is the language's, not a hand-kept one, so it cannot drift.
RESERVED = frozenset(keyword.kwlist) | {"self", "cls"}


def _owned_by_file(name: str, text: str) -> bool:
    escaped = re.escape(name)
    return bool(re.search(
        rf"(?:^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+{escaped}\b)"
        rf"|(?:^[ \t]*(?:from|import)[ \t]+[^\n]*\b{escaped}\b)"
        rf"|(?:^[ \t]*{escaped}[ \t]*(?::[^=\n]+)?=(?!=))",
        text, re.M,
    ))


def removed_identifiers(before: str, after: str, limit: int = 2) -> list[str]:
    """Names this edit deleted from the file completely, longest first.

    "Completely" is the important part: an edit that renames one of three call
    sites has not removed the name, and reporting it would be wrong.
    """
    gone = set(IDENTIFIER.findall(before)) - set(IDENTIFIER.findall(after))
    kept = [n for n in gone - RESERVED if _owned_by_file(n, before)]
    kept.sort(key=lambda n: (-len(n), n))
    return kept[:limit]


# The other half of the same idea. A rename *removes* a name and strands its
# references; a half-finished move *adds* a definition and leaves the original
# in place, which removes nothing at all — so the note above cannot fire, and
# the 30B ends a `cascade-move` run with `def slugify` in two files while
# reporting a clean move. Python only, deliberately: the definition syntax has
# to be known to be matched, and guessing it for other languages would produce
# confident nonsense.
DEFINITION = re.compile(r"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+([A-Za-z_]\w*)",
                        re.M)


def added_definitions(before: str, after: str, limit: int = 2) -> list[str]:
    """Functions and classes this write defines that the file did not define."""
    fresh = set(DEFINITION.findall(after)) - set(DEFINITION.findall(before))
    return sorted(fresh, key=lambda n: (-len(n), n))[:limit]


def defined_here(name: str, text: str) -> bool:
    """Did this file *define* the name, rather than borrow it from elsewhere?

    The distinction matters when asking "I removed this — who still uses it?".
    Rewriting `from src.util.text import slugify` into `...util.slug import...`
    removes the token `text`, which is a module path component, not a symbol this
    file owned; chasing it leads to reporting `README.md` for naming a file. Only
    a `def`, a `class` or a line-start binding counts as ownership.
    """
    escaped = re.escape(name)
    return bool(re.search(
        rf"(?:^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+{escaped}\b)"
        rf"|(?:^[ \t]*{escaped}[ \t]*(?::[^=\n]+)?=(?!=))",
        text, re.M,
    ))


def definition_pattern(name: str) -> str:
    """A grep for `def name` / `class name` at the start of a line."""
    return rf"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+{re.escape(name)}\b"


def unified_diff(path: str, before: str, after: str, context: int = 3) -> str:
    """A normal unified diff. Shown to the user, and returned to the model as
    the tool result so it can verify what it actually did."""
    diff = difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}",
        lineterm="", n=context,
    )
    return "\n".join(diff)


def nearest_lines(haystack: str, needle: str, limit: int = 3) -> list[tuple[int, str]]:
    """The closest actual lines to a failed anchor, as (line_number, text).

    This is the grep-widening lesson applied to editing: an anchor miss must
    hand back the real text to use, not just "no match". A model told only that
    it failed will retry with another guess; a model shown the true line can
    copy it.
    """
    first = next((ln for ln in needle.split("\n") if ln.strip()), needle).strip()
    if not first:
        return []
    scored: list[tuple[float, int, str]] = []
    for i, line in enumerate(haystack.split("\n"), start=1):
        if not line.strip():
            continue
        ratio = difflib.SequenceMatcher(None, first, line.strip()).ratio()
        if ratio >= 0.6:
            scored.append((ratio, i, line))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [(i, line) for _, i, line in scored[:limit]]
