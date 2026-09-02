"""Content search across the workspace (pure stdlib, no ripgrep needed)."""

from __future__ import annotations

import fnmatch
import re

from .. import vocab
from ..output import cap
from ..sandbox import MAX_SCAN_BYTES, ToolError, Workspace, looks_binary
from .base import Param, Tool

MAX_MATCHES = 80
MAX_LINE_CHARS = 300


def build(ws: Workspace) -> list[Tool]:
    return [
        Tool(
            name="grep",
            description=(
                "Search file contents for a regular expression across the workspace. "
                "Returns 'path:line: text' for each match. "
                "Use file_glob to restrict the search, e.g. '*.py'. "
                "This is the fastest way to locate where something is defined or used."
            ),
            params=[
                Param("pattern", "string",
                      "Regular expression to search for, e.g. 'def main' or 'TODO'.",
                      required=True),
                Param("path", "string",
                      "Directory to search under. Defaults to the whole workspace.",
                      default="."),
                Param("file_glob", "string",
                      "Only search files whose name matches this glob, e.g. '*.py'. "
                      "Empty means all text files.",
                      default=""),
                Param("ignore_case", "boolean",
                      "Case-insensitive search. Defaults to false.",
                      default=False),
                Param("files_only", "boolean",
                      "Return only the list of matching file names, not the matching "
                      "lines. Useful when a pattern matches too much.",
                      default=False),
            ],
            fn=lambda pattern, path, file_glob, ignore_case, files_only: _grep(
                ws, pattern, path, file_glob, ignore_case, files_only
            ),
        ),
    ]


# Widening to one of these finds half the repo and teaches the model nothing.
STOPWORDS = {
    "def", "class", "function", "const", "let", "var", "import", "from",
    "return", "self", "this", "true", "false", "none", "null", "public",
    "static", "void", "the", "and", "for", "with", "that", "value", "file",
    "project", "used", "does", "use", "set", "get", "new",
}


def split_alternatives(pattern: str) -> list[str]:
    """The top-level `a|b|c` branches of a pattern, or [] if there is only one.

    Searching three names in one call costs one pass over the files instead of
    three, which matters most where the budget is tightest — a research
    sub-agent gets four steps. The regex engine has always accepted this; what
    was missing was the model being told, and the result saying which name each
    hit belongs to. A flat list of matches for `A|B` cannot be read back: the
    model sees hits, assumes both names are present, and reports a `B` that is
    not there.

    Only splits at depth 0, so `(foo|bar)_id` and `[a|b]` stay one pattern. A
    single wrapping group is unwrapped first, since models write `(a|b)` about
    as often as `a|b`.
    """
    text = pattern.strip()
    if _wrapped_in_one_group(text):
        inner = text[1:-1]
        text = inner[2:] if inner.startswith("?:") else inner

    parts: list[str] = []
    current: list[str] = []
    depth = 0
    escaped = False
    in_class = False

    for ch in text:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if ch == "\\":
            current.append(ch)
            escaped = True
            continue
        if in_class:
            current.append(ch)
            if ch == "]":
                in_class = False
            continue
        if ch == "[":
            in_class = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "|" and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))

    parts = [p.strip() for p in parts]
    if len(parts) < 2 or any(not p for p in parts):
        return []
    return parts


def _wrapped_in_one_group(text: str) -> bool:
    """True for '(a|b)' but not for '(a)|(b)'."""
    if not (text.startswith("(") and text.endswith(")")):
        return False
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i == len(text) - 1
    return False


def _widenings(pattern: str) -> list[tuple[str, str]]:
    """Progressively broader retries for a pattern that found nothing.

    A weak model treats "No matches" as proof of absence and stops. So the tool
    widens the search itself and reports what it *did* find — turning a dead end
    into a lead without requiring the model to have the idea.

    Ordered most-specific first, so the narrowest useful lead wins.
    """
    out: list[tuple[str, str]] = []
    seen = {pattern}

    def add(p: str, why: str) -> None:
        p = (p or "").strip()
        if p and p not in seen and p.lower() not in STOPWORDS and len(p) > 2:
            seen.add(p)
            out.append((p, why))

    # `def foo` / `class Foo` -> `foo`. Most specific, so try it first.
    add(re.sub(r"^\s*(def|class|function|const|let|var)\s+", "", pattern),
        "without the declaration keyword")

    # A phrase lifted from the question ("tax rate") never appears in code.
    words = [w for w in re.split(r"[^A-Za-z0-9_]+", pattern)
             if len(w) > 2 and w.lower() not in STOPWORDS]
    for word in sorted(words, key=len, reverse=True)[:2]:
        add(word, "single word from your pattern")

    # snake_case / CamelCase -> its components. Try several, longest first:
    # searching `compute_tax_amount` when the code defines `compute_tax` must
    # still find it, and the longest fragment is not always the right one.
    parts = [p for w in words for p in re.split(r"_|(?<=[a-z])(?=[A-Z])", w)]
    for part in sorted(parts, key=len, reverse=True)[:3]:
        add(part, "word fragment")

    return out


def _grep(
    ws: Workspace,
    pattern: str,
    path: str,
    file_glob: str,
    ignore_case: bool,
    files_only: bool,
) -> str:
    root = ws.resolve(path)
    if not root.exists():
        raise ToolError(f"path does not exist: {ws.display(root)}")

    glob = (file_glob or "").strip()
    alts = split_alternatives(pattern)
    matches, matched_files, scanned, truncated_early, per_alt = _search(
        ws, pattern, root, glob, ignore_case, files_only, alts
    )

    scope = ws.display(root) + ("/" if root.is_dir() else "")
    where = f"{scope}" + (f" (files matching {glob!r})" if glob else "")

    # Zero files scanned is not absence, it is a broken filter — and reporting
    # it as "no matches" is the worst output this tool can produce.
    if scanned == 0 and glob:
        return (
            f"ERROR: file_glob {glob!r} matched no files at all under {where}, so "
            f"nothing was searched. This says nothing about whether {pattern!r} is "
            "present. Retry with no file_glob."
        )

    if not matched_files:
        return _widened_report(ws, pattern, root, glob, files_only, where,
                               scanned, ignore_case)

    breakdown = _per_term_breakdown(alts, per_alt) if alts else ""

    if files_only:
        header = f"{len(matched_files)} file(s) contain {pattern!r} (searched {scanned}):"
        return header + breakdown + "\n" + cap(matched_files, MAX_MATCHES, "files")

    header = (
        f"{len(matches)} match(es) for {pattern!r} in {len(matched_files)} file(s) "
        f"under {where}, searched {scanned} file(s):"
    )
    body = cap(matches, MAX_MATCHES, "matches")
    if truncated_early:
        body += "\n... search stopped early: too many matches. Use files_only=true or a narrower pattern."
    return header + breakdown + "\n" + body


def _per_term_breakdown(alts: list[str], per_alt: dict[str, set[str]]) -> str:
    """Report each searched name on its own line, including the ones that missed.

    The absent ones are the reason this exists. A combined search that finds two
    of three names still returns a wall of hits, and a model reading it reports
    all three as present — a false positive built out of a true result.
    """
    lines = []
    for term in alts:
        files = per_alt.get(term) or set()
        if files:
            lines.append(f"  {term}: found in {len(files)} file(s)")
        else:
            lines.append(f"  {term}: NO matches — do not report it as present")
    return "\n" + "\n".join(lines)


def _search(ws: Workspace, pattern: str, root, glob: str,
            ignore_case: bool, files_only: bool, alts: list[str] | None = None):
    """Core scan.

    Returns (matches, matched_files, files_scanned, stopped_early, per_term),
    where `per_term` maps each alternative of an `a|b|c` pattern to the files it
    matched. Attribution happens in this same pass — searching three names stays
    one walk over the workspace, which is the whole point of allowing it.
    """
    flags = re.IGNORECASE if ignore_case else 0
    try:
        rx = re.compile(pattern, flags)
        alt_rx = [(a, re.compile(a, flags)) for a in (alts or [])]
    except re.error as exc:
        raise ToolError(
            f"invalid regular expression {pattern!r}: {exc}. "
            "Remember to escape regex characters like ( ) [ ] . * + ?"
        ) from None

    candidates = [root] if root.is_file() else list(ws.walk(root))
    matches: list[str] = []
    matched_files: list[str] = []
    per_term: dict[str, set[str]] = {a: set() for a, _ in alt_rx}
    scanned = 0
    truncated_early = False

    # `file_glob` is fnmatch, not regex, so '|' is a literal character there and
    # '*.txt|*.md' matches nothing at all. Observed: a 7B carried the 'A|B' idea
    # over from `pattern`, scanned zero files, and read the empty result as
    # proof the thing did not exist. Accept the alternation instead of silently
    # searching nothing.
    globs = [g.strip() for g in (glob or "").split("|") if g.strip()]

    for file in candidates:
        if globs and not any(fnmatch.fnmatch(file.name, g) for g in globs):
            continue
        try:
            if file.stat().st_size > MAX_SCAN_BYTES:
                continue
        except OSError:
            continue
        if looks_binary(file):
            continue

        scanned += 1
        hit_in_file = False
        shown = ws.display(file)
        try:
            with open(file, "r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    if not rx.search(line):
                        continue
                    hit_in_file = True
                    for term, term_rx in alt_rx:
                        if term_rx.search(line):
                            per_term[term].add(shown)
                    # files_only still reads to the end of a multi-term file:
                    # stopping at the first hit would attribute the file to
                    # whichever name happened to appear first and report the
                    # others as absent.
                    if files_only and not alt_rx:
                        break
                    if files_only:
                        continue
                    text = line.rstrip("\n")
                    if len(text) > MAX_LINE_CHARS:
                        text = text[:MAX_LINE_CHARS] + " ...[long line truncated]"
                    matches.append(f"{shown}:{lineno}: {text}")
                    if len(matches) >= MAX_MATCHES * 4:
                        truncated_early = True
                        break
        except OSError:
            continue

        if hit_in_file:
            matched_files.append(shown)
        if truncated_early:
            break

    return matches, matched_files, scanned, truncated_early, per_term


def _widened_report(ws: Workspace, pattern: str, root, glob: str,
                    files_only: bool, where: str, scanned: int,
                    ignore_case: bool = False) -> str:
    """Nothing matched. Retry broader patterns and report the best lead found."""
    attempted: list[str] = []

    # Case first, before any lexical variant. A model searching a constant it
    # half-remembers writes `LICENSE`, and the file says `License:`; every
    # widening of a single all-caps word is that word again, so without this the
    # tool tries nothing at all and then announces the thing does not exist.
    candidates = list(_widenings(pattern))
    if not ignore_case:
        candidates.insert(0, (pattern, "the same pattern, ignoring case"))

    for candidate, why in candidates:
        # The case-fold retry is the same string as the original, and listing it
        # as `tried 'deploy'` reads like the tool ran the identical search twice
        # and is quietly claiming more than it did. On the honesty cases this
        # sentence is the whole output, so it has to say exactly what happened.
        attempted.append(repr(candidate) + ("" if candidate != pattern
                                            else " ignoring case"))
        matches, files, seen_files, _, _ = _search(
            ws, candidate, root, glob, True, files_only
        )
        if not files:
            continue
        # A term that hits most of the repo is noise, not a lead.
        if len(files) > max(3, int(seen_files * 0.6)):
            continue

        lines = files if files_only else matches
        body = cap(lines, MAX_MATCHES, "matches")
        return (
            f"No matches for {pattern!r} in {where} (searched {scanned} file(s)).\n"
            f"A broader search for {candidate!r} ({why}) found "
            f"{len(files)} file(s) — this is very likely what you want:\n"
            f"{body}\n"
            f"Use these results. Do not report that the thing does not exist."
        )

    tried = ", ".join(attempted) if attempted else "no variants"
    head = (
        f"No matches for {pattern!r} in {where}. Searched {scanned} text file(s), "
        f"and also tried {tried} with no result."
    )

    # Nothing spelled like the query exists. Fall back to the repo's own
    # vocabulary so the model picks a real symbol instead of inventing one.
    idx = vocab.index(ws, root)

    near = vocab.near_misses(pattern, idx)
    if near:
        lines = "\n".join(f"  {name}  ({where_})" for name, where_ in near)
        return (
            f"{head}\n"
            f"These identifiers in this workspace are spelled similarly — one of "
            f"them is very likely what you meant:\n{lines}\n"
            f"Search one of these names instead."
        )

    listing = vocab.definition_listing(idx)
    if listing:
        lines = "\n".join(f"  {item}" for item in listing)
        return (
            f"{head}\n"
            f"No identifier resembles {pattern!r}, so the word you searched for is "
            f"not the word this code uses. These are the symbols this workspace "
            f"actually defines:\n{lines}\n"
            f"Pick the one that means what you are looking for and search it. "
            f"If none of them fits, then the thing genuinely does not exist here."
        )

    return (
        f"{head}\n"
        "Before concluding it does not exist: search a single short identifier, "
        "drop file_glob, or list_files to see what is actually here."
    )
