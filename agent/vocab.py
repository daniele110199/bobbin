"""The workspace's own vocabulary, used to rescue searches that found nothing.

When a search fails there are two different misses, and they need different
answers:

  lexical  — `normalize` vs the code's `normalise_amount`, or a typo.
             Fixed by string similarity against identifiers that really exist.
  semantic — the question says "authenticate", the code says `sign_in`.
             No string similarity will ever bridge that.

For the semantic case we do not ask a model to invent synonyms — it would grep
words that do not exist. We show it the symbols this repo actually defines and
let it choose. The model supplies the meaning; the repo supplies the candidates,
so a suggestion can never be hallucinated.
"""

from __future__ import annotations

import difflib
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .sandbox import Workspace, looks_binary

# def/class (py), function/const/let/var/class (js/ts), plus MODULE_CONSTANT = x
DEFINITION_RE = re.compile(
    r"^\s*(?:async\s+)?(?:def|class|function|interface|type|struct|fn)\s+([A-Za-z_]\w{2,})"
    r"|^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_]\w{2,})"
    r"|^([A-Z][A-Z0-9_]{2,})\s*=",
    re.M,
)
IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w{2,}")

MAX_FILES = 400
MAX_SYMBOLS_SHOWN = 40
NEAR_MISS_CUTOFF = 0.8   # strict: a genuinely absent term must stay absent


@dataclass
class Index:
    #: symbol name -> file it is defined in
    symbols: dict[str, str] = field(default_factory=dict)
    #: every identifier seen anywhere, for near-miss matching
    words: set[str] = field(default_factory=set)


_CACHE: dict[str, Index] = {}
# The eval runner can run cases in threads (`--jobs`), all sharing one fixture
# root. Without this they each build the same index at once on a cold cache.
_CACHE_LOCK = threading.Lock()


def index(ws: Workspace, root: Path) -> Index:
    """Build (and cache) the vocabulary under `root`.

    Cached per root for the process lifetime. The tools here are read-only, so
    a stale index cannot cause a wrong edit — at worst a suggestion goes out of
    date within one session.
    """
    key = str(root)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

    idx = Index()
    for count, file in enumerate(ws.walk(root)):
        if count >= MAX_FILES:
            break
        try:
            if file.stat().st_size > 400_000 or looks_binary(file):
                continue
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        shown = ws.display(file)
        for match in DEFINITION_RE.finditer(text):
            name = match.group(1) or match.group(2) or match.group(3)
            if name and not name.startswith("_"):  # skip __init__ and privates
                idx.symbols.setdefault(name, shown)
        idx.words.update(IDENTIFIER_RE.findall(text))

    with _CACHE_LOCK:
        return _CACHE.setdefault(key, idx)


def query_terms(pattern: str) -> list[str]:
    """The searchable words inside a failed pattern."""
    from .tools.search import STOPWORDS
    return [w for w in re.split(r"[^A-Za-z0-9_]+", pattern)
            if len(w) > 2 and w.lower() not in STOPWORDS]


def near_misses(pattern: str, idx: Index, limit: int = 6) -> list[tuple[str, str]]:
    """Identifiers that look like the query but are spelled differently.

    Compared against each symbol's *parts* as well as its whole name: matching
    `normalize` against `normalise_amount` scores far too low as a whole string
    (length dominates the ratio), but `normalize` vs the `normalise` fragment is
    an obvious hit.
    """
    names = list(idx.symbols) or list(idx.words)

    # comparable token -> the full symbol name it belongs to
    tokens: dict[str, str] = {}
    for name in names:
        tokens.setdefault(name.lower(), name)
        for part in re.split(r"_|(?<=[a-z])(?=[A-Z])", name):
            if len(part) > 3:
                tokens.setdefault(part.lower(), name)

    found: dict[str, None] = {}
    for term in query_terms(pattern):
        low = term.lower()
        # Substring either way: `compute` -> `compute_tax`, `slugifying` -> `slugify`
        for key, original in tokens.items():
            if len(low) > 3 and (low in key or key in low):
                found.setdefault(original, None)
        for hit in difflib.get_close_matches(
            low, list(tokens), n=limit, cutoff=NEAR_MISS_CUTOFF
        ):
            found.setdefault(tokens[hit], None)

    return [(n, idx.symbols.get(n, "")) for n in list(found)[:limit]]


def definition_listing(idx: Index, limit: int = MAX_SYMBOLS_SHOWN) -> list[str]:
    """A sample of what this repo actually defines, for the semantic case."""
    items = sorted(idx.symbols.items())
    lines = [f"{name}  ({where})" for name, where in items[:limit]]
    if len(items) > limit:
        lines.append(f"... and {len(items) - limit} more symbols")
    return lines
