"""Static import checking: does every `from x import y` still find its `y`?

This is the narrow replacement for `run_command`, which was built, measured, and
removed (it cost the cascade suite one to two cases and let the model delete a
file with `python3 -c "import os; os.remove(...)"`, bypassing the diff, the human
gate and the undo journal all at once).

The failure worth catching does not need a running program:

    src/util/text.py   def slugify   -> renamed to make_slug
    src/api/feed.py    from src.util.text import slugify   <- now a lie

That is a fact about two syntax trees, so `ast` answers it and **nothing is
executed** — which matters twice over. Importing a module runs its top level, so
a checker that imports is a checker that can delete your files; and the machine
this project runs on has no pytest, so every "run the tests" affordance offered
to the model so far has been a step spent learning that the tests cannot run.

The mirror failure, found by the confirmation runs on `cascade-delete-symbol`:

    src/config.py          MAX_ITEMS = 50                  -> deleted
    src/store/orders.py    from src.config import MAX_ITEMS  <- deleted too
    src/store/orders.py    if len(lines) > MAX_ITEMS:      <- left behind

That tree raises `NameError`, and both existing guards were silent about it and
right to be: there is no import left to resolve, so the check above has nothing
to say, and `unfinished_reasons()` only fires on files the run never touched.
`undefined_names()` closes that gap — a name *used* where nothing binds it.

What it does not do, deliberately: no type checking, no call-signature checking,
no cross-checking of attribute access. Those need inference, inference needs to
be right, and a checker that reports plausible nonsense is worse than none —
this project has already measured what happens when a tool result reads
convincingly and is wrong.
"""

from __future__ import annotations

import ast
import builtins
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .edits import definition_pattern

# Names that are always available without being bound in the file. The module
# dunders are the ones an import system supplies; without them every
# `__file__` in a repo reads as undefined.
_ALWAYS_BOUND = frozenset(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__", "__path__", "__all__",
    "__class__",  # bound implicitly inside a class body's methods
}

# A module using any of these can bind names in ways no parser can follow, so it
# is not judged at all. Silence on a file we cannot reason about is the whole
# design rule here: one false positive teaches a model to ignore the tool.
_DYNAMIC = frozenset({"exec", "eval", "globals", "locals", "vars"})


@dataclass
class Problem:
    path: str      # workspace-relative
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def module_name(root: Path, file: Path) -> str:
    """`src/util/text.py` -> `src.util.text`, `src/util/__init__.py` -> `src.util`."""
    rel = file.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def top_level_names(tree: ast.Module, include_imports: bool = True) -> set[str]:
    """Every name a module binds at import time.

    Conditional imports (`try: import fast except ImportError: import slow`) bind
    names inside an `if`/`try` body, so those are walked too — missing them would
    report a working import as broken, and one false positive is enough to teach
    a model to ignore the tool.

    `include_imports=False` gives the names this module *defines* rather than
    re-exports, which is what makes "and nothing in the workspace defines it"
    below a fact rather than a guess about a chain of re-imports.
    """
    names: set[str] = set()

    def visit(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if not include_imports:
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        names.update(e.id for e in target.elts if isinstance(e, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Try):
                visit(node.body)
                visit(node.orelse)
                visit(node.finalbody)
                for handler in node.handlers:
                    visit(handler.body)
    visit(tree.body)
    return names


def bound_names(tree: ast.Module) -> set[str]:
    """Every name this module binds *anywhere*, in any scope.

    Deliberately scope-blind. Python's real rules would say a name bound in one
    function is not available in another, and honouring that would find more
    bugs — and would also report a name as undefined the moment this walk missed
    a binding form. Collapsing every scope into one set can only ever produce
    false negatives, which is the direction this project errs in: a missed
    `NameError` costs one case, a wrong accusation costs the tool's credibility.

    Store and Del contexts carry most of the work, which covers assignment,
    `for` targets, `with ... as`, walrus and comprehensions without needing a
    branch for each.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load):
                names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            if node.name:
                names.add(node.name)
        elif isinstance(node, ast.MatchMapping):
            if node.rest:
                names.add(node.rest)
        elif node.__class__.__name__ in ("TypeVar", "ParamSpec", "TypeVarTuple"):
            names.add(node.name)  # PEP 695 type parameters
    return names


def undefined_names(tree: ast.Module) -> list[tuple[str, int]]:
    """Names this module loads but never binds, as (name, first line) pairs.

    Returns nothing at all for a module that could bind names dynamically — a
    star-import brings in an unknown set, and `exec`/`globals()` can bind
    anything. A file this cannot reason about gets silence, not a guess.
    """
    loads: dict[str, int] = {}
    dynamic = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            dynamic = True
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in _DYNAMIC:
                dynamic = True
            loads.setdefault(node.id, node.lineno)
    if dynamic:
        return []
    known = bound_names(tree) | _ALWAYS_BOUND
    return sorted(((name, line) for name, line in loads.items() if name not in known),
                  key=lambda pair: (pair[1], pair[0]))


def _resolve(module: str, level: int, source: str, is_package: bool) -> str:
    """Turn a possibly-relative import into a module name we may know about.

    `from .text import x` means `pkg.text` in `pkg/views.py`, and also in
    `pkg/__init__.py` — a package's own `__init__` is *inside* the package, so it
    counts one level differently from a module beside it. Getting this backwards
    reports every working relative import as broken.
    """
    if not level:
        return module
    parts = source.split(".") if source else []
    base = parts if is_package else parts[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([p for p in base if p] + ([module] if module else []))


def _candidates(target: str, level: int, source: str,
                exports: dict[str, set[str]]) -> list[str]:
    """Every workspace module an absolute `from X import ...` might mean.

    A relative import has exactly one meaning, but an absolute one has two when a
    sibling module shares the name:

        google_maps/scheduler.py:  try:    from .config import CATEGORIES
                                   except ImportError:
                                           from config import CATEGORIES

    Run as a package that is the root `config.py`; run as a script from inside
    the directory it is `google_maps/config.py`, because `sys.path[0]` is the
    script's own folder. Both are live, so a name found in *either* is not
    missing — checking only the root reported 22 working imports as broken in a
    real repo, which is precisely the false-positive class this module exists to
    avoid.
    """
    out = [target] if target in exports else []
    if not level and "." in source:
        sibling = f"{source.rsplit('.', 1)[0]}.{target}"
        if sibling in exports and sibling not in out:
            out.append(sibling)
    return out


def definition_sites(ws, name: str) -> list[str]:
    """Every workspace file that defines `name` as a function or class."""
    pattern = re.compile(definition_pattern(name), re.M)
    out = []
    for file in sorted(ws.walk(ws.root)):
        if file.suffix != ".py":
            continue
        try:
            if pattern.search(file.read_text(encoding="utf-8", errors="replace")):
                out.append(ws.display(file))
        except OSError:
            continue
    return out


def files_mentioning(ws, name: str) -> list[str]:
    """Every workspace file that still contains `name` as a whole word."""
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    out = []
    for file in sorted(ws.walk(ws.root)):
        try:
            if pattern.search(file.read_text(encoding="utf-8", errors="replace")):
                out.append(ws.display(file))
        except OSError:
            continue
    return out


def check_workspace(ws, limit: int = 12) -> tuple[list[Problem], int]:
    """Parse every Python file under `ws` and check its intra-workspace imports.

    Returns (problems, files checked). Imports of anything outside the workspace
    — the standard library, an installed package — are ignored: this cannot know
    what they export, and guessing would produce exactly the confident nonsense
    the docstring above refuses to produce.
    """
    trees: dict[str, ast.Module] = {}
    paths: dict[str, str] = {}
    packages: set[str] = set()
    problems: list[Problem] = []
    checked = 0

    files = [f for f in ws.walk(ws.root) if f.suffix == ".py"]
    for file in sorted(files):
        rel = ws.display(file)
        checked += 1
        try:
            source = file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            problems.append(Problem(rel, 1, f"cannot be read: {exc}"))
            continue
        try:
            name = module_name(ws.root, file)
            trees[name] = ast.parse(source, filename=rel)
            paths[name] = rel
            if file.name == "__init__.py":
                packages.add(name)
        except SyntaxError as exc:
            problems.append(Problem(rel, exc.lineno or 1,
                                    f"is not valid Python: {exc.msg}"))

    exports = {name: top_level_names(tree) for name, tree in trees.items()}
    defines = {name: top_level_names(tree, include_imports=False)
               for name, tree in trees.items()}

    for name, tree in trees.items():
        rel = paths[name]
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _resolve(node.module or "", node.level or 0, name,
                              name in packages)
            candidates = _candidates(target, node.level or 0, name, exports)
            if not candidates:
                continue  # not ours to judge
            for alias in node.names:
                if alias.name == "*":
                    continue
                # `from pkg import submodule` is a module, not a name in pkg.
                if any(alias.name in exports[c] or f"{c}.{alias.name}" in exports
                       for c in candidates):
                    continue
                problems.append(Problem(
                    rel, node.lineno,
                    f"imports {alias.name!r} from {target}, which does not define "
                    f"it (see {paths[candidates[0]]})",
                ))

    problems.sort(key=lambda p: (p.path, p.line))
    problems += _undefined_problems(trees, paths, defines)
    return problems[:limit], checked


# Caps, per file and overall. A module that legitimately relies on something this
# cannot see should cost one line of noise, not push a real broken import out of
# the caller's problem budget — which is also why these are appended after the
# import problems rather than sorted in among them.
_MAX_UNDEFINED_PER_FILE = 3
_MAX_UNDEFINED = 6


def _undefined_problems(trees: dict[str, ast.Module], paths: dict[str, str],
                        defines: dict[str, set[str]]) -> list[Problem]:
    """Uses of names that nothing binds — the deleted-import failure.

    `AGENT_NO_UNDEFINED_CHECK=1` switches it off, so the "before" arm of its A/B
    is the same binary as the "after" arm.
    """
    if os.environ.get("AGENT_NO_UNDEFINED_CHECK"):
        return []
    out: list[Problem] = []
    for module in sorted(trees):
        found = undefined_names(trees[module])[:_MAX_UNDEFINED_PER_FILE]
        for name, line in found:
            # Where the workspace defines it, if anywhere. That distinction is
            # the difference between "you deleted the import" and "you deleted
            # the definition", and the model needs it to pick the right repair.
            elsewhere = sorted(paths[m] for m, names in defines.items()
                               if name in names and m != module)
            where = (f"defined in {elsewhere[0]} but not imported here"
                     if elsewhere else "and nothing in the workspace defines it")
            out.append(Problem(
                paths[module], line,
                f"uses {name!r}, which is not defined or imported in this file "
                f"({where})",
            ))
    return out[:_MAX_UNDEFINED]
