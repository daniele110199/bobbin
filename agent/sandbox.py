"""Workspace sandbox: every path a tool touches goes through here.

The single rule: a tool may never read outside the workspace root. Small models
hallucinate paths like "/etc/passwd" or "../../secrets" fairly often, so this is
enforced in one place rather than trusted to each tool.
"""

from __future__ import annotations

import os
from pathlib import Path

# Directories that are never worth showing a model: huge, generated, or noise.
IGNORED_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".venv", "venv", "env",
    "dist", "build", "target", ".next", ".cache", ".tox",
    ".idea", ".vscode",
}

IGNORED_FILE_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".dll", ".exe", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
    ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z", ".jar",
    ".mp3", ".mp4", ".mov", ".avi", ".woff", ".woff2", ".ttf",
    ".db", ".sqlite", ".sqlite3", ".parquet",
}

MAX_SCAN_BYTES = 2_000_000  # skip files larger than this when searching


class ToolError(Exception):
    """Raised when a tool is called with bad input. The message is shown to the
    model verbatim so it can correct itself on the next turn."""


class Workspace:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"workspace root is not a directory: {self.root}")

    def resolve(self, rel: str) -> Path:
        """Turn a model-supplied path into a real path inside the workspace."""
        if rel is None:
            rel = "."
        rel = str(rel).strip()
        if rel in ("", "."):
            return self.root

        # Expand '~' first, so '~/.ssh/id_rsa' is reported as a boundary
        # violation rather than as a missing directory literally named '~'.
        candidate = Path(rel).expanduser()
        if candidate.is_absolute():
            # Allow absolute paths only if they already point inside the root.
            target = candidate.resolve()
        else:
            target = (self.root / candidate).resolve()

        if target != self.root and self.root not in target.parents:
            raise ToolError(
                f"path {rel!r} is outside the workspace ({self.root}). "
                "Use a path relative to the workspace root."
            )
        return target

    def display(self, path: Path) -> str:
        """Path as the model should see it: relative to root, POSIX style."""
        try:
            rel = path.resolve().relative_to(self.root)
        except ValueError:
            return str(path)
        return rel.as_posix() or "."

    # -- filtering ---------------------------------------------------------

    def is_ignored_dir(self, name: str) -> bool:
        return name in IGNORED_DIRS or (name.startswith(".") and name not in {".", ".."})

    def is_ignored_file(self, path: Path) -> bool:
        return path.suffix.lower() in IGNORED_FILE_SUFFIXES

    def walk(self, start: Path):
        """Yield files under `start`, skipping ignored dirs and binary-ish files."""
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(d for d in dirnames if not self.is_ignored_dir(d))
            for name in sorted(filenames):
                p = Path(dirpath) / name
                if self.is_ignored_file(p):
                    continue
                yield p


def looks_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(8192)
    except OSError:
        return True
    return b"\x00" in chunk


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()
