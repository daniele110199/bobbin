"""Resolve a download path inside the export directory."""
from pathlib import Path

EXPORT_ROOT = Path("/srv/exports").resolve()


def export_path(user_supplied_name):
    candidate = (EXPORT_ROOT / user_supplied_name).resolve()
    if not candidate.is_relative_to(EXPORT_ROOT):
        raise ValueError("outside export root")
    return candidate
