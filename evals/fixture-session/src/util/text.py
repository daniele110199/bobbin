"""Text helpers."""

import re


def slugify(value):
    """Lowercase, strip punctuation, join words with hyphens."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower())
    return cleaned.strip("-")


def truncate(value, limit=40):
    """Shorten `value` to `limit` characters, marking the cut with an ellipsis."""
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"
