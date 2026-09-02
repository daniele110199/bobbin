"""Text helpers shared by the API layer."""

import re

_PUNCT = re.compile(r"[^a-z0-9]+")


def slugify(value):
    """Turn a title into a URL slug."""
    return _PUNCT.sub("-", value.strip().lower()).strip("-")
