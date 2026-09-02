"""A dictionary with a lifetime."""

from src.core.config import CACHE_TTL

_entries = {}


def put(key, value, now):
    """Store `value`, expiring at `now + CACHE_TTL`."""
    _entries[key] = (value, now + CACHE_TTL)


def get(key, now):
    """Return the value for `key`, or None once it has expired."""
    hit = _entries.get(key)
    return None if hit is None or hit[1] <= now else hit[0]
