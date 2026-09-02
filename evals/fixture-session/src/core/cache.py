"""A dictionary with a lifetime, used for stock levels."""

from src.core.config import CACHE_TTL

_entries = {}


def put(key, value, now):
    """Store `value` under `key`, expiring at `now + CACHE_TTL`."""
    _entries[key] = (value, now + CACHE_TTL)


def get(key, now):
    """Return the value for `key`, or None once it has expired."""
    hit = _entries.get(key)
    if hit is None or hit[1] <= now:
        return None
    return hit[0]


def drop(key):
    """Invalidate one key. Called on every write to a stock level."""
    _entries.pop(key, None)
