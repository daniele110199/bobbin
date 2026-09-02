"""The database handle. One connection, opened lazily."""

from src.core.config import DATABASE_URL

_connection = None


def connect():
    """Return the process-wide connection, opening it on first use."""
    global _connection
    if _connection is None:
        _connection = {"url": DATABASE_URL, "open": True}
    return _connection


def rows(table):
    """Every row in `table`. The fixture has no real driver behind it."""
    return connect().get(table, [])
