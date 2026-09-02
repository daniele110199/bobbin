"""Request handling."""

from src.core.config import PAGE_SIZE


def paginate(items):
    return [items[i:i + PAGE_SIZE] for i in range(0, len(items), PAGE_SIZE)]
