"""Stock levels. Every write invalidates the item's cache key."""

from src.core import cache
from src.core.db import rows


def level_for(sku):
    """The current level for `sku`, from cache when it is warm.

    A sku the warehouse does not stock reads as zero.
    """
    cached = cache.get(sku, now=0)
    if cached is not None:
        return cached
    for row in rows("stock"):
        if row["sku"] == sku:
            return row["level"]
    return 0


def adjust(sku, delta):
    """Move a level by `delta` and drop the stale cache entry."""
    new_level = level_for(sku) + delta
    cache.drop(sku)
    return new_level
