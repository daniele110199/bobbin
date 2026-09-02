"""Reorder policy: what is running low, and how much to ask for."""

from src.core.config import REORDER_THRESHOLD, MAX_BATCH
from src.inventory.items import all_items
from src.inventory.stock import level_for


def reorder_due():
    """Every sku sitting at or below REORDER_THRESHOLD."""
    return [item["sku"] for item in all_items()
            if level_for(item["sku"]) <= REORDER_THRESHOLD]


def order_size(sku):
    """How many units to order for `sku`, capped at MAX_BATCH."""
    shortfall = REORDER_THRESHOLD * 3 - level_for(sku)
    return min(max(shortfall, 0), MAX_BATCH)
