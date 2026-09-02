"""Item records — the catalogue side of inventory."""

from src.core.db import rows


def all_items():
    """Every item record known to the warehouse."""
    return rows("items")


def describe(sku):
    """A one-line description of `sku`, or a placeholder for a stray sku."""
    for item in all_items():
        if item["sku"] == sku:
            return f"{item['sku']}: {item['name']}"
    return f"{sku}: unlisted item"
