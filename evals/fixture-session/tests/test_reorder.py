"""Reorder policy tests."""

from src.core.config import MAX_BATCH
from src.inventory.reorder import order_size


def test_order_size_is_capped():
    assert order_size("anything") <= MAX_BATCH


def test_order_size_is_never_negative():
    assert order_size("anything") >= 0
