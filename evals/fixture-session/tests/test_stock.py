"""Stock level tests."""

from src.inventory.stock import adjust, level_for


def test_level_for_unstocked_sku_is_zero():
    assert level_for("no-such-sku") == 0


def test_adjust_moves_the_level():
    assert adjust("widget", 5) == level_for("widget") + 5
