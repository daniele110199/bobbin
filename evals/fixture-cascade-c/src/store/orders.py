"""Order placement."""

from src.store.models import Order
from src.store.pricing import apply_discount


def place_order(customer, lines, percent=0):
    if len(lines) > MAX_ITEMS:
        raise ValueError("too many items")
    order = Order(customer, lines)
    return {"order": order, "due": apply_discount(order, percent)}
