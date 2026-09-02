"""Human-readable order summaries."""

from src.config import CURRENCY
from src.store.models import Order
from src.store.pricing import apply_discount


def render_summary(order: Order, percent=0):
    due = apply_discount(order, percent)
    return f"{order.customer.name}: {due} {CURRENCY}"
