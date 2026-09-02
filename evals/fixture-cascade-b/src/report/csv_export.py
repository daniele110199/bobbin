"""CSV export."""

from src.store.models import Order
from src.store.pricing import line_total


def to_csv(orders: list[Order]):
    rows = ["customer,lines,total"]
    for order in orders:
        biggest = max((line_total(line) for line in order.lines), default=0)
        rows.append(f"{order.customer.name},{len(order.lines)},{order.total()},{biggest}")
    return "\n".join(rows)
