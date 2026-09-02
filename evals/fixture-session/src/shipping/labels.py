"""Label booking."""

from src.shipping.carriers import cheapest, quote, record_failure
from src.util.text import slugify


def book(order_id, weight_kg):
    """Book a label for `order_id` with whichever carrier is cheapest."""
    carrier = cheapest(weight_kg)
    return {
        "reference": slugify(f"{carrier} {order_id}"),
        "carrier": carrier,
        "price": quote(carrier, weight_kg),
    }


def reject(carrier, reason):
    """A carrier refused the label; count it and report the new total."""
    return {"carrier": carrier, "reason": reason,
            "failures": record_failure(carrier)}
