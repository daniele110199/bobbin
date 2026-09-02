"""Price calculations."""

from src.config import TAX_RATE


def line_total(line):
    """The price of one order line."""
    _, price = line
    return price


def apply_discount(order, percent):
    """Return the order total after `percent` off, tax included."""
    net = order.total() * (1 - percent / 100)
    return round(net * (1 + TAX_RATE), 2)
