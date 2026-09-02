"""Tax calculation rules."""

STANDARD_RATE = 0.22
REDUCED_RATE = 0.10


def compute_tax(amount, reduced=False):
    """Return the tax owed on `amount`."""
    rate = REDUCED_RATE if reduced else STANDARD_RATE
    return round(amount * rate, 2)
