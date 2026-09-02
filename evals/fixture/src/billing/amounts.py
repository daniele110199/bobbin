"""Amount helpers."""


def normalise_amount(value):
    """Round a monetary value to two decimals."""
    return round(float(value), 2)
