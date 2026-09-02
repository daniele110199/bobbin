"""Carrier selection. Three failures in a row and a carrier is skipped."""

CARRIERS = {
    "ravenpost": {"base": 4.20, "per_kg": 0.85},
    "tortoise": {"base": 2.10, "per_kg": 1.60},
    "kestrel": {"base": 6.00, "per_kg": 0.40},
}

_failures = {name: 0 for name in CARRIERS}


def quote(name, weight_kg):
    """What `name` charges to carry `weight_kg`."""
    terms = CARRIERS[name]
    return round(terms["base"] + terms["per_kg"] * weight_kg, 2)


def cheapest(weight_kg):
    """The cheapest carrier still in the running for this weight."""
    live = [n for n in CARRIERS if _failures[n] < 3]
    return min(live, key=lambda n: quote(n, weight_kg))


def record_failure(name):
    """Count a rejected label against `name`."""
    _failures[name] += 1
    return _failures[name]
