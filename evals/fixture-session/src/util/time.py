"""Time helpers. Seconds since the epoch, no timezone handling."""

SECONDS_PER_DAY = 86400


def days_between(earlier, later):
    """Whole days between two epoch timestamps."""
    return int((later - earlier) // SECONDS_PER_DAY)


def is_stale(timestamp, now, ttl):
    """True once `timestamp` is older than `ttl` seconds."""
    return (now - timestamp) >= ttl
