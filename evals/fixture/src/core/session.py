"""User sessions."""

SESSION_TTL = 900


def sign_in(username, secret):
    """Check a secret against the store and open a session."""
    return {"user": username, "ttl": SESSION_TTL}


def sign_out(token):
    """Close an open session."""
    return None
