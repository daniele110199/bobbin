"""HTTP client with a retry budget."""

from src.core.config import RETRY_LIMIT


def fetch(url, attempt=1):
    if attempt > RETRY_LIMIT:
        raise RuntimeError("gave up on " + url)
    return {"url": url, "attempt": attempt}
