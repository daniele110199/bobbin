"""Token auth. Every route goes through `require_token`."""

from src.core.config import TOKEN_HEADER


def require_token(handler):
    """Reject a request that carries no token, then call `handler`."""
    def wrapped(request):
        if not getattr(request, TOKEN_HEADER, None):
            raise PermissionError("missing token")
        return handler(request)
    return wrapped
