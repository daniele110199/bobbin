"""The fetch_url tool: read a page off the web, opt-in, never in the eval registry.

Registered only when the caller passes `--allow-web`, and *not* part of either
default registry. That is not caution about the network, it is the project's own
measured rule: a tool schema is prompt text charged on every request, advertising
one has already cost this project four cases, and none of the 40-odd eval cases
can use this one — the fixtures are offline, self-contained repos. Adding it to
the default set would be pure schema cost against zero schema benefit, and would
invalidate every number on record in the bargain.

The narrow-affordance pattern, for the same reason `check_imports` replaced
`run_command`: one parameter, no shell, no interpreter, no redirect chain into
another scheme, and a hard cap on what comes back.
"""

from __future__ import annotations

import html
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse

from ..output import cap
from ..sandbox import ToolError
from .base import Param, Tool

TIMEOUT_S = 15
MAX_BYTES = 512_000
MAX_LINES = 200
USER_AGENT = "llm-agent-project/1.0 (+local coding agent)"

# Loopback and the link-local metadata address are the two that turn a "read a
# doc page" tool into a way out of the sandbox: one reaches whatever the user is
# running locally, the other is the cloud credential endpoint.
BLOCKED_HOSTS = {
    "localhost", "127.0.0.1", "::1", "0.0.0.0",
    "169.254.169.254", "metadata.google.internal",
}

_SCRIPT = re.compile(r"<(script|style)\b.*?</\1>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_BLANK = re.compile(r"\n\s*\n\s*\n+")


def build() -> list[Tool]:
    return [
        Tool(
            name="fetch_url",
            description=(
                "Fetch a web page or file over http(s) and return it as text. "
                "Use it for documentation, a changelog, an error message you do "
                "not recognise, or an API reference — anything the workspace "
                "cannot tell you. HTML is reduced to text. Nothing is executed, "
                "and the page is not saved to the workspace."
            ),
            params=[
                Param("url", "string",
                      "Absolute http:// or https:// URL to fetch.",
                      required=True),
            ],
            fn=_fetch,
        ),
    ]


def _check(url: str) -> str:
    """Validate before opening anything. Raises ToolError the model can act on."""
    url = (url or "").strip()
    if not url:
        raise ToolError("fetch_url needs a url, e.g. 'https://docs.python.org/3/'.")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolError(
            f"fetch_url only speaks http and https, not {parsed.scheme or 'a bare path'!r}. "
            "Pass an absolute URL like 'https://example.com/page'."
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise ToolError(f"{url!r} has no host in it.")
    if host in BLOCKED_HOSTS:
        raise ToolError(
            f"{host} is not reachable from this tool: it is the local machine or a "
            "cloud metadata endpoint, not the public web."
        )
    return url


def _to_text(body: bytes, content_type: str) -> str:
    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
    try:
        text = body.decode(charset, errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")
    if "html" in content_type:
        text = _SCRIPT.sub(" ", text)
        text = _TAG.sub(" ", text)
        text = html.unescape(text)
        text = "\n".join(line.strip() for line in text.splitlines())
        text = _BLANK.sub("\n\n", text)
    return text.strip()


def _fetch(url: str) -> str:
    url = _check(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        # No custom opener, so urllib's default redirect handler applies — and it
        # only ever follows http(s), which is the property that matters here.
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            content_type = (response.headers.get("Content-Type") or "").lower()
            body = response.read(MAX_BYTES + 1)
            final = response.geturl()
    except urllib.error.HTTPError as exc:
        return f"ERROR: {url} returned HTTP {exc.code} ({exc.reason})."
    except urllib.error.URLError as exc:
        return f"ERROR: could not reach {url}: {exc.reason}."
    except (TimeoutError, OSError) as exc:
        return f"ERROR: could not reach {url}: {exc}."

    truncated = len(body) > MAX_BYTES
    text = _to_text(body[:MAX_BYTES], content_type)
    if not text:
        return f"{final} returned {len(body)} bytes of {content_type or 'unknown type'}, no readable text."
    header = final if final == url else f"{final}  (redirected from {url})"
    lines = [header, ""] + text.splitlines()
    if truncated:
        lines.append(f"... truncated at {MAX_BYTES} bytes.")
    return cap(lines, MAX_LINES, "lines")
