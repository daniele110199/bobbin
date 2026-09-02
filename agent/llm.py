"""Minimal Ollama client over the stdlib. No third-party packages."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_HOST = "http://localhost:11434"


class LLMError(RuntimeError):
    pass


def installed_models(host: str = DEFAULT_HOST, timeout: int = 5) -> list[str]:
    """Model names as `ollama list` shows them: most recently modified first.

    `/api/tags` returns them in exactly that order, so this needs no `ollama`
    binary on PATH — which matters, because the whole point of reaching for it
    is the person who just cloned this and has not read anything yet.
    """
    req = urllib.request.Request(f"{host.rstrip('/')}/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return [m["name"] for m in body.get("models", []) if m.get("name")]


def resolve_model(name: str, host: str = DEFAULT_HOST) -> str:
    """Check a model the user named against what is actually installed.

    Ollama's own 404 arrives after the banner has printed and reads like a bug
    report; this arrives before anything starts and says what to do. It also
    catches the likelier mistake, which is not a typo: `bobbin "what does this
    do?"` puts a prompt where the model goes, and a message listing three model
    names is a faster explanation than a 404.

    `ollama list` prints `foo:latest` but Ollama accepts a bare `foo`, so an
    implicit `:latest` is not a mismatch.
    """
    try:
        names = installed_models(host)
    except Exception:
        return name          # unreachable: let the request fail with its own error
    if name in names or f"{name}:latest" in names:
        return name
    listed = "\n".join(f"  {n}" for n in names) or "  (none)"
    hint = ("\n\nThat looks like a prompt rather than a model name — put it "
            "after -p." if " " in name.strip() else "")
    raise LLMError(
        f"'{name}' is not installed.\n\nYou have:\n{listed}\n\n"
        f"Pull it with `ollama pull {name}`, or run with no model to use the "
        f"first one listed.{hint}"
    )


def default_model(host: str = DEFAULT_HOST) -> str:
    """Whatever `ollama list` shows first.

    A pinned default is right for the eval harness, where a floating model would
    make two sittings incomparable, and wrong here: it made the first command a
    stranger types fail unless they happened to have pulled a specific 18GB
    model. Both failure modes below name the fix in the message, because someone
    meeting this project for the first time should not have to read the source
    to get past it.
    """
    try:
        names = installed_models(host)
    except Exception as exc:                        # not running, wrong port, ...
        raise LLMError(
            f"could not reach Ollama at {host} ({exc}). Start it with `ollama "
            f"serve`, or point elsewhere with --host, or name a model with "
            f"--model."
        ) from exc
    if not names:
        raise LLMError(
            f"Ollama at {host} has no models installed. Pull one first, e.g. "
            f"`ollama pull qwen2.5-coder:7b`."
        )
    return names[0]


@dataclass
class ToolCall:
    name: str
    arguments: dict
    call_id: str | None = None


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall]
    raw: dict
    # True when the model ignored the tool protocol and we recovered the call
    # from prose. Worth measuring: it is the clearest signal of a weak model.
    recovered_from_text: bool = False


TOOL_PROTOCOL_UNSUPPORTED = "does not support tools"


def render_tools_as_prose(tools: list[dict]) -> str:
    """Describe the tools in the prompt, for a model Ollama will not send them to.

    Some models are registered without tool support and the API rejects the
    `tools` parameter outright with HTTP 400 — `dolphin3` is one. That is not the
    same as being unable to use tools: the 7Bs here have *never* once used the
    native protocol, and every call they make is recovered from prose by
    `_parse_tool_calls_from_text`. So the schemas go in the prompt instead and
    the recovery does what it already does.

    The format asked for is the one that parser handles most reliably — a bare
    JSON object — and the descriptions are trimmed to their first line, because a
    tool description is prompt text charged at the same rate as any other, and
    this project has already paid four cases for over-describing one.
    """
    lines = [
        "TOOLS",
        "",
        "You cannot call tools through the API. To use one, reply with a single "
        "JSON object and nothing else:",
        '{"name": "grep", "arguments": {"pattern": "compute_tax"}}',
        "",
        "The result comes back as the next message. Available tools:",
    ]
    for tool in tools:
        fn = tool.get("function") or {}
        params = fn.get("parameters") or {}
        required = set(params.get("required") or [])
        args = ", ".join(
            name if name in required else f"{name}?"
            for name in (params.get("properties") or {})
        )
        summary = (fn.get("description") or "").strip().splitlines()
        lines.append(f'- {fn.get("name")}({args}) — {summary[0] if summary else ""}')
    return "\n".join(lines)


class OllamaClient:
    def __init__(self, model: str, host: str = DEFAULT_HOST,
                 num_ctx: int = 16384, temperature: float = 0.0,
                 timeout: int = 600, prose_tools: bool = False):
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        # Set on construction, latched the first time Ollama refuses the `tools`
        # parameter, or forced with `AGENT_PROSE_TOOLS=1`. Reported by the eval
        # harness, because a run where the model never saw a schema is not
        # comparable with one where it did.
        #
        # The env switch exists to separate two things that are otherwise
        # confounded: a model that scores badly on this path may be a weak model,
        # or the path itself may be worse than the API's own template. Forcing it
        # on a model with a known baseline answers that.
        self.prose_tools = prose_tools or bool(os.environ.get("AGENT_PROSE_TOOLS"))

    def build_payload(self, messages: list[dict],
                      tools: list[dict] | None = None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if tools and self.prose_tools:
            payload["messages"] = messages + [
                {"role": "system", "content": render_tools_as_prose(tools)}
            ]
        elif tools:
            payload["tools"] = tools
        return payload

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Reply:
        payload = self.build_payload(messages, tools)

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            # "<model> does not support tools" is a fact about the registry entry,
            # not about the model's ability, so fall back to describing the tools
            # in the prompt and retry once. Latched, so the cost is one request
            # per process, and recorded, so a run is never quietly incomparable.
            if (tools and not self.prose_tools
                    and TOOL_PROTOCOL_UNSUPPORTED in detail):
                self.prose_tools = True
                return self.chat(messages, tools)
            raise LLMError(f"Ollama returned HTTP {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise LLMError(
                f"cannot reach Ollama at {self.host} ({exc.reason}). "
                "Is `ollama serve` running?"
            ) from None

        message = body.get("message") or {}
        content = message.get("content") or ""
        calls = _parse_tool_calls(message)
        recovered = False

        # Some models ignore the native tool protocol and describe the call in
        # prose instead. Recover it rather than losing the turn.
        #
        # The tool names come from the schemas we just sent, never from a list
        # kept here: a hand-maintained copy would drift and start recognising
        # tools that no longer exist. When `tools` is None — the forced final
        # answer after the step budget runs out — there are no names, so the
        # loosest recovery shape is off, which is what we want there anyway.
        if not calls and content:
            calls = _parse_tool_calls_from_text(content, _schema_names(tools))
            if calls:
                content = ""
                recovered = True

        return Reply(content=content, tool_calls=calls, raw=body,
                     recovered_from_text=recovered)


def _parse_tool_calls(message: dict) -> list[ToolCall]:
    out: list[ToolCall] = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        out.append(ToolCall(
            name=fn.get("name", ""),
            arguments=args if isinstance(args, dict) else {},
            call_id=call.get("id"),
        ))
    return out


def _schema_names(tools: list[dict] | None) -> tuple[str, ...]:
    """The tool names out of the schema list handed to `chat()`."""
    out = []
    for tool in tools or []:
        name = (tool.get("function") or {}).get("name")
        if isinstance(name, str) and name:
            out.append(name)
    return tuple(out)


def _parse_tool_calls_from_text(text: str,
                                known_names: tuple[str, ...] = ()) -> list[ToolCall]:
    """Best-effort recovery of tool calls emitted as text instead of protocol.

    Small models do this constantly, and the *same* model does it inconsistently
    from one prompt to the next. Observed shapes, all handled here:
      - Qwen XML:     <function=grep><parameter=pattern>x</parameter></function>
      - Qwen JSON:    <tool_call>{"name": ...}</tool_call>
      - fenced:       ```json\\n{"name": ...}\\n```
      - buried:       "Let's search for it.\\n{"name": ..., "arguments": {...}}"
      - recited:      "1. `grep pattern=X files_only=true`"   (needs known_names)

    For the JSON shapes we scan for balanced objects anywhere in the string, then
    keep only the ones that look like a call (a name plus a dict of arguments),
    so a model quoting unrelated JSON does not trigger a phantom tool call.
    """
    xml_calls = _parse_xml_tool_calls(text)
    if xml_calls:
        return xml_calls

    candidates: list[str] = []

    if "<tool_call>" in text:
        for chunk in text.split("<tool_call>")[1:]:
            candidates.append(chunk.split("</tool_call>")[0].strip())
    else:
        candidates = _balanced_json_objects(text)

    out: list[ToolCall] = []
    for blob in candidates:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        fn = obj.get("function") if isinstance(obj.get("function"), dict) else {}
        name = obj.get("name") or fn.get("name")
        args = obj.get("arguments", obj.get("parameters", fn.get("arguments")))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if args is None:
            args = {}

        if isinstance(name, str) and name and isinstance(args, dict):
            out.append(ToolCall(name=name, arguments=args))

    # Last resort, and only with a list of real tool names to check against.
    return out or _parse_recited_tool_calls(text, known_names)


_ARG = re.compile(r"([a-z_][a-z0-9_]*)[ \t]*=[ \t]*('[^']*'|\"[^\"]*\"|[^\s`]+)")


def _parse_recited_tool_calls(text: str,
                              known_names: tuple[str, ...]) -> list[ToolCall]:
    """Recover calls the model *narrated* rather than made.

        1. `grep pattern=normalise_amount files_only=true`
        2. `read_file` the most likely hit to confirm before answering
        3. `edit_file path=a.py old_string='def f(x):' new_string='def g(x):'`

    Measured on `qwen2.5-coder:7b`: this is its single biggest failure once the
    task is imperative. It answers with the recipe block out of
    `prompts/playbook.md`, identifier substituted, having called nothing — 1 of
    22 read-only cases and 3 of 8 edit cases. The `qwen3-coder:30b` never does
    it. What it writes is a perfectly good call in a format nothing parsed.

    Two rules keep this from firing on ordinary prose, and both matter:

      * the name must be a tool that was actually offered this turn, and
      * it must be followed by at least one `key=value` pair.

    The second is doing the real work. Line 2 above is *not* recovered, and must
    not be: a bare tool name in a sentence ("`read_file` the most likely hit",
    "the grep tool lives in search.py") is prose, and turning it into an
    argument-less call would spend a step to earn a missing-argument error.
    Checked against all 60 answers in the four baseline result files: 4 calls
    recovered, every one of them in a failing case, zero in a passing one.

    Caveat worth remembering: a recited plan is an *intention*. Recovering it
    executes a call the model only said it would make — which is exactly what
    the human gate on the write tools is there to arbitrate.

    Every recited step is executed, which is the configuration the +2 above was
    measured in. Returning only the *first* — a recited plan is a sequence, and
    step 3 is written against what step 1 was expected to return — is a live idea
    that would only pay off on the 7B, so it is deliberately not built: see the
    note in README's next steps.

    Set `AGENT_NO_RECITED_CALLS=1` to switch this shape off. That exists so the
    "before" half of an A/B is the same binary as the "after" half; nothing in
    the agent sets it.
    """
    if not known_names or os.environ.get("AGENT_NO_RECITED_CALLS"):
        return []

    # Longest first so a name that is a prefix of another cannot shadow it.
    alternation = "|".join(
        re.escape(n) for n in sorted(known_names, key=len, reverse=True)
    )
    pattern = re.compile(
        r"\b(" + alternation + r")\b[ \t]+"
        r"((?:[a-z_][a-z0-9_]*[ \t]*=[ \t]*(?:'[^']*'|\"[^\"]*\"|[^\s`]+)[ \t]*)+)"
    )

    out: list[ToolCall] = []
    for name, blob in pattern.findall(text):
        args = {key: _unquote(value) for key, value in _ARG.findall(blob)}
        if args:
            out.append(ToolCall(name=name, arguments=args))
    return out


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


_XML_FUNCTION = re.compile(r"<function=([^>\s]+)\s*>(.*?)(?:</function>|\Z)", re.S)
_XML_PARAM = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)(?:</parameter>|\Z)", re.S)


def _parse_xml_tool_calls(text: str) -> list[ToolCall]:
    """Parse Qwen's XML-ish call format.

        <function=grep>
        <parameter=pattern>def compute_tax</parameter>
        <parameter=file_glob>*.py</parameter>
        </function>

    Values arrive as strings (`<parameter=depth>2</parameter>`); the registry's
    argument coercion turns them into the declared types. Closing tags are
    optional so a response truncated mid-call still yields a usable call.
    """
    out: list[ToolCall] = []
    for name, body in _XML_FUNCTION.findall(text):
        args = {
            key.strip(): value.strip()
            for key, value in _XML_PARAM.findall(body)
        }
        cleaned = name.strip().strip('"\'')
        if cleaned:
            out.append(ToolCall(name=cleaned, arguments=args))
    return out


def _balanced_json_objects(text: str) -> list[str]:
    """Every brace-balanced {...} substring in `text`, string-literal aware."""
    out: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start:i + 1])
                    start = -1
    return out
