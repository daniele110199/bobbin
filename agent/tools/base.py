"""Tool registry + the argument-coercion layer.

Small models get JSON *shape* right but JSON *types* wrong all the time:
they send "10" instead of 10, "true" instead of true, or invent extra keys.
Rejecting those wastes a whole turn. Instead we coerce what we can and give a
precise error only when we truly can't proceed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..sandbox import ToolError


@dataclass
class Param:
    name: str
    type: str                      # "string" | "integer" | "boolean"
    description: str
    required: bool = False
    default: Any = None
    enum: list[str] | None = None


@dataclass
class Tool:
    name: str
    description: str
    params: list[Param]
    fn: Callable[..., str]
    # Callable either way; `False` keeps it out of the schema list sent to the
    # model. A schema is prompt text charged on every request, so a tool nothing
    # ever calls is a standing tax — and this project has measured a tool
    # description costing four cases. See `undo_edit` in edit.py.
    advertised: bool = True

    def schema(self) -> dict:
        """OpenAI / Ollama function-calling schema."""
        props: dict[str, dict] = {}
        for p in self.params:
            spec: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                spec["enum"] = p.enum
            props[p.name] = spec
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": [p.name for p in self.params if p.required],
                },
            },
        }

    def call(self, raw_args: dict | None) -> str:
        return self.fn(**self._coerce(raw_args or {}))

    def _coerce(self, raw: dict) -> dict:
        by_name = {p.name: p for p in self.params}
        clean: dict[str, Any] = {}

        for key, value in raw.items():
            p = by_name.get(key)
            if p is None:
                continue  # silently drop hallucinated params
            clean[key] = _coerce_value(p, value)

        for p in self.params:
            if p.name in clean:
                continue
            if p.required:
                raise ToolError(
                    f"missing required argument {p.name!r} for tool {self.name!r}"
                )
            clean[p.name] = p.default
        return clean


def _coerce_value(p: Param, value: Any) -> Any:
    if value is None:
        return p.default

    if p.type == "integer":
        if isinstance(value, bool):
            raise ToolError(f"{p.name} must be an integer, got a boolean")
        if isinstance(value, int):
            return value
        try:
            return int(str(value).strip())
        except ValueError:
            raise ToolError(f"{p.name} must be an integer, got {value!r}") from None

    if p.type == "boolean":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
        raise ToolError(f"{p.name} must be true or false, got {value!r}")

    text = value if isinstance(value, str) else str(value)
    if p.enum and text not in p.enum:
        raise ToolError(f"{p.name} must be one of {p.enum}, got {text!r}")
    return text


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self.tools.get(name)
        if tool is None:
            known = ", ".join(sorted(self.tools)) or "(none)"
            raise ToolError(f"unknown tool {name!r}. Available tools: {known}")
        return tool

    def schemas(self) -> list[dict]:
        """What the model is told about. Unadvertised tools stay dispatchable —
        a REPL user, or a recovered prose call, can still reach them."""
        return [t.schema() for t in self.tools.values() if t.advertised]

    def subset(self, names: list[str]) -> "Registry":
        """A registry with only `names` in it, for a phase that may not use the rest.

        A tool a phase must not use is better removed than forbidden in prose: a
        7B ignores "do not call list_files" and then burns a step of a four-step
        budget on it. What is not in the schema list cannot be called.
        """
        missing = [n for n in names if n not in self.tools]
        if missing:
            known = ", ".join(sorted(self.tools)) or "(none)"
            raise ToolError(
                f"cannot restrict to unknown tool(s): {', '.join(missing)}. "
                f"Available tools: {known}"
            )
        return Registry({n: self.tools[n] for n in names})

    def dispatch(self, name: str, args: dict | None) -> tuple[str, bool]:
        """Run a tool. Returns (text_for_the_model, ok).

        Errors are returned as text, not raised: the model needs to *read* the
        error and retry. A crashing tool would end the conversation instead.
        """
        try:
            return self.get(name).call(args), True
        except ToolError as exc:
            return f"ERROR: {exc}", False
        except Exception as exc:  # noqa: BLE001 - surface anything to the model
            return f"ERROR: {type(exc).__name__}: {exc}", False
