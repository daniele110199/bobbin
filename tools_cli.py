#!/usr/bin/env python3
"""Drive the tools by hand, with no model involved.

This exists so that when the agent misbehaves you can always answer the
question "was it the tool layer or the model?" in one command:

    ./tools_cli.py list_files path=. depth=2
    ./tools_cli.py read_file path=agent/loop.py offset=1 limit=40
    ./tools_cli.py find_files pattern=*.py
    ./tools_cli.py grep pattern="def build" file_glob=*.py
    ./tools_cli.py --schemas

The web tools need `--allow-web`, exactly as the agent does:

    ./tools_cli.py --allow-web web_search query="argparse add_argument"
    ./tools_cli.py --allow-web fetch_url url=https://docs.python.org/3/

`web_search` scrapes a page it does not control, so it is the tool most likely
to break on its own one day. This is where you find out, without a model in the
way: if the search works here, the problem is the model.
"""

from __future__ import annotations

import argparse
import json
import sys

from agent.sandbox import Workspace
from agent.tools import build_registry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tool", nargs="?", help="tool name")
    ap.add_argument("args", nargs="*", help="key=value arguments")
    ap.add_argument("--root", default=".", help="workspace root (default: cwd)")
    ap.add_argument("--schemas", action="store_true",
                    help="print the JSON schemas sent to the model")
    ap.add_argument("--allow-web", action="store_true",
                    help="also register web_search and fetch_url")
    opts = ap.parse_args()

    ws = Workspace(opts.root)
    registry = build_registry(ws, allow_web=opts.allow_web)

    if opts.schemas:
        print(json.dumps(registry.schemas(), indent=2))
        return 0

    if not opts.tool:
        print("Available tools:\n")
        for name, tool in sorted(registry.tools.items()):
            params = " ".join(
                f"{p.name}" + ("" if p.required else "?") for p in tool.params
            )
            print(f"  {name} {params}")
            print(f"      {tool.description.strip().splitlines()[0]}")
        print("\nRun a tool:  ./tools_cli.py grep pattern='def build'")
        return 0

    kwargs: dict[str, str] = {}
    for item in opts.args:
        if "=" not in item:
            print(f"bad argument {item!r}, expected key=value", file=sys.stderr)
            return 2
        key, value = item.split("=", 1)
        kwargs[key] = value

    text, ok = registry.dispatch(opts.tool, kwargs)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
