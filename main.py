#!/usr/bin/env python3
"""Chat with a local model that can actually use the tools.

    bobbin                                    # first model in `ollama list`
    bobbin qwen3-coder:30b                    # or name one
    bobbin --root ~/some/repo
    bobbin -p "what does this project do?"    # one-shot, no REPL
    bobbin --allow-edits                      # let it write, asking first
    bobbin --allow-edits --dry-run            # show the diffs, apply nothing
    bobbin --mode research -p "trace what happens when the CLI runs"
"""

from __future__ import annotations

import argparse
import json
import select
import signal
import sys
from pathlib import Path

from agent.edits import EditSession
from agent.llm import LLMError, OllamaClient, default_model, resolve_model
from agent.loop import Agent, Trace
from agent.persist import load_session, restore_session, save_session
from agent.research import ResearchAgent
from agent.sandbox import Workspace
from agent.tools import build_registry
from agent.tools.web import origin

DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RED = "\033[31m"
GREEN = "\033[32m"
BOLD = "\033[1m"
RESET = "\033[0m"


def colour_diff(diff: str) -> str:
    out = []
    for line in diff.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            out.append(f"{GREEN}{line}{RESET}")
        elif line.startswith("-") and not line.startswith("---"):
            out.append(f"{RED}{line}{RESET}")
        elif line.startswith("@@"):
            out.append(f"{CYAN}{line}{RESET}")
        else:
            out.append(f"{DIM}{line}{RESET}")
    return "\n".join(out)


def make_approver(dry_run: bool, assume_yes: bool):
    """The human gate. The model makes one call; the decision is the user's.

    On a non-tty (piped input, `-p` with no terminal) there is nobody to ask, so
    it refuses rather than applying unattended writes.
    """
    def approve(path: str, diff: str) -> bool:
        print(f"\n{BOLD}proposed change to {path}{RESET}")
        print(colour_diff(diff))
        if dry_run:
            print(f"{DIM}--dry-run: not applied{RESET}", flush=True)
            return False
        if assume_yes:
            print(f"{DIM}--yes: applied{RESET}", flush=True)
            return True
        if not sys.stdin.isatty():
            print(f"{DIM}not a terminal, cannot confirm: not applied{RESET}", flush=True)
            return False
        try:
            reply = input(f"{BOLD}apply this change? [y/N] {RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return reply in ("y", "yes")

    return approve


def make_post_approver(dry_run: bool, assume_yes: bool, ask=None):
    """The human gate for `http_post`, and the same bargain as the write gate.

    The model composes the address and the body; the user is the one who decides
    whether it leaves the machine. Both are printed in full (the body truncated
    only for display) because "approve a POST" means nothing if you cannot see
    what is being posted.

    On a non-tty it refuses, exactly as the write gate does. That is the whole
    difference from `fetch_url`, which deliberately has no gate: a GET can be
    repeated and walked page by page, and a POST cannot be taken back.

    **Answering `a` grants the origin for the rest of the session**, which is
    what makes an iterate-on-an-API loop usable: an agent querying one GraphQL
    endpoint five times should not ask five times. The grant is per *origin* and
    not per host, so a yes to `https://api.example.com` is not a yes to the
    cleartext version of it, and it is held in memory only — a resumed session
    starts with nothing granted, because an approval given days ago in another
    sitting is not consent for this one.

    It is narrower than `--yes`, which is the point: `--yes` approves everything,
    including an address the model just invented. A standing grant approves the
    one endpoint the user has actually looked at. A request that rides a standing
    grant still prints in full — a POST nobody can see is worse than a prompt.

    `ask` is injectable so the policy can be tested without a terminal.
    """
    standing: set[str] = set()

    def prompt(where: str) -> str:
        if not sys.stdin.isatty():
            print(f"{DIM}not a terminal, cannot confirm: not sent{RESET}", flush=True)
            return "n"
        try:
            return input(f"{BOLD}send this request? "
                         f"[y = once / a = always to {where} / N = no] "
                         f"{RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "n"

    ask_user = prompt if ask is None else ask

    def approve(url: str, body: str, content_type: str) -> bool:
        where = origin(url)
        shown = body if len(body) <= 2000 else body[:2000] + "\n... (truncated for display)"
        print(f"\n{BOLD}proposed POST to {url}{RESET}")
        print(f"{DIM}{content_type}, {len(body.encode('utf-8'))} bytes{RESET}")
        print(shown)
        if dry_run:
            print(f"{DIM}--dry-run: not sent{RESET}", flush=True)
            return False
        if assume_yes:
            print(f"{DIM}--yes: sent{RESET}", flush=True)
            return True
        if where in standing:
            print(f"{DIM}{where} approved earlier this session: sent{RESET}", flush=True)
            return True
        reply = ask_user(where)
        if reply in ("a", "always"):
            standing.add(where)
            print(f"{DIM}{where} approved for the rest of this session{RESET}", flush=True)
            return True
        return reply in ("y", "yes")

    return approve


def make_steer_poll():
    """Hand the loop anything the user typed while the model was working.

    The terminal is in canonical mode, so a line the user types during a turn is
    buffered until they press Enter and `select` then reports the descriptor
    readable — no raw mode, no thread, and nothing to clean up. Returns None when
    there is nothing to say, which is almost always.
    """
    def poll() -> str | None:
        if not sys.stdin.isatty():
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        return sys.stdin.readline().strip() or None
    return poll


def make_trace(verbose: bool) -> Trace:
    def on_call(name: str, args: dict) -> None:
        pretty = ", ".join(f"{k}={v!r}" for k, v in args.items() if v not in (None, ""))
        print(f"{CYAN}→ {name}({pretty}){RESET}", flush=True)

    def on_result(name: str, text: str, ok: bool) -> None:
        lines = text.splitlines()
        color = "" if ok else RED
        if verbose:
            body = text
        else:
            body = "\n".join(lines[:6])
            if len(lines) > 6:
                body += f"\n  {DIM}... {len(lines) - 6} more lines (use -v to see all){RESET}"
        print(f"{color}{DIM}{body}{RESET}\n", flush=True)

    def on_thinking(text: str) -> None:
        print(f"{DIM}{text}{RESET}", flush=True)

    def on_phase(label: str) -> None:
        print(f"{BLUE}▸ {label}{RESET}", flush=True)

    return Trace(on_tool_call=on_call, on_tool_result=on_result,
                 on_thinking=on_thinking, on_phase=on_phase)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default=None,
                    help="Ollama model to use, e.g. qwen3-coder:30b "
                         "(default: whatever `ollama list` shows first)")
    ap.add_argument("--model", dest="model_flag", default=None,
                    help=argparse.SUPPRESS)   # the older spelling, still honoured
    ap.add_argument("--root", default=".", help="workspace root the agent may read")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="fixed step budget (default: scale it to the task)")
    ap.add_argument("-p", "--prompt", help="run one prompt and exit")
    ap.add_argument("--playbook", default="default",
                    help="'default', 'none', or a path to a playbook .md file")
    ap.add_argument("--mode", default="direct", choices=["direct", "research"],
                    help="'direct' = one loop; 'research' = survey/plan/subagents/synthesise")
    ap.add_argument("--subtasks", type=int, default=3,
                    help="research mode: how many subtasks to plan (default 3)")
    ap.add_argument("--gather-steps", type=int, default=4,
                    help="research mode: tool steps per subagent (default 4)")
    ap.add_argument("--allow-edits", action="store_true",
                    help="register write_file/edit_file/undo_edit. Every change "
                         "shows you a diff and asks before it is applied")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --allow-edits: print every proposed diff but "
                         "apply none of them")
    ap.add_argument("--yes", action="store_true",
                    help="with --allow-edits: apply without asking. It can "
                         "overwrite anything under --root")
    ap.add_argument("--allow-web", action="store_true",
                    help="register web_search and fetch_url, so the agent can "
                         "look up and read a page off the web when the workspace "
                         "cannot answer. Off by default: a tool schema is prompt "
                         "charged on every request, and no eval case can use these")
    ap.add_argument("--allow-post", action="store_true",
                    help="additionally register http_post, for data behind an "
                         "endpoint that only answers POST. Implies --allow-web. "
                         "Every request is shown to you in full and sent only if "
                         "you approve it; answer 'a' to allow that one endpoint "
                         "for the rest of the session")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print full tool output")
    ap.add_argument("--session", metavar="PATH",
                    help="save the conversation, edit journal and turn "
                         "boundaries here after every turn, and resume from it "
                         "if the file already exists")
    ap.add_argument("--dump-messages", action="store_true",
                    help="on exit, dump the raw message list (to debug the protocol)")
    ap.add_argument("--dump-dossier", action="store_true",
                    help="research mode: print the verified evidence pack after each answer")
    opts = ap.parse_args()

    if (opts.dry_run or opts.yes) and not (opts.allow_edits or opts.allow_post):
        raise SystemExit("--dry-run and --yes only mean anything with "
                         "--allow-edits or --allow-post")
    if opts.dry_run and opts.yes:
        raise SystemExit("--dry-run and --yes contradict each other")
    if opts.allow_edits and opts.mode == "research":
        # Research sub-agents get a restricted registry and are forbidden to
        # answer; handing them write tools is a separate design question.
        raise SystemExit("--allow-edits is not supported in research mode yet")

    if opts.model and opts.model_flag:
        raise SystemExit("give the model once: either `bobbin <model>` or --model, "
                         "not both")
    named = opts.model or opts.model_flag
    picked_for_you = named is None
    try:
        opts.model = default_model(opts.host) if picked_for_you \
            else resolve_model(named, opts.host)
    except LLMError as exc:
        raise SystemExit(str(exc))

    ws = Workspace(opts.root)
    session = EditSession(approve=make_approver(opts.dry_run, opts.yes)) \
        if opts.allow_edits else None
    registry = build_registry(
        ws, session,
        allow_web=opts.allow_web or opts.allow_post,
        post_approve=make_post_approver(opts.dry_run, opts.yes) if opts.allow_post
        else None)
    client = OllamaClient(opts.model, host=opts.host, num_ctx=opts.num_ctx)
    trace = make_trace(opts.verbose)
    if opts.mode == "research":
        agent = ResearchAgent(client=client, registry=registry, workspace=ws,
                              max_subtasks=opts.subtasks,
                              gather_steps=opts.gather_steps, trace=trace)
    else:
        agent = Agent(client=client, registry=registry, workspace=ws,
                      max_steps=opts.max_steps, trace=trace,
                      playbook=opts.playbook, session=session)

    chosen = f"{opts.model}{' (first in ollama list)' if picked_for_you else ''}"
    print(f"{DIM}model={chosen}  root={ws.root}  mode={opts.mode}  "
          f"tools={', '.join(sorted(registry.tools))}{RESET}")
    if opts.allow_edits:
        state = "dry run, nothing will be written" if opts.dry_run else (
            "applying WITHOUT asking" if opts.yes else "you will be asked before each change")
        print(f"{YELLOW}edits enabled — {state}{RESET}")

    if opts.session:
        stored = Path(opts.session)
        if stored.is_file():
            restore_session(agent, load_session(stored))
            turns = len(agent.turn_marks)
            print(f"{DIM}resumed {stored} — {turns} turn(s), "
                  f"{len(session.history) if session else 0} edit(s) in the "
                  f"journal{RESET}")
        else:
            print(f"{DIM}session will be saved to {stored}{RESET}")

    if not opts.prompt and sys.stdin.isatty():
        agent.steer_poll = make_steer_poll()

    def answer(prompt: str) -> None:
        # First Ctrl-C asks the turn to stop at its next safe point; a second one
        # gives up on being tidy. Restored afterwards, so Ctrl-C at the prompt
        # still means "quit" and not "stop a turn that is not running".
        hits = [0]

        def on_sigint(signum, frame):
            hits[0] += 1
            if hits[0] == 1:
                agent.interrupt()
                print(f"\n{YELLOW}stopping at the next safe point — "
                      f"Ctrl-C again to abort{RESET}", flush=True)
            else:
                raise KeyboardInterrupt
        previous = signal.signal(signal.SIGINT, on_sigint)
        try:
            print(f"\n{YELLOW}{agent.ask(prompt)}{RESET}\n")
        finally:
            signal.signal(signal.SIGINT, previous)
            if opts.session:
                save_session(agent, opts.session)
        if opts.dump_dossier and isinstance(agent, ResearchAgent):
            print(f"{DIM}--- evidence ---\n{agent.report.dossier}{RESET}\n")
            for finding in agent.report.findings:
                for claim in finding.dropped:
                    print(f"{RED}{DIM}dropped [{finding.subtask.id}]: {claim}{RESET}")

    try:
        if opts.prompt:
            answer(opts.prompt)
        else:
            print(f"{DIM}Ask a question. Ctrl-D or /quit to exit.{RESET}")
            print(f"{DIM}While it works: type a line + Enter to steer it, "
                  f"Ctrl-C to stop it.{RESET}\n")
            while True:
                try:
                    line = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not line:
                    continue
                if line in ("/quit", "/exit"):
                    break
                print()
                answer(line)
    finally:
        if opts.dump_messages:
            json.dump(agent.messages, sys.stderr, indent=2)
            print(file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
