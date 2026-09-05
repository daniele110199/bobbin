#!/usr/bin/env python3
"""Run the eval suite across one or more local models.

    python3 -m evals.run                                  # all cases, default model
    python3 -m evals.run --models qwen3-coder:30b,nemotron-3.5-lightning
    python3 -m evals.run --cases locate-tax,sentinel      # by id
    python3 -m evals.run --cases tag:honesty              # by tag
    python3 -m evals.run --allow-edits --cases tag:multi-turn   # sessions, not questions
    python3 -m evals.run --show-fails                     # print failing answers
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):  # allow `python3 evals/run.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "evals"

from agent.edits import EditSession
from agent.llm import OllamaClient
from agent.loop import Agent, compaction_enabled, scope_check_enabled
from agent.research import ResearchAgent
from agent.sandbox import Workspace
from agent.tools import build_registry

from .cases import ALL_CASES, CASES, EDIT_CASES, Case, _norm_model
from .stability import load, noise_floor
from .webfixture import FixtureWeb
from .score import (Score, diff_snapshots, honesty_problem,
                    honesty_problem_own_words, score_answer, score_workspace,
                    snapshot_tree, unfinished_problems)

EVALS = Path(__file__).resolve().parent
FIXTURE = EVALS / "fixture"
RESULTS = EVALS / "results"

# Every case runs against its own throwaway copy of the fixture, never the
# fixture itself. Once a tool can write, a single bad run would otherwise
# corrupt the only thing all the scores are measured against — and this project
# is not a git repo, so there is no `git checkout` to undo it.
#
# The copy root is deterministic rather than random (`mkdtemp`) so that two runs
# of the same case see a byte-identical system prompt. These models are brittle
# to formatting, so a path that changes length between runs is a real source of
# nondeterminism, not a theoretical one.
SANDBOX_ROOT = Path(tempfile.gettempdir()) / "llm-agent-evals"

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def select_cases(spec: str | None, allow_edits: bool = False) -> list[Case]:
    """Pick cases. With no `--cases`, the default is the read-only 22 unless
    edits are enabled — so a bare `evals.run` keeps meaning what it always did.

    The cascade cases are excluded from that default too: they run against a
    different fixture and are not comparable with anything measured so far, so
    they have to be asked for by `--cases tag:cascade`."""
    if not spec:
        return CASES + EDIT_CASES if allow_edits else CASES
    if spec.startswith("tag:"):
        want = spec[4:]
        return [c for c in ALL_CASES if want in c.tags]
    wanted = {s.strip() for s in spec.split(",")}
    chosen = [c for c in ALL_CASES if c.id in wanted]
    unknown = wanted - {c.id for c in chosen}
    if unknown:
        raise SystemExit(f"unknown case id(s): {', '.join(sorted(unknown))}")
    return chosen


DEFAULT_NUM_CTX = 16384


def resolve_num_ctx(case: Case, opts, model: str = "") -> int:
    """The context window this case runs at, for `model`.

    An explicit `--num-ctx` beats a case's pin, for the same reason
    `--max-steps` beats `Case.max_steps`: sweeping the parameter is how the pin
    gets found, and a case that could not be swept would freeze its own regime.
    Below that, a per-model pin beats the case's single pin — the window that
    puts one model in the regime a case is built to measure does not put another
    one there, which is what the two-model baseline showed at 2560.
    """
    return opts.num_ctx or case.num_ctx_for_model(model) or DEFAULT_NUM_CTX


def build_agent(model: str, ws: Workspace, opts, max_steps: int | None = None,
                num_ctx: int | None = None, case: Case | None = None):
    client = OllamaClient(model, host=opts.host,
                          num_ctx=num_ctx or opts.num_ctx or DEFAULT_NUM_CTX)
    # Auto-approve: the human gate is a property of the REPL, not of the agent.
    # Scoring measures whether the model can produce a correct edit, and a
    # prompt nobody is there to answer would measure nothing. Safe because the
    # workspace is a throwaway copy.
    session = EditSession() if opts.allow_edits else None
    # The web tools are per case, never suite-wide. The read-only 22 must keep
    # seeing exactly the four tools every number on record was measured against,
    # and a schema is prompt text charged on every request — the same arithmetic
    # that keeps them behind a flag in `main.py`.
    #
    # `http_post` is auto-approved for the same reason writes are: the human gate
    # is a property of the REPL, and a prompt nobody is there to answer would
    # measure nothing. The requests go to the offline fixture, not the internet.
    registry = build_registry(
        ws, session,
        allow_web=bool(case and case.allow_web),
        post_approve=(lambda url, body, ctype: True)
        if (case and case.allow_post) else None)
    if opts.mode == "research":
        return ResearchAgent(
            client=client, registry=registry, workspace=ws,
            max_subtasks=opts.subtasks, gather_steps=opts.gather_steps,
        )
    return Agent(
        client=client, registry=registry, workspace=ws,
        # An explicit `--max-steps` beats a case's own pin: sweeping the budget
        # of a case built around one is how that pin gets re-tuned, and the flag
        # is documented as pinning every case.
        max_steps=opts.max_steps or max_steps, playbook=opts.playbook,
        session=session,
    )


def case_sandbox(model: str, case: Case) -> Path:
    """A fresh copy of the case's fixture for one (model, case).

    Deterministic path, and the copy is always named `fixture/` whichever tree it
    came from: the workspace root goes into the system prompt, and these models
    are brittle enough to formatting that a path of a different length is a real
    source of nondeterminism.

    That warning was written about the *fixture* name and left the **model** name
    free to do the same thing. `nemotron-3.5-lightning` and
    `nemotron-3.5-lightning:latest` are one model, and they produced paths seven
    characters apart — which is a different system prompt, which is a different
    run. Measured 2026-08-27 on one binary: `cascade-signature` passes with the
    tag typed (9 steps, two files edited) and fails without it (7 steps, three
    files, `src/api/views.py` touched), twice each. The stored corpus is split
    135/170 between the two spellings, so half of it was never comparable with
    the other half and nothing said so.

    `:latest` is normalised away, the same direction `_pin_for()` already
    normalises for the step pins. Only that tag: `qwen3-coder:30b` keeps its tag,
    because there the tag names a different model.
    """
    source = EVALS / case.fixture
    if not source.is_dir():
        raise SystemExit(f"case {case.id}: no such fixture: {source}")
    canonical = model[:-len(":latest")] if model.endswith(":latest") else model
    slug = canonical.replace(":", "-").replace("/", "-")
    root = SANDBOX_ROOT / slug / case.id / "fixture"
    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, root)
    return root


def context_size(agent) -> int:
    """Characters sitting in the agent's message list — what the next turn carries.

    Chars, not tokens: there is no tokenizer in the stdlib and this project does
    not pip install one. `chars // 4` is reported alongside as an estimate,
    which is accurate enough for the only question asked of it — which turn of a
    session crosses the context window, and by how much. A one-shot case has one
    reading and it means nothing; a session has a curve.
    """
    total = 0
    for message in getattr(agent, "messages", []) or []:
        total += len(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            args = (call.get("function") or {}).get("arguments") or {}
            total += len(json.dumps(args, default=str))
    return total


def run_turn(agent, case: Case, turn, index: int, root: Path) -> dict:
    """One user message, scored against the tree as this turn left it.

    The snapshot is taken here rather than once per run, and that is the whole
    of per-turn attribution: `may_touch` then means "what *this turn* was
    allowed to change". Without it every turn after the first would inherit the
    blame for its predecessors' edits, and `may_touch=[]` — the assertion that a
    question did not turn into a write — could not be stated at all.
    """
    before = snapshot_tree(root)
    answer = agent.ask(turn.prompt)
    changes = diff_snapshots(before, snapshot_tree(root))

    file_problems = score_workspace(turn, root, changes)
    score = score_answer(turn, answer, case_id=case.id)
    dishonest = honesty_problem(answer, file_problems)
    dishonest_own = honesty_problem_own_words(answer, file_problems)
    gating_lie = dishonest_own if case.honesty_own_words else dishonest
    collateral = [p for p in file_problems if p not in unfinished_problems(file_problems)]
    st = agent.stats
    chars = context_size(agent)
    return {
        "turn": index,
        "prompt": turn.prompt,
        "answer": answer,
        "passed": (score.passed and not collateral and not gating_lie
                   if case.score_honesty else
                   score.passed and not file_problems),
        "missing": score.missing,
        "forbidden": score.forbidden,
        "file_problems": file_problems,
        "changed_files": changes,
        "dishonest": dishonest,
        "dishonest_own_words": dishonest_own,
        "budget": st.budget,
        "steps": st.steps,
        "tool_calls": st.names,
        "tool_calls_detail": [{"name": n, "args": a} for n, a in st.tool_calls],
        "tool_errors": st.tool_errors,
        "repeat_blocks": st.repeat_blocks,
        "recovered_text_calls": st.recovered_text_calls,
        "absence_challenges": st.absence_challenges,
        "quoted_absences": st.quoted_absences,
        "presupposition_challenges": st.presupposition_challenges,
        "scope_challenges": st.scope_challenges,
        "interrupted": st.interrupted,
        "steers": st.steers,
        "context_notices": st.context_notices,
        "compactions": st.compactions,
        "digest_previews": st.digest_previews,
        "fabrications": st.fabrications,
        "verify_nudges": st.verify_nudges,
        "repair_turns": st.repair_turns,
        "broken_at_end": st.broken_at_end,
        "unfinished_flags": st.unfinished_flags,
        "unaddressed_flags": st.unaddressed_flags,
        "auto_reviews": st.auto_reviews,
        "creations_blocked": st.creations_blocked,
        "budget_exhausted": st.budget_exhausted,
        "llm_error": st.llm_error,
        "duration_s": round(st.duration_s, 1),
        # What this turn left for the next one to carry. The agent never counts
        # it and nothing bounds it, so on a long session the oldest content —
        # the first turn's answer, and eventually the system prompt — is dropped
        # by the server with nobody told. This column is where that becomes
        # visible, and it is the number any compaction work has to move.
        "context_chars": chars,
        "context_tokens_est": chars // 4,
    }


def run_case(model: str, case: Case, opts) -> dict:
    root = case_sandbox(model, case)
    before = snapshot_tree(root)

    ws = Workspace(root)
    num_ctx = resolve_num_ctx(case, opts, model)
    agent = build_agent(model, ws, opts, max_steps=case.budget_for_model(model),
                        num_ctx=num_ctx, case=case)

    # One agent for every turn, which is the point of a multi-turn case: turn 2
    # is answered by something carrying turn 1's messages, its edit journal and
    # its read gate. Building a fresh agent per turn would measure the same
    # thing the suite already measures, twice.
    turns = case.turn_list()
    multi = len(turns) > 1
    if multi and isinstance(agent, ResearchAgent):
        raise SystemExit(f"case {case.id}: multi-turn is not supported in research mode")
    rows = [run_turn(agent, case, turn, i, root)
            for i, turn in enumerate(turns, start=1)]

    def gathered(key: str) -> list:
        """One turn's list, or every turn's tagged with which turn it came from."""
        if not multi:
            return rows[0][key]
        return [f"turn {r['turn']}: {item}" for r in rows for item in r[key]]

    def total(key: str) -> int:
        return sum(r[key] for r in rows)

    changes = diff_snapshots(before, snapshot_tree(root))
    research = {}
    if isinstance(agent, ResearchAgent):
        rep = agent.report
        research = {
            "plan_source": rep.plan_source,
            "subtasks": [
                {"id": t.id, "question": t.question, "grep": t.hint,
                 "grounded": t.grounded}
                for t in rep.subtasks
            ],
            "grounded_hints": rep.grounded_hints,
            "verified_facts": rep.verified_facts,
            "dropped_claims": rep.dropped_claims,
            "dropped": [d for f in rep.findings for d in f.dropped],
            "model_calls": rep.model_calls,
            "dossier_chars": len(rep.dossier),
            "dossier": rep.dossier,
        }
    return {
        "research": research,
        "case": case.id,
        "tags": case.tags,
        "model": model,
        # What the model *could* have called, not just what it did. A tool that
        # is offered and never used is the interesting number: nemotron scores
        # like qwen and calls `check_imports` a tenth as often, and the case it
        # fails is the one that tool answers. Without this the difference is
        # only visible by reading traces by hand.
        "tools_available": sorted(agent.registry.tools),
        # What the run was allowed to spend. A scaled budget makes "exhausted"
        # ambiguous without it, and the A/B of the sizing rule is unreadable.
        # Sized per turn, so a session reports the largest of them and the
        # per-turn rows carry the rest.
        "budget": max(r["budget"] for r in rows),
        # An edit case is only correct if the disk is correct. A model that says
        # "done" and writes nothing must not score a pass.
        #
        # An honesty case is the one exception, and it has to be: it is built to
        # end with the work undone, so it is scored on whether the answer admits
        # that. Collateral damage still fails it — being honest about unfinished
        # work does not license rewriting files the case forbids.
        #
        # A session passes only if every turn of it passed. There is no partial
        # credit on purpose: a follow-up answered correctly on top of a first
        # turn that went wrong is not a working session, and averaging the two
        # would hide exactly the failure this suite was built to see.
        "passed": all(r["passed"] for r in rows),
        "missing": gathered("missing"),
        "forbidden": gathered("forbidden"),
        "file_problems": gathered("file_problems"),
        # "The disk is wrong and the answer says it is right." Both columns are
        # always recorded, and they are the number a reporting mechanism is meant
        # to move when the pass rate cannot see it. Which one *gates* is per-case
        # (`Case.honesty_own_words`, opted into only by `edit-honesty-budget`);
        # every other case keeps the as-returned definition each number on record
        # was measured under. See `evals/score.py`.
        "dishonest": next((r["dishonest"] for r in rows if r["dishonest"]), None),
        "dishonest_own_words": next(
            (r["dishonest_own_words"] for r in rows if r["dishonest_own_words"]), None),
        # Across the whole session, so a file created in one turn and deleted in
        # a later one nets out — the same question the single-turn column asks.
        "changed_files": changes,
        # The last thing the user was told. Every stored-artifact instrument in
        # this repo (`evals.rescore`, `evals.replay`) reads this key, so on a
        # one-turn case it stays exactly what it always was.
        "answer": rows[-1]["answer"],
        "turn_count": len(rows),
        "num_ctx": num_ctx,
        "context_chars": max(r["context_chars"] for r in rows),
        "context_tokens_est": max(r["context_tokens_est"] for r in rows),
        "steps": total("steps"),
        "tool_calls": [name for r in rows for name in r["tool_calls"]],
        "tool_calls_detail": [d for r in rows for d in r["tool_calls_detail"]],
        "tool_errors": total("tool_errors"),
        "repeat_blocks": total("repeat_blocks"),
        "recovered_text_calls": total("recovered_text_calls"),
        "absence_challenges": total("absence_challenges"),
        "quoted_absences": total("quoted_absences"),
        "presupposition_challenges": total("presupposition_challenges"),
        "scope_challenges": total("scope_challenges"),
        # A run stopped by its user is not a run that failed, and a
        # scored suite has to be able to tell them apart.
        "interrupted": any(r["interrupted"] for r in rows),
        "steers": total("steers"),
        "context_notices": total("context_notices"),
        "compactions": total("compactions"),
        "digest_previews": gathered("digest_previews"),
        "fabrications": total("fabrications"),
        "verify_nudges": total("verify_nudges"),
        "repair_turns": total("repair_turns"),
        "broken_at_end": rows[-1]["broken_at_end"],
        "unfinished_flags": gathered("unfinished_flags"),
        "unaddressed_flags": gathered("unaddressed_flags"),
        "auto_reviews": total("auto_reviews"),
        # Creations the guard refused. Counted in both arms of its A/B: with
        # `AGENT_NO_CREATE_GUARD=1` the write goes through and this stays 0, so
        # the arms are told apart by the counter as well as by the disk.
        "creations_blocked": total("creations_blocked"),
        "budget_exhausted": any(r["budget_exhausted"] for r in rows),
        "llm_error": next((r["llm_error"] for r in rows if r["llm_error"]), None),
        "duration_s": round(sum(r["duration_s"] for r in rows), 1),
        # Only on a session: a turn row repeats the whole trace, and doubling
        # the size of every stored result to say "1 turn" would be a poor trade.
        **({"turns": rows} if multi else {}),
    }


def print_case_line(row: dict) -> None:
    mark = f"{GREEN}pass{RESET}" if row["passed"] else f"{RED}FAIL{RESET}"
    detail = ""
    if row.get("turns"):
        marks = "".join(f"{GREEN}o{RESET}" if t["passed"] else f"{RED}x{RESET}"
                        for t in row["turns"])
        mark = f"{mark} {marks}"
    if not row["passed"]:
        bits = []
        if row["missing"]:
            bits.append(f"missing {row['missing']}")
        if row["forbidden"]:
            bits.append(f"forbidden {row['forbidden']}")
        if row.get("file_problems"):
            bits.append("; ".join(row["file_problems"]))
        detail = f"  {DIM}{'; '.join(bits)}{RESET}"
    print(f"  {mark}  {row['case']:<20} "
          f"{DIM}{len(row['tool_calls'])} calls, {row['duration_s']}s{RESET}{detail}",
          flush=True)


def run_model(model: str, cases: list[Case], opts) -> list[dict]:
    """Every case for one model, optionally several at a time.

    Cases are independent — separate Workspace, separate Agent, no shared state
    but the vocabulary cache — so they parallelise cleanly. The work is waiting
    on Ollama's HTTP socket, so threads are enough.

    `--jobs 1` stays the exact serial path it always was, and is what any number
    quoted as a result should come from. Concurrent requests get batched by the
    server, which changes the arithmetic behind a token and can therefore change
    an answer: on a model as formatting-sensitive as a 7B that is a real risk,
    not a theoretical one. Use `--jobs` to iterate, then confirm serially.
    """
    if opts.jobs <= 1:
        rows = []
        for case in cases:
            row = run_case(model, case, opts)
            rows.append(row)
            print_case_line(row)
        return rows

    order = {c.id: i for i, c in enumerate(cases)}
    rows = []
    with ThreadPoolExecutor(max_workers=opts.jobs) as pool:
        futures = {pool.submit(run_case, model, c, opts): c for c in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad case must not kill the run
                row = {"case": case.id, "tags": case.tags, "model": model,
                       "passed": False, "missing": [f"crashed: {exc}"],
                       "forbidden": [], "file_problems": [],
                       "changed_files": {"created": [], "deleted": [], "modified": []},
                       "answer": "", "steps": 0, "tool_calls": [],
                       "tool_calls_detail": [], "tool_errors": 0, "repeat_blocks": 0,
                       "recovered_text_calls": 0, "absence_challenges": 0,
                       "quoted_absences": 0, "presupposition_challenges": 0,
                       "scope_challenges": 0,
                       "context_notices": 0, "compactions": 0,
                       "digest_previews": [],
                       "fabrications": 0, "budget_exhausted": False,
                       "llm_error": str(exc), "duration_s": 0.0, "research": {}}
            rows.append(row)
            print_case_line(row)
    return sorted(rows, key=lambda r: order[r["case"]])


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    passed = sum(r["passed"] for r in rows)
    research = [r["research"] for r in rows if r.get("research")]
    extra = {}
    if research:
        extra = {
            "avg_subtasks": sum(len(r["subtasks"]) for r in research) / len(research),
            "plan_fallbacks": sum(r["plan_source"] == "fallback" for r in research),
            "grounded_hints": sum(r["grounded_hints"] for r in research),
            "verified_facts": sum(r["verified_facts"] for r in research),
            "dropped_claims": sum(r["dropped_claims"] for r in research),
            "avg_dossier_chars": sum(r["dossier_chars"] for r in research) / len(research),
            "avg_model_calls": sum(r["model_calls"] for r in research) / len(research),
        }
    return {
        "cases": n,
        "passed": passed,
        "pass_rate": passed / n,
        **extra,
        "avg_tool_calls": sum(len(r["tool_calls"]) for r in rows) / n,
        "tool_errors": sum(r["tool_errors"] for r in rows),
        "repeat_blocks": sum(r["repeat_blocks"] for r in rows),
        "recovered": sum(r["recovered_text_calls"] for r in rows),
        "challenges": sum(r["absence_challenges"] for r in rows),
        "quoted": sum(r.get("quoted_absences", 0) for r in rows),
        "presup": sum(r.get("presupposition_challenges", 0) for r in rows),
        "scope": sum(r.get("scope_challenges", 0) for r in rows),
        "notices": sum(r.get("context_notices", 0) for r in rows),
        "compactions": sum(r.get("compactions", 0) for r in rows),
        "fabrications": sum(r["fabrications"] for r in rows),
        "verify_nudges": sum(r.get("verify_nudges", 0) for r in rows),
        "repair_turns": sum(r.get("repair_turns", 0) for r in rows),
        "creations_blocked": sum(r.get("creations_blocked", 0) for r in rows),
        # Runs that finished with import damage the agent itself
        # caused. The pass rate cannot tell "ran out of room" from
        # "left the repo half-migrated"; this can.
        "broken_at_end": sum(bool(r.get("broken_at_end")) for r in rows),
        "exhausted": sum(r["budget_exhausted"] for r in rows),
        # Runs where the request named something the run never went near —
        # the "never begun" shape, which leaves nothing damaged for the
        # three artifact-keyed guards to see.
        "unaddressed": sum(bool(r.get("unaddressed_flags")) for r in rows),
        # Cases whose prose was fine but whose workspace was not, plus how many
        # cases wrote anything at all. On the read-only suite both must be 0.
        "disk_fails": sum(bool(r.get("file_problems")) for r in rows),
        # Of those, the ones whose answer claimed the work was done anyway. This
        # is the honesty axis on edits, and the one a reporting mechanism moves.
        "dishonest": sum(bool(r.get("dishonest")) for r in rows),
        "dishonest_own_words": sum(bool(r.get("dishonest_own_words")) for r in rows),
        "cases_that_wrote": sum(
            bool(any(r.get("changed_files", {}).values())) for r in rows
        ),
        "avg_s": sum(r["duration_s"] for r in rows) / n,
        "tool_use": tool_use(rows),
    }


def tool_use(rows: list[dict]) -> dict[str, dict]:
    """Per tool: how often it was called, and in how many cases it was used.

    Keyed on every tool that was *available*, so a tool nobody touched shows up
    as a zero rather than as an absent row. That is the measurement this table
    exists for — a mechanism can generalise in availability and not in uptake,
    and a missing row reads as "no such tool" instead of "never reached for".
    """
    available: set[str] = set()
    for row in rows:
        available.update(row.get("tools_available") or [])
        available.update(row.get("tool_calls") or [])

    out: dict[str, dict] = {}
    for name in sorted(available):
        calls = sum((r.get("tool_calls") or []).count(name) for r in rows)
        used_in = sum(bool(name in (r.get("tool_calls") or [])) for r in rows)
        out[name] = {"calls": calls, "cases": used_in,
                     "per_case": calls / len(rows) if rows else 0.0}
    return out


def print_report(all_rows: list[dict], models: list[str], cases: list[Case],
                 show_fails: bool) -> None:
    print(f"\n{BOLD}Summary{RESET}")
    head = f"{'model':<26} {'pass':>9} {'calls':>6} {'err':>4} {'rept':>5} {'recov':>6} {'chal':>5} {'qtd':>4} {'psp':>4} {'fab':>4} {'vfy':>4} {'rpr':>4} {'brk':>4} {'unadd':>6} {'exh':>4} {'avg s':>7}"
    print(head)
    print("-" * len(head))
    for model in models:
        rows = [r for r in all_rows if r["model"] == model]
        s = summarise(rows)
        if not s:
            continue
        print(f"{model:<26} {s['passed']:>3}/{s['cases']:<3}{s['pass_rate']*100:>4.0f}% "
              f"{s['avg_tool_calls']:>6.1f} {s['tool_errors']:>4} {s['repeat_blocks']:>5} "
              f"{s['recovered']:>6} {s['challenges']:>5} {s['quoted']:>4} {s['presup']:>4} {s['fabrications']:>4} {s['verify_nudges']:>4} {s['repair_turns']:>4} {s['broken_at_end']:>4} {s['unaddressed']:>6} {s['exhausted']:>4} {s['avg_s']:>7.1f}")

    print(f"\n{DIM}calls=avg tool calls/case  err=tool errors  rept=blocked repeats  "
          f"recov=recovered from prose  chal=absence challenged  qtd=absence spared because it was quoted out of the workspace, not claimed (counted in both arms)  psp=asked to change a name the run itself defined  fab=fabricated tool output rejected  vfy=answers sent back over broken imports  rpr=repair turns granted  brk=runs that ENDED with a tree the agent broke  unadd=runs that never went near something the request named  exh=budget exhausted{RESET}")

    print(f"\n{BOLD}Tool use{RESET}")
    tools = sorted({t for r in all_rows
                    for t in (r.get("tools_available") or []) + (r.get("tool_calls") or [])})
    if tools:
        width = max(len(t) for t in tools) + 2
        head = f"{'tool':<{width}}" + "".join(
            f"{m.split(':')[0][:16]:>18}" for m in models)
        print(head)
        print("-" * len(head))
        for tool in tools:
            line = f"{tool:<{width}}"
            for model in models:
                rows = [r for r in all_rows if r["model"] == model]
                stats = tool_use(rows).get(tool)
                if not stats or not rows:
                    line += f"{'-':>18}"
                    continue
                cell = f"{stats['per_case']:.1f}/case {stats['cases']}/{len(rows)}"
                # A tool that was offered and never called is the finding, so it
                # is coloured rather than left to be spotted in a column of nines.
                colour = DIM if stats["calls"] else RED
                line += f"{colour}{cell:>18}{RESET}"
            print(line)
        print(f"\n{DIM}calls per case, and the number of cases that used the tool "
              f"at least once. Every tool the registry offered is listed, so a "
              f"zero means offered and never reached for — not absent.{RESET}")

    if any(r.get("research") for r in all_rows):
        print(f"\n{BOLD}Research phases{RESET}")
        head = (f"{'model':<26} {'subtasks':>9} {'planfb':>7} {'grnd':>5} "
                f"{'facts':>6} {'dropped':>8} {'dossier':>8} {'calls':>6}")
        print(head)
        print("-" * len(head))
        for model in models:
            s = summarise([r for r in all_rows if r["model"] == model])
            if "avg_subtasks" not in s:
                continue
            print(f"{model:<26} {s['avg_subtasks']:>9.1f} {s['plan_fallbacks']:>7} "
                  f"{s['grounded_hints']:>5} {s['verified_facts']:>6} "
                  f"{s['dropped_claims']:>8} {s['avg_dossier_chars']:>8.0f} "
                  f"{s['avg_model_calls']:>6.1f}")
        print(f"\n{DIM}subtasks=avg planned/case  planfb=plan unparseable, fell back  "
              f"grnd=grep terms corrected against the repo vocabulary  "
              f"facts=claims backed by tool output  dropped=claims deleted as unsupported  "
              f"dossier=avg chars handed to the answer phase  calls=avg model calls/case{RESET}")

    # The noise floor, from the runs already on disk. `print_report` runs before
    # this sitting's file is written, so it is strictly historical.
    #
    # It is here because every table in this project quotes a pass rate with
    # nothing beside it, and a bare "5/6 against 4/6" reads as a result. Measured
    # over 3729 stored runs, roughly one fixed-configuration group in five flips
    # outcome, and the worst cases are near a coin. A column that says so is the
    # cheapest defence there is against reading a sitting as a finding.
    # Two numbers, not one, and the second is the reason: `web-search-then-fetch`
    # reads **0.00 within a sitting** over 26 pairs — it failed consistently on one
    # day and passed consistently on the next. Its whole instability is *between*
    # sittings, so the within-sitting floor calls the most treacherous case in the
    # suite perfectly stable. Pooled, it reads 0.44.
    #
    # A case where those two disagree is the dangerous shape: reproducible enough
    # inside one sitting to look like a fact, and different next week. Quoting
    # only one of them would have hidden exactly the trap that produced this
    # column.
    floor = pooled = {}
    try:
        groups = load(RESULTS)
        floor = noise_floor(groups)
        pooled = noise_floor(groups, same_sitting=False)
    except Exception:  # noqa: BLE001 - a report must never fail over its own footnote
        pass

    width = max(len(c.id) for c in cases) + 2
    print(f"\n{BOLD}Per case{RESET}")
    print(f"{'case':<{width}}"
          + "".join(f"{m.split(':')[0][:12]:>14}" for m in models)
          + f"{'noise':>14}")
    for case in cases:
        line = f"{case.id:<{width}}"
        for model in models:
            row = next((r for r in all_rows
                        if r["model"] == model and r["case"] == case.id), None)
            mark = "-" if row is None else (f"{GREEN}pass{RESET}" if row["passed"]
                                            else f"{RED}FAIL{RESET}")
            line += " " * (14 - 4) + mark
        # The worst rate over the models actually in this run, so the column
        # answers "how much should I trust the cells to my left".
        def worst(table):
            rates = [table[(case.id, _norm_model(m))]["rate"] for m in models
                     if (case.id, _norm_model(m)) in table]
            return max(rates) if rates else None

        within, across = worst(floor), worst(pooled)
        if within is None and across is None:
            line += f"{DIM}{'-':>14}{RESET}"
        else:
            cell = (f"{within:.2f}" if within is not None else "-") + "/" + \
                   (f"{across:.2f}" if across is not None else "-")
            hidden = (within is not None and across is not None
                      and within < 0.2 <= across)
            colour = RED if (within or 0) >= 0.2 or hidden else DIM
            line += f"{colour}{cell:>14}{RESET}"
        print(line)
    if floor:
        print(f"\n{DIM}noise = P(two runs of this case under an identical recorded "
              f"configuration disagree), as within-one-sitting/pooled-across-"
              f"sittings, over every stored run. '-' means too few pairs to say. "
              f"At 0.20 either number, a single rep is close to worthless and a "
              f"one-cell difference is not a result. **A low first number beside a "
              f"high second one is the worst case**: reproducible inside a sitting, "
              f"different next week — quote it across sittings and you will find "
              f"something that is not there. See `python3 -m evals.stability "
              f"--noise`.{RESET}")

    sessions = [r for r in all_rows if r.get("turns")]
    if sessions:
        print(f"\n{BOLD}Turns{RESET}  {DIM}(sessions only){RESET}")
        head = (f"{'case':<26} {'model':<20} {'turn':>4} {'ok':>3} {'steps':>6} "
                f"{'calls':>6} {'ctx~tok':>8} {'num_ctx':>8} {'s':>6}")
        print(head)
        print("-" * len(head))
        for row in sessions:
            for turn in row["turns"]:
                over = turn["context_tokens_est"] > row.get("num_ctx", 0)
                ctx = f"{turn['context_tokens_est']}"
                print(f"{row['case']:<26} {row['model'].split(':')[0][:20]:<20} "
                      f"{turn['turn']:>4} "
                      f"{(GREEN + 'ok' + RESET) if turn['passed'] else (RED + 'X' + RESET):>3} "
                      f"{turn['steps']:>6} {len(turn['tool_calls']):>6} "
                      f"{(RED if over else DIM)}{ctx:>8}{RESET} "
                      f"{row.get('num_ctx', 0):>8} {turn['duration_s']:>6.1f}")
        print(f"\n{DIM}ctx~tok = characters in the message list after that turn, "
              f"divided by four. It is an estimate — there is no tokenizer here — "
              f"and it is red once the session no longer fits the window it was "
              f"given, which is the point at which the server starts dropping the "
              f"oldest messages with nobody told.{RESET}")

    fails = [r for r in all_rows if not r["passed"]]
    if fails and show_fails:
        print(f"\n{BOLD}Failures{RESET}")
        for r in fails:
            print(f"\n{RED}{r['case']} [{r['model']}]{RESET}")
            if r["missing"]:
                print(f"  missing:   {r['missing']}")
            if r["forbidden"]:
                print(f"  forbidden: {r['forbidden']}")
            for problem in r.get("file_problems", []):
                print(f"  {RED}disk:{RESET}      {problem}")
            if r.get("turns"):
                # The last answer is rarely the interesting one on a session:
                # the turn that broke it is.
                for t in r["turns"]:
                    tag = f"{GREEN}ok{RESET}" if t["passed"] else f"{RED}X{RESET}"
                    body = t["answer"].replace("\n", " ")
                    print(f"  {tag} turn {t['turn']}: {DIM}{t['prompt'][:70]}{RESET}")
                    print(f"     tools:  {t['tool_calls']}")
                    print(f"     answer: {DIM}{body[:300]}{RESET}")
                continue
            print(f"  tools:     {r['tool_calls']}")
            answer = r["answer"].replace("\n", " ")
            print(f"  answer:    {DIM}{answer[:400]}{RESET}")


def result_header(stamp: str, models: list[str], opts) -> dict:
    """The identifying half of a result file: what ran, and in which arm.

    `switches` is the new part. Every mechanism here ships behind an `AGENT_*`
    switch and every claim is an A/B, but until 2026-08-25 the only record of
    which arm produced a file was its *name* — so a stored-artifact instrument
    could not tell a guard's own success from a run where the guard was never on,
    and `evals.presuppose` was reading both as the same thing. Files written
    before this say nothing about their arm, which is why the instruments carry an
    "unknown" bucket instead of assuming "off".
    """
    return {"timestamp": stamp, "models": models, "mode": opts.mode,
            "allow_edits": opts.allow_edits,
            # The *effective* state, not just the environment. Compaction's
            # default flipped on 2026-08-27, so an empty `switches` means
            # "compaction off" in every file written before that date and
            # "compaction on" in every file after it — an instrument reading the
            # env alone would silently mix the two arms, which is the mistake
            # this header was added to stop.
            "compaction": compaction_enabled(),
            "scope_check": scope_check_enabled(),
            "switches": {k: v for k, v in sorted(os.environ.items())
                         if k.startswith("AGENT_")}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen3-coder:30b",
                    help="comma-separated Ollama model names")
    ap.add_argument("--cases", help="comma-separated case ids, or tag:<name>")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=None,
                    help=f"context window (default {DEFAULT_NUM_CTX}). Given "
                         "explicitly it also overrides a case's own pin")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="pin every case to this budget; default is per-case, then scaled to the task")
    ap.add_argument("--playbook", default="default",
                    help="'default', 'none' (to A/B without it), or a path to a .md file")
    ap.add_argument("--mode", default="direct", choices=["direct", "research"],
                    help="'direct' = one loop; 'research' = survey/plan/subagents/synthesise")
    ap.add_argument("--subtasks", type=int, default=3,
                    help="research mode: subtasks per question (default 3)")
    ap.add_argument("--gather-steps", type=int, default=4,
                    help="research mode: tool steps per subagent (default 4)")
    ap.add_argument("--allow-edits", action="store_true",
                    help="register the write tools and auto-approve their changes. "
                         "Required by tag:edit cases; changes the read-only "
                         "numbers, so never mix the two in one quoted result")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run this many cases at once (default 1). Much faster, but "
                         "batched requests can change an answer — quote serial numbers")
    ap.add_argument("--show-fails", action="store_true",
                    help="print the answer for each failing case")
    ap.add_argument("--out", help="write JSON results here (default: evals/results/)")
    opts = ap.parse_args()

    models = [m.strip() for m in opts.models.split(",") if m.strip()]
    cases = select_cases(opts.cases, opts.allow_edits)

    # Fail loudly rather than reporting a suite of edit cases the agent had no
    # tools to attempt. Every one would "fail" for the same uninteresting reason.
    if not opts.allow_edits:
        # `edit` is the eight-case suite's own tag and cannot be reused, so a
        # case that writes but belongs to another suite says so with `writes`.
        # (The cascade and repair cases predate this and are not tagged either
        # way — running those without `--allow-edits` still fails silently and
        # uninterestingly, which is worth fixing separately.)
        needs_edits = [c.id for c in cases
                       if "edit" in c.tags or "writes" in c.tags]
        if needs_edits:
            raise SystemExit(
                f"these cases need --allow-edits: {', '.join(needs_edits)}"
            )

    # The offline web, started only if a selected case asks for it and stopped
    # before the results are written. Nothing reaches the real internet: a score
    # that depended on a stranger's uptime, or on what a search engine ranked
    # this morning, would not be a measurement of this agent.
    web_cases = [c for c in cases if c.allow_web]

    print(f"{BOLD}{len(cases)} case(s) x {len(models)} model(s){RESET}  "
          f"mode={opts.mode}  playbook={opts.playbook}  "
          f"edits={'on' if opts.allow_edits else 'off'}  "
          f"fixture={FIXTURE}"
          + (f"  {DIM}offline web for {len(web_cases)} case(s){RESET}"
             if web_cases else ""))

    all_rows: list[dict] = []
    started = time.monotonic()
    offline_web = FixtureWeb().start() if web_cases else None
    try:
        for model in models:
            print(f"\n{BOLD}{model}{RESET}"
                  + (f"  {DIM}({opts.jobs} at a time){RESET}" if opts.jobs > 1 else ""))
            all_rows += run_model(model, cases, opts)
    finally:
        if offline_web is not None:
            offline_web.stop()

    print_report(all_rows, models, cases, opts.show_fails)
    print(f"\n{DIM}total wall time {time.monotonic() - started:.0f}s{RESET}")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(opts.out) if opts.out else RESULTS / f"{stamp}.json"
    out.write_text(json.dumps(result_header(stamp, models, opts) | {
        "jobs": opts.jobs,
        "max_steps": opts.max_steps,
        "playbook": opts.playbook,
        "num_ctx": opts.num_ctx or DEFAULT_NUM_CTX,
        "summary": {m: summarise([r for r in all_rows if r["model"] == m]) for m in models},
        "rows": all_rows,
    }, indent=2))
    print(f"{DIM}results -> {out}{RESET}")

    return 0 if all(r["passed"] for r in all_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
