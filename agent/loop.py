"""The agent loop.

This is the piece aider does not really have. The contract is:

    model emits a structured tool call
      -> we execute it ourselves
      -> the result goes back as a `tool` message
      -> the model gets another turn

and that repeats until the model answers in prose or we hit the step budget.
Everything else in this project exists to make that loop reliable.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .edits import RESERVED, EditSession
from .imports import check_workspace, definition_pattern
from .review import (
    full_report, path_requested, request_identifiers, unaddressed_requests,
    unfinished_reasons,
)
from .llm import LLMError, OllamaClient, ToolCall
from .prompts import DEFAULT, load_system_prompt
from .sandbox import MAX_SCAN_BYTES, Workspace, looks_binary, read_text
from .tools import Registry

MAX_STEPS = 12

# What a cascade costs, and why 12 is the wrong number for one. `MAX_STEPS` was
# tuned on read-only questions with a single answer: find the file, read it,
# reply. A rename across a package is a different shape of work — every file
# that mentions the symbol needs a read and an edit, and the measured runs bear
# that out: the cascade suite lands between 16 and 28 steps, and the first
# `cascade-move` run that ever passed needed 19. A budget that cannot fit the
# task does not make the model concise, it makes it stop halfway and then report
# the half as the whole, which is the failure the whole reporting layer exists
# to catch.
#
# The size of the task is knowable before the first call: the request names the
# symbols, and the tree says how many files carry them. Two steps per file is
# the read plus the edit; the base stays on top for orientation — the listing,
# the greps, the answer.
STEPS_PER_FILE = 2

# A ceiling, because the budget's other job is stopping a runaway loop, and that
# job does not go away because the task is large. 28 is the top of the measured
# cascade range, not a guess about what a bigger task might want.
MAX_BUDGET = 28


# Reading the whole tree to size one request is only sane because a workspace is
# a workspace, not a filesystem — but the cap is here so a large one degrades to
# the base budget instead of stalling before the first tool call.
FILE_SCAN_CAP = 500

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def task_files(request: str, ws: Workspace) -> set[str]:
    """Workspace files the request appears to be about.

    Two ways a name counts, and the difference is the whole design:

        named as code   `apply_discount`, `PurchaseOrder`, `` `slugify` `` —
                        `request_identifiers()`, the same syntactic reading the
                        completeness check is judged on. Trusted outright.
        defined here    any other word in the prompt that this tree *defines* as
                        a function or a class. `Order` is not code-shaped and
                        never will be — one capitalised word is indistinguishable
                        from prose — but a repo with `class Order` in it settles
                        the question the prompt cannot.

    The looser second rule belongs here and nowhere else. `unaddressed_requests()`
    spends a false positive on *accusing the model of skipping work*, so it may
    only use shape; a budget spends one on two extra steps. Sizing can afford to
    guess where an accusation cannot, and the tree — not the sentence — is what
    keeps the guess honest.

    One pass over the tree, because the alternative is a walk per candidate word
    and a prompt has plenty of those.
    """
    trusted = list(request_identifiers(request))
    loose = [name for name in dict.fromkeys(m.group() for m in _TOKEN.finditer(request))
             if len(name) > 2 and name not in trusted and name not in RESERVED]
    texts: dict[str, str] = {}
    for file in sorted(ws.walk(ws.root)):
        if len(texts) >= FILE_SCAN_CAP:
            break
        try:
            texts[ws.display(file)] = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

    def mentions(name: str) -> set[str]:
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        return {path for path, text in texts.items() if pattern.search(text)}

    files: set[str] = set()
    for name in trusted:
        files |= mentions(name)
    for name in loose:
        defined = re.compile(definition_pattern(name), re.M)
        if any(defined.search(text) for path, text in texts.items()
               if path.endswith(".py")):
            files |= mentions(name)
    return files


def budget_for(request: str, ws: Workspace, base: int = MAX_STEPS) -> int:
    """How many steps this request is worth, from the names in it and the tree.

    Never raises and never returns less than `base`: a sizing heuristic that can
    shrink a budget could fail a task the old constant would have finished, and
    one that can throw would fail it outright. Both directions of that asymmetry
    are deliberate — this may only ever hand out *more* room.

    `AGENT_FIXED_STEPS=1` pins it to `base` — the arm that runs every number
    measured before 2026-08-19.
    """
    if os.environ.get("AGENT_FIXED_STEPS"):
        return base
    try:
        touched = task_files(request, ws)
    except Exception:
        return base
    return max(base, min(base + STEPS_PER_FILE * len(touched), MAX_BUDGET))

# A false negative is the worst failure this agent can produce: "it isn't there"
# is indistinguishable from "I didn't look". When the model claims absence, we
# make it search once more before we accept the claim.
NEGATIVE_PHRASES = (
    "does not exist", "doesn't exist", "not exist", "not found",
    "could not find", "couldn't find", "cannot find", "can't find",
    "not defined", "not configured", "not specified", "not mentioned",
    "not present", "not available", "no such", "there is no", "there are no",
    "no file named", "not explicitly",
)

CHALLENGE = """You have claimed that something is not there. That claim is only \
acceptable if you searched for it, and so far you have called: {calls}.

Call `grep` once more with a single short identifier and no file_glob — the \
shortest distinctive word, not a phrase. If that search also returns nothing, \
repeat your answer and it will be accepted."""

# **A subject-bearing version of this challenge was built, measured and removed
# on 2026-08-23.** The defect it targeted is real and still open: this text names
# no subject ("something is not there"), and in a session the model resolves that
# against the most salient *earlier* claim — `multi-absence-subject` turn 2 comes
# back with turn 1's answer verbatim on qwen. The fix quoted the offending
# sentence back and told it not to answer an earlier question.
#
# It cost more than it bought. On nemotron, `edit-nonexistent` — the case this
# guard exists for — went **4/4 pass and 0 writes under this text to 0/4 pass and
# 4/4 writes under the variant**, inventing `ENABLE_TELEMETRY = False` and
# reporting success. That held after restoring the "repeat your answer and it
# will be accepted" licence, so it is not the missing licence alone; something
# about quoting the claim back turns a refusal into an action on that model.
# Deterministic, four reps a cell, both arms one binary, one sitting.
#
# The trade was a session-scoped wrong-turn answer against a false negative on
# the honesty axis, and this project ranks the false negative worse. The case and
# its fixture are kept — the bug is documented by a failing test-case, not by a
# comment. See `evals/cases.py: multi-absence-subject` and the README.
# The window is not infinite and nothing in this loop used to look at it.
# `self.messages` grows without bound, Ollama drops the oldest messages once the
# session no longer fits, and neither the loop nor the model is told — which is
# the whole failure. Measured over 23 stored sessions (`python3 -m evals.context`):
# a turn costs ~900-960 est tokens, so the 16384 default holds ~17 turns of
# ordinary work, and **71% of turns that ran over their window failed** against
# 9% of turns that did not.
#
# The reserve is **one turn**, not a percentage. In the eight short pinned runs on
# disk the first turn to cross the window is the same turn that first fails, or
# later — so a trigger that waits for the crossing has already lost the turn it
# was meant to protect. p90 growth is ~1400 est tokens across both models.
#
# The message is a *notice*, not a summary. The failure at the pin is not lost
# memory: turn 12 answers "I don't see any reference to a tracking id" with
# **zero tool calls** while the id is on disk one grep away, and the same session
# recovers a fact six turns back *by going to look*. Silent truncation teaches a
# model that its history is complete; the cheap half of compaction is saying
# otherwise. Measured: 11/12 -> 12/12, two models, two sittings, four cells.
CONTEXT_RESERVE = 1400

# Worded to be true in both regimes it fires in. Simulated over the stored curves
# before it was ever run: at qwen's pin the trigger is due from turn 2 while the
# session does not actually cross until turn 6, so "the earliest turns have been
# dropped" would be false on four of the turns carrying it — and a notice that
# lies about the history is a strange way to teach a model to distrust its
# memory. "May no longer be" is accurate at both stages.
CONTEXT_NOTICE = """Note on this conversation: it is at or past the limit of \
what you can see, so the earliest turns may no longer be visible to you.

Do not answer from memory about earlier turns. If this question refers to \
something established earlier, find it again with `grep` or `read_file` before \
answering, exactly as if you were seeing it for the first time."""


def context_tokens(messages: list[dict]) -> int:
    """A rough token count for a message list: characters // 4.

    The same estimate `evals.context` uses on stored runs, deliberately — the
    trigger has to be sized on the curve it will run against. There is no
    tokenizer here and this project does not pip install one, which is an
    argument for the reserve above, not for a tokenizer.
    """
    total = 0
    for message in messages or []:
        total += len(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            args = (call.get("function") or {}).get("arguments") or {}
            total += len(json.dumps(args, default=str))
    return total // 4


def context_notice_due(messages: list[dict], num_ctx: int,
                       reserve: int = CONTEXT_RESERVE) -> bool:
    """Whether the next turn is at risk of being answered from a truncated history.

    Predictive, not reactive: `fill + one turn > window`, because by the time
    `fill > window` is true the turn that needed the warning is already lost.
    """
    if not num_ctx:
        return False
    return context_tokens(messages) + reserve > num_ctx


# --- stage two: summarise and drop ----------------------------------------
#
# The notice above works, and measuring *why* it works drew the line this
# mechanism is for. `multi-long-session`'s deep fact is on disk, so "do not
# answer from memory, go and look" is followable and one grep recovers it. A
# session is also full of facts that exist nowhere but the conversation — what
# the user said the deploy target is, which approach they picked, what they said
# not to touch. When those turns fall out of the window there is nothing to look
# at, and the notice's advice is unfollowable by construction.
#
# So the loop stops letting the server decide what to forget and does it itself:
# when the window is about to bind, the oldest turns are replaced by a digest of
# them. Everything after the cut is untouched, and the digest is a real message
# in the list, which is what makes this different from the notice — the fact
# survives in context instead of being pointed at.
#
# On by default since 2026-08-27, `AGENT_NO_COMPACT=1` is the off arm — see
# `compaction_enabled()` for what that shipped on. No new "would have fired"
# counter: the trigger is `context_notice_due()` itself, so `context_notices`
# already reports it in both arms.
COMPACT_KEEP_TURNS = 2

# The digest call is one model call at the session's own window, so its input has
# to fit beside its output. Head and tail are kept and the middle elided: the
# oldest turns are where a session's premises are stated, the newest are what the
# current work depends on, and the middle is mostly tool traffic that the
# workspace still holds.
DIGEST_INPUT_CHARS = 6000

# The digest is found again by this exact prefix, so locating it never depends on
# the rest of the header's wording.
DIGEST_MARKER = "Summary of the earlier part of this session."

# Carried facts plus new material, capped. Elided in the middle when it grows:
# the head is where the session's premises were stated and the tail is the work
# in progress.
DIGEST_MAX_CHARS = 2400

# The body starts after this line. Slicing by `len(header)` worked only while
# there was exactly one header; with a second wording under test, the carry
# forward has to find the body without knowing which header is above it.
DIGEST_BODY_MARK = "RECORD:"

SESSION_DIGEST_HEADER = """Summary of the earlier part of this session. Those \
turns are no longer in your context; this is what is left of them.

Facts recorded here came from the user and cannot be found by searching the \
workspace, so treat them as given. Anything about files or code should be \
re-checked with `grep` or `read_file` before you rely on it."""

# Deliberately not "summarise the conversation". A general summary of a coding
# session is mostly a list of files touched — which is the half the workspace
# still holds — and the half that cannot be recovered is a sentence the user
# said once. The prompt asks for that half first and by name.
SESSION_DIGEST_PROMPT = """Below is the earlier part of a conversation between a \
user and a coding agent. It is about to be dropped from the agent's context, and \
your summary is all that will remain of it.

Record, in this order:
1. Every fact the user stated — values, names, targets, identifiers, numbers. \
Copy them exactly, character for character.
2. Every instruction, decision or preference the user gave.
3. What has been done to the workspace so far, in one line each.

Facts from the user are the only part that cannot be recovered by searching the \
workspace later, so never drop one to save room. Do not add anything that is not \
in the text below, and do not explain the agent's reasoning.

CONVERSATION:
{transcript}"""


# The header above ends by telling the model to re-check anything about files or
# code with `grep` before relying on it — true, and harmless while the digest sat
# at index 1. Restated, that sentence lands immediately before the question, and
# nemotron's failing turns look exactly like it being obeyed: 11 to 16 tool calls
# and then "No matches for 'deploy' in any file". This variant keeps the half
# that says the user's facts are given and drops the half that sends the model
# looking. `AGENT_COMPACT_HEADER=facts`.
SESSION_DIGEST_HEADER_FACTS = """Summary of the earlier part of this session. \
Those turns are no longer in your context; this is what is left of them.

Facts recorded here came from the user. They are not in the workspace and no \
search will find them, so treat them as given and answer from them directly."""


def scope_check_enabled() -> bool:
    """On by default since 2026-08-28; `AGENT_NO_SCOPE_CHECK=1` is the off arm.

    See `SCOPE_CHALLENGE` for what it shipped on.
    """
    return not os.environ.get("AGENT_NO_SCOPE_CHECK")


def compaction_enabled() -> bool:
    """On by default since 2026-08-27; `AGENT_NO_COMPACT=1` is the off arm.

    Shipped on the measurement, not the idea. `AGENT_COMPACT=1` was the switch
    while it was being priced, and what it selected is what runs now: the digest
    restated before the request, under the facts-only header, carried forward
    verbatim under the one-digest invariant. On that build, three reps a cell and
    both model families: `multi-stated-fact` **3/3 at 12/12** on each,
    `multi-long-session` 12/12 on each with no trace of the regression an earlier
    build cost it, and `multi-context-pressure` **0/3 -> 3/3** on qwen — the case
    whose zero-tool-call false absence started the whole line of work. The
    one-shot suites cannot reach it: `tag:edit` 8/8 and `tag:honesty` 3/3 on both
    models with `compactions=0`, because no case in them gets near its window.

    The cost is one model call per compaction, four to seven in a twelve-turn
    session at these pins, and none of them is a loop step.
    """
    return not os.environ.get("AGENT_NO_COMPACT")


def digest_header() -> str:
    """The facts-only header, unless the measured-worse one is asked for.

    Restated immediately before the request, the re-check clause in
    `SESSION_DIGEST_HEADER` is the last instruction the model reads, and nemotron
    obeys it: 1/3 with it, 3/3 without, every passing run answering in one step
    with no tool calls.
    """
    return (SESSION_DIGEST_HEADER
            if os.environ.get("AGENT_COMPACT_HEADER") == "recheck"
            else SESSION_DIGEST_HEADER_FACTS)


def render_transcript(messages: list[dict]) -> str:
    """The message list as plain text for the digest call.

    Tool *results* are included but tool *arguments* are not: the results are
    what the answers were built from, while the arguments are recoverable by
    running the call again.
    """
    lines = []
    for message in messages or []:
        role = message.get("role", "?")
        content = (message.get("content") or "").strip()
        calls = message.get("tool_calls") or []
        if calls:
            named = ", ".join((c.get("function") or {}).get("name", "?")
                              for c in calls)
            content = (content + f" [called {named}]").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def elide_middle(text: str, cap: int = DIGEST_INPUT_CHARS) -> str:
    if len(text) <= cap:
        return text
    half = cap // 2
    return f"{text[:half]}\n[...]\n{text[-half:]}"


def claims_absence(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in NEGATIVE_PHRASES)


# `claims_absence()` reads the answer and nothing else, so it cannot tell the
# agent's own failed search from a sentence it is quoting out of a file it just
# read. Summarising a document that says "There is no second cache" trips it, and
# then `CHALLENGE` — which names no subject — sends the model back to the most
# salient *earlier* claim in the session. Measured 2026-08-23: on
# `multi-absence-subject` the challenge fires on **every run of both models**,
# passes included; whether the derailed turn is scored a failure is the coin flip.
#
# The fix is in the trigger, not the challenge text. Two previous attempts
# rewrote the challenge and one of them cost nemotron `edit-nonexistent` 4/4 -> 0/4
# with writes (see `agent-tuning-dead-ends`); the text is a two-sided lever and
# the trigger is not.
#
# A phrase is an *echo* when it is both in what this turn's tools returned and in
# the workspace itself. Both halves are load-bearing, and the corpus proved the
# second one in a single run: grep's own no-match diagnostic — "No matches for
# 'ENABLE_TELEMETRY' ... so the word you searched for is not the word this code
# uses" — contains the phrase an *honest* refusal uses, so scoring against tool
# output alone classified `edit-nonexistent`'s correct answer as a quotation and
# would have disarmed the guard on the case it exists for. Workspace text is the
# repo talking; tool prose is the loop talking, and only the former can be quoted.
#
# Priced over 3079 stored answers before it ran once (`python3 -m evals.absence`):
# the trigger fires on 333, and this suppresses **9 — every one of them
# `multi-absence-subject`**, with 0 on the `honesty` and `edit` tags, all 42
# `edit-nonexistent` fires and all 48 `edit-honesty-budget` fires still challenged.
QUOTED_SCAN_FILES = 400


def quoted_absence(answer: str, tool_output: str, workspace) -> bool:
    """True when every negative phrase in `answer` was quoted, not claimed.

    Conservative by construction: one phrase the tools never produced means the
    agent is speaking for itself and the challenge still goes out. A false
    negative on the honesty axis — "it isn't there" when nobody looked — is the
    failure this project ranks worst, so the asymmetry runs that way on purpose.
    """
    phrases = [p for p in NEGATIVE_PHRASES if p in answer.lower()]
    if not phrases:
        return False
    lowered = tool_output.lower()
    if not all(phrase in lowered for phrase in phrases):
        return False
    return all(_phrase_in_workspace(workspace, phrase) for phrase in phrases)


def _phrase_in_workspace(workspace, phrase: str) -> bool:
    """Is this literal phrase in the repo's own text?

    Bounded the way the search tools are bounded — same `MAX_SCAN_BYTES`, same
    binary sniff, a file ceiling — because this runs inside a turn and a guard
    that costs a full tree walk on a large repo would be its own bug. It only
    runs when the guard was about to fire, which is ~11% of answers on record.
    """
    try:
        root = workspace.root
    except AttributeError:
        return False
    scanned = 0
    for path in sorted(root.rglob("*")):
        if scanned >= QUOTED_SCAN_FILES:
            break
        try:
            if not path.is_file() or path.stat().st_size > MAX_SCAN_BYTES:
                continue
            if looks_binary(path):        # takes the path, reads its own chunk
                continue
            text = read_text(path)
        except OSError:
            continue
        scanned += 1
        if phrase in text.lower():
            return True
    return False


# **Warm invention**, the session suite's own finding and the one failure in it
# with no mechanism behind it. `multi-refuse-followup`: cold, this model refuses
# to change a constant that does not exist (3/3); one successful turn in front of
# it, the same model greps twice, reads the file, sees no such constant, and
# writes it anyway — "I've successfully added the RETRY_BACKOFF variable" (3/3).
# Agreeableness after a success. Nothing else in the loop fires: the tree is
# consistent, imports resolve, and `unaddressed_requests()` is correctly silent
# because the run did touch the name the request named.
#
# The shape is mechanical, so the detection can be. A request that says *change*
# X **presupposes** X; if nothing bound X before this turn and something binds it
# now, the agent has answered a question about the world by changing the world.
#
# Priced over the corpus before it was built: 269 runs whose request presupposes
# a name, **18 that created a binding for it, and all 18 failed their case** —
# `edit-nonexistent` x16 and `multi-refuse-followup` x2, no fires on any passing
# run. That is better retrospective precision than the create guard had, which is
# exactly why it is **off by default** (`AGENT_PRESUPPOSITION_GUARD=1`): the
# create guard also priced well retrospectively and then failed on the first real
# request, because it keyed on the *shape of the request* and eval prompts are the
# weakest possible sample of that. This guard keys on request phrasing too.
#
# The first attempt at this measurement was blind to its own true positives:
# `definition_pattern()` matches `def`/`class` only, so `RETRY_BACKOFF = 2` was
# not a "definition" and the price came back 0/85. A detector that cannot see the
# case it was built for reports a clean bill of health.
PRESUPPOSING = re.compile(r"\b(change|set|update|modify|adjust|bump)\b", re.I)
CREATING = re.compile(r"\b(add|create|introduce|new|define)\b", re.I)
RENAMING = re.compile(r"\brenam\w*\b", re.I)


def presupposed_names(request: str) -> set[str]:
    """Names the request treats as already existing.

    A creation verb anywhere in the request disqualifies the whole thing rather
    than the individual name: "change the timeout and add a retry constant" is
    one request, the model is allowed to create in it, and a guard that fires on
    half a sentence would be worse than no guard.

    **Constants only, on purpose.** The pattern is ALL-CAPS identifiers, so
    "change the slugify function" is out of scope. Every fire in the corpus is a
    constant (`RETRY_BACKOFF`, `ENABLE_TELEMETRY`), lower-case names collide with
    ordinary English in a way that would need real parsing to separate, and a
    guard is worth more narrow and silent than broad and wrong.
    """
    if not PRESUPPOSING.search(request) or CREATING.search(request) \
            or RENAMING.search(request):
        return set()
    return {n for n in re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", request)
            if n not in ("NOTE", "TODO", "HTTP", "JSON", "HTML")}


def _binding(name: str) -> re.Pattern:
    """`def`/`class`, or an assignment — the half the first price was blind to."""
    return re.compile(
        rf"^[ \t]*(?:(?:async[ \t]+)?(?:def|class)[ \t]+{re.escape(name)}\b"
        rf"|{re.escape(name)}[ \t]*(?::[^=\n]+)?=(?!=))", re.M)


def invented_bindings(request: str, ws: Workspace,
                      before: dict[str, str | None]) -> list[str]:
    """Presupposed names that had no binding before this turn and have one now.

    `before` holds only the files this turn touched, which is the whole point: a
    name that was defined elsewhere in the tree all along is not an invention, so
    the pre-turn text of an untouched file is simply its current text.
    """
    names = presupposed_names(request)
    if not names:
        return []
    # Capped like `_phrase_in_workspace`, and for the same reason: this runs
    # inside a turn, on every edited turn whose request presupposes a name.
    try:
        now = {}
        for f in sorted(ws.walk(ws.root)):
            if len(now) >= QUOTED_SCAN_FILES:
                break
            if not looks_binary(f):
                now[ws.display(f)] = read_text(f)
    except OSError:
        return []
    # The tree as it stood before the turn: current text everywhere, except the
    # files this turn touched, which show what they held when the turn began.
    # `None` means the file did not exist, so it is absent from the pre-tree.
    #
    # Both sides are normalised through the workspace before they are compared.
    # The model spells the same file `src/core/config.py` or `./src/core/config.py`
    # as it pleases, and a raw dict lookup misses on the second spelling — the
    # substitution then silently no-ops, pre == now, and the guard reports that
    # nothing was invented. A guard that depends on how a path was typed is not a
    # guard; this is the same defect `_pin_for()` had with `:latest`.
    pre = dict(now)
    for path, text in before.items():
        try:
            key = ws.display(ws.resolve(path))
        except Exception:
            key = path
        if text is None:
            pre.pop(key, None)
        else:
            pre[key] = text
    now_text, pre_text = "\n".join(now.values()), "\n".join(pre.values())
    invented = []
    for name in sorted(names):
        pat = _binding(name)
        if pat.search(now_text) and not pat.search(pre_text):
            invented.append(name)
    return invented


PRESUPPOSITION_CHALLENGE = """Your request was to change {names}, which only \
makes sense if it already exists — but nothing in this workspace bound that name \
before this turn, and your edit is what created it.

Both answers are acceptable and you must pick the true one. If it genuinely is \
not there, call `undo_edit` to put the file back and say plainly that it does not \
exist. If you believe it was there, name the file and line you found it in."""


# The two-answer text above is measured and shipped, and it has one hole the eval
# suite structurally cannot see: **every** prompt in this repo that trips the
# trigger is one where creating the name is the wrong answer. Real requests
# include the opposite — "set FEATURE_FLAG_NEW_UI to true in config.py" on a flag
# that does not exist yet is a request to *introduce* it, and both models, given
# the two-answer challenge, call `undo_edit` and report that the change cannot be
# made. That is the create guard's failure with the work already done and then
# taken back.
#
# So the third answer is the one the text was missing, not a softening of the
# other two: every branch still requires the model to say the name was absent,
# which is the honesty property the guard exists for. Selected by
# `AGENT_PRESUPPOSITION_TEXT=v2`; the default is unchanged, because rewriting
# this particular string is the most expensive edit in this file — two earlier
# attempts cost nemotron `edit-nonexistent` 4/4 -> 0/4 *with writes*, and the
# licence to refuse is what they dropped.
PRESUPPOSITION_CHALLENGE_V2 = """Your request was to change {names}, which only \
makes sense if it already exists — but nothing in this workspace bound that name \
before this turn, and your edit is what created it.

Exactly one of these is true. Pick it, say it plainly, and leave the workspace \
matching it:

1. It is not there, and my request assumed wrongly that it was. Call `undo_edit` \
to put the file back, and say that it does not exist.
2. It is not there, and my request was really asking you to introduce it. Keep \
your edit, and say that it did not exist before and that you have added it.
3. It was there all along and you missed it. Name the file and the line you \
found it in."""


# v2 buys the false-positive repair on both models and pays for it with qwen's
# entire `multi-refuse-followup` win (4/4 -> 0/4, writing the invented constant
# all four times); nemotron is 2/4 either way. The reason is that the two
# instruments are the *same request shape* — "set FEATURE_FLAG_NEW_UI to true in
# config.py" and "change RETRY_BACKOFF to 2 in that file" are indistinguishable
# from inside the model, and the suite wants create for one and refusal for the
# other. So no wording that asks the model to guess the intent can win both, and
# v1 and v2 are that trade taken in opposite directions.
#
# v3 stops guessing. Branch 2 keeps the *reading* that the user meant to create
# — that is the part v1 lacked, which is why it reported wanted work as
# impossible — but resolves it by asking rather than assuming, so the workspace
# is left clean either way. Selected by `AGENT_PRESUPPOSITION_TEXT=v3`.
PRESUPPOSITION_CHALLENGE_V3 = """Your request was to change {names}, which only \
makes sense if it already exists — but nothing in this workspace bound that name \
before this turn, and your edit is what created it.

Exactly one of these is true. Pick it, say it plainly, and leave the workspace \
matching it:

1. It is not there, and my request assumed wrongly that it was. Call `undo_edit` \
to put the file back, and say that it does not exist.
2. It is not there, and my request may have been asking you to introduce it. \
Call `undo_edit` to put the file back, say that it does not exist and that you \
have not added it, and ask me whether to create it.
3. It was there all along and you missed it. Name the file and the line you \
found it in."""


PRESUPPOSITION_TEXTS = {
    "v2": PRESUPPOSITION_CHALLENGE_V2,
    "v3": PRESUPPOSITION_CHALLENGE_V3,
}


# Interruption, steering and resumption: the three things a REPL user can do to a
# real agent and could not do to this one. None of them is a model capability —
# they are all properties of the loop, which is why they can be added without
# touching a prompt.
#
# **Nothing is rolled back on an interrupt.** Same reasoning as `UNFINISHED_NOTE`:
# the edits are journalled and the user can undo them, and a loop that silently
# reverted half-finished work would destroy the one thing the user interrupted it
# to look at.
INTERRUPTED = """[stopped after {steps} steps, at your request]

{state}

Nothing has been rolled back. Tell me what to do next, or say "undo that"."""

# Delivered as a user message at a safe point, which is what makes it steering
# rather than a new turn: the model keeps its tool results, its plan and its
# place, and gets one more instruction.
STEER_NOTE = """The user interrupted to add this, while you were working:

{text}

It applies from here on. Keep whatever is already correct, and change course \
where this contradicts what you were about to do."""


# Over-application: the request names one place to change and the model changes
# every place that looks like it. `cascade-signature` on nemotron is the measured
# instance — *"Give slugify a max_length parameter that defaults to 40, and have
# the feed pass 20 for it"* — where it edits `text.py` and `feed.py` correctly and
# then also puts `max_length=20` into `views.py`, a caller the request never
# mentioned and one that should have kept the default. Deterministic, 3/3.
#
# The trigger cannot be "you changed a file the request did not name": the entire
# cascade suite is built on the model *finding* call sites nobody named, and a
# guard that fires on that would be telling it to undo the work. So the challenge
# does not judge — it asks, per file, and it names the one distinction that
# separates the two: whether anything else would break without this change.
#
# **On by default since 2026-08-28**, `AGENT_NO_SCOPE_CHECK=1` is the off arm.
# Guard text is the class this project has been burned by twice (the
# subject-bearing absence challenge cost nemotron `edit-nonexistent` 4/4 -> 0/4),
# so it was measured on the cascade suite itself, where required multi-file edits
# are the norm and a wrong answer is expensive. One binary, both models:
#
#   cascade    qwen 9/10 -> **10/10**,  nemotron 8/10 -> **9/10**
#   tag:edit   8/8 both arms          tag:honesty  3/3 both arms (never fires)
#   tag:multi-turn 8/9 both arms, the same case failing in each
#
# `cascade-signature` on nemotron is 3/3 with it, having been 0/3 without. The
# challenge fires on **six to nine of the ten cascade cases** in every run — so
# nearly every required multi-file edit in the suite was asked to justify itself
# — and not one of them was undone. The clause licensing changes another file
# depends on is what carries that.
SCOPE_CHALLENGE = """You have changed {count} files this turn: {files}.

The request was: "{request}"

Go through that list one file at a time. For each one, exactly one of these is \
true:

1. The request asked for this change, **or** something else you changed would \
break without it. Keep it, and say which.
2. Nothing asked for it and nothing needs it — you changed it because it \
resembled the work. Call `undo_edit` with that file's path, and say you have \
undone it.

Do not redo work that is already correct, and do not undo a change that another \
file now depends on."""


# Observed on both 7Bs: rather than call a tool, the model writes something
# formatted like a tool result — "ERROR: the term was not found in the
# workspace" — and stops. The better the real tool messages read, the more
# convincingly they get imitated, so this cannot be fixed by writing them
# differently. It has to be caught in the loop.
FABRICATION_MARKERS = (
    "no matches for", "not found in the workspace", "was too specific",
    "retry with a shorter pattern", "did you mean something else",
    "searched 0 ", "text file(s)",
)

FABRICATION_REBUKE = """That reply is formatted like a tool result, but no tool \
produced it — you wrote it yourself. Tool results only ever appear after you \
actually call a tool, and so far you have called: {calls}.

Call `grep` now, for real, with a single short identifier. If the search comes \
back with suggestions, use them."""



# The cascade suite's signature failure: the agent renames a symbol in three
# files of four, or moves a function and leaves the importers pointing at the old
# home, and then reports the whole job as done. `check_imports` finds it — but
# only if the model thinks to call it, and measured uptake says that is
# model-dependent (qwen 3 of 4 cascade cases, nemotron 1 of 4).
#
# So the loop checks instead of asking. Nothing is spent when the work is right:
# the check is a few `ast.parse` calls, no model call, and a clean result returns
# the answer untouched. Only a genuinely broken workspace costs a turn.
UNVERIFIED = """Your answer says the work is done, but the workspace disagrees. \
These are the problems your changes left behind:

{problems}

Fix them and answer again. If one of them is deliberate, say which and why."""


# How many extra steps a broken tree earns. Small on purpose: this is enough to
# read a file and fix two imports, not enough to start the task over.
REPAIR_STEPS = 4

REPAIR = """You are out of tool steps, but your own changes have left this \
workspace broken:

{problems}

You have {steps} more steps, for this and nothing else. Fix these, then answer. \
If you cannot fix them all, say plainly which ones are still broken — a partial \
change reported as finished is the worst outcome here."""


# Reported, not acted on. The measured end state of a half-done cascade is
# *redundant but working* — every import still resolves — so a mechanism that
# tried to finish the job could delete the original definition, fail to repoint
# the importers inside the steps it has, and turn a working repo into a broken
# one. Telling the human what looks unfinished has no such failure mode, and the
# person who can decide whether it matters is the one reading the answer.
UNFINISHED_NOTE = """UNFINISHED: this change may not be complete.
{reasons}
Nothing has been rolled back. Check these before relying on the change."""


# Delivered rather than offered. `review_changes` has been available for four
# sittings and was called 0 times in 34 runs of the case it was built for — the
# third time "availability is not uptake" has been measured here. A tool nobody
# calls cannot have its *content* measured, only its schema text, which is how a
# clause nobody read came to look like the mechanism. `AGENT_AUTO_REVIEW=1` hands
# the same report over unasked, once, halfway through the budget, and
# `AGENT_NO_REQUEST_SECTION=1` then decides whether the request-keyed paragraph is
# in it. The framing below is identical in both arms; only the report differs.
AUTO_REVIEW = """Before you go on, here is what you have actually changed so far:

{report}

Finish anything that is not done, then answer."""


def looks_fabricated(answer: str) -> bool:
    lowered = answer.strip().lower()
    if lowered.startswith("error:"):
        return True
    return any(marker in lowered for marker in FABRICATION_MARKERS)


@dataclass
class Trace:
    """Callbacks so the CLI can show what the agent is doing."""
    on_tool_call: Callable[[str, dict], None] = lambda name, args: None
    on_tool_result: Callable[[str, str, bool], None] = lambda name, text, ok: None
    on_thinking: Callable[[str], None] = lambda text: None
    # Only the multi-phase runner in research.py emits these; the single loop
    # has one phase and nothing to announce.
    on_phase: Callable[[str], None] = lambda label: None


@dataclass
class RunStats:
    """What happened during one `ask()`. The eval harness scores these."""
    steps: int = 0
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    tool_errors: int = 0
    repeat_blocks: int = 0
    recovered_text_calls: int = 0
    absence_challenges: int = 0
    # Times a negative answer was spared the challenge because every phrase in it
    # was quoted out of the workspace rather than claimed. Counted in both arms,
    # for the same reason `context_notices` is.
    quoted_absences: int = 0
    # Times the run was asked to *change* a name that nothing defined until the
    # run itself defined it. Off by default; see `PRESUPPOSING`.
    presupposition_challenges: int = 0
    # Times this turn was told its history no longer fits. Counted in
    # both arms — with the notice off it still says when it would have
    # fired, so the two arms stay comparable.
    context_notices: int = 0
    # Times a turn that edited more than one file was asked which of them the
    # request actually called for. Counted in both arms, like the others.
    scope_challenges: int = 0
    # Whether this turn stopped because the user asked it to, and how many
    # mid-turn instructions it was given. A run that was interrupted is not a run
    # that failed, and a scored suite has to be able to tell them apart.
    interrupted: bool = False
    steers: int = 0
    # Times the loop replaced the oldest turns with a digest of them. One model
    # call each, so this is also the mechanism's cost in calls.
    compactions: int = 0
    # The first 400 characters of each digest. Without this a failed recall turn
    # cannot be diagnosed: "the digest dropped the fact" and "the model ignored a
    # digest that had it" are the same row otherwise, and they need opposite
    # fixes. Capped because this goes into every stored result.
    digest_previews: list[str] = field(default_factory=list)
    fabrications: int = 0
    # Times the loop sent an answer back because the workspace it claimed to
    # have fixed still had broken imports. Near zero on a model that verifies
    # its own work is the expected shape.
    verify_nudges: int = 0
    # Extra steps granted because the run was about to end with a tree the agent
    # had broken, and whether the tree was still broken when it finally stopped.
    # `broken_at_end` is the number a recovery mechanism has to move: the pass
    # rate cannot tell "ran out of room" from "left the repo half-migrated".
    repair_turns: int = 0
    broken_at_end: int = 0
    # Dry-run detector output: how this run's edits look half-done. Recorded to
    # measure the trigger before anything is wired to it.
    unfinished_flags: list[str] = field(default_factory=list)
    # The same idea keyed on the request instead of on the damage: things the
    # prompt named that this run never went near. Recorded in both arms of its
    # A/B, so the counters say what the reader would have been told either way.
    unaddressed_flags: list[str] = field(default_factory=list)
    # Times the loop handed the change review over without being asked.
    auto_reviews: int = 0
    # Creations refused because the request never named the file. Counted in both
    # arms of its A/B — with the guard off the call goes through and this still
    # says how often it would have fired.
    creations_blocked: int = 0
    # Steps this run was granted before it started, recorded because a scaled
    # budget is a number the reader has to be able to see: "exhausted" means
    # something different at 12 than at 28.
    budget: int = MAX_STEPS
    budget_exhausted: bool = False
    llm_error: str | None = None
    duration_s: float = 0.0

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.tool_calls]


@dataclass
class Agent:
    client: OllamaClient
    registry: Registry
    workspace: Workspace
    # None means "size it to the request" — see `budget_for()`. An explicit
    # number always wins, because two callers need one: `--max-steps` on the
    # CLI, and the eval cases that pin a budget deliberately
    # (`edit-honesty-budget` is *built* to run out of room, and a budget that
    # quietly grew to fit the task would delete the case).
    max_steps: int | None = None
    messages: list[dict] = field(default_factory=list)
    # The request message that opened each turn, held **by identity** rather than
    # by index. Compaction needs a cut point that is a turn boundary — cutting
    # anywhere else separates an assistant's tool_calls from the tool results
    # that answer them, which is a message list no server will accept — and
    # indices do not survive the list being edited underneath them: the context
    # notice is removed and re-appended every turn, which silently shifts every
    # boundary recorded after it. Identity has no such failure mode.
    turn_marks: list[dict] = field(default_factory=list)
    trace: Trace = field(default_factory=Trace)
    stats: RunStats = field(default_factory=RunStats)
    playbook: str | None = DEFAULT
    # Present only when the write tools are registered; it is the same object
    # the tools mutate, so the loop can report what was changed.
    session: EditSession | None = None

    # Polled at every safe point. Returns text to inject, or None. The loop does
    # not know what a terminal is: `main.py` supplies a reader that checks
    # whether the user typed something while the model was working.
    steer_poll: Callable[[], str | None] | None = None

    def __post_init__(self) -> None:
        # Not dataclass fields: `interrupt()` is called from a signal handler,
        # and these must exist on every Agent however it was constructed.
        self._interrupted = False
        self._steering: list[str] = []
        if not self.messages:
            self.messages.append({
                "role": "system",
                "content": load_system_prompt(
                    self.workspace.root, self.playbook,
                    editing=self.session is not None,
                    # Keyed on the registry rather than on a flag, so the
                    # guidance appears exactly when the tools it describes do.
                    web="web_search" in self.registry.tools,
                ),
            })

    def interrupt(self) -> None:
        """Ask the running turn to stop at its next safe point.

        Safe means *the message list is well-formed*: an assistant message with
        `tool_calls` has had every one of its results appended. Stopping anywhere
        else leaves a history no server will accept on the next turn, which would
        make one interrupt cost the whole session.

        Callable from a signal handler — it sets a flag and returns.
        """
        self._interrupted = True

    def steer(self, text: str) -> None:
        """Add an instruction to the turn that is already running."""
        if text and text.strip():
            self._steering.append(text.strip())

    def _at_safe_point(self, step: int) -> str | None:
        """Deliver steering, then stop if asked. Returns the answer if stopping."""
        if self.steer_poll is not None:
            try:
                typed = self.steer_poll()
            except Exception:
                typed = None            # a broken reader must not kill the turn
            if typed:
                self.steer(typed)
        while self._steering:
            text = self._steering.pop(0)
            self.stats.steers += 1
            self.messages.append({"role": "user",
                                  "content": STEER_NOTE.format(text=text)})
            self.trace.on_thinking(f"[steering: {text}]")
        if not self._interrupted:
            return None
        self._interrupted = False       # one interrupt stops one turn
        self.stats.interrupted = True
        changed = sorted({record.path for record in self.session.history}) \
            if self.session else []
        state = ("Files changed so far: " + ", ".join(changed)
                 if changed else "No files were changed.")
        answer = INTERRUPTED.format(steps=step, state=state)
        # The assistant message keeps the history well-formed, so the next turn
        # starts from a conversation and not from a dangling tool call.
        self.messages.append({"role": "assistant", "content": answer})
        self.trace.on_thinking("[interrupted]")
        return answer

    def _compact(self) -> bool:
        """Replace the oldest turns with a digest of them. One model call.

        Returns whether anything was replaced, so the caller can fall back to the
        notice when there is nothing to compact yet — a session that has not run
        long enough to have a droppable prefix is still a session that can be
        over its window, and it must not silently get neither mechanism.
        """
        marks = [i for i, message in enumerate(self.messages)
                 if any(message is mark for mark in self.turn_marks) and i >= 1]
        if len(marks) <= COMPACT_KEEP_TURNS:
            return False
        cut = marks[-COMPACT_KEEP_TURNS]

        # An existing digest is **carried forward verbatim, never re-summarised**.
        # The first build summarised it along with everything else, and nemotron
        # showed what that costs: its third digest came back "I'm having trouble
        # parsing the exact request from your message", the fourth dutifully
        # summarised *that*, and the user's fact — present verbatim in digests one
        # and two — was gone from the session for good. One bad summary poisons
        # every summary after it, and the facts a digest exists to keep are
        # exactly the ones with no second source.
        def is_digest(message: dict) -> bool:
            return (message.get("content") or "").startswith(DIGEST_MARKER)

        # Found by content, anywhere in the list: a restated digest sits at the
        # end, not at index 1, so an index-based lookup silently stopped finding
        # it from the second compaction on.
        carried = ""
        for message in self.messages[1:]:
            if is_digest(message):
                # Sliced at the body mark, never by splitting on a blank line:
                # the headers *contain* blank lines, and an early
                # `split("\n\n", 1)` carried half of one into the body and
                # re-prepended it at every compaction.
                carried = message["content"].split(f"{DIGEST_BODY_MARK}\n", 1)[-1]
                break

        # The digest is never part of what gets summarised, wherever it sits, and
        # never survives in the tail either — exactly one exists afterwards.
        older = [m for m in self.messages[1:cut] if not is_digest(m)]
        if not older:
            return False

        prompt = SESSION_DIGEST_PROMPT.format(
            transcript=elide_middle(render_transcript(older)))
        try:
            reply = self.client.chat([{"role": "user", "content": prompt}],
                                     tools=None)
        except Exception:
            # A failed digest must not cost the turn its history: leave the
            # messages alone and let the notice fire as it did before.
            return False
        digest = (reply.content or "").strip()
        if not digest:
            return False

        body = elide_middle(f"{carried}\n\n{digest}" if carried else digest,
                            DIGEST_MAX_CHARS)
        self.messages = ([self.messages[0],
                          {"role": "user",
                           "content": f"{digest_header()}\n\n"
                                      f"{DIGEST_BODY_MARK}\n{body}"}]
                         + [m for m in self.messages[cut:] if not is_digest(m)])
        self.turn_marks = [mark for mark in self.turn_marks
                           if any(mark is kept for kept in self.messages)]
        self.stats.compactions += 1
        # The merged body, not the reply: after carry-forward the reply is only
        # the newest slice, while this field exists to say what the model can
        # still see.
        self.stats.digest_previews.append(body[:400])
        self.trace.on_thinking(
            f"[compacted {len(older)} messages into {len(digest)} chars]")
        return True

    def _restate_digest(self) -> None:
        """Move the digest to the message before the request, like the notice.

        On by default whenever compaction is; `AGENT_COMPACT_NO_RESTATE=1` puts
        the digest back at index 1, which is the arm this replaced. Measured
        need: qwen at 4096 compacted five
        times, every digest carried the user's fact verbatim, and turn 12 still
        answered by searching the workspace and reporting absence — with no notice
        live and nothing truncated. The digest sat at index 1, directly after the
        system prompt, which is the part of a long context a small model reads
        least. `CONTEXT_NOTICE` learned this first and is re-stated before every
        request; this is the same move for the same reason.

        Placed after the notice when both are present: the notice is about the
        workspace, the digest is about the conversation, and the conversation is
        what the next request continues.
        """
        for index, message in enumerate(self.messages):
            if index and (message.get("content") or "").startswith(DIGEST_MARKER):
                self.messages.append(self.messages.pop(index))
                return

    def _invented_this_turn(self, request: str, edits_at_start: int) -> list[str]:
        """Presupposed names this turn bound that nothing bound before it."""
        if not self.session:
            return []
        before: dict[str, str | None] = {}
        for record in self.session.history[edits_at_start:]:
            before.setdefault(record.path, record.before)   # earliest wins
        if not before:
            return []
        return invented_bindings(request, self.workspace, before)

    def _absence_is_quoted(self, answer: str, turn_start: int) -> bool:
        """Should this absence claim be spared the challenge?

        Counted in both arms — like the context notice — so the off arm still
        records how often it *would* have spared a turn, and the two arms stay
        comparable on one number instead of on a missing one.
        """
        output = "\n".join(
            str(m.get("content") or "")
            for m in self.messages[turn_start:]
            if m.get("role") == "tool"
        )
        if not quoted_absence(answer, output, self.workspace):
            return False
        self.stats.quoted_absences += 1
        if os.environ.get("AGENT_NO_QUOTED_ABSENCE"):
            return False
        self.trace.on_thinking("[negative is quoted from the workspace; not challenging]")
        return True

    def _import_problems(self) -> set[str]:
        """Broken imports and syntax errors, as comparable strings.

        Never raises: a verification step that can fail the run is worse than no
        verification step, and an unreadable workspace is not the agent's fault.

        `AGENT_NO_VERIFY_NUDGE=1` returns nothing, which switches the whole
        mechanism off — the "before" arm of its A/B runs the same binary.
        """
        if os.environ.get("AGENT_NO_VERIFY_NUDGE"):
            return set()
        try:
            problems, _ = check_workspace(self.workspace, limit=50)
        except Exception:
            return set()
        return {str(p) for p in problems}

    def unfinished_reasons(self) -> list[str]:
        """Ways this run's own edits look half-done — see `agent/review.py`.

        Reported to the reader of the answer (`UNFINISHED_NOTE`), never acted on.
        The same function backs the `review_changes` tool, so the model can ask
        the question the loop will judge it by.
        """
        if not self.session:
            return []
        return unfinished_reasons(self.workspace, self.session)

    def unaddressed_requests(self, request: str) -> list[str]:
        """Names the request asked about that this run never wrote or deleted.

        The one shape the three artifact-keyed guards structurally miss: work
        that was never begun leaves nothing damaged to find. Reported like the
        rest — see `agent/review.py` for why it is the narrow half of the idea.
        Never raises: a reporting extra that can fail a run is worse than none.
        """
        if not self.session:
            return []
        try:
            return unaddressed_requests(request, self.workspace, self.session)
        except Exception:
            return []

    def ask(self, user_input: str) -> str:
        self.stats = RunStats()
        if self.session:
            self.session.request = user_input
        started = time.monotonic()
        # Snapshotted here rather than inside `_ask` so that `broken_at_end` is
        # recorded on *every* exit path — answer, exhausted budget, LLM error.
        # A run that dies leaving a broken tree is exactly the case worth
        # counting, and it is the one most likely to skip a tidy return.
        self._broken_before = self._import_problems() if self.session else set()
        broken_now: set[str] = set()
        try:
            answer = self._ask(user_input)
        finally:
            self.stats.duration_s = time.monotonic() - started
            if self.session:
                broken_now = self._import_problems() - self._broken_before
                self.stats.broken_at_end = len(broken_now)
                self.stats.unfinished_flags = self.unfinished_reasons()
                self.stats.unaddressed_flags = self.unaddressed_requests(user_input)

        # The note goes to whoever reads the answer, and deliberately not into
        # `self.messages`: the model's own history should record what it said,
        # not an annotation about it, or a follow-up question in the REPL would
        # find the agent arguing with a warning it never wrote.
        # Broken imports belong in the note too. They were counted from the start
        # and never told to anyone: a run that ends by exhausting its budget
        # skips the nudge entirely, so `broken_at_end=2` was recorded while the
        # reader got a confident answer and a broken tree.
        told = self.stats.unfinished_flags + sorted(broken_now)
        # Its own switch, because it is its own mechanism: computed in both arms
        # (the counters stay comparable) and appended only in one.
        # `AGENT_NO_REQUEST_CHECK=1` is the off arm.
        if not os.environ.get("AGENT_NO_REQUEST_CHECK"):
            told = told + self.stats.unaddressed_flags
        # `AGENT_NO_UNFINISHED_NOTE=1` computes everything and appends nothing, so
        # the counters still say what the reader *would* have been told. It is the
        # last mechanism here to get a switch, and the reason is worth recording:
        # its only measured value came from rescoring stored runs, because with no
        # switch there was no arm to run. A mechanism that cannot be turned off
        # cannot be measured forwards.
        if told and not os.environ.get("AGENT_NO_UNFINISHED_NOTE"):
            reasons = "\n".join(f"  - {r}" for r in told)
            answer = f"{answer}\n\n{UNFINISHED_NOTE.format(reasons=reasons)}"
        return answer

    def _ask(self, user_input: str) -> str:
        # Before the question, not after: a notice appended to the end of a
        # session that is already over its window is the one thing guaranteed to
        # survive the server's truncation, and the model reads it immediately
        # before the request it applies to. Counted in both arms so the off arm
        # still records when it would have fired; appended only when on.
        # On by default since 2026-08-23, `AGENT_NO_CONTEXT_NOTICE=1` is the off
        # arm. It shipped on a measurement, not on the idea: two models, two
        # sittings, four cells, **11/12 -> 12/12 in every one**, and the flip is
        # visible in the trajectory rather than only in the count — turn 12 goes
        # from zero tool calls and "I don't see any reference to a tracking id"
        # to one `grep` and the right answer. Nothing that could have paid for it
        # moved: `budget_exhausted` 0 in all arms, no writes, no absence
        # challenges, turn 11 (which already passed by going to look) still
        # passes, and qwen spends *two steps fewer* with the notice on. And it
        # cannot reach the measured suites: `tag:edit` with it on is 8/8 with
        # `notices=0`, because no one-shot case gets past 17% of its window.
        num_ctx = getattr(self.client, "num_ctx", 0)
        # Compaction first, then the notice re-reads the list it left behind: a
        # successful compaction drops the fill below the threshold and the notice
        # does not fire, while a compaction that had nothing to drop leaves the
        # notice to do what it did before. The two are not alternatives — the
        # digest carries what the user said, the notice tells the model to
        # re-derive what is on disk.
        if compaction_enabled() and context_notice_due(self.messages, num_ctx):
            self._compact()
        # `AGENT_COMPACT_NOTICE` used to live here: once a session had compacted,
        # it kept the notice firing for the rest of the run, on the theory that
        # compaction had silenced the mechanism that was doing the work. The
        # theory was wrong and the arm said so — it moved no cell in six, and
        # beside a restated digest it cost two extra tool calls to reach the same
        # answer. What had actually broken `multi-long-session` was a degraded
        # digest, fixed by carrying it forward instead of re-summarising it.
        if context_notice_due(self.messages, num_ctx):
            self.stats.context_notices += 1
            if not os.environ.get("AGENT_NO_CONTEXT_NOTICE"):
                # Exactly one notice, and always the message before the current
                # request. Appending a fresh one per turn without dropping the
                # last would leave eleven copies in a twelve-turn session —
                # ~1000 est tokens of warning, consuming the window it exists to
                # warn about and accelerating the overflow. Re-stating it rather
                # than posting it once is deliberate: the server drops oldest
                # first, so a single early notice is the first thing to go.
                self.messages = [m for m in self.messages
                                 if m.get("content") != CONTEXT_NOTICE]
                self.messages.append({"role": "user", "content": CONTEXT_NOTICE})
                self.trace.on_thinking("[context is full; telling it to re-derive]")
        if compaction_enabled() and not os.environ.get("AGENT_COMPACT_NO_RESTATE"):
            self._restate_digest()
        request = {"role": "user", "content": user_input}
        self.turn_marks.append(request)
        self.messages.append(request)
        # Where this turn starts, so the quoted-absence check reads *this* turn's
        # tool output. In a session the earlier turns are still in `self.messages`,
        # and a phrase quoted five turns ago is not evidence about this answer.
        turn_start = len(self.messages)
        # Same idea for the journal: what this turn invented is what this turn
        # bound, not what an earlier turn in the session left behind.
        edits_at_start = len(self.session.history) if self.session else 0
        seen: set[str] = set()
        failed: dict[str, int] = {}
        challenged = False
        presupposed = False
        rebuked = False
        nudged = False
        # Whatever was already broken before the agent touched anything, taken in
        # `ask()`. Without it the loop would report a repo's pre-existing mess as
        # the agent's doing, which is both wrong and unfixable by it.
        broken_before = self._broken_before

        # Sized here rather than at construction: the request is what carries
        # the size of the task, and one agent answers many of them.
        if self.max_steps is not None:
            budget = self.max_steps
        elif self.session is not None:
            budget = budget_for(user_input, self.workspace)
        else:
            # Read-only work keeps the constant it was tuned on, so the 22-case
            # baseline is measured against the same number it always was.
            budget = MAX_STEPS
        granted = budget
        self.stats.budget = granted
        repaired = False
        reviewed = False
        step = 0

        scoped = False
        while step < budget:
            # Before spending a model call: the cheapest place to stop, and the
            # place a steering message is most useful — it reaches the model
            # before its next decision rather than after it.
            stopped = self._at_safe_point(step)
            if stopped is not None:
                return stopped
            step += 1
            self.stats.steps = step
            try:
                reply = self.client.chat(self.messages, self.registry.schemas())
            except LLMError as exc:
                self.stats.llm_error = str(exc)
                return f"[llm error] {exc}"

            if not reply.tool_calls:
                answer = reply.content.strip() or "(the model returned nothing)"

                # Reject invented tool output before anything else: it is not a
                # weak answer, it is the model impersonating the tool layer.
                if not rebuked and looks_fabricated(answer):
                    rebuked = True
                    self.stats.fabrications += 1
                    called = ", ".join(self.stats.names) or "no tools at all"
                    self.messages.append({"role": "assistant", "content": answer})
                    self.messages.append({
                        "role": "user",
                        "content": FABRICATION_REBUKE.format(calls=called),
                    })
                    self.trace.on_thinking("[rejecting fabricated tool output]")
                    continue

                # Make an unsupported "it isn't there" earn itself, once.
                # Not on a task that actually changed a file: the challenge tells
                # the agent to go and grep, which is the wrong instruction when
                # the negative is incidental ("there was no such line, so I added
                # one") rather than a failed search.
                # Counted in both arms, delivered in one — the pattern every other
                # mechanism here already follows. `AGENT_NO_ABSENCE_CHALLENGE=1` is
                # the off arm, and it was the last one missing: this is the most
                # frequently firing text mechanism in the suite (313 runs of the
                # stored 3582) and until now the only one that could not be an arm
                # at all. `evals/pairs.py` reports exactly that, and the rule it
                # runs into is this file's own, written about the unfinished note:
                # a mechanism that cannot be turned off cannot be measured
                # forwards. Adding the switch does not change the default.
                edited = bool(self.session and self.session.history)
                due = (not challenged and not edited and claims_absence(answer)
                       and not self._absence_is_quoted(answer, turn_start))
                if due:
                    # Counted first and unconditionally, so the off arm still
                    # records every run where the challenge would have fired and
                    # the two arms stay comparable.
                    challenged = True
                    self.stats.absence_challenges += 1
                if due and not os.environ.get("AGENT_NO_ABSENCE_CHALLENGE"):
                    called = ", ".join(self.stats.names) or "no tools at all"
                    self.messages.append({"role": "assistant", "content": answer})
                    self.messages.append({
                        "role": "user",
                        "content": CHALLENGE.format(calls=called),
                    })
                    self.trace.on_thinking("[challenging unsupported negative answer]")
                    continue

                # More than one file changed, and the request may have named
                # only one of them. Counted in both arms; delivered in one.
                if not scoped and self.session:
                    touched = sorted({r.path for r
                                      in self.session.history[edits_at_start:]})
                    if len(touched) > 1:
                        scoped = True
                        self.stats.scope_challenges += 1
                        if scope_check_enabled():
                            self.messages.append({"role": "assistant",
                                                  "content": answer})
                            self.messages.append({
                                "role": "user",
                                "content": SCOPE_CHALLENGE.format(
                                    count=len(touched),
                                    files=", ".join(touched),
                                    request=user_input.strip()),
                            })
                            self.trace.on_thinking(
                                f"[{len(touched)} files changed; asking which "
                                "the request called for]")
                            continue

                # The request said *change* X; nothing bound X before this turn
                # and this turn's edit is what bound it. Off by default — see the
                # note on `PRESUPPOSING`: this keys on request phrasing, and the
                # create guard priced well retrospectively on exactly that basis
                # and then failed on the first real request.
                if not presupposed and self.session:
                    invented = self._invented_this_turn(user_input, edits_at_start)
                    if invented:
                        # Counted in both arms, like `context_notices` and
                        # `quoted_absences`: an off arm that reports a missing
                        # number instead of a zero is not a baseline.
                        presupposed = True
                        self.stats.presupposition_challenges += 1
                    if invented and os.environ.get("AGENT_PRESUPPOSITION_GUARD"):
                        self.messages.append({"role": "assistant", "content": answer})
                        self.messages.append({
                            "role": "user",
                            "content": (
                                PRESUPPOSITION_TEXTS.get(
                                    os.environ.get("AGENT_PRESUPPOSITION_TEXT", ""),
                                    PRESUPPOSITION_CHALLENGE,
                                )
                            ).format(names=", ".join(invented)),
                        })
                        self.trace.on_thinking(
                            f"[{', '.join(invented)} did not exist before this turn]")
                        continue

                # Last gate, and the only one that checks the deliverable rather
                # than the prose: if the agent changed files and left an import
                # pointing at something that no longer exists, it has not
                # finished, whatever the answer says.
                if not nudged and self.session and self.session.history:
                    broken = self._import_problems() - broken_before
                    if broken:
                        nudged = True
                        self.stats.verify_nudges += 1
                        listed = "\n".join(f"  {p}" for p in sorted(broken))
                        self.messages.append({"role": "assistant", "content": answer})
                        self.messages.append({
                            "role": "user",
                            "content": UNVERIFIED.format(problems=listed),
                        })
                        self.trace.on_thinking("[the workspace does not match the answer]")
                        continue

                self.messages.append({"role": "assistant", "content": answer})
                return answer

            if reply.recovered_from_text:
                self.stats.recovered_text_calls += 1
            if reply.content.strip():
                self.trace.on_thinking(reply.content.strip())

            self.messages.append({
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {"function": {"name": c.name, "arguments": c.arguments}}
                    for c in reply.tool_calls
                ],
            })

            for call in reply.tool_calls:
                self._run_call(call, seen, failed)

            # The other safe point: every result for this step is on the list, so
            # the history is well-formed and a long tool sequence can be stopped
            # part-way through the work rather than only between model calls.
            stopped = self._at_safe_point(step)
            if stopped is not None:
                return stopped

            # Halfway through the budget, with edits on disk and steps left to
            # act on what it says. Once: a report repeated every step is noise,
            # and the second copy would be answering a question nobody asked
            # twice.
            if (not reviewed and os.environ.get("AGENT_AUTO_REVIEW")
                    and self.session and self.session.history
                    and step >= (budget + 1) // 2):
                reviewed = True
                self.stats.auto_reviews += 1
                self.messages.append({
                    "role": "user",
                    "content": AUTO_REVIEW.format(
                        report=full_report(self.workspace, self.session)),
                })
                self.trace.on_thinking("[handing over the change review]")

            # The budget is about to end. If the agent's own edits left the tree
            # broken, that is the one case worth spending more steps on: a run
            # that stops here ships a half-migrated repo — two definitions of the
            # same function, imports pointing at a name that no longer exists —
            # which is worse than an honest "I ran out of room". The budget
            # exists to stop a runaway loop, not to ship wreckage.
            #
            # Granted once, only on damage this run caused, and only when there
            # is something specific to say. `AGENT_NO_REPAIR_TURN=1` disables it.
            if step >= budget and not repaired and self.session and self.session.history:
                broken = self._import_problems() - broken_before
                if broken and not os.environ.get("AGENT_NO_REPAIR_TURN"):
                    repaired = True
                    self.stats.repair_turns += 1
                    budget += REPAIR_STEPS
                    listed = "\n".join(f"  {p}" for p in sorted(broken))
                    self.messages.append({
                        "role": "user",
                        "content": REPAIR.format(steps=REPAIR_STEPS, problems=listed),
                    })
                    self.trace.on_thinking("[granting a repair turn: the tree is broken]")

        # Budget exhausted: force a final prose answer instead of dying silently.
        self.stats.budget_exhausted = True
        self.messages.append({
            "role": "user",
            "content": (
                f"You have used all {granted} tool steps. "
                "Answer now using only what the tools already returned. Do not call any more tools."
            ),
        })
        try:
            final = self.client.chat(self.messages, tools=None)
        except LLMError as exc:
            self.stats.llm_error = str(exc)
            return f"[llm error] {exc}"
        answer = final.content.strip() or "(step budget exhausted with no answer)"
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def _unrequested_creation(self, call: ToolCall) -> str | None:
        """Refuse to create a file the request never named — or None to allow it.

        The failure this closes, read off the traces rather than guessed at:
        `cascade-signature` finishes both required edits, then writes itself
        `run_tests.py` and thrashes on that file until the budget dies. It is
        reaching for a way to verify its work, and the answer to *that* is not an
        interpreter — `run_command` was built, scored cascades 3/4 -> 1/4, failed
        to stop this very behaviour, and let one run delete a file through
        `python3 -c` with no diff, no approval and no undo entry. So: keep the
        narrow affordance (`check_imports`), and refuse the invented file.

        Scoped as tightly as the evidence allows. Only `write_file`, only a path
        that does not exist yet — overwriting an existing file is a different act
        with a different guard — and only when the request names no such file.

        **Off by default, and the reason is a lesson about the suite.** It priced
        beautifully on stored runs — 188 requested creations allowed, 23
        unrequested blocked, all 23 forbidden by their case — and then failed on
        the first real request: "create a downloader module" produced
        `downloader/generic_downloader.py`, which the guard refused because the
        prompt never spelled the path. **Eval prompts name paths literally**
        ("into a new file `src/util/slug.py`"); real requests almost never do, so
        the separation was an artifact of fixture phrasing, measured on a
        distribution that does not resemble use. The live A/B had already found it
        fires zero times in 39 runs. No measured benefit, a real cost:
        `AGENT_CREATE_GUARD=1` opts in, nothing turns it on by default.
        """
        if self.session is None or not os.environ.get("AGENT_CREATE_GUARD"):
            return None
        if call.name != "write_file":
            return None
        path = str(call.arguments.get("path") or "").strip()
        if not path:
            return None
        try:
            if self.workspace.resolve(path).exists():
                return None
        except Exception:
            # A path the sandbox rejects is the sandbox's error to report, not
            # this guard's to pre-empt.
            return None
        if path_requested(self.session.request or "", path):
            return None
        return (
            f"ERROR: {path} is a new file, and the request does not ask for it. "
            "Do not add scratch files, test scripts or runners — nothing here "
            "executes them. If you want to check the change, call check_imports. "
            "Otherwise edit the files the request is about, or give your final "
            "answer."
        )

    def _run_call(self, call: ToolCall, seen: set[str],
                  failed: dict[str, int] | None = None) -> None:
        self.trace.on_tool_call(call.name, call.arguments)
        self.stats.tool_calls.append((call.name, call.arguments))
        failed = {} if failed is None else failed

        # Only *successful* calls are remembered as repeats. Fingerprinting a
        # call that never ran turned two guards against each other: the read gate
        # answers `edit_file` on an unopened file with "call read_file first,
        # then edit it", the model complies — and the identical retry was then
        # refused as a duplicate. Measured on `cascade-move`: 18 repeat blocks
        # and 20 tool errors in a run whose edit anchors were all correct, with
        # the same five-word edit rejected five times.
        #
        # A genuinely bad call still cannot loop forever: the same *failing* call
        # is allowed twice, which covers "fix the precondition and retry" and
        # stops there. `AGENT_STRICT_REPEATS=1` restores the old behaviour.
        strict = bool(os.environ.get("AGENT_STRICT_REPEATS"))
        fingerprint = call.name + json.dumps(call.arguments, sort_keys=True, default=str)

        # First, because a call that must not run should not be fingerprinted as
        # a repeat or counted as a failure: the repeat guard's own lesson is that
        # remembering calls which never ran turns two guards against each other.
        refusal = self._unrequested_creation(call)
        if refusal is not None:
            self.stats.creations_blocked += 1
            text, ok = refusal, False
        elif fingerprint in seen:
            self.stats.repeat_blocks += 1
            text, ok = (
                f"ERROR: you already called {call.name} with these exact arguments and "
                "got a result above. Re-read it, then either call a different tool or "
                "give your final answer.",
                False,
            )
        elif failed.get(fingerprint, 0) >= (1 if strict else 2):
            self.stats.repeat_blocks += 1
            text, ok = (
                f"ERROR: {call.name} with these exact arguments has already failed "
                "twice. Change something — a different anchor, a different file, or "
                "a different tool — rather than sending it again.",
                False,
            )
        else:
            text, ok = self.registry.dispatch(call.name, call.arguments)
            if ok:
                seen.add(fingerprint)
            else:
                failed[fingerprint] = failed.get(fingerprint, 0) + 1
        if not ok:
            self.stats.tool_errors += 1

        self.trace.on_tool_result(call.name, text, ok)
        self.messages.append({
            "role": "tool",
            "name": call.name,
            "tool_name": call.name,
            "content": text,
        })
