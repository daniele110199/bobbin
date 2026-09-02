"""Multi-phase research: survey -> plan -> investigate -> synthesise.

The single-loop agent in `loop.py` answers "find X". It cannot answer "I have a
problem, tell me how this works", because one loop has one context: every grep
dump, every file it opened and every dead end stay in front of it until it
answers. A 7B with 32k context is full after three files, and the failure mode
in the README follows directly — it reads one file, does not find the answer,
and stops.

This module splits that into phases, each with its own context and its own tool
subset:

    survey       no model at all — the directory tree and the symbols this
                 repo actually defines, computed directly
    plan         one model call, sees the survey, emits 2-3 subtasks
    investigate  one throwaway sub-agent per subtask, isolated context,
                 4 steps, cannot answer — only report facts
    synthesise   one agent, sees a dossier of *verified* facts, answers

Two properties do the real work, and neither is a prompt rule:

1. **Context isolation.** A sub-agent's grep dumps die with it. The parent
   never sees them, only the handful of facts they produced. Reading five
   files costs the final context roughly five lines.

2. **Citation checking.** A sub-agent's report is not trusted. Every path,
   identifier and number in it is checked against the raw tool output that
   sub-agent actually received, and any sentence containing something that was
   never in a tool result is deleted before the parent sees it. The parent is
   therefore reading evidence, not prose. This is the same principle as
   `vocab.py` — the repo supplies the facts, the model supplies the meaning —
   applied to the model's own summary.

A dropped claim is not a lost answer: the searches that sub-agent ran are
reported as fact in its place, so "nothing verified here" stays visible instead
of turning into a confident wrong sentence.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import vocab
from .llm import LLMError, OllamaClient
from .loop import Agent, RunStats, Trace
from .prompts import load_research_prompt
from .sandbox import Workspace
from .tools import Registry

MAX_SUBTASKS = 3
GATHER_STEPS = 4
SYNTH_STEPS = 6

# Restricted tool subsets per phase. A tool a phase must not use is removed from
# its schema list rather than forbidden in prose — a small model ignores the
# prose and burns one of four steps on it.
#
# Gatherers have no `list_files`: the survey already told the planner what the
# tree looks like, and a sub-agent with a four-step budget that starts by
# wandering the directory tree has spent a quarter of it learning nothing.
# The synthesiser has no `find_files` either: at that point it is confirming one
# fact, not starting a new exploration.
GATHER_TOOLS = ["grep", "read_file", "find_files"]
SYNTH_TOOLS = ["grep", "read_file"]

SURVEY_DEPTH = 2
SURVEY_MAX_ENTRIES = 120
SURVEY_IN_DOSSIER_LINES = 30   # the tree the answer phase gets to keep
MAX_EVIDENCE_LINES = 8      # raw lines shown per tool call in the dossier
MAX_SUBTASK_LINES = 20      # raw lines shown per subtask in the dossier


# --- phase 1: survey ------------------------------------------------------
# No model call. This cannot be wrong and cannot be hallucinated, so it is the
# one part of the plan that is free.

def survey(ws: Workspace, registry: Registry) -> str:
    """The structure brief: what is here, and what it defines."""
    tree, _ = registry.dispatch("list_files", {"path": ".", "depth": SURVEY_DEPTH})
    entries = tree.splitlines()
    if len(entries) > SURVEY_MAX_ENTRIES:
        entries = entries[:SURVEY_MAX_ENTRIES] + [
            f"... and {len(entries) - SURVEY_MAX_ENTRIES} more entries"
        ]

    idx = vocab.index(ws, ws.root)
    symbols = vocab.definition_listing(idx)

    parts = ["FILES:", *entries]
    if symbols:
        parts += ["", "SYMBOLS THIS WORKSPACE DEFINES:"]
        parts += [f"  {s}" for s in symbols]
    return "\n".join(parts)


# --- phase 2: plan --------------------------------------------------------

@dataclass
class Subtask:
    id: str
    question: str
    hint: str = ""          # the identifier to grep first
    grounded: bool = False  # True if `hint` was corrected against the vocabulary


_SUBTASK_RE = re.compile(
    r"^\s*(?:[-*\d.\s]*)?SUBTASK\s*:\s*(.+?)\s*(?:\|\s*GREP\s*:\s*(.*?))?\s*$",
    re.I | re.M,
)


def parse_plan(text: str, limit: int = MAX_SUBTASKS) -> list[Subtask]:
    """Pull `SUBTASK: ... | GREP: ...` lines out of whatever the model replied.

    A line format, not JSON: the 7Bs get JSON *shape* wrong often enough that a
    plan phase built on it would fail on the models this project exists for.
    Anything that is not a SUBTASK line — preamble, apology, markdown fence — is
    ignored rather than parsed.
    """
    out: list[Subtask] = []
    for question, hint in _SUBTASK_RE.findall(text or ""):
        question = question.strip().strip("*_`")
        hint = (hint or "").strip().strip("*_`\"'")
        # A hint with spaces is a phrase, not an identifier; keep the first word.
        if " " in hint:
            hint = hint.split()[0]
        if question:
            out.append(Subtask(id=f"S{len(out) + 1}", question=question, hint=hint))
        if len(out) >= limit:
            break
    return out


def default_plan(question: str, limit: int = MAX_SUBTASKS) -> list[Subtask]:
    """The plan to use when the model's plan is unusable.

    Derived from the question's own content words, so the investigate phase
    still runs — a failed plan call degrades the research, it does not skip it.
    """
    terms = vocab.query_terms(question)[:limit]
    if not terms:
        return [Subtask(id="S1", question=question)]
    return [
        Subtask(id=f"S{i}", question=f"find where {term!r} appears and what it is",
                hint=term)
        for i, term in enumerate(terms, 1)
    ]


def ground_plan(subtasks: list[Subtask], idx: vocab.Index) -> list[Subtask]:
    """Replace planned search terms that do not exist with ones that do.

    The planner is a language model asked for an identifier, so it sometimes
    supplies the question's word (`normalize`, `authenticate`) instead of the
    code's (`normalise_amount`, `sign_in`). Sending a sub-agent to grep a word
    that is not in the repo wastes its whole budget, and it is cheap to check
    here: the vocabulary index already knows every identifier that exists.
    """
    lowered = {w.lower() for w in idx.words}
    for task in subtasks:
        if not task.hint or task.hint.lower() in lowered:
            continue
        near = vocab.near_misses(task.hint, idx, limit=1)
        if near:
            task.hint = near[0][0]
            task.grounded = True
    return subtasks


# --- phase 3: investigate -------------------------------------------------

@dataclass
class Evidence:
    """One tool call a sub-agent made, and what came back."""
    tool: str
    args: dict
    text: str
    ok: bool

    def signature(self) -> str:
        shown = ", ".join(f"{k}={v!r}" for k, v in self.args.items()
                          if v not in (None, "", False))
        return f"{self.tool}({shown})"


class EvidenceLog:
    """Records a sub-agent's real tool traffic, via the existing Trace hooks.

    Deliberately not a change to the loop: what a sub-agent is allowed to report
    has to be checkable against what the tools actually returned, and the loop
    already announces both.
    """

    def __init__(self) -> None:
        self.items: list[Evidence] = []
        self._pending: tuple[str, dict] | None = None

    def trace(self) -> Trace:
        return Trace(on_tool_call=self._call, on_tool_result=self._result)

    def _call(self, name: str, args: dict) -> None:
        self._pending = (name, dict(args or {}))

    def _result(self, name: str, text: str, ok: bool) -> None:
        args = self._pending[1] if self._pending and self._pending[0] == name else {}
        self._pending = None
        self.items.append(Evidence(tool=name, args=args, text=text, ok=ok))

    def raw(self) -> str:
        """Everything the tools returned. What a claim gets checked against."""
        return "\n".join(e.text for e in self.items if e.ok)


@dataclass
class Finding:
    subtask: Subtask
    facts: list[str] = field(default_factory=list)    # verified FACT lines
    dropped: list[str] = field(default_factory=list)  # deleted for lack of evidence
    evidence: list[Evidence] = field(default_factory=list)
    steps: int = 0
    calls: list[tuple[str, dict]] = field(default_factory=list)


# --- citation checking ----------------------------------------------------
# A claim is checkable if it contains something a tool result would have to have
# shown: a path, a code identifier, or a value. Ordinary English is not checked —
# the point is to catch invented facts, not to grade the prose around them.

_PATH_RE = re.compile(
    r"\b[\w./-]*\w+\.(?:py|md|txt|json|toml|ya?ml|cfg|ini|js|jsx|ts|tsx|rs|go|java|c|h|cpp|sh)\b",
    re.I,
)
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_CAMEL_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b")
_CONST_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)*\b")
# Two or more digits, or anything with a decimal point. A bare single digit is
# usually counting ("2 files"), not quoting, and checking it costs good facts.
_NUMBER_RE = re.compile(r"\b\d*\.\d+\b|\b\d{2,}\b")
# ... unless it is plainly counting something.
_COUNTED = re.compile(
    r"\b(?:file|files|match|matches|line|lines|result|results|place|places|"
    r"time|times|entry|entries|occurrence|occurrences)\b", re.I,
)


def checkable_atoms(claim: str) -> list[str]:
    """The parts of a claim that a tool result must contain for it to stand."""
    atoms: list[str] = []
    for rx in (_PATH_RE, _SNAKE_RE, _CAMEL_RE, _CONST_RE):
        atoms += rx.findall(claim)
    for match in _NUMBER_RE.finditer(claim):
        trailing = claim[match.end():match.end() + 12]
        if not _COUNTED.match(trailing.strip()):
            atoms.append(match.group())
    return atoms


def unsupported(claim: str, evidence: str) -> list[str]:
    """Which checkable atoms of `claim` never appeared in the tool output."""
    haystack = evidence.lower()
    missing: list[str] = []
    for atom in checkable_atoms(claim):
        if atom.lower() in haystack:
            continue
        # A path may be quoted short ("tax.py" for "src/billing/tax.py").
        tail = atom.rsplit("/", 1)[-1]
        if tail != atom and tail.lower() in haystack:
            continue
        if atom not in missing:
            missing.append(atom)
    return missing


_FACT_RE = re.compile(r"^\s*(?:[-*\d.\s]*)?FACT\s*:\s*(.+?)\s*$", re.I | re.M)
_NOTHING_RE = re.compile(r"^\s*(?:nothing found|none|n/?a)\.?\s*$", re.I)


def verify(report: str, evidence: str) -> tuple[list[str], list[str]]:
    """Split a sub-agent's report into (facts backed by tool output, deleted).

    Falls back to treating every prose line as a claim when the model ignored
    the FACT format, which the 7Bs do about a third of the time. The format is
    a convenience for reading; the check does not depend on it.
    """
    claims = [c.strip() for c in _FACT_RE.findall(report or "")]
    if not claims:
        claims = [line.strip(" -*\t") for line in (report or "").splitlines()
                  if len(line.strip()) > 12]

    kept: list[str] = []
    dropped: list[str] = []
    for claim in claims:
        if _NOTHING_RE.match(claim):
            continue
        missing = unsupported(claim, evidence)
        if missing:
            dropped.append(f"{claim}  [unsupported: {', '.join(missing[:4])}]")
        elif claim not in kept:
            kept.append(claim)
    return kept, dropped


# --- the dossier ----------------------------------------------------------

_GREP_LINE = re.compile(r"^\S.*?:\d+:")        # path:line: text
_READ_LINE = re.compile(r"^\s*\d+\|")          # numbered file line
# grep's per-term report for a name that matched nothing. Indented, so it is not
# a hit line, and it is the single most valuable line in the result: it is the
# only evidence of absence a tool ever produces.
_ABSENCE_LINE = re.compile(r"NO matches")


def _relevant(text: str, terms: list[str], limit: int) -> list[str]:
    """The lines of a tool result worth putting in front of the next phase.

    A `read_file` result is up to 400 numbered lines and the parent context is
    the scarce resource this whole module exists to protect, so quote the lines
    that mention what the subtask was about, not the file.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    absences = [ln.strip() for ln in lines if _ABSENCE_LINE.search(ln)]
    hits = [ln for ln in lines if _GREP_LINE.match(ln) or _READ_LINE.match(ln)]
    if not hits:
        return (absences + lines)[:limit]

    lowered = [t.lower() for t in terms if t]
    if lowered and len(hits) > limit:
        wanted = [ln for ln in hits if any(t in ln.lower() for t in lowered)]
        if wanted:
            hits = wanted
    # Absences first: they never survive a truncation otherwise, and a dossier
    # that shows only the names that were found is how "two of three" becomes
    # "all three" in the answer.
    return (absences + hits)[:limit]


def survey_block(brief: str, limit: int = SURVEY_IN_DOSSIER_LINES) -> str:
    """The survey, as the dossier's first entry.

    It was already computed for the planner, it costs no model call, and being
    a `list_files` result it cannot be wrong. Throwing it away before the answer
    phase was a real bug and cost a real case: `top-level-dirs` asks what the
    top-level directories are, gatherers have no `list_files` by design, and so
    the one phase that *had* the answer never passed it on.

    It goes in as evidence rather than as prose in the prompt, because that is
    what it is — and because the prompt is where a 7B charges for every word.
    """
    lines = [ln for ln in brief.splitlines() if ln.strip() and ln.strip() != "FILES:"]
    if "SYMBOLS THIS WORKSPACE DEFINES:" in brief:
        lines = lines[:lines.index("SYMBOLS THIS WORKSPACE DEFINES:")]
    if len(lines) > limit:
        lines = lines[:limit] + [f"... and {len(lines) - limit} more entries"]
    body = "\n".join(f"    {ln}" for ln in lines)
    return ("[S0] what this workspace contains "
            "(from list_files, not from a model)\n" + body)


def dossier(findings: list[Finding], brief: str = "") -> str:
    """The only thing the synthesis phase sees of the investigation."""
    blocks: list[str] = [survey_block(brief)] if brief.strip() else []
    for f in findings:
        terms = [f.subtask.hint] + vocab.query_terms(f.subtask.question)
        lines = [f"[{f.subtask.id}] {f.subtask.question}"]

        if f.facts:
            lines += [f"  FACT: {fact}" for fact in f.facts]
        else:
            searched = ", ".join(e.signature() for e in f.evidence) or "nothing"
            lines.append(f"  (no verified fact. searched: {searched})")

        quoted = 0
        for e in f.evidence:
            if not e.ok or quoted >= MAX_SUBTASK_LINES:
                continue
            excerpt = _relevant(e.text, terms,
                                min(MAX_EVIDENCE_LINES, MAX_SUBTASK_LINES - quoted))
            if not excerpt:
                continue
            quoted += len(excerpt)
            lines.append(f"  from {e.signature()}:")
            lines += [f"    {ln}" for ln in excerpt]

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# --- the whole thing ------------------------------------------------------

@dataclass
class ResearchReport:
    """What each phase did. The eval harness reads this."""
    survey_chars: int = 0
    plan_source: str = "model"        # "model" | "fallback"
    subtasks: list[Subtask] = field(default_factory=list)
    grounded_hints: int = 0
    findings: list[Finding] = field(default_factory=list)
    dossier: str = ""
    dropped_claims: int = 0
    verified_facts: int = 0
    model_calls: int = 0


@dataclass
class ResearchAgent:
    """Same interface as `Agent`: `.ask(question) -> str`, plus `.stats`."""
    client: OllamaClient
    registry: Registry
    workspace: Workspace
    max_subtasks: int = MAX_SUBTASKS
    gather_steps: int = GATHER_STEPS
    max_steps: int = SYNTH_STEPS
    trace: Trace = field(default_factory=Trace)
    stats: RunStats = field(default_factory=RunStats)
    report: ResearchReport = field(default_factory=ResearchReport)
    #: the synthesis phase's transcript, for `--dump-messages`. The sub-agents'
    #: transcripts are deliberately not kept — discarding them is the point.
    messages: list[dict] = field(default_factory=list)

    def ask(self, question: str) -> str:
        self.stats = RunStats()
        self.report = ResearchReport()
        started = time.monotonic()
        try:
            return self._ask(question)
        finally:
            self.stats.duration_s = time.monotonic() - started

    def _ask(self, question: str) -> str:
        root = self.workspace.root

        self.trace.on_phase("survey")
        brief = survey(self.workspace, self.registry)
        self.report.survey_chars = len(brief)

        self.trace.on_phase("plan")
        subtasks = self._plan(question, brief, root)
        idx = vocab.index(self.workspace, self.workspace.root)
        subtasks = ground_plan(subtasks, idx)
        self.report.subtasks = subtasks
        self.report.grounded_hints = sum(t.grounded for t in subtasks)
        for task in subtasks:
            self.trace.on_phase(f"  {task.id}: {task.question}"
                                + (f"  [grep {task.hint}]" if task.hint else ""))

        self.trace.on_phase(f"investigate ({len(subtasks)} subtasks)")
        findings = [self._investigate(task, question, root) for task in subtasks]
        self.report.findings = findings
        self.report.verified_facts = sum(len(f.facts) for f in findings)
        self.report.dropped_claims = sum(len(f.dropped) for f in findings)

        pack = dossier(findings, brief)
        self.report.dossier = pack

        self.trace.on_phase(
            f"synthesise ({self.report.verified_facts} verified facts, "
            f"{self.report.dropped_claims} claims dropped, {len(pack)} chars)"
        )
        return self._synthesise(question, pack, root)

    # -- phases ------------------------------------------------------------

    def _plan(self, question: str, brief: str, root) -> list[Subtask]:
        prompt = load_research_prompt(
            "plan", root=root, question=question, survey=brief,
            max_subtasks=self.max_subtasks,
        )
        try:
            reply = self.client.chat([{"role": "user", "content": prompt}], tools=None)
            self.report.model_calls += 1
            self.stats.steps += 1
        except LLMError as exc:
            self.stats.llm_error = str(exc)
            self.report.plan_source = "fallback"
            return default_plan(question, self.max_subtasks)

        subtasks = parse_plan(reply.content, self.max_subtasks)
        if not subtasks:
            self.report.plan_source = "fallback"
            return default_plan(question, self.max_subtasks)
        return subtasks

    def _investigate(self, task: Subtask, question: str, root) -> Finding:
        log = EvidenceLog()
        hint = (f"\nStart by searching for `{task.hint}` — it is a name that really "
                f"exists in this workspace.\n") if task.hint else ""
        prompt = load_research_prompt(
            "gather", root=root, question=question, subtask=task.question, hint=hint,
        )

        sub = Agent(
            client=self.client,
            registry=self.registry.subset(GATHER_TOOLS),
            workspace=self.workspace,
            max_steps=self.gather_steps,
            messages=[{"role": "system", "content": prompt}],
            trace=log.trace(),
        )
        report = sub.ask(task.question)
        self.report.model_calls += sub.stats.steps
        self._absorb(sub.stats)

        # A dead sub-agent reports nothing. Its "[llm error] ..." string carries
        # no checkable atom, so verify() would wave it through as a fact.
        if sub.stats.llm_error:
            return Finding(subtask=task, evidence=log.items, steps=sub.stats.steps,
                           calls=sub.stats.tool_calls)

        facts, dropped = verify(report, log.raw())
        return Finding(subtask=task, facts=facts, dropped=dropped,
                       evidence=log.items, steps=sub.stats.steps,
                       calls=sub.stats.tool_calls)

    def _synthesise(self, question: str, pack: str, root) -> str:
        prompt = load_research_prompt(
            "synthesise", root=root, question=question, dossier=pack,
        )
        final = Agent(
            client=self.client,
            registry=self.registry.subset(SYNTH_TOOLS),
            workspace=self.workspace,
            max_steps=self.max_steps,
            messages=[{"role": "system", "content": prompt}],
            trace=self.trace,
        )
        answer = final.ask(question)
        self.report.model_calls += final.stats.steps
        self._absorb(final.stats)
        self.messages = final.messages
        return answer

    def _absorb(self, other: RunStats) -> None:
        """Roll a phase's stats into the run total, so evals score one row."""
        s = self.stats
        s.steps += other.steps
        s.tool_calls += other.tool_calls
        s.tool_errors += other.tool_errors
        s.repeat_blocks += other.repeat_blocks
        s.recovered_text_calls += other.recovered_text_calls
        s.absence_challenges += other.absence_challenges
        s.fabrications += other.fabrications
        s.budget_exhausted = s.budget_exhausted or other.budget_exhausted
        s.llm_error = s.llm_error or other.llm_error
