#!/usr/bin/env python3
"""This agent against aider on the fixture suite, same models, one scorer.

Three arms per case:

  ours          the loop, as it ships
  aider-told    `aider <files the prompt names> --message "<prompt>"` — aider's
                intended use, and what a user who knows the file would type
  aider-find    `aider --map-tokens 1024 --message "<prompt>"` with no files
                added, so its repo map has to locate them

**Scored on disk only**, for both tools: the case's `FileCheck`s plus
`may_touch`. The prose half of the eval (`expect_all`, denial patterns) is
dropped, because those regexes were written against our agent's answers and
would be scoring style rather than work. That costs us `edit-nonexistent`'s
"say it does not exist" — on disk it becomes "change nothing", which is the
honest tool-neutral reading of the same requirement.

Only the files the *prompt* names are handed over, never `may_touch`: on a
cascade case `may_touch` is the answer key.

One rep is 114 runs, about 2.8 hours. Launch it detached:

    setsid nohup evals/compare/versus.py out.json > rep.log 2>&1 < /dev/null &
"""
import json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.loop import task_files
from agent.sandbox import Workspace
from evals.cases import CASCADE_B_CASES, CASCADE_CASES, EDIT_CASES, REPAIR_CASES
from evals.run import build_agent
from evals.score import diff_snapshots, score_workspace, snapshot_tree

EVALS = ROOT / "evals"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("versus.json")
MODELS = (os.environ.get("VERSUS_MODELS") or
          "qwen3-coder:30b,nemotron-3.5-lightning").split(",")
CASES = EDIT_CASES + CASCADE_CASES + CASCADE_B_CASES + REPAIR_CASES
ONLY = set(filter(None, (os.environ.get("VERSUS_ONLY") or "").split(",")))
REP = int(os.environ.get("VERSUS_REP", 1))
AIDER_TIMEOUT = 420


def fresh(case, git: bool = False) -> Path:
    """aider is built around git.

    With `--no-git` it prints "Repo-map: disabled" and cannot locate a file
    nobody handed it, which would make the find column a test of a feature that
    was switched off. So every aider run gets a real repository.
    """
    root = Path(tempfile.mkdtemp()) / "repo"
    shutil.copytree(EVALS / case.fixture, root)
    if git:
        for command in (["git", "init", "-q"],
                        ["git", "-c", "user.email=e@x", "-c", "user.name=n",
                         "add", "-A"],
                        ["git", "-c", "user.email=e@x", "-c", "user.name=n",
                         "commit", "-qm", "fixture"]):
            subprocess.run(command, cwd=root, capture_output=True)
    return root


def deliverable(changes: dict) -> dict:
    """Neither tool's own bookkeeping is part of the deliverable.

    aider writes `.aider.chat.history.md` and a tags cache into the repo it
    works in, and a `.gitignore` for them; scoring those as unrequested edits
    fails it for existing. Match on every path component, not the basename: the
    cache is `.aider.tags.cache.v4/cache.db`.
    """
    def keep(path: str) -> bool:
        parts = Path(path).parts
        return not (any(part.startswith(".aider") for part in parts)
                    or parts[0] in (".git", ".gitignore"))
    return {key: [p for p in paths if keep(p)] for key, paths in changes.items()}


def named_files(case, root: Path) -> list[str]:
    """Paths the prompt spells out, and only those.

    Not `task_files()`, which resolves a *symbol* to every file that defines or
    uses it: on `cascade-rename` that returns all four call sites, which is the
    answer key. A user adding files to an aider chat can only add the ones the
    request mentions; finding the rest is the work being measured.
    """
    found = []
    for candidate in re.findall(r"[\w./-]+\.(?:py|md|txt|json|yml|toml)",
                                case.prompt):
        if (root / candidate).is_file():
            found.append(candidate)
    return sorted(set(found))


def run_ours(case, model, root: Path) -> dict:
    opts = SimpleNamespace(host="http://127.0.0.1:11434", num_ctx=None,
                           allow_edits=True, mode="direct", max_steps=None,
                           playbook="default", subtasks=2, gather_steps=6)
    agent = build_agent(model, Workspace(root), opts,
                        max_steps=case.budget_for_model(model))
    answer = agent.ask(case.prompt)
    return {"answer": answer[:400], "steps": agent.stats.steps}


def run_aider(case, model, root: Path, files: list[str]) -> dict:
    command = ["aider", "--model", f"ollama_chat/{model}", "--yes",
               "--no-check-update", "--no-auto-commits",
               "--no-show-model-warnings"]
    command += ["--map-tokens", "0" if files else "1024"]
    command += files
    command += ["--message", case.prompt]
    env = dict(os.environ, OLLAMA_API_BASE="http://127.0.0.1:11434")
    try:
        done = subprocess.run(command, cwd=root, env=env, capture_output=True,
                              text=True, timeout=AIDER_TIMEOUT)
        tail = (done.stdout or "")[-400:]
    except subprocess.TimeoutExpired:
        tail = "[timed out]"
    return {"answer": tail, "steps": None}


def main() -> int:
    rows = []
    for model in MODELS:
        for case in [c for c in CASES if not ONLY or c.id in ONLY]:
            for arm in ("ours", "aider-told", "aider-find"):
                root = fresh(case, git=arm != "ours")
                before = snapshot_tree(root)   # after git init, so .git is not "created"
                started = time.monotonic()
                try:
                    if arm == "ours":
                        extra = run_ours(case, model, root)
                    else:
                        files = named_files(case, root) if arm == "aider-told" else []
                        extra = run_aider(case, model, root, files)
                        if arm == "aider-told" and not files:
                            extra["note"] = "prompt names no file; nothing to add"
                except Exception as exc:                       # a crash is a fail
                    extra = {"answer": f"[crashed: {exc}]", "steps": None}
                changes = deliverable(diff_snapshots(before, snapshot_tree(root)))
                problems = score_workspace(case, root, changes)
                row = {"case": case.id, "model": model, "arm": arm, "rep": REP,
                       "passed": not problems, "problems": problems,
                       "seconds": round(time.monotonic() - started, 1), **extra}
                rows.append(row)
                print(f"{'PASS' if row['passed'] else 'fail'}  {model.split(':')[0][:12]:12} "
                      f"{arm:11} {case.id:26} {row['seconds']:6.1f}s", flush=True)
                shutil.rmtree(root.parent, ignore_errors=True)
                OUT.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
