#!/usr/bin/env python3
"""The same head-to-head, on a real repository, judged by its own test suite.

`pallets/click` at 36baa15: 17 source files, 12,674 lines, 1991 tests that run in
under three seconds. Ten times the size of any fixture in this project, with an
import graph and a public API that were not designed to be refactored by anyone.

**The oracle is not ours.** A task passes only if

  1. the change is actually on disk (a grep-level assertion, no regexes over
     prose), and
  2. `pytest` is still green.

Neither half was written by this project or by aider, which is the point: our own
suite can only say whether our agent does what we expected, and this can say
whether either tool can edit a codebase without breaking it.

The tasks are the shapes the fixture suite measures — a cross-file rename, a move,
a signature change, a single-file edit, and a request for something that does not
exist — asked of code nobody wrote for the purpose.

Selection is by environment, so a single lost cell can be re-run without editing
the file (that mattered: the first version of this harness lived in /tmp and a
reboot took it):

    REALREPO_ONLY=real-nonexistent REALREPO_MODELS=nemotron-3.5-lightning \
        python3 realrepo.py out.json
"""
import json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repo root, wherever it was cloned
from agent.sandbox import Workspace
from evals.run import build_agent

HERE = Path(__file__).resolve().parent
PRISTINE = HERE / "click-probe"
PYTEST = HERE / "clickenv/bin/python"
MODELS = (os.environ.get("REALREPO_MODELS") or
          "qwen3-coder:30b,nemotron-3.5-lightning").split(",")
ONLY = set(filter(None, (os.environ.get("REALREPO_ONLY") or "").split(",")))
AIDER_TIMEOUT = 900
STEPS = int(os.environ.get("REALREPO_STEPS", 40))   # generous and equal


def hits(root: Path, pattern: str, where: str = "src tests") -> int:
    """Matching lines in Python files under `where`.

    `-H` is not optional: grep omits the filename prefix when it is given a
    single file, so a filter keyed on `".py:"` silently counted zero for every
    check scoped to one file — reporting `change=False` for tools that had made
    the change correctly. Found because a one-line default change failed for all
    three arms at once, which is not a thing three different tools do.
    """
    out = subprocess.run(["grep", "-rnH", pattern] + where.split(),
                         cwd=root, capture_output=True, text=True)
    return sum(1 for line in out.stdout.splitlines()
               if line.split(":", 1)[0].endswith(".py"))


def tests_pass(root: Path) -> bool:
    """`pytest` inside the copy — importing the *copy's* source.

    Without an explicit `PYTHONPATH` a test run inside a copy imports the
    installed package and reports 1991 passing however broken the copy is.
    Verified by breaking a copy on purpose: green without this line, red with
    it. An oracle that cannot see the change it is judging is worse than no
    oracle.
    """
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    done = subprocess.run([str(PYTEST), "-m", "pytest", "-q", "-x", "--no-header",
                           "-p", "no:cacheprovider"],
                          cwd=root, env=env, capture_output=True, text=True,
                          timeout=900)
    return done.returncode == 0


TASKS = [
    {
        "id": "real-rename-across-files",
        "prompt": ("Rename the function split_arg_string to split_argument_string "
                   "everywhere it is used, including the tests."),
        "files": ["src/click/shell_completion.py"],
        "check": lambda r: (hits(r, "split_arg_string") == 0
                            and hits(r, "def split_argument_string") == 1),
    },
    {
        "id": "real-rename-internal",
        "prompt": ("Rename the function join_options to format_option_list "
                   "everywhere it is used."),
        "files": ["src/click/formatting.py"],
        "check": lambda r: (hits(r, "join_options") == 0
                            and hits(r, "def format_option_list") == 1),
    },
    {
        "id": "real-move-function",
        "prompt": ("Move the make_str function from src/click/utils.py to "
                   "src/click/_compat.py and update everything that imports it."),
        "files": ["src/click/utils.py", "src/click/_compat.py"],
        "check": lambda r: (hits(r, "def make_str", "src/click/_compat.py") == 1
                            and hits(r, "def make_str", "src/click/utils.py") == 0),
    },
    {
        "id": "real-signature",
        "prompt": ("Give split_arg_string a keyword-only parameter posix that "
                   "defaults to True, and have src/click/parser.py pass "
                   "posix=True explicitly at every call site."),
        "files": ["src/click/shell_completion.py", "src/click/parser.py"],
        "check": lambda r: (hits(r, "posix", "src/click/shell_completion.py") >= 1
                            and hits(r, "posix=True", "src/click/parser.py") >= 1),
    },
    {
        "id": "real-single-file",
        "prompt": ("In src/click/formatting.py, change the default width of "
                   "wrap_text from 78 to 100."),
        "files": ["src/click/formatting.py"],
        "check": lambda r: hits(r, "width: int = 100", "src/click/formatting.py") == 1,
    },
    {
        "id": "real-nonexistent",
        "prompt": ("Change the value of MAX_ARGUMENT_COUNT in src/click/core.py "
                   "to 20."),
        "files": ["src/click/core.py"],
        # Nothing to do: the name is not in the codebase. Passing means the tree
        # is untouched — the disk-level reading of "say so instead of inventing".
        "check": lambda r: hits(r, "MAX_ARGUMENT_COUNT") == 0,
        "must_not_edit": True,
    },
]


def fresh(git: bool) -> Path:
    root = Path(tempfile.mkdtemp()) / "click"
    shutil.copytree(PRISTINE, root, ignore=shutil.ignore_patterns(".git"))
    if git:
        for command in (["git", "init", "-q"],
                        ["git", "-c", "user.email=e@x", "-c", "user.name=n", "add", "-A"],
                        ["git", "-c", "user.email=e@x", "-c", "user.name=n",
                         "commit", "-qm", "base"]):
            subprocess.run(command, cwd=root, capture_output=True)
    return root


def dirty(root: Path) -> bool:
    """Did anything under src/ change?

    Scoped to `src/` on purpose: aider writes its own bookkeeping files
    (`.aider.*`) into the working tree, and counting those as unrequested edits
    failed it on the do-nothing task for something that is not an edit to the
    code.
    """
    out = subprocess.run(["diff", "-rq", str(PRISTINE / "src"), str(root / "src")],
                         capture_output=True, text=True)
    return bool(out.stdout.strip())


def run_ours(task, model, root):
    opts = SimpleNamespace(host="http://127.0.0.1:11434", num_ctx=None,
                           allow_edits=True, mode="direct", max_steps=STEPS,
                           playbook="default", subtasks=2, gather_steps=6)
    agent = build_agent(model, Workspace(root), opts)
    answer = agent.ask(task["prompt"])
    return {"answer": answer[:300], "steps": agent.stats.steps}


def run_aider(task, model, root, files):
    command = ["aider", "--model", f"ollama_chat/{model}", "--yes",
               "--no-check-update", "--no-auto-commits", "--no-show-model-warnings",
               "--map-tokens", "0" if files else "1024", *files,
               "--message", task["prompt"]]
    env = dict(os.environ, OLLAMA_API_BASE="http://127.0.0.1:11434")
    try:
        done = subprocess.run(command, cwd=root, env=env, capture_output=True,
                              text=True, timeout=AIDER_TIMEOUT)
        return {"answer": (done.stdout or "")[-300:], "steps": None}
    except subprocess.TimeoutExpired:
        return {"answer": "[timed out]", "steps": None}


def main() -> int:
    out = Path(sys.argv[1])
    rows = []
    for model in MODELS:
        for task in [t for t in TASKS if not ONLY or t["id"] in ONLY]:
            for arm in ("ours", "aider-told", "aider-find"):
                root = fresh(git=arm != "ours")
                started = time.monotonic()
                try:
                    if arm == "ours":
                        extra = run_ours(task, model, root)
                    else:
                        files = task["files"] if arm == "aider-told" else []
                        extra = run_aider(task, model, root, files)
                except Exception as exc:
                    extra = {"answer": f"[crashed: {exc}]", "steps": None}
                changed = dirty(root)
                try:
                    did_it = bool(task["check"](root))
                except Exception:
                    did_it = False
                if task.get("must_not_edit"):
                    did_it = did_it and not changed
                try:
                    green = tests_pass(root)
                except Exception:
                    green = False
                rows.append({"task": task["id"], "model": model, "arm": arm,
                             "change_made": did_it, "tests_pass": green,
                             "passed": did_it and green, "touched_src": changed,
                             "seconds": round(time.monotonic() - started, 1), **extra})
                print(f"{'PASS' if rows[-1]['passed'] else 'fail'}  "
                      f"{model.split(':')[0][:12]:12} {arm:11} {task['id']:26} "
                      f"change={did_it!s:5} tests={green!s:5} {rows[-1]['seconds']:6.0f}s",
                      flush=True)
                shutil.rmtree(root.parent, ignore_errors=True)
                out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
