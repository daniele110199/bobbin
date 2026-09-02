# Against aider

What this project is worth measured against the tool people already use, on the
same tasks, with the same local models, scored by one harness.

*Method notes for [bobbin](../README.md).*

---

## How it was run, before any numbers

Everything here is a config choice that moves the result, so it goes first
rather than in a footnote.

| | |
|---|---|
| aider version | **0.86.2**, via `--model ollama_chat/<model>`, `OLLAMA_API_BASE=127.0.0.1:11434` |
| aider flags | `--yes --no-check-update --no-auto-commits --no-show-model-warnings` (fixtures also `--no-git --no-analytics`) |
| **timeout** | **900s** per run on the real repo, 420s on the fixtures. **7 of the 24 aider runs on the real repo hit it**, all on nemotron. A longer ceiling would change aider's column |
| models | `qwen3-coder:30b` and `nemotron-3.5-lightning` (32.9B), Ollama, one RTX 4060 (8GB) — so the 32B runs partly on CPU for both tools |
| step budget (ours) | 40 on the real repo; the case's own pin on fixtures |
| reps | **one sample per cell**, except the budget experiment below. This matters — see [what the numbers do not carry](#what-the-numbers-do-not-carry) |

Three arms, because how you invoke aider is most of what it scores:

- **ours** — the loop as it ships, `--allow-edits`, auto-approved
- **aider-told** — the files the *prompt itself names* are added to the chat.
  aider's intended use, and what a user who knows the file would type
- **aider-find** — no files added, `--map-tokens 1024`, so aider's repo map has
  to locate them

On cascade cases only the files the prompt names are ever handed over, never the
full `may_touch` set — on a cascade, `may_touch` is the answer key.

## Fixtures: 19 cases × 2 models

Scored **on disk only**, for both tools: the case's file checks plus
`may_touch`. The prose half of our eval (`expect_all`, denial patterns) is
dropped, because those regexes were written against our agent's answers and
would score style rather than work.

| shape | ours | aider-told | aider-find |
|---|---|---|---|
| single-file edits | 15/16 | 15/16 | 14/16 |
| cascades + repair | **19/22** | 5/22 | 4/22 |
| **all** | **34/38** | 20/38 | 18/38 |

Per model, ours / told / find: qwen3-coder **17/19** / 11/19 / 9/19; nemotron
**17/19** / 9/19 / 9/19.

The margin is entirely cascades. On single-file edits the three arms are the
same tool. That is the expected shape — a multi-file refactor is exactly the
work a loop does and a single patch round does not.

## Real repo: `pallets/click` @36baa15

12,674 lines across 17 source files, **judged by its own 1991 tests**. A task
passes only if the change is on disk (a grep-level assertion, no regexes over
prose) *and* `pytest` is still green. Neither half of that oracle was written by
this project or by aider, which is the point.

| | ours | aider-told | aider-find |
|---|---|---|---|
| qwen3-coder:30b | 2/6 | 3/6 | 3/6 |
| nemotron 32.9B | **4/6** | 1/6 | 1/6 |
| **both** | **6/12** | 4/12 | 4/12 |

**The fixture advantage does not transfer.** Four-to-one on our own cases
becomes 6-to-4 on real code, which on twelve cells is inside the noise.
Everyone passes the single-file edit. Cross-file renames succeed roughly once in
six attempts across both tools.

Two things in that table are real rather than noise:

- **The do-nothing task.** Asked to change `MAX_ARGUMENT_COUNT`, a constant that
  does not exist in click, our agent explored and left the tree untouched on
  both models. aider timed out at 900s in both arms **with `src/` modified and
  the test suite red** — on a task whose correct answer is to change nothing.
  This is the fixture suite's `edit-nonexistent` shape reproducing on code
  nobody wrote for us.
- **Speed, the other way.** aider is faster with qwen (median 108s against our
  296s). With the 32B model it is the one that becomes impractical: 7 of its 12
  nemotron runs never finished inside 900 seconds.

## Was 40 steps the constraint? No.

qwen failed both cross-file renames on click having used its full budget (40/40
and 44/40), so the obvious hypothesis was that budgets sized on 24-file fixtures
do not transfer. A first probe re-ran them at 80 steps: one passed. That looked
like a confirmation and it was not — both re-runs finished in 8 and 13 steps,
never reaching even the *old* ceiling, so the extra budget was untested.

The matched design, 3 reps at each budget, everything else fixed:

| qwen, real repo | 40 steps | 80 steps |
|---|---|---|
| `real-rename-across-files` | 0/3 | 0/3 |
| `real-rename-internal` | 1/3 | 1/3 |
| runs that reached their budget | **4/6** | **0/6** |

Doubling the budget changed neither pass rate, and at 80 no run ever reached its
ceiling — the agent stops on its own at 33–53 steps. **The 40-step ceiling was a
symptom of runs going nowhere, not the cause of the failures.** Pooling every
sample: `real-rename-across-files` is 0/7, `real-rename-internal` 3/7.

The finding underneath it is the uncomfortable one: across those 12 runs the
agent modified `src/` in 11 and turned click's suite **red in 8**, while
reporting success. The same shape of task is 19/22 on 24-file fixtures.

So the honest headline is not "our budgets don't transfer", which was fixable.
It is: **cascade competence at 24-file scale does not survive at 12k lines** —
for us at any budget tried, and for aider at any invocation tried.

## What the numbers carry

- Parity with aider on real code with small local models.
- A large, reproducible advantage on multi-file refactors at fixture scale.
- A specific, repeated advantage on *not* doing the wrong thing when the request
  presupposes something false.
- A documented method with its negative results attached.

## What the numbers do not carry

- **General superiority.** They do not show it and we do not claim it.
- **Statistical weight per cell.** The fixture table and 8 of the 12 real-repo
  cells are **one sample each**. The budget experiment above is exactly why that
  caveat is not boilerplate: a single sample pointed at the wrong conclusion
  until three reps overturned it.
- **A verdict on aider tuned by someone who uses it daily.** These are default
  invocations by someone who does not. The harness ships so that can be
  corrected — if a cell is misconfigured, change it and re-run.

## Four harness bugs, and why the harness ships

Each of these silently favoured a different side, and each was caught only
because a result looked wrong:

1. aider's own bookkeeping files (`.aider.*`) scored as unrequested edits,
   failing it on the do-nothing task for something that is not an edit to the
   code. `dirty()` is now scoped to `src/`.
2. The venv had click installed editable from the pristine checkout, so `pytest`
   inside a copy imported the *original* package and reported 1991 passing
   however broken the copy was. Fixed with an explicit `PYTHONPATH`, and
   verified by breaking a copy on purpose: green without the fix, red with it.
3. `grep` omits the filename prefix when given a single file, so a check keyed
   on `".py:"` counted zero for every single-file assertion — reporting
   "no change" for tools that had made the change correctly. Caught because one
   one-line task failed for all three arms at once, which is not a thing three
   different tools do.
4. The single-rep budget probe above, which pointed at the wrong conclusion.

Numbers from a comparison like this are worth exactly what the harness is, so
both are public.

## Reproducing it

```bash
evals/compare/setup.sh                  # clone click @36baa15, build the test venv
python3 evals/compare/realrepo.py results.json          # the real-repo matrix
python3 evals/compare/budget_matched.py budget.json     # the 40-vs-80 experiment
python3 evals/compare/score.py                          # every table on this page
```

`evals/compare/results/recovered.py` holds the cells that were run before a
reboot took the raw rows; they carry `source: recovered-from-session-log` and
contain only what the run actually printed — arm, pass/fail, wall seconds.
