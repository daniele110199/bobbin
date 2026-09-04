# bobbin

**A small coding agent for small local models.** A dependency-free agent
runtime for local models via Ollama. Stdlib only: no `pip install`, no
`requests`, no `ollama` package, no ripgrep.

![bobbin renaming a function across four files](media/demo.gif)

*A real run, unedited: `qwen3-coder:30b` renaming `slugify` to `make_slug`
across four files, then leaving the vendored copy in `node_modules/` alone.
90 seconds of real time, played at 2.6x with long pauses clipped. The frames
are the bytes the agent actually wrote ([how it was made](media/README.md)).
It does not pass every time; the rates are in the tables below.*

It is one loop:

```
model emits a structured tool call
  -> we execute it ourselves
  -> the result goes back as a `tool` message
  -> the model gets another turn
```

Repeat until the model answers in prose or the step budget runs out. Everything
in `agent/` exists to make that loop survive a 7B.

## What it is for

Small local coding models, 7B to 32B on one consumer GPU, fail in the agent
loop in specific, repeatable ways: they read one file and give up, they loop on
the same call forever, they emit tool calls as prose, they rename a symbol in
one file and report the refactor done. Almost every fix in this project is a
change to the *environment*, the tools, the errors, the loop, rather than to
the prompt or the model.

That bet is measured rather than asserted. The 3,354 runs behind it are **in
this repository**, under `evals/results/`; not a summary of them, the runs
themselves, so any number quoted here can be recomputed rather than taken on
trust. The things that were built, measured, and **removed** for zero benefit
are written down next to the things that worked.

## Where it stands against aider

Same tasks, same local models, one harness, scored on disk. Full method,
disclosures and caveats: **[docs/comparison.md](docs/comparison.md)**.

On this project's own fixtures, 19 cases × 2 models, **three reps per cell,
342 runs:**

| shape | ours | aider-told | aider-find |
|---|---|---|---|
| single-file edits | 46/48 | 45/48 | 42/48 |
| cascades + repair | **61/66** | 11/66 | 10/66 |
| **all** | **107/114** | 56/114 | 52/114 |

On single-file edits the three arms are the same tool. The entire margin is
cascades, where it is close to six to one.

On `pallets/click` at 12,674 lines, judged by its own 1991 tests. **Three reps
per cell, 108 runs:**

| | ours | aider-told | aider-find |
|---|---|---|---|
| qwen3-coder:30b | 6/18 | **9/18** | **9/18** |
| nemotron 32.9B | **12/18** | 3/18 | 4/18 |
| **both** | **18/36** | 12/36 | 13/36 |

**Read the second table before the first, and read it by row.** The fixture
advantage does not transfer, and on real code the answer depends on which model
you run:

- **On qwen3-coder, aider wins.** 9/18 against our 6/18, entirely on one task:
  moving a function between modules, which aider does 3 times out of 3 and we
  fail 3 times out of 3.
- **On nemotron, we win by a lot.** 12/18 against 3/18, and aider hit the
  900-second ceiling on 19 of its 36 runs with that model.

Aggregated, that is 50% against 33%, which is parity with a wide spread, not a
lead. At the level of individual tasks: of 12 cells, we win 4, aider wins 1,
and 7 are ties (mostly tasks neither tool can do).

**These outcomes are stable, not noisy.** 33 of the 36 cells were unanimous
across all three reps, so the failures are capability, not luck. That also
means the earlier single-sample table happened to be right, which we could not
have known without paying for the reps.

So the claim this project makes is narrow and specific:

- **parity with aider on real code with small local models**, not superiority,
  and behind it on one of the two models tested;
- **a large advantage on multi-file refactors at fixture scale** (107/114
  against 56/114), which does not survive the jump to 12k lines;
- **a specific advantage on not doing the wrong thing**: asked to change a
  constant that does not exist in click, this agent left the tree untouched in
  all 6 runs across both models, while on nemotron aider modified `src/` and
  left the suite red in 5 of 6;
- and a documented method **with its negative results attached**, which is the
  part that is usually missing.

Caveats belong next to the numbers, not under them: aider 0.86.2 at default
settings by someone who does not use it daily, and a 900s timeout that 19 of its
72 real-repo runs hit, all on nemotron. Both tables are **3 reps per cell**, 450
runs in total. The harness ships so any of that can be corrected and re-run.

## What you need

- **Python 3.9+**: the standard library and nothing else. There is no
  `requirements.txt` because there is nothing to install. (Developed and tested
  on 3.14; no syntax or API newer than 3.9 is used, but older versions are
  untested.)
- **[Ollama](https://ollama.com) running locally**, and at least one model
  pulled:

  ```bash
  ollama pull qwen2.5-coder:7b    # small and fine to start with
  ollama pull qwen3-coder:30b     # what most numbers here were measured on
  ```

  With no `--model`, the agent uses whatever `ollama list` shows first, so
  the model you just pulled is the one it runs. Point elsewhere with `--host`
  if Ollama is not on `127.0.0.1:11434`.

Then clone and run. There is no build step and nothing to compile:

```bash
git clone <this repo> && cd llm-agent-project
python3 tests.py        # 755 tests, no network, no model needed
./install.sh            # symlinks `bobbin` into ~/.local/bin
bobbin qwen2.5-coder:7b
```

`install.sh` only makes a symlink; there is no package to install, so `git
pull` updates the command. If you would rather not touch `~/.local/bin`, skip
it: `./bobbin` and `./main.py` work identically from the checkout.

Reproducing the [aider comparison](docs/comparison.md) additionally needs
`aider` on `PATH`, plus `git` and network for `evals/compare/setup.sh`.

## Quickstart

```bash
bobbin                                   # first model in `ollama list`, current dir
bobbin qwen3-coder:30b                   # or name one
bobbin --root ~/some/repo
bobbin -p "what does this project do?"   # one-shot
bobbin --allow-edits                     # editing on, every diff confirmed
bobbin --mode research -p "trace what happens when the CLI runs"
```

Naming a model you have not pulled fails immediately with the list of the ones
you have, rather than a 404 from Ollama halfway through.

Drive the tools with no model involved. The fastest way to answer "was that the
tool layer or the model?":

```bash
./tools_cli.py grep pattern='def build' file_glob='*.py'
./tools_cli.py --schemas                    # exact JSON the model receives
```

```bash
python3 tests.py                            # 755 tests, no network
python3 -m evals.run --cases tag:edit       # score a suite
```

`bobbin --help` lists every flag.

## The five things that make small models work here

Each one came from watching a 7B fail.

1. **Errors are returned to the model, never raised**: a tool error becomes a
   `tool` message that says how to fix the call.
2. **Argument coercion**: models send `"10"` for an int and invent parameters;
   coerce what you can, drop what you can't.
3. **Text-emitted tool calls are recovered**: the 7Bs never use the native tool
   protocol. This is the single most load-bearing piece of the project.
4. **Every result is budget-capped and truncation is announced**, so the model
   knows its view was partial.
5. **Repeated-call detection**: without it, small models loop forever.

## The comparison in full

[docs/comparison.md](docs/comparison.md) has the method behind the two tables
above: how aider was invoked, the timeout that 7 of its 24 real-repo runs hit,
the matched-rep experiment that overturned the flattering explanation, and the
four harness bugs found along the way, each of which favoured a different
side.

## Layout

```
agent/       the loop, tools, sandbox, edit session, context handling
prompts/     system contract, playbook, per-phase research prompts
evals/       cases, fixtures, runner, scoring, 3,354 stored runs
evals/compare/   the aider head-to-head and its real-repo oracle
bobbin         the command (a symlinked launcher; `install.sh` links it)
main.py      chat REPL
tools_cli.py run tools without a model
tests.py     755 tests, no network
```

## Licence

MIT. See [LICENSE](LICENSE).

## Reproducing the comparison

```bash
evals/compare/setup.sh                              # clone click, build the oracle
python3 evals/compare/realrepo.py results.json      # the real-repo matrix
python3 evals/compare/budget_matched.py budget.json # the 40-vs-80 experiment
python3 evals/compare/score.py                      # every table above
```

Requires `aider` on `PATH` and Ollama serving the two models. The click checkout
and its venv are not committed; `setup.sh` rebuilds both and verifies the oracle
reports 1991 passing tests.
