# A small local-model coding agent

A dependency-free agent runtime for local models via Ollama. Stdlib only:
no `pip install`, no `requests`, no `ollama` package, no ripgrep.

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

On this project's own fixtures: 19 cases × 2 models:

| shape | ours | aider-told | aider-find |
|---|---|---|---|
| single-file edits | 15/16 | 15/16 | 14/16 |
| cascades + repair | **19/22** | 5/22 | 4/22 |
| **all** | **34/38** | 20/38 | 18/38 |

On `pallets/click` at 12,674 lines, judged by its own 1991 tests:

| | ours | aider-told | aider-find |
|---|---|---|---|
| qwen3-coder:30b | 2/6 | 3/6 | 3/6 |
| nemotron 32.9B | **4/6** | 1/6 | 1/6 |
| **both** | **6/12** | 4/12 | 4/12 |

**Read the second table before the first.** The fixture advantage does not
transfer: four-to-one on our own cases becomes 6-to-4 on real code, which on
twelve cells is inside the noise. Cross-file renames succeed about once in six
attempts *for both tools*. We tested the flattering explanation, that our step
budgets were sized on 24-file fixtures, with a matched 3-rep experiment, and it
is false: doubling the budget changed nothing, and the agent stops on its own
well short of the larger ceiling.

So the claim this project makes is narrow and specific:

- **parity with aider on real code with small local models**, not superiority;
- **a large, reproducible advantage on multi-file refactors at fixture scale**;
- **a specific advantage on not doing the wrong thing** , asked to change a
  constant that does not exist in click, this agent left the tree untouched on
  both models, while aider timed out with `src/` modified and the test suite
  red;
- and a documented method **with its negative results attached**, which is the
  part that is usually missing.

Caveats belong next to the numbers, not under them: aider 0.86.2 at default
settings by someone who does not use it daily, a 900s timeout that 7 of 24 aider
runs hit, and **one sample per cell** everywhere except the budget experiment.
The harness ships so any of that can be corrected and re-run.

## What you need

- **Python 3.9+** , the standard library and nothing else. There is no
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

Then clone and run , there is no build step and nothing to compile:

```bash
git clone <this repo> && cd llm-agent-project
python3 tests.py        # 755 tests, no network, no model needed
./install.sh            # symlinks `bobbin` into ~/.local/bin
bobbin qwen2.5-coder:7b
```

`install.sh` only makes a symlink , there is no package to install, so `git
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

Drive the tools with no model involved , the fastest way to answer "was that the
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

1. **Errors are returned to the model, never raised** , a tool error becomes a
   `tool` message that says how to fix the call.
2. **Argument coercion** , models send `"10"` for an int and invent parameters;
   coerce what you can, drop what you can't.
3. **Text-emitted tool calls are recovered** , the 7Bs never use the native tool
   protocol. This is the single most load-bearing piece of the project.
4. **Every result is budget-capped and truncation is announced** , so the model
   knows its view was partial.
5. **Repeated-call detection** , without it, small models loop forever.

## The comparison in full

[docs/comparison.md](docs/comparison.md) has the method behind the two tables
above: how aider was invoked, the timeout that 7 of its 24 real-repo runs hit,
the matched-rep experiment that overturned the flattering explanation, and the
four harness bugs found along the way , each of which favoured a different
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

MIT , see [LICENSE](LICENSE).

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
