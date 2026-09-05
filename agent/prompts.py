"""Loading the prompt layer from `prompts/`.

The system contract and the tool playbook are text files, not Python string
constants, so they can be tuned and A/B'd against `evals/run.py` without
touching code. `--playbook none` drops the playbook, which is how you measure
whether it is actually earning its place.
"""

from __future__ import annotations

import os
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_FILE = PROMPTS_DIR / "system.md"
PLAYBOOK_FILE = PROMPTS_DIR / "playbook.md"
EDITING_FILE = PROMPTS_DIR / "editing.md"
WEB_FILE = PROMPTS_DIR / "web.md"

RESEARCH_DIR = PROMPTS_DIR / "research"

DEFAULT = "default"
NONE = "none"

# system.md opens by calling the workspace read-only, which stops being true the
# moment the write tools are registered. It is patched here rather than reworded
# in the file so the read-only prompt stays byte-for-byte what the 22-case
# baseline was measured against — these models are sensitive enough that
# re-wrapping a line has flipped a case.
READ_ONLY_PHRASE = "inside a read-only workspace at"
WRITABLE_PHRASE = "inside a workspace at"


def load_system_prompt(root: str | Path, playbook: str | None = DEFAULT,
                       editing: bool = False, web: bool = False) -> str:
    """Assemble the system prompt. `playbook` is 'default', 'none', or a path.

    A missing file raises rather than silently degrading: a prompt that quietly
    lost its playbook would make every eval number after it meaningless.

    `web` appends `web.md`, the same way `editing` appends `editing.md`, and for
    the same reason: the guidance is only true when the tools are registered, and
    a suite that never sees those tools must not pay for the text. Every number
    on record was measured without it.

    **Off by default, because it was measured and it did not work.** It was
    written for a real failure: `web-search-then-fetch` fails 11 of 12 runs on
    qwen3-coder:30b, identically every time — twelve steps, eleven of them
    searching a workspace that does not contain the word `quaystone`,
    `web_search` as the last call, budget exhausted, `fetch_url` never called,
    and an answer claiming it "cannot access the actual documentation from the
    tools available to me".

    One rep a side against two off-reps, and the text is worse than nothing:

    * **nemotron went 2/2 to 0/1, and the trajectory says why.** Without the text
      it ran `web_search` then `fetch_url` and passed. With it, it ran `grep`,
      `web_search`, and answered off the snippet. The bullet telling it to search
      the web first landed; the bullet telling it to open the page did not, and
      the model that needed no guidance at all was the one the text broke.
    * **On qwen3-coder it changed nothing** — the same twelve calls in the same
      order. Its failure is therefore not a guidance gap: it does not ignore the
      web tools for want of being told, so more prose will not reach it. What is
      left is structural — eleven calls spent on a workspace with no match in it,
      against a budget of twelve.

    Kept, switched off, rather than deleted: the diagnosis it produced is worth
    more than the text, and a future attempt should be measured against this one
    rather than starting from the same blank page. `AGENT_WEB_PLAYBOOK=1` turns
    it on.
    """
    parts = [_read(SYSTEM_FILE)]

    if playbook not in (None, NONE):
        path = PLAYBOOK_FILE if playbook == DEFAULT else Path(playbook)
        parts.append(_read(path))

    if editing:
        parts.append(_read(EDITING_FILE))

    if web and os.environ.get("AGENT_WEB_PLAYBOOK"):
        parts.append(_read(WEB_FILE))

    # str.replace, not str.format: the playbook contains literal braces.
    text = "\n\n".join(parts).replace("{root}", str(root))
    if editing:
        text = text.replace(READ_ONLY_PHRASE, WRITABLE_PHRASE)
    return text


def load_research_prompt(name: str, **fields: object) -> str:
    """Load `prompts/research/<name>.md` and substitute {placeholders}.

    Same deal as the playbook: the phase prompts are text files so they can be
    tuned and measured without touching the loop. str.replace, not str.format —
    these files contain literal braces in their examples.
    """
    text = _read(RESEARCH_DIR / f"{name}.md")
    for key, value in fields.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def _read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
