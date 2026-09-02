"""Loading the prompt layer from `prompts/`.

The system contract and the tool playbook are text files, not Python string
constants, so they can be tuned and A/B'd against `evals/run.py` without
touching code. `--playbook none` drops the playbook, which is how you measure
whether it is actually earning its place.
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_FILE = PROMPTS_DIR / "system.md"
PLAYBOOK_FILE = PROMPTS_DIR / "playbook.md"
EDITING_FILE = PROMPTS_DIR / "editing.md"

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
                       editing: bool = False) -> str:
    """Assemble the system prompt. `playbook` is 'default', 'none', or a path.

    A missing file raises rather than silently degrading: a prompt that quietly
    lost its playbook would make every eval number after it meaningless.
    """
    parts = [_read(SYSTEM_FILE)]

    if playbook not in (None, NONE):
        path = PLAYBOOK_FILE if playbook == DEFAULT else Path(playbook)
        parts.append(_read(path))

    if editing:
        parts.append(_read(EDITING_FILE))

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
