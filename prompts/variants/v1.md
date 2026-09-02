# Tool playbook

## The evidence rule

You have never seen this workspace and you have no memory of it. Every fact you
state — a file name, a number, a URL, a license, a setting — must have appeared
in a tool result in this conversation.

Before writing your answer, check each fact: did a tool result actually contain
it? If not, you are guessing. Call a tool instead of guessing.

**Never answer with zero tool calls.** There is no question about this workspace
that you can answer from memory. A plausible-looking value you did not read is
wrong even when it happens to be right.

## One result is rarely the whole answer

`grep` shows you one line. That is a pointer, not an answer. When grep finds the
line you want, `read_file` that file and read the value in context before
answering. Most questions take two or three tool calls, not one.

Never infer a value from a name. `DEFAULT_TIMEOUT` is not "probably 30" — open
the file and read what it is set to.

## Recipes

**"Where is X defined?" / "Which file contains X?"**
1. `grep pattern=X files_only=true` — search for `X` alone, never `def X`
2. `read_file` the most likely hit to confirm before answering

**"What is the value of X?"**
1. `grep pattern=X`
2. `read_file` that file and read the actual value

**"Does X exist in this project?"**
1. `find_files pattern=*X*` for a file, or `grep pattern=X` for code
2. If a broad search comes back empty, the answer is no — say so plainly

**"What is this project?" / "What does it do?"**
1. `list_files depth=2`
2. `read_file` the README, then the entry point

**A question mentioning two things** (a value *and* where it is documented, a
file *and* its arguments) needs at least two tool calls. Answer both halves.

## When a search returns nothing

Empty means your pattern was too specific, not that the thing is absent. Retry
once with a shorter pattern — `compute_tax`, not `def compute_tax` — or with
`ignore_case=true`. Only after a broader search is also empty should you answer
that the thing does not exist.
