You are answering a question about a read-only workspace at {root}.

Researchers have already searched the workspace for you. Below is what they
found, together with the raw tool output each finding came from. This evidence
is real — it was produced by the tools, not written by a model, and every claim
in it was checked against that output before you were shown it.

{dossier}

Now answer this question:

{question}

- Treat the evidence as fact. Every path, value and number in it is confirmed.
- If the evidence already answers the question, answer now. Do not re-run a
  search that is already above.
- If exactly one fact is still missing, call `read_file` or `grep` once to get
  it, then answer.
- Answer the whole question. If it asks for a file *and* a value, give both.
- If the evidence genuinely does not contain the answer, say so plainly rather
  than filling the gap yourself.
