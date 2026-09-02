You are planning an investigation of a read-only workspace at {root}.

QUESTION: {question}

This is what the workspace actually contains:

{survey}

Break the question into at most {max_subtasks} separate things that have to be
looked up. Each one must be answerable on its own by searching for ONE
identifier — and that identifier must appear in the listing above.

Reply with one line per lookup, and nothing else:

SUBTASK: <what to find out> | GREP: <one identifier to search first>

Example, for "what tax does an invoice add?":

SUBTASK: find where the tax rate is defined | GREP: STANDARD_RATE
SUBTASK: find how the invoice applies the tax | GREP: InvoiceBuilder

Rules:
- Fewer lines is better. Two good lookups beat four vague ones.
- The GREP term is a real identifier from the listing above, never a phrase
  from the question. If the question says "authenticates" and the listing says
  `sign_in`, write `sign_in`.
- Do not plan a lookup for something the listing already tells you.
