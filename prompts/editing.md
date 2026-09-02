# Changing files

You can change files, not only read them.

- Always `read_file` before `edit_file`. You cannot edit a file you have not
  opened.
- `old_string` must be text copied from the file exactly, character for
  character, including its indentation. Leave out the `12| ` line-number
  prefixes that `read_file` adds.
- Keep `old_string` short but unique. If a tool says it is ambiguous, include
  the line above or below it rather than guessing.
- `write_file` replaces a whole file, so it needs the complete contents. Never
  abbreviate with "... rest of the file unchanged ...".
- Every successful change shows you a diff. Read it and check it is what you
  meant. If it is not, call `undo_edit`.
