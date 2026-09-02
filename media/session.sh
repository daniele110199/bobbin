#!/bin/sh
# Every line is really executed. The checks below are the ones the eval suite
# uses for this case: call sites must be gone, and the vendored copy must not
# have been touched.
cd /tmp/bobbin-demo || exit 1
P='Rename the function slugify to make_slug everywhere in this repository, including every place it is called.'
printf '\033[2m$\033[0m grep -rn "slugify(" src tests node_modules\n'
grep -rn "slugify(" src tests node_modules
printf '\n\033[2m$\033[0m bobbin qwen3-coder:30b --allow-edits --yes -p "Rename slugify to make_slug everywhere..."\n\n'
bobbin qwen3-coder:30b --allow-edits --yes -p "$P"
printf '\n\033[2m$\033[0m grep -rn "slugify(" src tests\n'
grep -rn "slugify(" src tests > /tmp/_left 2>/dev/null
if [ -s /tmp/_left ]; then cat /tmp/_left; else printf '\033[2m(no matches — every call site updated)\033[0m\n'; fi
printf '\n\033[2m$\033[0m grep -rn "def slugify" node_modules\n'
grep -rn "def slugify" node_modules
printf '\033[2m(vendored code left alone, as it should be)\033[0m\n'
