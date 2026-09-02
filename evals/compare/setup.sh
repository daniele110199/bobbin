#!/bin/sh
# Rebuild the real-repo oracle: pallets/click at the pinned commit, plus a venv
# that can run its 1991 tests. Neither is committed — both are ~200MB and both
# are reproducible from here.
#
# The install is deliberately NOT editable: an editable install of the pristine
# checkout makes pytest inside a working copy import the *original* package and
# report green however broken the copy is. realrepo.py also sets PYTHONPATH per
# run, which is the belt to this braces.
set -e
cd "$(dirname "$0")"
COMMIT=36baa15ff831b939a22bc527cd76ce653ef6f66d

if [ ! -d click-probe ]; then
    git clone -q https://github.com/pallets/click.git click-probe
    (cd click-probe && git checkout -q "$COMMIT" && rm -rf .git)
fi

if [ ! -d clickenv ]; then
    python3 -m venv clickenv
    ./clickenv/bin/pip install -q pytest
    (cd click-probe && ../clickenv/bin/pip install -q .)
fi

PY="$PWD/clickenv/bin/python"
cd click-probe
PYTHONPATH="$PWD/src" "$PY" -m pytest -q --no-header -p no:cacheprovider | tail -1
echo "oracle ready — expect '1991 passed'"
