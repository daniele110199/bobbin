#!/bin/sh
# Put `bobbin` on your PATH.
#
# A symlink rather than a package install: this project's whole claim is that it
# needs nothing but the standard library, and a `pip install` step would be the
# first thing to contradict that. The link points at the checkout, so `git pull`
# updates the command with no reinstall.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${1:-$HOME/.local/bin}

mkdir -p "$BIN"
ln -sf "$HERE/bobbin" "$BIN/bobbin"
echo "linked $BIN/bobbin -> $HERE/bobbin"

case ":$PATH:" in
    *":$BIN:"*)
        echo
        echo "Try it:  bobbin --help" ;;
    *)
        echo
        echo "$BIN is not on your PATH. Add it:"
        echo
        echo "    echo 'export PATH=\"$BIN:\$PATH\"' >> ~/.zshrc   # or ~/.bashrc"
        echo "    exec \$SHELL" ;;
esac
