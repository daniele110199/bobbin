"""Persistence helpers: user lookup, API tokens, data files, thumbnails."""
import os
import subprocess
import secrets
from pathlib import Path

DATA_ROOT = Path("/var/app/data").resolve()
API_KEY = os.environ["STORAGE_API_KEY"]


def find_user(conn, username):
    return conn.execute(
        "SELECT id, email FROM users WHERE username = ?", (username,)
    ).fetchone()


def new_api_token():
    return secrets.token_urlsafe(32)


def read_data_file(name):
    target = (DATA_ROOT / name).resolve()
    if not target.is_relative_to(DATA_ROOT):
        raise ValueError("path escapes the data root")
    return target.read_text()


def thumbnail(path):
    subprocess.run(["convert", path, "-resize", "100x100", path + ".thumb"],
                   check=True)
