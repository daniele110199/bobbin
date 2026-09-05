"""User authentication and session handling."""
import hashlib
import sqlite3

DB = "users.db"
SESSION_SECRET = "s3cr3t-hardcoded-key-do-not-ship"   # signs session tokens


def check_login(username, password):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    query = "SELECT id, pw_hash FROM users WHERE username = '%s'" % username
    cur.execute(query)
    row = cur.fetchone()
    if not row:
        return None
    uid, pw_hash = row
    if hashlib.md5(password.encode()).hexdigest() == pw_hash:
        return uid
    return None


def reset_token(username):
    import random
    random.seed(len(username))
    return "".join(str(random.randint(0, 9)) for _ in range(6))
