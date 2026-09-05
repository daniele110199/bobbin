"""Customer lookup."""
import sqlite3


def customer_by_email(conn, email):
    q = "SELECT id, name FROM customers WHERE email = '%s'" % email
    return conn.execute(q).fetchone()
