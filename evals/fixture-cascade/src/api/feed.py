"""RSS feed rendering."""

from src.util.text import slugify


def feed_entry(title, body):
    return {"id": slugify(title), "title": title, "body": body}
