"""HTTP views."""

from src.core.config import PAGE_SIZE
from src.util.text import slugify


def article_url(title):
    return "/articles/" + slugify(title)


def page_count(total):
    return (total + PAGE_SIZE - 1) // PAGE_SIZE
