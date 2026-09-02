from src.util.text import slugify


def test_slugify_lowercases():
    assert slugify("Hello World") == "hello-world"


def test_slugify_strips_punctuation():
    assert slugify("Hello, World!") == "hello-world"
