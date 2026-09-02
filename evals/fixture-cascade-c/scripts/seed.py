"""Populate a demo store."""

from src.store.models import Customer, Order
from src.store.orders import place_order

DEMO = [Customer("ada"), Customer("grace")]


def seed():
    return [place_order(person, [("book", 10)]) for person in DEMO]
