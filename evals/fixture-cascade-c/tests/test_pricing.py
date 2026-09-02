from src.config import TAX_RATE
from src.store.models import Customer, Order
from src.store.pricing import apply_discount


def test_apply_discount_includes_tax():
    order = Order(Customer("ada"), [("book", 100)])
    assert apply_discount(order, 0) == round(100 * (1 + TAX_RATE), 2)


def test_apply_discount_takes_percent_off():
    order = Order(Customer("ada"), [("book", 100)])
    assert apply_discount(order, 50) < apply_discount(order, 0)
