from src.store.models import Customer
from src.store.orders import place_order


def test_place_order_returns_an_order():
    result = place_order(Customer("ada"), [("book", 10)])
    assert result["order"].customer.name == "ada"
