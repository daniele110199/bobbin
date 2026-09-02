"""Domain objects."""


class Customer:
    def __init__(self, name):
        self.name = name


class Order:
    def __init__(self, customer, lines):
        self.customer = customer
        self.lines = lines

    def total(self):
        return sum(price for _, price in self.lines)
