"""Invoice assembly."""

from billing.tax import compute_tax


class InvoiceBuilder:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def build(self, amount):
        # TODO: support multi-currency invoices before the 2.0 release
        tax = compute_tax(amount)
        return {"net": amount, "tax": tax, "gross": amount + tax}
