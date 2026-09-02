from billing.tax import compute_tax


def test_standard_rate():
    assert compute_tax(100) == 22.0


def test_reduced_rate():
    assert compute_tax(100, reduced=True) == 10.0
