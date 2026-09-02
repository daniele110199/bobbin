"""Command line entry point for Ledger."""

import argparse

from billing.invoice import InvoiceBuilder
from core.config import API_ENDPOINT


def main():
    parser = argparse.ArgumentParser(prog="ledger")
    parser.add_argument("--amount", type=float, required=True)
    args = parser.parse_args()

    builder = InvoiceBuilder(endpoint=API_ENDPOINT)
    print(builder.build(args.amount))


if __name__ == "__main__":
    main()
