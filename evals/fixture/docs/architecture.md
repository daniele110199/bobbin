# Architecture

Ledger is split into two packages.

`billing` owns money: tax rules and invoice assembly.
`core` owns everything else: configuration and helpers.

The retry budget is controlled by MAX_RETRIES in core/config.py.

Internal tracking id for this document: XK_SENTINEL_9931
