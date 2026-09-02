# Depot

A small warehouse service: it tracks stock, decides when to reorder, and books
shipping labels with whichever carrier is cheapest for the parcel.

    src/api          HTTP surface and auth
    src/core         config, database and cache plumbing
    src/inventory    items, stock levels, reorder policy
    src/shipping     carriers and label booking
    src/util         text and time helpers
    tests            the test suite
    docs             architecture notes and the on-call runbook

Reorder behaviour is governed by `REORDER_THRESHOLD` in `core/config.py`.
