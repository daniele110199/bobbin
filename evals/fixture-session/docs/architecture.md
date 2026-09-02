# Architecture

Internal tracking id for this document: ZQ_SENTINEL_4417

Depot is split into four packages.

- `api` owns the HTTP surface. Every route is authenticated by `api/auth.py`
  before it reaches a handler; every route needs a token.
- `core` owns plumbing: configuration, the database handle and the cache.
  Nothing in `core` imports from `inventory` or `shipping`, which keeps the
  dependency direction one-way.
- `inventory` owns what is on the shelves: item records, stock counts, and the
  reorder policy that decides when a line needs replenishing.
- `shipping` owns everything that leaves the building: carrier selection and
  label booking.

Stock counts are cached, so a write to `stock.py` invalidates the cache key for
that item. The cache lifetime itself lives in `core/config.py`.
