# Design notes

The cache in `src/core/cache.py` holds computed values for `CACHE_TTL` seconds.
That lifetime is set in `src/core/config.py` and read nowhere else. There is no
second cache, and there are no background refreshes: an entry expires and the
next caller recomputes it.

The retry budget is separate and lives in the same config file.
