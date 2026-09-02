# On-call runbook

Escalation code for the warehouse pager rotation: PAGER-7731

## The queue is backing up

Check `reorder_due()` first. A slow reorder pass blocks the nightly batch, and
the batch holds the write lock the API needs.

## A carrier is rejecting labels

`shipping/carriers.py` keeps a per-carrier failure count. Three failures in a
row and the carrier is skipped for the rest of the run; the next cheapest one
takes over. This is silent by design — check the counts before assuming a bug.

## Cache looks stale

`CACHE_TTL` in `core/config.py` is the only lifetime the cache has.
