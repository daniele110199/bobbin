#!/usr/bin/env python3
"""Completed head-to-head cells, recovered from the 2026-08-25/30 session log.

The original run wrote full rows (answers, step counts, per-check detail) to
/tmp; a reboot on 2026-09-01 took them. What survives is the scored table each
run printed, so that is what this file holds: arm, pass/fail, wall seconds, and
nothing invented to fill the gaps. `source` says so on every row.

Re-running would cost ~4 GPU-hours and would not reproduce these rows anyway —
they are one sample each, which is the protocol these were run under.
"""
import json
from pathlib import Path

# case: (ours, aider-told, aider-find) as (passed, seconds); None = never run
FIXTURES = {
 "qwen3-coder:30b": {
  "cascade-doc":              ((1,61),  (0,25),  (0,14)),
  "cascade-move":             ((1,108), (0,21),  (0,71)),
  "cascade-rename":           ((1,112), (0,30),  (0,28)),
  "cascade-signature":        ((0,108), (1,49),  (1,29)),
  "edit-constant":            ((1,23),  (1,17),  (1,10)),
  "edit-create":              ((1,24),  (1,19),  (1,7)),
  "edit-keeps-rest":          ((1,23),  (1,16),  (1,9)),
  "edit-nonexistent":         ((1,36),  (1,15),  (0,15)),
  "edit-rate":                ((1,23),  (1,20),  (1,14)),
  "edit-rename":              ((1,28),  (1,17),  (1,10)),
  "edit-trap":                ((1,24),  (1,24),  (1,15)),
  "edit-two-files":           ((1,61),  (1,24),  (1,17)),
  "cascade-delete-symbol":    ((0,73),  (1,40),  (1,36)),
  "cascade-move-function":    ((1,138), (0,26),  (0,17)),
  "cascade-rename-class":     ((1,183), (0,46),  (0,28)),
  "cascade-rename-constant":  ((1,85),  (0,35),  (0,30)),
  "cascade-rename-method":    ((1,115), (0,39),  (0,42)),
  "cascade-split-module":     ((1,137), (0,21),  (0,15)),
  "repair-half-deleted":      ((1,106), (1,31),  (0,18)),
 },
 "nemotron-3.5-lightning": {
  "cascade-doc":              ((1,102), (0,224), (0,266)),
  "cascade-move":             ((1,240), (0,72),  (0,117)),
  "cascade-rename":           ((0,541), (0,265), (0,238)),
  "cascade-signature":        ((1,98),  (1,224), (1,226)),
  "edit-constant":            ((1,65),  (1,58),  (1,31)),
  "edit-create":              ((1,51),  (1,60),  (1,31)),
  "edit-keeps-rest":          ((1,36),  (1,38),  (1,60)),
  "edit-nonexistent":         ((0,156), (0,43),  (0,91)),
  "edit-rate":                ((1,30),  (1,28),  (1,59)),
  "edit-rename":              ((1,36),  (1,46),  (1,47)),
  "edit-trap":                ((1,36),  (1,150), (1,146)),
  "edit-two-files":           ((1,53),  (1,34),  (1,39)),
  "cascade-delete-symbol":    ((1,113), (0,217), (0,218)),
  "cascade-move-function":    ((1,191), (0,117), (0,203)),
  "cascade-rename-class":     ((1,237), (0,420), (0,220)),
  "cascade-rename-constant":  ((1,107), (0,177), (0,183)),
  "cascade-rename-method":    ((1,103), (0,152), (0,166)),
  "cascade-split-module":     ((1,136), (0,59),  (0,120)),
  "repair-half-deleted":      ((1,119), (1,164), (1,201)),
 },
}

REAL = {
 "qwen3-coder:30b": {
  "real-rename-across-files": ((0,501),  (0,171), (0,96)),
  "real-rename-internal":     ((0,231),  (0,35),  (0,36)),
  "real-move-function":       ((0,2331), (1,62),  (1,58)),
  "real-signature":           ((0,104),  (0,101), (0,280)),
  "real-single-file":         ((1,39),   (1,28),  (1,22)),
  "real-nonexistent":         ((1,823),  (1,30),  (1,12)),
 },
 "nemotron-3.5-lightning": {
  "real-rename-across-files": ((1,289),  (0,900), (0,900)),   # aider hit its ceiling
  "real-rename-internal":     ((0,166),  (0,114), (0,357)),
  "real-move-function":       ((1,302),  (0,900), (0,536)),
  "real-signature":           ((0,320),  (0,900), (0,904)),
  "real-single-file":         ((1,47),   (1,96),  (1,99)),
  "real-nonexistent":         (None, None, None),             # lost to the reboot
 },
}

ARMS = ("ours", "aider-told", "aider-find")


def rows(table, key):
    out = []
    for model, cases in table.items():
        for case, cells in cases.items():
            for arm, cell in zip(ARMS, cells):
                if cell is None:
                    continue
                out.append({key: case, "model": model, "arm": arm,
                            "passed": bool(cell[0]), "seconds": float(cell[1]),
                            "source": "recovered-from-session-log"})
    return out


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    (here / "fixtures_recovered.json").write_text(
        json.dumps(rows(FIXTURES, "case"), indent=2))
    (here / "realrepo_recovered.json").write_text(
        json.dumps(rows(REAL, "task"), indent=2))
    print("wrote fixtures_recovered.json, realrepo_recovered.json")
