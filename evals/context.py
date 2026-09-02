"""What the stored runs already know about the context window.

The third stored-artifact instrument, after `evals.rescore` (recompute a metric
over old answers) and `evals.replay` (re-run old edits against a new detector).
This one needs no model and no fixture at all: every multi-turn row records
`context_tokens_est` per turn and the `num_ctx` it ran in, so the shape of the
overflow — how fast a session grows, when it crosses, whether crossing predicts
failure, and how much warning a trigger would have had — is already on disk.

    python3 -m evals.context

The estimate is characters // 4. There is no tokenizer here and this project
does not pip install one, so every number below is an estimate with the same
bias in both arms. That is fine for sizing a trigger and wrong for anything
that needs to be exact — which is the argument for a *reserve*, not for a
tokenizer.
"""

from __future__ import annotations

import glob
import json
import os
import statistics as st

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def load_sessions(pattern: str | None = None) -> list[dict]:
    """Every stored row that is a session: >1 turn, with per-turn context."""
    out = []
    for path in sorted(glob.glob(pattern or os.path.join(RESULTS, "*.json"))):
        try:
            data = json.load(open(path))
        except (ValueError, OSError):
            continue
        for row in data.get("rows", []):
            turns = row.get("turns") or []
            if len(turns) > 1 and all("context_tokens_est" in t for t in turns):
                out.append({"file": os.path.basename(path), "model": row["model"],
                            "case": row["case"], "num_ctx": row.get("num_ctx", 0),
                            "turns": turns})
    return out


def growth(sessions: list[dict]) -> dict[str, list[int]]:
    """Est tokens added by one turn, per model. The unit the loop grows in."""
    per: dict[str, list[int]] = {}
    for s in sessions:
        prev = 0
        for turn in s["turns"]:
            delta = turn["context_tokens_est"] - prev
            prev = turn["context_tokens_est"]
            if delta > 0:
                per.setdefault(s["model"].split(":")[0], []).append(delta)
    return per


def reserve_for(deltas: list[int], quantile: float = 0.9) -> int:
    """How much room the next turn needs. A trigger without this fires late."""
    ordered = sorted(deltas)
    return ordered[max(0, int(len(ordered) * quantile) - 1)]


def crossing_vs_outcome(sessions: list[dict]) -> dict[bool, tuple[int, int]]:
    """{over_window: (failed, total)} — does crossing predict a failed turn?"""
    tally: dict[bool, list[int]] = {True: [0, 0], False: [0, 0]}
    for s in sessions:
        for turn in s["turns"]:
            over = turn["context_tokens_est"] > s["num_ctx"] > 0
            tally[over][1] += 1
            if not turn["passed"]:
                tally[over][0] += 1
    return {k: tuple(v) for k, v in tally.items()}


def trigger_report(sessions: list[dict], reserve: int) -> list[dict]:
    """When a `fill + reserve > window` trigger fires, against what happened.

    `fires` is the turn *after* which the loop would act; `crosses` is the turn
    whose context no longer fits; `fails` is the first turn that failed. A
    trigger is only useful if it fires no later than the crossing.
    """
    out = []
    for s in sessions:
        window = s["num_ctx"]
        est = [t["context_tokens_est"] for t in s["turns"]]
        first = lambda pred: next((i + 1 for i, e in enumerate(est) if pred(e)), None)
        out.append({
            "model": s["model"], "case": s["case"], "num_ctx": window,
            "fires": first(lambda e: e + reserve > window),
            "crosses": first(lambda e: e > window),
            "fails": next((t["turn"] for t in s["turns"] if not t["passed"]), None),
        })
    return out


def compaction_report(sessions: list[dict], reserve: int, keep: int,
                      system_est: int, digest_est: int = 250) -> list[dict]:
    """What summarise-and-drop would have done to the sessions already on disk.

    No model is called, so the *content* of a digest cannot be simulated — only
    its arithmetic. That is still the question worth asking first: on the real
    growth curves, does replacing the prefix with a digest actually get a session
    back under its window, or do two fat turns plus a digest still overflow? If
    the answer is no, the mechanism cannot work at this `keep` and no GPU time
    should be spent finding that out.

    Every number is an estimate on an estimate (`chars // 4`, and a digest length
    assumed rather than generated), so read the direction and the ordering, not
    the magnitudes.
    """
    out = []
    for session in sessions:
        num_ctx = session.get("num_ctx") or 0
        est = [t["context_tokens_est"] for t in session["turns"]]
        if not num_ctx or not est:
            continue
        saved, fires, first_fire = 0, 0, None
        over_turns: list[int] = []
        peak = 0
        for index, cumulative in enumerate(est):
            fill = cumulative - saved
            if fill + reserve > num_ctx:
                # A prefix only exists once `keep` turns are already behind the
                # one being dropped — the loop's own precondition.
                if index >= keep + 1:
                    kept_growth = cumulative - est[index - keep]
                    new_fill = system_est + digest_est + kept_growth
                    if new_fill < fill:          # never make it worse
                        fires += 1
                        first_fire = first_fire or index + 1
                        saved = cumulative - new_fill
                        fill = new_fill
                if fill + reserve > num_ctx:
                    over_turns.append(index + 1)
            peak = max(peak, fill)
        out.append({"file": session["file"], "model": session["model"],
                    "case": session["case"], "num_ctx": num_ctx,
                    "turns": len(est), "fires": fires, "first_fire": first_fire,
                    "peak_after": peak, "still_over": len(over_turns),
                    "over_turns": over_turns,
                    "peak_before": max(est)})
    return out


def main() -> None:
    sessions = load_sessions()
    if not sessions:
        print("no stored sessions with per-turn context")
        return
    pressured = [s for s in sessions if 0 < s["num_ctx"] <= 8192]
    print(f"{len(sessions)} sessions on disk, {len(pressured)} of them pinned under pressure\n")

    print("=== how fast a session grows (est tokens per turn) ===")
    per = growth(sessions)
    for model, deltas in sorted(per.items()):
        p90 = reserve_for(deltas)
        print(f"  {model:24} n={len(deltas):3}  median={st.median(deltas):5.0f}"
              f"  p90={p90:5.0f}  max={max(deltas):5.0f}")
        for window in (16384,):
            print(f"  {'':24} -> {window} fits ~{window / st.median(deltas):.0f} turns"
                  f" of ordinary work, ~{window / p90:.0f} at p90")

    print("\n=== does crossing the window predict a failed turn? ===")
    for over, (failed, total) in sorted(crossing_vs_outcome(sessions).items()):
        label = "over window " if over else "under window"
        pct = f"{failed / total * 100:.0f}%" if total else "-"
        print(f"  {label}: {failed:3} failed / {total:3} turns  ({pct})")
    print("  (confounded: later turns are both fuller and harder. Direction, not effect size.)")

    print("\n=== a trigger sized at p90 growth, against what happened ===")
    for model, deltas in sorted(per.items()):
        res = reserve_for(deltas)
        for row in trigger_report([s for s in pressured
                                   if s["model"].split(":")[0] == model], res):
            late = row["fires"] and row["crosses"] and row["fires"] > row["crosses"]
            print(f"  {model:22} win={row['num_ctx']:5} reserve={res:5}"
                  f"  fires=t{row['fires']}  crosses=t{row['crosses']}"
                  f"  fails=t{row['fails']}  {'LATE' if late else 'in time'}")

    print("\n=== what summarise-and-drop would have done (arithmetic only) ===")
    from agent.loop import COMPACT_KEEP_TURNS, CONTEXT_RESERVE
    from agent.prompts import load_system_prompt
    system_est = len(load_system_prompt(os.path.join(os.path.dirname(__file__),
                                                     "fixture-session"),
                                        editing=True)) // 4
    print(f"  system prompt is {system_est} est tokens, kept across every compaction;"
          f" keep={COMPACT_KEEP_TURNS} turns, digest assumed 250")
    for row in compaction_report(pressured, CONTEXT_RESERVE, COMPACT_KEEP_TURNS,
                                 system_est):
        verdict = ("never relieved" if row["still_over"] == row["turns"]
                   else f"still over at t{','.join(str(n) for n in row['over_turns'])}"
                   if row["over_turns"] else "clear after the first fire")
        print(f"  {row['model'].split(':')[0]:22} {row['case'][:22]:24}"
              f" win={row['num_ctx']:5} fires={row['fires']}"
              f" first=t{row['first_fire']}"
              f"  peak {row['peak_before']} -> {row['peak_after']}  ({verdict})")

    unpressured = [s for s in sessions if s["num_ctx"] > 8192]
    if unpressured:
        peak = max(t["context_tokens_est"] / s["num_ctx"]
                   for s in unpressured for t in s["turns"])
        print(f"\n=== false fires on ordinary sessions ===\n"
              f"  peak fill across {len(unpressured)} unpinned sessions: {peak * 100:.1f}%"
              f" of the window — nothing fires, and nothing measured moves.")


if __name__ == "__main__":
    main()
