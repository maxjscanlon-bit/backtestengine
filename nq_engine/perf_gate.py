"""Trailing performance gate. Pure wrapper over any event strategy's shadow
trade ledger. Implements the 8-point spec:

1. Reads ONLY the shadow ledger (every signal simulated to completion).
2. No lookahead: at each decision, uses only shadow trades with
   exit_time strictly earlier than the decision time.
3. Gate rules: list of (window, min_winrate); ALL must pass.
4. Optional Markov rule: P(win | last resolved outcome) estimated from the
   ledger up to the decision time (expanding), must exceed a threshold.
5. warmup_mode: 'trade' (default, take trades until largest window fills)
   or 'block'. Partial windows are never evaluated silently.
6. Diagnostics: one row per signal with each rule's computed value,
   per-rule pass/fail, final decision, and the shadow outcome.
7. Base strategy untouched: this receives its trade list, returns take/skip.
8. Caller registers the full gate config in the trial log.
"""

import numpy as np
import pandas as pd


def apply_gate(shadow_trades, index, gate_rules, markov_threshold=None,
               warmup_mode="trade"):
    """Run the gate over a shadow ledger.

    shadow_trades: DataFrame with entry_time, exit_time, result (1 win / -1 loss)
    index: full bar index for rebuilding the filtered pnl series
    gate_rules: list of (window, min_winrate) tuples; all must pass
    markov_threshold: None (off) or float; if set, require
        P(win | last outcome state) > threshold, estimated expanding
    warmup_mode: 'trade' or 'block'

    Returns {'trades', 'pnl', 'diagnostics'}.
    """
    if warmup_mode not in ("trade", "block"):
        raise ValueError(warmup_mode)
    t = shadow_trades.sort_values("exit_time").reset_index(drop=True)
    # decision order = entry order; sequential engines make these identical,
    # but sort explicitly and verify the no-lookahead property per decision.
    t = t.sort_values("entry_time").reset_index(drop=True)
    outcomes = (t["result"] == 1).astype(int).values
    entry_times = t["entry_time"].values
    exit_times = t["exit_time"].values
    n = len(t)
    max_win = max(w for w, _ in gate_rules) if gate_rules else 0

    diag_rows = []
    take = np.zeros(n, dtype=bool)

    for i in range(n):
        decision_time = entry_times[i]
        # resolved strictly before this signal's decision time
        resolved_mask = exit_times < decision_time
        resolved = outcomes[resolved_mask]
        row = {"timestamp": t["entry_time"].iloc[i],
               "n_resolved": int(resolved_mask.sum())}
        rules_pass = []
        warm = len(resolved) >= max_win
        if not warm:
            row["warmup"] = True
            decision = warmup_mode == "trade"
            for w, thr in gate_rules:
                row[f"wr_{w}"] = np.nan
                row[f"pass_{w}@{thr}"] = None
        else:
            row["warmup"] = False
            for w, thr in gate_rules:
                wr = float(resolved[-w:].mean())
                ok = wr >= thr
                row[f"wr_{w}"] = round(wr, 4)
                row[f"pass_{w}@{thr}"] = bool(ok)
                rules_pass.append(ok)
            decision = all(rules_pass) if rules_pass else True
            if markov_threshold is not None and len(resolved) >= 2:
                prev = resolved[-1]
                trans_from_prev = [resolved[k + 1] for k in range(len(resolved) - 1)
                                   if resolved[k] == prev]
                p = float(np.mean(trans_from_prev)) if trans_from_prev else np.nan
                ok = (p == p) and p > markov_threshold
                row["markov_state"] = "W" if prev == 1 else "L"
                row["markov_p"] = round(p, 4) if p == p else np.nan
                row["pass_markov"] = bool(ok)
                decision = decision and ok
        row["decision"] = "take" if decision else "skip"
        row["shadow_outcome"] = int(outcomes[i])
        diag_rows.append(row)
        take[i] = decision

    taken = t[take].copy()
    pnl = pd.Series(0.0, index=index)
    if len(taken):
        add = taken.groupby("exit_time")["net_pnl"].sum()
        pnl.loc[add.index] += add.values
    diagnostics = pd.DataFrame(diag_rows)
    return {"trades": taken, "pnl": pnl, "diagnostics": diagnostics}


def gate_value_report(diagnostics):
    """Requirement 6 payoff: outcome distribution of passed vs failed signals,
    post-warmup only. The gate has value iff passed outcomes beat failed."""
    d = diagnostics[~diagnostics["warmup"]]
    if d["decision"].nunique() < 2:
        return {"note": "gate never split signals post-warmup", "n": len(d)}
    g = d.groupby("decision")["shadow_outcome"].agg(["mean", "count"])
    took = g.loc["take"] if "take" in g.index else None
    skip = g.loc["skip"] if "skip" in g.index else None
    out = {
        "winrate_taken": round(float(took["mean"]), 4),
        "n_taken": int(took["count"]),
        "winrate_skipped": round(float(skip["mean"]), 4),
        "n_skipped": int(skip["count"]),
        "separation": round(float(took["mean"] - skip["mean"]), 4),
    }
    # two-proportion z for the separation
    p = d["shadow_outcome"].mean()
    se = np.sqrt(p * (1 - p) * (1 / took["count"] + 1 / skip["count"]))
    out["separation_z"] = round(float((took["mean"] - skip["mean"]) / se), 2) if se > 0 else np.nan
    return out
