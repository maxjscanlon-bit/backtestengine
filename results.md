# NQ 5-min Research Log

Frozen record of every hypothesis tested. Verdicts are final once written.

| Date | Hypothesis | Params | Data split | Expectancy | PF | p-value | Verdict |
|------|-----------|--------|-----------|-----------|----|---------|---------|
| 2026-08-07 | z-score reversal, plumbing run | lb=48, z=-1.5, hold=6, RTH, 2 ticks/side | TRAIN | -$40.03/trade | 0.909 | n/a | Baseline/plumbing. Naive YM-analog params lose on NQ 5m. Counts as trial 1. |
| 2026-08-07 | z-score reversal, optimizer verification | 2x2x2 grid (lb 24/48, z -1.5/-2.0, hold 3/6), RTH, first 60k TRAIN bars | TRAIN subset | best -$61.05/trade | 0.847 | DSR 0.0006 | REJECT. Optimizer mechanical check. Trials 2-9. Long-only reversal family loses across entire grid on this slice. |

## Corrections log

**2026-08-07 audit.** Two critical defects found and fixed. All CPCV numbers produced before commit
`28ba085` are void.

1. *Seam contamination.* `cpcv_evaluate` sliced the price frame to discontiguous fold indices and
   recomputed rolling features on the stitched result. Rolling windows spanned month-long gaps and
   `close.diff()` at each seam fabricated a price move worth up to 3,345 points. Measured distortion
   on real TRAIN folds reached 61% of reported fold P&L. Fixed: signals are now computed once on the
   full contiguous series and scored via `engine.backtest_subset`, which drops the first bar of every
   contiguous block. Regression tests added.
2. *Back-adjustment bias in percentage returns.* Panama adjustment lifts 2021 closes from ~15,090 to
   ~18,469, compressing pct returns in early history relative to late. Signals now use point
   differences, which are invariant to the additive offset. Regression test added.

**2026-08-07, second pass.** Remaining two audit items fixed, and a third defect found while fixing
them. All results before commit `c6e1bf2` are void.

3. *Session boundaries.* Daily P&L was grouped by calendar date, splitting each Globex session at
   midnight and fabricating 264 low-activity pseudo-days across the sample. Daily Sharpe, and
   therefore DSR and every CPCV summary, was computed on the wrong denominator. Fixed via
   `sessions.session_group`, which runs 18:00-17:00 ET per CME trade-date convention.
4. *RTH off-by-one.* Under right-labeled bars the filter `>= 09:30 and < 16:00` included the 09:30
   bar, which covers 09:25-09:30 and is pre-open, and dropped the 16:00 bar, which is the last real
   RTH bar. Fixed via `sessions.rth_mask`, keeping (09:30, 16:00]. RTH days are now exactly 78 bars.
5. *Bar misalignment (found during 4).* Databento `ohlcv-1m` stamps `ts_event` at the interval START,
   verified empirically: the last bar before the 17:00 break is 16:59 and the first after the 18:00
   open is 18:00. The aggregation used `closed='right'`, so every 5-minute bar covered T-4 to T+1
   instead of T-5 to T. The entire dataset was shifted one minute. Fixed to `closed='left'` and the
   continuous series was rebuilt from source. Bar count 355,374 -> 354,081. Splits re-locked. Holdout
   was never read, so re-locking costs nothing.

Seven regression tests added covering all three. Suite is 27/27.

| 2026-08-07 | z-score reversal, optimizer re-verification post-fix | same 2x2x2 grid, RTH, first 60k TRAIN bars | TRAIN subset | best -$61.54/trade | 0.845 | DSR 0.0005 | REJECT. Re-run of the voided trials 2-9, not new trials. Whole grid loses. Long-only only. |
| 2026-08-07 | z-score reversal, optimizer re-verification post-audit | same 2x2x2 grid, RTH, first 60k TRAIN bars | TRAIN subset | best -$61.25/trade | n/a | DSR 0.0016 | REJECT. Third re-run of voided trials 2-9, not new trials. Ledger stays at 9. Whole grid loses, long-only only. |
| 2026-08-07 | z-score two-sided, reversal + momentum | 3 lb x 3 z x 3 hold x 2 modes = 54 params, RTH, 2 ticks/side | full TRAIN | best +$23.43/trade (momentum 48/2.5/12) | 1.047 | DSR 0.051 | REJECT. Trials 10-63. Every reversal variant loses. Momentum weakly positive at extreme z but flagged SPIKE, neighbors negative, CPCV 5/28 paths positive. Directional finding: NQ 5m extreme moves continue, not revert. |
| 2026-08-07 | ZigZag M/W pattern trader (user's NinjaScript port), Default mode | 4 periods x 3 TP x 3 SL = 36 params, all sessions, conservative fills, 2 ticks/side | full TRAIN | best -$15.78/trade (period 40, TP 100, SL -61.8) | 0.966 | DSR 0.0024 | REJECT. Trials 64-99. Every variant negative. Conservative fills (trade-through entries, SL-first, no same-bar TP) vs the source's optimistic pseudo-engine likely explains gap vs NT results. |
| 2026-08-07 | M/W entries with alternative exits (time exit no-stop, far-pivot stop) | 12 time (4 periods x 3 holds) + 4 farstop = 16 params, 2 ticks/side | full TRAIN | best -$16.10/trade (time, period 5, hold 12) | 0.948 | DSR 0.0002 | REJECT. Trials 100-115. Exit architecture does not rescue the entries. Closes the M/W Default family: entries carry no edge on NQ 5m. |
| 2026-08-07 | ZigZag M/W Inverse mode (breakout: M breakdown short, W breakout long) | 36 params (4 periods x 3 TP x 3 SL), stop entries, 2 ticks/side | full TRAIN | best -$52.70/trade (period 10, 100/-100) | 0.881 | DSR 0.000 | REJECT. Trials 116-151. Uniformly worse than Default. Pivot breakouts on NQ 5m are fade-the-trigger events: price that breaks a pattern pivot tends to snap back. Entire M/W family closed, both directions. |
| 2026-08-07 | M/W Default + Markov meta-filter on shadow outcomes | 4 periods x 2 brackets x 4 rules = 32 params, shadow-ledger design, no lookahead | full TRAIN | best +$130.58/trade (period 40, 100/-61.8, roll20>=0.45, 145 trades) | 1.315 | DSR 0.039 | REJECT at gate. Trials 152-183. First family with positive in-sample cells and a mechanism (period-40 outcome streaking, diagnostic z=+2.4). CPCV OOS mean sharpe -0.61, frac positive 0.39 (best seen but far below 0.7 bar). Streaking likely proxies vol regime; revisit via regime conditioning, not outcome memory. |
| 2026-08-07 | Cypher harmonic pattern (XABCD, Fib-constrained, D entry, X stop) | 4 periods x 2 TP fracs x 2 SL buffers = 16 params, 2 ticks/side | full TRAIN | best +$82.51/trade (period 20, 0.618, 0.10, 77 trades) | 1.204 | DSR 0.014 | REJECT at gate. Trials 184-199. Best OOS profile of any family: CPCV median sharpe +0.08, frac positive 0.50 (bar is 0.7), but mean -0.24 and mean OOS pnl negative. Positive cells are thin (77-171 trades/3yr). Pattern verdict consistent: near-coin-flip after friction, not enough sample to prove more. |
| 2026-08-07 | Trailing performance gate (full 8-pt spec) on Cypher P10/P20 | 12 configs: 5 window-rule sets + markov, 2 bases, warmup=trade | full TRAIN | best +$54.87/trade (P10, g8_50, 144 trades) | 1.180 | DSR 0.026 | REJECT. Trials 200-211. Requirement-6 separation diagnostic: NO gate config separates winners from losers (best z=+1.02, only significant z is NEGATIVE: P20 g4+20 z=-2.32; markov rule anti-predictive, skipped 69% wr vs taken 56%). CPCV frac-positive 0.71 traces to the unfiltered P10 base (0.61 alone), not the gate (paired: gate improved 10/28 folds). The "is it working now" hypothesis is directly falsified on Cypher: trailing outcomes carry no information about the next trade. Unfiltered Cypher P10 is now the single most interesting object in the ledger: OOS median sharpe +0.55 with zero conditioning. |

**2026-08-08 gap-fill fix.** Independent per-trade verification (battery 4) caught 3 trades exiting at levels the exit bar never traded: bars gapping through TP were filled at the level instead of the (better) open. Same defect symmetric on stops (worse fill). Fixed in cypher.py and mw.py: levels gapped through now fill at the open, both directions. All 171 P10 trades re-verified 6/6 checks. Frozen-P10 stats improved: expectancy $39.25 -> $58.84, PF 1.13 -> 1.20, sharpe 0.37 -> 0.54.

| 2026-08-08 | Cypher P10 promotion battery (frozen: period 10, tp 0.382, slb 0.10) | friction 0-4 ticks, 3 regimes, 28 CPCV folds, 171-trade independent verification | full TRAIN | +$58.84/trade at 2 ticks, +$38.84 at 4 | 1.200 | n/a (no new search) | Battery results: survives 4 ticks/side positive; CPCV folds mean +0.49 median +0.63 frac-pos 0.64; regimes +0.36/-0.12/+1.03 (middle third flat, not inverted); tradability: 57 tr/yr, maxDD -$6.8k, worst streak 7. No new trials (frozen params, no selection). |

**2026-08-08 VAL shot, criterion pre-committed before execution.** Cypher P10 frozen (period 10, tp_frac 0.382, sl_buffer 0.10, 2 ticks/side, gap-aware fills). PASS requires BOTH: total VAL P&L > 0 AND session Sharpe > 0. One run, no re-tries, no parameter changes regardless of outcome. Fail -> archive the strategy.
