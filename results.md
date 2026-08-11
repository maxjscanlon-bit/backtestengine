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

**2026-08-10 VAL RESULT — PASS.** Cypher P10, frozen params, single run, no re-tries.

| Metric | TRAIN | VAL (2024-08 to 2025-08) |
|---|---|---|
| Total P&L | $10,062.07 | **$2,863.53** |
| Session Sharpe | 0.54 | **0.445** |
| Trades | 171 (57/yr) | 68 |
| Win rate | 56.7% | 47.1% |
| Profit factor | 1.200 | 1.132 |
| Expectancy/trade | $58.84 | $42.11 |
| Max drawdown | -$6,786 | -$5,634 |

Both pre-committed criteria met (P&L > 0, Sharpe > 0). Performance degraded ~28% on expectancy
vs TRAIN, which is normal and expected out-of-sample decay, not a red flag. Directional structure
held: long 49 trades +$6,521 (win 53.1%), short 19 trades -$3,658 (win 31.6%). The long/short
asymmetry observed on TRAIN replicated independently on unseen data — shorts lost money in both
periods. This is now a two-period observation, not a TRAIN artifact.

Also confirmed this date: full C#/NinjaScript parity. NT pseudo ledger reported 171 trades,
503.10 net points, $10,062.07 vs Python $10,062.07 — exact to the cent. NT real-order layer
(its own fill engine, first-entry-per-pattern only): 76 trades, avg $57.30/trade vs Python
$58.84/trade, PF 1.16 vs 1.20, win 55.3% vs 56.7%. Edge survives realistic fills.

HOLDOUT (2025-08 to 2026-08) remains locked and unread.

| 2026-08-10 | Cypher P10 LONG-ONLY | period 10, tp 0.382, slb 0.10, 2 ticks, bullish patterns only | TRAIN + VAL | TRAIN $168.09/trade, VAL $199.87/trade | 1.676 / 1.719 | DSR 0.227 | Trials 212-213. Dramatically better on both periods: TRAIN Sharpe 1.197 (vs 0.543), CPCV frac-positive 0.964 (vs 0.643), maxDD halved. VAL Sharpe 1.666. **BUT VAL IS NO LONGER A CLEAN TEST FOR THIS VARIANT** — the long-only idea came from observing short-side losses on both TRAIN and VAL. VAL is now in-sample-by-selection. DSR 0.227 still fails the 0.95 gate at 213 trials. HOLDOUT is the only uncontaminated data remaining and must NOT be spent casually. |

**2026-08-10 HOLDOUT shot, criterion pre-committed before execution.**
Strategy frozen: Cypher, period 10, tp_frac 0.382, sl_buffer_frac 0.10, d_ret 0.786,
B 0.382-0.618, C 1.272-1.414, LONG ONLY, 2 ticks/side, conservative gap-aware fills, 1 contract.
PASS requires ALL of: total P&L > 0, session Sharpe >= 0.50, profit factor >= 1.20,
trades >= 30, max drawdown not worse than -$8,000.
Partial pass counts as FAIL. One run, no re-tries, no parameter changes.
PASS -> sim forward test then small live. FAIL -> archived permanently.
Stated expectation before running: Sharpe 0.4-0.9, 40-55 trades, meaningful chance of failing.

**2026-08-10 HOLDOUT RESULT — FAIL. Strategy archived.**

| Metric | TRAIN | VAL (contaminated) | HOLDOUT |
|---|---|---|---|
| Total P&L | $16,136 | $10,393 | **-$7,963.53** |
| Session Sharpe | 1.197 | 1.666 | **-1.261** |
| Profit factor | 1.676 | 1.719 | **0.539** |
| Win rate | 59.4% | 55.8% | **32.3%** |
| Expectancy/trade | $168.09 | $199.87 | **-$256.89** |
| Max drawdown | -$3,140 | -$2,131 | **-$13,114** |
| Trades | 96 | 52 | 31 |

Four of five criteria failed. Only the trade-count minimum passed. Result is not marginal,
it is a sign flip: the strategy lost more per trade on HOLDOUT than it made on TRAIN.
Win rate collapsed from 59.4% to 32.3% and drawdown was 4x the TRAIN figure.

Per pre-commitment: ARCHIVED. No tweaking, no re-running with different exits, no rescue.

Final ledger: 213 trials, 0 survivors. HOLDOUT is now spent and this dataset is finished
as an evaluation tool. Any future strategy tested on this data carries the full 213-trial
selection burden plus a consumed holdout, and cannot be honestly validated here.

| 2026-08-10 | Cypher vol-scaled brackets (ATR-multiple SL/TP) | 18 variants: mode atr/hybrid x atr_sl 1.5/2.0/3.0 x atr_tp 1.0/1.5/2.5, ATR=EWMA96 shifted 1 bar, both directions | TRAIN ONLY | best $104.10/trade (atr, sl 1.5, tp 1.0, 191 trades) | 1.943 | DSR 0.694 | Trials 214-231. HYPOTHESIS GENERATION ONLY, NOT VALIDATED. Best variant TRAIN Sharpe 1.883 vs 0.543 structural, CPCV 28/28 folds positive, DSR 0.694 (best ever, still below 0.95). Pure-ATR brackets beat hybrid across the board, and the ATR mode is monotone in a sensible way (tighter TP -> higher win rate). **CANNOT BE VALIDATED ON THIS DATASET**: the vol-scaling hypothesis was derived from the HOLDOUT autopsy, so testing it on VAL or HOLDOUT is circular. Requires fresh data (Databento 2010-2021, untouched) for any honest verdict. |

**2026-08-11 Independent engine verification (backtesting.py 0.6.6).**
Third implementation, third-party library, own order lifecycle and fill logic.
Compared on 60,000 TRAIN bars, friction 0, gross points.

| | nq_engine | backtesting.py |
|---|---|---|
| Trades | 38 | 63 |
| Gross points | 238.46 | 322.33 |
| Win rate | 68.4% | 68.3% |
| Per trade | 6.28 | 5.12 |

AGREES: entry prices identical to 0 decimals on all 29 shared trades. Exit times agree
27/29. Win rates agree to 0.1pt. Pattern detection, ATR math and level computation are
now confirmed across three independent implementations.

DIFFERS (documented convention, not bugs): exit prices agree only 4/29 because nq_engine
fills gapped levels at the bar open (worse for stops, better for targets) while
backtesting.py fills at the level. Trade count differs because backtesting.py fills limits
on touch while nq_engine requires trade-through and cancels resting orders when the stop
zone is reached first.

CONCLUSION: no logic defect found. Fill-convention choice moves per-trade results ~20%,
so all expectancy point estimates carry at least that much model uncertainty. This is the
honest resolution limit of OHLC backtesting and matches what the NT real-order run showed.
