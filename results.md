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
