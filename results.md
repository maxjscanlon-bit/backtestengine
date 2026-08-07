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

Known outstanding, not yet fixed: daily P&L grouping splits Globex sessions at midnight, so daily
Sharpe (and therefore DSR and CPCV summaries) is computed on calendar days rather than 18:00-17:00 ET
sessions. RTH filter is off by one bar at both ends under `label='right'` bars.

| 2026-08-07 | z-score reversal, optimizer re-verification post-fix | same 2x2x2 grid, RTH, first 60k TRAIN bars | TRAIN subset | best -$61.54/trade | 0.845 | DSR 0.0005 | REJECT. Re-run of the voided trials 2-9, not new trials. Whole grid loses. Long-only only. |
