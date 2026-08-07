# NQ 5-min Research Log

Frozen record of every hypothesis tested. Verdicts are final once written.

| Date | Hypothesis | Params | Data split | Expectancy | PF | p-value | Verdict |
|------|-----------|--------|-----------|-----------|----|---------|---------|
| 2026-08-07 | z-score reversal, plumbing run | lb=48, z=-1.5, hold=6, RTH, 2 ticks/side | TRAIN | -$40.03/trade | 0.909 | n/a | Baseline/plumbing. Naive YM-analog params lose on NQ 5m. Counts as trial 1. |
| 2026-08-07 | z-score reversal, optimizer verification | 2x2x2 grid (lb 24/48, z -1.5/-2.0, hold 3/6), RTH, first 60k TRAIN bars | TRAIN subset | best -$61.05/trade | 0.847 | DSR 0.0006 | REJECT. Optimizer mechanical check. Trials 2-9. Long-only reversal family loses across entire grid on this slice. |
