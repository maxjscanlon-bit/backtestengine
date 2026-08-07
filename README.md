# nq-research

Vectorized 5-min NQ backtesting engine. Research in Python, execution in NinjaTrader 8 (C#).

Workflow: data integrity check, locked TRAIN/VAL/HOLDOUT split, hypothesis testing on TRAIN,
permutation + friction stress validation, VAL confirmation, one-shot HOLDOUT, then C# port.

- nq_engine/data.py: loading, integrity checks, splits
- nq_engine/engine.py: backtest core, friction, next-close fills
- nq_engine/stats.py: permutation tests, sub-period stability
- nq_engine/signals.py: hypothesis definitions
- results.md: frozen log of everything tested

Data files (parquet splits) live in data/ and are gitignored if large.
