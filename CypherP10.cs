#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

// ============================================================================
// CypherP10 - parity port of the Python research engine's Cypher strategy.
//
// PURPOSE: verify trade-for-trade agreement between the Python backtester and
// this C# implementation on identical data, BEFORE trusting either.
//
// TWO LAYERS, mirroring the MWBase pseudo/real architecture:
//   PSEUDO LEDGER (always on): simulates every trade under the Python engine's
//     exact conservative fill rules and writes CypherPseudoTrades.csv to the
//     NinjaTrader user folder. THIS is what gets diffed against Python.
//   REAL ORDERS (EnableRealOrders, default false): places actual NT orders.
//     NT's own fill engine has different intrabar assumptions, so real-order
//     results are expected to differ slightly; the pseudo ledger is the
//     parity reference.
//
// FILL RULES (identical to Python engine):
//   - Entry: limit at D fills only if the bar trades STRICTLY through it
//     (Low < D for longs, High > D for shorts). Fill price = D.
//   - Pending order invalidated if price reaches the stop zone before entry.
//   - Exits: SL checked FIRST. No TP allowed on the entry bar.
//   - Gap fills: a level gapped through fills at the bar OPEN
//     (worse for stops, better for targets).
//   - Friction: FrictionTicks per side charged in the pseudo ledger.
//
// IMPORTANT: run Strategy Analyzer ONLY on 2021-08-08 through 2024-08-06
// (the TRAIN window). Later dates are reserved validation data.
// Calculate must remain OnBarClose for parity.
// ============================================================================

namespace NinjaTrader.NinjaScript.Strategies.QPG
{
    public class CypherP10 : Strategy
    {
        private List<double> zzVals = new List<double>();
        private List<int> zzDirs = new List<int>();
        private double curVal = double.NaN;
        private int curDir = 0;
        private int dir = 0;

        private class PendingOrder
        {
            public double Entry, TP, SL, X, A, B, C;
            public int Dir;
        }
        private class ActiveTrade
        {
            public double Entry, TP, SL, X, A, B, C;
            public int Dir;
            public int EntryBar;
            public DateTime EntryTime;
        }

        private PendingOrder pending = null;
        private ActiveTrade active = null;
        private StreamWriter ledger = null;
        private int pseudoCount = 0;
        private double pseudoPnl = 0;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Cypher harmonic, parity port of Python research engine";
                Name = "CypherP10";
                Calculate = Calculate.OnBarClose;        // REQUIRED for parity
                IsExitOnSessionCloseStrategy = false;
                EntryHandling = EntryHandling.UniqueEntries;
                BarsRequiredToTrade = 20;

                ZigZagPeriod = 10;
                TpFrac = 0.382;
                SlBufferFrac = 0.10;
                BLo = 0.382; BHi = 0.618;
                CLo = 1.272; CHi = 1.414;
                DRet = 0.786;
                FrictionTicks = 2.0;
                EnableRealOrders = false;
            }
            else if (State == State.Configure)
            {
                zzVals.Clear(); zzDirs.Clear();
                curVal = double.NaN; curDir = 0; dir = 0;
                pending = null; active = null;
                pseudoCount = 0; pseudoPnl = 0;
            }
            else if (State == State.DataLoaded)
            {
                string path = Path.Combine(NinjaTrader.Core.Globals.UserDataDir, "CypherPseudoTrades.csv");
                ledger = new StreamWriter(path, false);
                ledger.WriteLine("entry_time,exit_time,dir,X,A,B,C,entry,exit,tp_level,sl_level,result,net_points,net_pnl,bars_held");
                Print(string.Format("[CypherP10] pseudo ledger -> {0}", path));
            }
            else if (State == State.Terminated)
            {
                if (ledger != null)
                {
                    ledger.Flush(); ledger.Close(); ledger = null;
                    Print(string.Format("[CypherP10] DONE. pseudo trades {0}, pseudo pnl {1:F2}", pseudoCount, pseudoPnl));
                }
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < ZigZagPeriod) return;

            double hi = High[0], lo = Low[0], op = Open[0];

            // ---- ZigZag pivots, identical to Python ----
            bool hasPH = hi > MAX(High, ZigZagPeriod - 1)[1];
            bool hasPL = lo < MIN(Low, ZigZagPeriod - 1)[1];

            if (hasPH && !hasPL) dir = 1;
            else if (hasPL && !hasPH) dir = -1;

            if (!double.IsNaN(curVal))
            {
                if (curDir == 1 && hasPH && hi > curVal) curVal = hi;
                else if (curDir == -1 && hasPL && lo < curVal) curVal = lo;
            }
            if (hasPH || hasPL)
            {
                if (!double.IsNaN(curVal) && dir != curDir)
                {
                    zzVals.Add(curVal); zzDirs.Add(curDir);
                    curVal = double.NaN;
                }
                if (double.IsNaN(curVal))
                {
                    curVal = dir == 1 ? hi : lo;
                    curDir = dir;
                }
            }

            // ---- Cypher detection on last four confirmed pivots ----
            if (zzVals.Count >= 4 && active == null)
            {
                int n = zzVals.Count;
                double X = zzVals[n - 4], A = zzVals[n - 3], B = zzVals[n - 2], C = zzVals[n - 1];
                int dX = zzDirs[n - 4], dA = zzDirs[n - 3], dB = zzDirs[n - 2], dC = zzDirs[n - 1];
                double xa = A - X;
                bool bull = dX == -1 && dA == 1 && dB == -1 && dC == 1 && xa > 0;
                bool bear = dX == 1 && dA == -1 && dB == 1 && dC == -1 && xa < 0;
                if (bull || bear)
                {
                    double rb = (A - B) / xa;
                    double rc = (C - X) / xa;
                    if (rb >= BLo && rb <= BHi && rc >= CLo && rc <= CHi)
                    {
                        double D = C - DRet * (C - X);
                        double slb = Math.Abs(C - X) * SlBufferFrac;
                        if (bull)
                            pending = new PendingOrder { Entry = D, Dir = 1, SL = X - slb, TP = D + TpFrac * (C - D), X = X, A = A, B = B, C = C };
                        else
                            pending = new PendingOrder { Entry = D, Dir = -1, SL = X + slb, TP = D - TpFrac * (D - C), X = X, A = A, B = B, C = C };
                        if (EnableRealOrders) SubmitRealEntry();
                    }
                }
            }

            // ---- Invalidate pending if price reaches the stop zone first ----
            if (pending != null && active == null)
            {
                if ((pending.Dir == 1 && lo < pending.SL) || (pending.Dir == -1 && hi > pending.SL))
                {
                    pending = null;
                    if (EnableRealOrders) CancelRealEntry();
                }
            }

            // ---- Activation: strict trade-through ----
            bool enteredThisBar = false;
            if (active == null && pending != null)
            {
                bool hit = (pending.Dir == 1 && lo < pending.Entry) ||
                           (pending.Dir == -1 && hi > pending.Entry);
                if (hit)
                {
                    active = new ActiveTrade
                    {
                        Entry = pending.Entry, TP = pending.TP, SL = pending.SL,
                        Dir = pending.Dir, X = pending.X, A = pending.A,
                        B = pending.B, C = pending.C,
                        EntryBar = CurrentBar, EntryTime = Time[0]
                    };
                    pending = null;
                    enteredThisBar = true;
                }
            }

            // ---- Exits: SL first, no TP on entry bar, gap fills at open ----
            if (active != null)
            {
                double exitPx = double.NaN;
                int result = 0;
                if (active.Dir == 1)
                {
                    if (lo <= active.SL) { exitPx = Math.Min(op, active.SL); result = -1; }
                    else if (!enteredThisBar && hi >= active.TP) { exitPx = Math.Max(op, active.TP); result = 1; }
                }
                else
                {
                    if (hi >= active.SL) { exitPx = Math.Max(op, active.SL); result = -1; }
                    else if (!enteredThisBar && lo <= active.TP) { exitPx = Math.Min(op, active.TP); result = 1; }
                }
                if (!double.IsNaN(exitPx))
                {
                    double fricPts = 2.0 * FrictionTicks * TickSize;
                    double pts = (exitPx - active.Entry) * active.Dir - fricPts;
                    double dollars = pts * Instrument.MasterInstrument.PointValue;
                    pseudoCount++; pseudoPnl += dollars;
                    if (ledger != null)
                        ledger.WriteLine(string.Format(
                            "{0:yyyy-MM-dd HH:mm:ss},{1:yyyy-MM-dd HH:mm:ss},{2},{3:F2},{4:F2},{5:F2},{6:F2},{7:F4},{8:F4},{9:F4},{10:F4},{11},{12:F4},{13:F2},{14}",
                            active.EntryTime, Time[0], active.Dir, active.X, active.A,
                            active.B, active.C, active.Entry, exitPx, active.TP,
                            active.SL, result, pts, dollars, CurrentBar - active.EntryBar));
                    active = null;
                }
            }
        }

        // ---- Real order plumbing (off by default; pseudo ledger is the parity reference) ----
        private void SubmitRealEntry()
        {
            if (pending == null) return;
            if (pending.Dir == 1)
                EnterLongLimit(0, true, 1, pending.Entry, "CypherEntry");
            else
                EnterShortLimit(0, true, 1, pending.Entry, "CypherEntry");
            SetProfitTarget("CypherEntry", CalculationMode.Price, pending.TP);
            SetStopLoss("CypherEntry", CalculationMode.Price, pending.SL, false);
        }

        private void CancelRealEntry()
        {
            foreach (Order o in Orders)
                if (o.Name == "CypherEntry" &&
                    (o.OrderState == OrderState.Working || o.OrderState == OrderState.Accepted || o.OrderState == OrderState.Submitted))
                    CancelOrder(o);
        }

        #region Properties
        [NinjaScriptProperty, Range(3, 500)]
        [Display(Name = "ZigZag Period", Order = 1, GroupName = "01 Strategy")]
        public int ZigZagPeriod { get; set; }

        [NinjaScriptProperty, Range(0.01, 5.0)]
        [Display(Name = "TP Fraction of CD", Order = 2, GroupName = "01 Strategy")]
        public double TpFrac { get; set; }

        [NinjaScriptProperty, Range(0.0, 2.0)]
        [Display(Name = "SL Buffer Fraction of XC", Order = 3, GroupName = "01 Strategy")]
        public double SlBufferFrac { get; set; }

        [NinjaScriptProperty, Range(0.0, 1.0)]
        [Display(Name = "B Retrace Min", Order = 4, GroupName = "02 Ratios")]
        public double BLo { get; set; }

        [NinjaScriptProperty, Range(0.0, 1.0)]
        [Display(Name = "B Retrace Max", Order = 5, GroupName = "02 Ratios")]
        public double BHi { get; set; }

        [NinjaScriptProperty, Range(1.0, 3.0)]
        [Display(Name = "C Extension Min", Order = 6, GroupName = "02 Ratios")]
        public double CLo { get; set; }

        [NinjaScriptProperty, Range(1.0, 3.0)]
        [Display(Name = "C Extension Max", Order = 7, GroupName = "02 Ratios")]
        public double CHi { get; set; }

        [NinjaScriptProperty, Range(0.0, 1.0)]
        [Display(Name = "D Retracement of XC", Order = 8, GroupName = "02 Ratios")]
        public double DRet { get; set; }

        [NinjaScriptProperty, Range(0.0, 10.0)]
        [Display(Name = "Friction Ticks Per Side (pseudo ledger)", Order = 9, GroupName = "03 Accounting")]
        public double FrictionTicks { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Real Orders", Order = 10, GroupName = "04 Execution")]
        public bool EnableRealOrders { get; set; }
        #endregion
    }
}
