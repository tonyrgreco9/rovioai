import unittest

import numpy as np
import pandas as pd

from rovio_trader import config, fixtures as F, indicators, instruments as I, strategies as S
from rovio_trader.data import daily_frame, synthetic
from rovio_trader.events import EarningsEvent, EconCalendar

CFG = config.load()
U = lambda s: pd.Timestamp(s, tz="UTC")


def stock_frame(closes, vols, opens=None):
    days = pd.bdate_range("2025-01-02", periods=len(closes))
    c = np.asarray(closes, float)
    o = np.asarray(opens, float) if opens is not None else np.concatenate([[c[0]], c[:-1]])
    return daily_frame(days, o, np.maximum(o, c) + 0.2, np.minimum(o, c) - 0.2, c, vols)


class Causality(unittest.TestCase):
    """A signal at bar t must be identical whether or not later bars exist."""

    def check(self, strat, inst, df, extra=None, cut_frac=0.8):
        full, _ = strat.build(inst, df, extra)
        cut = int(len(df) * cut_frac)
        part, _ = strat.__class__(CFG).build(inst, df.iloc[:cut], extra)
        limit = df.index[cut - 1]
        self.assertEqual([s for s in full if pd.Timestamp(s["time"]) <= limit], part)
        return full

    def test_all_strategies(self):
        n = 0
        n += len(self.check(S.Breakout(CFG), I.stock("X", "alpaca"), synthetic("X", "stock", "1d", "2025-12-12", 900)))
        n += len(self.check(S.OpeningRange(CFG), I.future("MES", "2099-12-18", "ibkr"), synthetic("MES", "futures", "5m", "2025-12-12", 78 * 60)))
        n += len(self.check(S.FxPullback(CFG), I.forex("EUR/USD", "oanda"), synthetic("EUR/USD", "forex", "1h", "2025-12-12", 4000)))
        n += len(self.check(S.BtcBreakout(CFG), I.crypto("BTC/USD", "alpaca"), synthetic("BTC/USD", "crypto", "4h", "2025-12-12", 2500)))
        self.assertGreater(n, 20)

    def test_resample_only_exposes_completed_coarse_bars(self):
        df = synthetic("BTC/USD", "crypto", "4h", "2025-12-12", 60)
        d = S.resample(df, "1D")
        for end, row in d.iterrows():
            self.assertTrue((df.index[df["start"].dt.floor("1D") == row["start"].floor("1D")] <= end).all())
        vals = S.asof(d["close"], df.index)
        # a 4h bar ending mid-day must not see that day's close
        i = 3
        self.assertNotEqual(df.index[i].hour, 0)
        day_close = d["close"][d.index > df.index[i]].iloc[0]
        self.assertFalse(vals[i] == day_close and not np.isnan(vals[i]) and d.index[d["close"] == day_close][0] > df.index[i])

    def test_indicators(self):
        s = pd.Series([1, 2, 3, 4, 5.0])
        self.assertEqual(indicators.sma(s, 3).tolist()[2:], [2, 3, 4])
        self.assertEqual(indicators.rsi(pd.Series(np.arange(1, 60.0))).iloc[-1], 100)


class BreakoutRule(unittest.TestCase):
    def setUp(self):
        self.s, self.inst = S.Breakout(CFG), I.stock("X", "alpaca")

    def frame(self, last_close=104.0, last_vol=2_000_000):
        return stock_frame([100 + 0.3 * np.sin(i) for i in range(70)] + [last_close], [1_000_000] * 70 + [last_vol])

    def test_signal_needs_all_three_conditions(self):
        sig, _ = self.s.build(self.inst, self.frame(), None)
        self.assertEqual(len(sig), 1)
        x = sig[0]
        self.assertEqual((x["side"], x["strategy"]), (1, "breakout"))
        self.assertLess(x["stop"], x["ref_price"])
        self.assertTrue(all(c["passed"] for c in x["inputs"]) and len(x["inputs"]) == 3)
        self.assertEqual(S.Breakout(CFG).build(self.inst, self.frame(last_vol=1_400_000), None)[0], [])   # volume 1.4x < 1.5x
        self.assertEqual(S.Breakout(CFG).build(self.inst, self.frame(last_close=100.2), None)[0], [])    # no new high

    def test_below_moving_average_blocks(self):
        closes = [140 - 1.0 * i for i in range(45)] + [96.0] * 25 + [99.0]     # new 20-day high but below the 50-day average
        df = stock_frame(closes, [1e6] * 70 + [3e6])
        self.assertGreater(df["close"].iloc[-1], df["high"].iloc[-21:-1].max())
        self.assertEqual(self.s.build(self.inst, df, None)[0], [])

    def test_insufficient_history(self):
        self.assertEqual(self.s.build(self.inst, self.frame().iloc[-30:], None)[0], [])
        snap = self.s.snapshot(self.inst, self.frame().iloc[-30:], U("2030-01-01"))
        self.assertFalse(snap[0]["passed"])

    def test_gap_and_spread_checks(self):
        sig = self.s.build(self.inst, self.frame(), None)[0][0]
        ok = self.s.entry_checks(sig, self.inst, {"bid": 104.4, "ask": 104.45}, None, {})
        self.assertTrue(all(c["passed"] for c in ok))
        gap = self.s.entry_checks(sig, self.inst, {"bid": 109.0, "ask": 109.05}, None, {})
        self.assertFalse(gap[0]["passed"])
        wide = self.s.entry_checks(sig, self.inst, {"bid": 104.0, "ask": 104.5}, None, {})
        self.assertFalse(wide[1]["passed"])


class EarningsRule(unittest.TestCase):
    def setUp(self):
        self.s, self.inst = S.Earnings(CFG), I.stock("X", "alpaca")
        closes = [100 + 0.3 * np.sin(i) for i in range(40)]
        self.days = pd.bdate_range("2025-01-02", periods=42)
        self.df = stock_frame(closes + [104.0, 104.5], [1e6] * 40 + [3e6, 1e6])
        self.rel = self.days[40]          # the high-volume up day

    def ev(self, **kw):
        base = dict(symbol="X", release_date=f"{self.rel:%Y-%m-%d}", session="bmo", eps_actual=1.2, eps_estimate=1.0, actual_basis="adjusted",
                    estimate_basis="adjusted", estimate_asof=f"{self.days[38]:%Y-%m-%d}", source="t")
        base.update(kw)
        return EarningsEvent(**base)

    def test_before_open_release_uses_same_day(self):
        kind, sig, t = self.s.evaluate(self.ev(), self.inst, self.df)
        self.assertEqual(kind, "signal")
        self.assertEqual(sig["meta"]["confirmation_bar"], f"{self.rel:%Y-%m-%d}")
        self.assertAlmostEqual(sig["meta"]["pre_release_close"], float(self.df["close"].iloc[39]))

    def test_after_close_release_uses_next_session(self):
        prev = self.days[39]
        kind, sig, _ = self.s.evaluate(self.ev(release_date=f"{prev:%Y-%m-%d}", session="amc", estimate_asof=f"{self.days[37]:%Y-%m-%d}"), self.inst, self.df)
        self.assertEqual(kind, "signal")
        self.assertEqual(sig["meta"]["confirmation_bar"], f"{self.rel:%Y-%m-%d}")
        self.assertAlmostEqual(sig["meta"]["pre_release_close"], float(self.df["close"].iloc[39]))

    def test_waits_until_reaction_session_completes(self):
        self.assertEqual(self.s.evaluate(self.ev(), self.inst, self.df.iloc[:40])[0], "wait")

    def test_unverifiable_inputs_are_skipped_with_reason(self):
        cases = {"release time": dict(session=""), "release time ": dict(session="during"), "missing": dict(eps_estimate=None),
                 "basis mismatch": dict(estimate_basis="gaap"), "EPS basis": dict(actual_basis=""), "no timestamp": dict(estimate_asof=""),
                 "not verifiably before": dict(estimate_asof=f"{self.rel:%Y-%m-%d}")}
        for needle, kw in cases.items():
            kind, why, _ = self.s.evaluate(self.ev(**kw), self.inst, self.df)
            self.assertEqual(kind, "skip", kw)
            self.assertIn(needle.strip(), why)

    def test_no_beat_weak_price_or_volume(self):
        self.assertIn("no qualifying beat", self.s.evaluate(self.ev(eps_actual=0.9), self.inst, self.df)[1])
        weak = self.df.copy()
        weak.iloc[40, weak.columns.get_loc("volume")] = 1.2e6
        self.assertIn("volume", self.s.evaluate(self.ev(), self.inst, weak)[1])
        down = self.df.copy()
        down.iloc[40, down.columns.get_loc("close")] = 99.0
        self.assertIn("not above the pre-release close", self.s.evaluate(self.ev(), self.inst, down)[1])

    def test_etfs_excluded(self):
        self.assertFalse(self.s.applies(I.stock("SPY", "alpaca", is_etf=True)))
        self.assertTrue(self.s.applies(self.inst))


class OtherMarkets(unittest.TestCase):
    def setUp(self):
        cfg = config.load(None, F.OVERRIDES)
        self.cfg = cfg
        self.insts, self.ex, self.data, _, self.cal = F.build(cfg)
        self.by = {i.symbol: i for i in self.insts}

    def test_opening_range_long_short_and_cutoffs(self):
        s = S.OpeningRange(self.cfg)
        long_, _ = s.build(self.by["MESZ5"], self.ex["MESZ5"], None)
        self.assertEqual([(x["side"], x["time"]) for x in long_], [(1, "2025-12-11T15:05:00Z")])       # first bar after 10:00 ET that closes outside
        x = long_[0]
        self.assertGreater(x["target"], x["ref_price"])
        self.assertEqual(x["flatten_at"], "2025-12-11T20:45:00Z")                                      # 15:45 ET
        self.assertEqual(x["stop"] % 0.25, 0)
        short, _ = s.build(self.by["MESH6"], self.ex["MESH6"], None)
        self.assertEqual([x["side"] for x in short], [-1])
        self.assertLess(short[0]["target"], short[0]["ref_price"])
        self.assertEqual(s.build(self.by["MNQZ5"], self.ex["MNQZ5"], None)[0], [])                     # stays inside the range: no trade
        no_short = S.OpeningRange(config.load(None, {"strategies": {"orb": {"allow_short": False}}}))
        self.assertEqual(no_short.build(self.by["MESH6"], self.ex["MESH6"], None)[0], [])
        q = {"bid": x["ref_price"] - 0.125, "ask": x["ref_price"] + 0.125}
        late = s.entry_checks(x, self.by["MESZ5"], q, U("2025-12-11 19:45"), {})
        self.assertFalse(late[3]["passed"])
        near_expiry = s.entry_checks(x, self.by["MESZ5"], q, U("2025-12-15 15:05"), {})
        self.assertFalse(near_expiry[2]["passed"])

    def test_fx_pullback_both_directions_and_calendar(self):
        s = S.FxPullback(self.cfg)
        jpy, _ = s.build(self.by["USD/JPY"], self.ex["USD/JPY"], None)
        eur, _ = s.build(self.by["EUR/USD"], self.ex["EUR/USD"], None)
        self.assertEqual([(x["side"], x["time"]) for x in jpy], [(1, "2025-12-02T14:00:00Z")])
        self.assertEqual([(x["side"], x["time"]) for x in eur], [(-1, "2025-12-03T12:00:00Z")])
        self.assertEqual(s.build(self.by["GBP/USD"], self.ex["GBP/USD"], None)[0], [])
        x, inst = eur[0], self.by["EUR/USD"]
        q = {"bid": x["ref_price"] - 0.00006, "ask": x["ref_price"] + 0.00006}
        now = U("2025-12-03 12:00")
        self.cal.asof = now
        self.assertIn("blackout", s.entry_checks(x, inst, q, now, {"calendar": self.cal})[-1]["name"])
        self.cal.asof = now - pd.Timedelta(hours=30)
        self.assertIn("stale", s.entry_checks(x, inst, q, now, {"calendar": self.cal})[-1]["name"])
        self.assertIn("unavailable", s.entry_checks(x, inst, q, now, {"calendar": None})[-1]["name"])
        self.assertIn("unavailable", s.entry_checks(x, inst, q, now, {"calendar": EconCalendar()})[-1]["name"])
        wide = {"bid": x["ref_price"] - 0.0003, "ask": x["ref_price"] + 0.0003}
        self.assertFalse(s.entry_checks(x, inst, wide, now, {"calendar": self.cal})[0]["passed"])
        self.assertEqual(self.by["USD/JPY"].pip, 0.01)
        self.assertEqual(inst.pip, 0.0001)

    def test_btc_breakout_needs_daily_trend(self):
        s = S.BtcBreakout(self.cfg)
        sig, _ = s.build(self.by["BTC/USD"], self.ex["BTC/USD"], None)
        self.assertEqual(sig[0]["time"], "2025-12-06T00:00:00Z")
        self.assertTrue(all(pd.Timestamp(x["time"]) >= U("2025-12-06") for x in sig))     # Friday's new highs came before the daily trend turned
        self.assertEqual(pd.Timestamp(sig[0]["time"]).day_name(), "Saturday")


class Instruments(unittest.TestCase):
    def test_futures_contract_resolution_and_rollover(self):
        cs = I.quarterly_contracts("MES", "ibkr", U("2025-12-01"))
        self.assertEqual([c.symbol for c in cs], ["MESZ5", "MESH6", "MESM6"])
        self.assertEqual((cs[0].multiplier, cs[0].tick_size, cs[0].tick_value, cs[0].expiry), (5.0, 0.25, 1.25, "2025-12-19"))
        self.assertEqual(I.active_contract(cs, U("2025-12-11"), 7).symbol, "MESZ5")
        self.assertEqual(I.active_contract(cs, U("2025-12-12"), 7).symbol, "MESH6")
        self.assertEqual(I.future("MNQ", "2026-03-20", "ibkr").tick_value, 0.5)

    def test_rounding_and_conversion(self):
        self.assertEqual(I.crypto("BTC/USD", "a").round_qty(0.12349), 0.1234)
        self.assertEqual(I.stock("X", "a").round_qty(79.99), 79)
        self.assertAlmostEqual(I.usd_per_quote("JPY", {"USD/JPY": 150.0}), 1 / 150)
        self.assertEqual(I.usd_per_quote("USD", {}), 1.0)
        with self.assertRaises(KeyError):
            I.usd_per_quote("CHF", {})


if __name__ == "__main__":
    unittest.main()
