import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from rovio_trader import backtest, indicators, risk, signals
from rovio_trader.brokers import AlpacaBroker, BrokerError, ManualBroker, Order, PaperBroker
from rovio_trader.config import Settings
from rovio_trader.data import DataError, SyntheticProvider, asset_class, normalize_symbol, parse_yahoo_chart
from rovio_trader.engine import Engine, RiskRejected
from rovio_trader.server import make_handler

PROV = SyntheticProvider(end="2026-11-10")


def frame(closes, opens=None):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D", tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    o = pd.Series(opens, index=idx, dtype=float) if opens is not None else c
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1000.0})


class Indicators(unittest.TestCase):
    def test_sma_ema(self):
        s = pd.Series([1, 2, 3, 4, 5.0])
        self.assertEqual(indicators.sma(s, 3).tolist()[2:], [2, 3, 4])
        self.assertTrue(np.isnan(indicators.sma(s, 3).iloc[1]))
        self.assertAlmostEqual(indicators.ema(pd.Series([5.0] * 30), 10).iloc[-1], 5.0)

    def test_rsi_extremes(self):
        self.assertEqual(indicators.rsi(pd.Series(np.arange(1, 60.0))).iloc[-1], 100)
        self.assertAlmostEqual(indicators.rsi(pd.Series(np.arange(60, 1, -1.0))).iloc[-1], 0)
        self.assertEqual(indicators.rsi(pd.Series([7.0] * 40)).iloc[-1], 50)

    def test_rsi_known_value(self):
        # Wilder's textbook example; first RSI value is ~70.5 (ours is EMA-seeded, so compare loosely after warm-up)
        c = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00,
             46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57, 43.42, 42.66, 43.13]
        r = indicators.rsi(pd.Series(c)).iloc[-1]
        self.assertTrue(30 < r < 45, r)

    def test_bollinger_and_atr(self):
        df = PROV.history("AAPL", "1y")
        mid, up, lo = indicators.bollinger(df["close"])
        ok = mid.notna()
        self.assertTrue(((up >= mid) & (mid >= lo))[ok].all() and ok.sum() > 200)
        self.assertTrue((indicators.atr(df).dropna() > 0).all())
        self.assertTrue(indicators.adx(df).dropna().between(0, 100).all())

    def test_no_lookahead(self):
        """Indicator values at bar t must not change when later bars are appended."""
        df = PROV.history("MSFT", "2y")
        full, part = indicators.compute_all(df), indicators.compute_all(df.iloc[:300])
        pd.testing.assert_frame_equal(full.iloc[:300], part)
        pd.testing.assert_series_equal(signals.votes(full)["score"].iloc[:300], signals.votes(part)["score"])


class Signals(unittest.TestCase):
    def test_uptrend_buy_downtrend_sell(self):
        up = frame(100 * 1.004 ** np.arange(400) * (1 + 0.01 * np.sin(np.arange(400))))
        dn = frame(100 * 0.996 ** np.arange(400) * (1 + 0.01 * np.sin(np.arange(400))))
        self.assertIn("BUY", signals.analyze(up)["action"])
        self.assertIn("SELL", signals.analyze(dn)["action"])

    def test_short_history(self):
        self.assertEqual(signals.analyze(frame([1.0] * 10))["action"], "NO DATA")
        a = signals.analyze(PROV.history("AAPL", "6mo"))
        self.assertTrue(any("200" in w for w in a["warnings"]))
        self.assertEqual(sum(signals.WEIGHTS.values()), 100)
        self.assertTrue(-100 <= a["score"] <= 100)

    def test_levels(self):
        a = signals.analyze(PROV.history("SPY", "2y"))
        self.assertLess(a["levels"]["stop"], a["price"])
        self.assertGreater(a["levels"]["target"], a["price"])
        json.dumps(a)


class Backtest(unittest.TestCase):
    def test_buy_hold_matches_market_less_costs(self):
        df = PROV.history("SPY", "5y")
        r = backtest.run(df, "buy_hold", cost_bps=0, slippage_bps=0)
        # enters at the 2nd bar's open
        expect = df["close"].iloc[-1] / df["open"].iloc[1] - 1
        self.assertAlmostEqual(r["stats"]["total_return"], expect, places=3)

    def test_fills_next_open_and_costs(self):
        # Flat at 100, then gap to 110 and stay. A strategy that turns on at the close of bar 2 must NOT earn the gap on bar 3... it buys at bar 3's open.
        closes = [100.0] * 100 + [110.0] * 50
        df = frame(closes, opens=[100.0] * 100 + [110.0] * 50)
        orig = backtest.STRATEGIES["sma_cross"]
        backtest.STRATEGIES["_t"] = ("t", lambda i: (i["close"] > 105).astype(float))
        try:
            r = backtest.run(df, "_t", cost_bps=10, slippage_bps=0)
        finally:
            del backtest.STRATEGIES["_t"]
        self.assertEqual(orig, backtest.STRATEGIES["sma_cross"])
        self.assertAlmostEqual(r["stats"]["total_return"], -0.001, places=6)  # only the entry cost, none of the gap
        self.assertEqual(r["trades"][0]["entry_time"], df.index[101].isoformat())

    def test_costs_hurt(self):
        df = PROV.history("AAPL", "5y")
        a = backtest.run(df, "macd", 0, 0)["stats"]["total_return"]
        b = backtest.run(df, "macd", 20, 20)["stats"]["total_return"]
        self.assertGreater(a, b)

    def test_all_strategies_serialisable(self):
        for sym in ("BTC-USD", "FXAIX", "EURUSD=X"):
            for k in backtest.STRATEGIES:
                r = backtest.run(PROV.history(sym, "5y"), k)
                json.dumps(r)
                self.assertLessEqual(r["stats"]["max_drawdown"], 0)
        with self.assertRaises(ValueError):
            backtest.run(PROV.history("SPY", "5y"), "nope")


class Risk(unittest.TestCase):
    cfg = risk.RiskConfig()

    def test_position_size(self):
        # 1% of 100k = $1000 risk; stop 2*ATR(5) = $10 away -> 100 shares
        self.assertEqual(risk.position_size(self.cfg, "AAPL", 100_000, 100_000, 100, 5), 100)
        # tight ATR -> capped by 20% position limit (less headroom)
        self.assertEqual(risk.position_size(self.cfg, "AAPL", 100_000, 100_000, 100, 0.1), 196)
        self.assertEqual(risk.position_size(self.cfg, "AAPL", 100_000, 500, 100, 5), 4)  # cash-limited
        self.assertEqual(risk.position_size(self.cfg, "BTC-USD", 100_000, 100_000, 50_000, 2500), 0.2)

    def check(self, **kw):
        base = dict(side="buy", symbol="AAPL", qty=10, price=100, equity=100_000, cash=50_000, positions={}, kill_switch=False, day_start_equity=100_000)
        base.update(kw)
        return risk.check(self.cfg, **base)

    def test_checks(self):
        self.assertEqual(self.check(), [])
        self.assertTrue(self.check(kill_switch=True))
        self.assertTrue(self.check(side="sell", kill_switch=True))
        self.assertTrue(self.check(qty=600, cash=10_000))            # cash
        self.assertTrue(self.check(qty=300))                         # > 20% of equity
        self.assertTrue(self.check(side="sell", qty=5))              # no shorting
        self.assertEqual(self.check(side="sell", qty=5, positions={"AAPL": 5}), [])
        self.assertTrue(self.check(positions={f"S{i}": 1 for i in range(10)}))
        self.assertTrue(self.check(equity=96_000))                   # daily loss stop
        self.assertEqual(self.check(side="sell", qty=5, positions={"AAPL": 5}, equity=90_000), [])  # exits still allowed
        self.assertTrue(self.check(qty=0))


class Data(unittest.TestCase):
    def test_symbols(self):
        self.assertEqual(normalize_symbol(" eur/usd "), "EURUSD=X")
        self.assertEqual(normalize_symbol("btc/usd"), "BTC-USD")
        self.assertEqual([asset_class(s) for s in ("EURUSD=X", "BTC-USD", "FXAIX", "AAPL", "BRK-B")], ["forex", "crypto", "mutual_fund", "equity", "equity"])
        for bad in ("", "AAPL; rm", "../x", "A" * 20):
            with self.assertRaises(DataError):
                normalize_symbol(bad)

    def test_parse_yahoo(self):
        payload = {"chart": {"error": None, "result": [{"timestamp": [1700000000, 1700086400, 1700172800], "indicators": {
            "quote": [{"open": [10, None, 12], "high": [11, None, 13], "low": [9, None, 11], "close": [10, None, 12], "volume": [100, None, 300]}],
            "adjclose": [{"adjclose": [5, None, 12]}]}}]}}
        df = parse_yahoo_chart(payload, "X")
        self.assertEqual(len(df), 2)                       # null bar dropped
        self.assertEqual(df["close"].tolist(), [5.0, 12.0])  # adjusted
        self.assertEqual(df["high"].iloc[0], 5.5)
        with self.assertRaises(DataError):
            parse_yahoo_chart({"chart": {"result": None, "error": {"description": "No data found"}}}, "X")

    def test_synthetic_deterministic(self):
        a = SyntheticProvider(end="2026-11-10").history("SPY", "1y")
        pd.testing.assert_frame_equal(a, PROV.history("SPY", "1y"))
        self.assertTrue((a["high"] >= a[["open", "close"]].max(axis=1)).all() and (a["low"] <= a[["open", "close"]].min(axis=1)).all())
        with self.assertRaises(DataError):
            PROV.history("SPY", "99y")


class Brokers(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_paper(self):
        b = PaperBroker(self.dir / "p.json", lambda s: 100.0, 10_000, slippage_bps=0)
        b.submit(Order("AAPL", "buy", 50))
        b.submit(Order("AAPL", "buy", 10))
        self.assertEqual(b.account()["cash"], 4000)
        self.assertEqual(b.account()["equity"], 10_000)
        with self.assertRaises(BrokerError):
            b.submit(Order("AAPL", "buy", 100))
        with self.assertRaises(BrokerError):
            b.submit(Order("AAPL", "sell", 61))
        b.submit(Order("AAPL", "sell", 60))
        self.assertEqual(b.positions(), [])
        self.assertEqual(PaperBroker(self.dir / "p.json", lambda s: 100.0).account()["cash"], 10_000)  # persisted
        self.assertEqual(len(b.orders()), 3)

    def test_manual_ticket_flow(self):
        b = ManualBroker(self.dir / "m.json", lambda s: 100.0, 10_000)
        o = b.submit(Order("FXAIX", "buy", 10))
        self.assertEqual(o.status, "ticket")
        self.assertEqual(b.account()["cash"], 10_000)       # nothing booked until the fill is reported
        b.confirm_fill(o.id, 101.5)
        self.assertEqual(b.account()["cash"], 10_000 - 1015)
        with self.assertRaises(BrokerError):
            b.confirm_fill(o.id, 101.5)
        o2 = b.submit(Order("FXAIX", "sell", 10))
        self.assertEqual(b.cancel(o2.id)["status"], "cancelled")

    def test_alpaca(self):
        with self.assertRaises(BrokerError):
            AlpacaBroker(env={})
        calls = []

        class Resp:
            def __init__(self, d): self.d = d
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self, *a): return json.dumps(self.d).encode()

        def opener(req, timeout=0):
            calls.append((req.get_method(), req.full_url, req.data, dict(req.header_items())))
            return Resp({"id": "abc", "cash": "5", "equity": "7"})

        b = AlpacaBroker(env={"ALPACA_API_KEY_ID": "unit-test-id", "ALPACA_API_SECRET_KEY": "unit-test-value"}, opener=opener)
        self.assertFalse(b.live)
        self.assertNotIn("unit-test", repr(b))
        self.assertEqual(b.account(), {"cash": 5.0, "equity": 7.0})
        o = b.submit(Order("BTC-USD", "buy", 0.25))
        m, url, data, _ = calls[-1]
        self.assertTrue(url.startswith("https://paper-api.alpaca.markets/v2/orders"))
        body = json.loads(data)
        self.assertEqual((body["symbol"], body["qty"], body["time_in_force"], o.status), ("BTC/USD", "0.25", "gtc", "submitted"))
        with self.assertRaises(BrokerError):
            b.submit(Order("FXAIX", "buy", 1))
        self.assertTrue(AlpacaBroker(env={"ALPACA_API_KEY_ID": "a", "ALPACA_API_SECRET_KEY": "b", "ALPACA_LIVE": "true"}).live)
        self.assertFalse(AlpacaBroker(env={"ALPACA_API_KEY_ID": "a", "ALPACA_API_SECRET_KEY": "b", "ALPACA_LIVE": "yes"}).live)


def make_engine(tmp, broker="paper", watch=("SPY", "BTC-USD", "FXAIX", "EURUSD=X")):
    s = Settings(broker=broker, demo=True, watchlist=list(watch), state_dir=Path(tmp))
    cls = ManualBroker if broker == "manual" else PaperBroker
    return Engine(s, PROV, cls(Path(tmp) / "b.json", PROV.last_price, 100_000))


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.e = make_engine(self.tmp)

    def test_propose_approve(self):
        props = self.e.generate_proposals()
        self.assertTrue(props and all(p["side"] == "buy" for p in props))
        for p in props:
            self.assertLessEqual(p["est_value"], 20_000)
        self.assertEqual(self.e.broker.positions(), [])            # proposing never trades
        done = self.e.approve(props[0]["id"])
        self.assertEqual(done["status"], "executed")
        self.assertEqual(len(self.e.broker.positions()), 1)
        with self.assertRaises(BrokerError):
            self.e.approve(props[0]["id"])
        if len(props) > 1:
            self.assertEqual(self.e.reject(props[1]["id"])["status"], "rejected")
        again = self.e.generate_proposals()
        self.assertNotIn(props[0]["symbol"], [p["symbol"] for p in again])   # already held
        self.assertEqual(make_engine(self.tmp).state["proposals"], self.e.state["proposals"])  # persisted

    def test_kill_switch_and_limits(self):
        self.e.set_kill_switch(True)
        with self.assertRaises(RiskRejected):
            self.e.submit_order("SPY", "buy", 1)
        self.e.set_kill_switch(False)
        with self.assertRaises(RiskRejected) as cm:
            self.e.submit_order("SPY", "buy", 100_000)
        self.assertGreaterEqual(len(cm.exception.reasons), 2)
        with self.assertRaises(RiskRejected):
            self.e.submit_order("SPY", "sell", 1)
        self.assertEqual(self.e.submit_order("spy", "buy", 1)["status"], "filled")

    def test_auto_cycle_and_sell_on_negative_score(self):
        out = self.e.run_cycle(auto=True)
        self.assertTrue(out["executed"] and all(x["ok"] for x in out["executed"]), out["executed"])
        e2 = make_engine(tempfile.mkdtemp(), watch=("FXAIX",))
        self.assertLess(e2.analyze("FXAIX")["score"], 0)
        e2.submit_order("FXAIX", "buy", 10)
        props = e2.generate_proposals()
        self.assertEqual([(p["side"], p["qty"]) for p in props], [("sell", 10)])

    def test_manual_broker_creates_tickets(self):
        e = make_engine(tempfile.mkdtemp(), broker="manual")
        props = e.generate_proposals()
        e.approve(props[0]["id"])
        self.assertEqual(e.broker.positions(), [])
        self.assertEqual(e.broker.orders()[0]["status"], "ticket")
        self.assertFalse(e.info()["executes"])


class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = make_engine(tempfile.mkdtemp())
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.e))
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def call(self, path, body=None, headers=None):
        h = {"Content-Type": "application/json"} if body is not None else {}
        h.update(headers or {})
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None, headers=h)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read()) if "json" in r.headers["Content-Type"] else r.read()
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_endpoints(self):
        self.assertEqual(self.call("/")[0], 200)
        self.assertEqual(len(self.call("/api/scan")[1]), 4)
        st, ch = self.call("/api/chart?symbol=SPY&range=1y")
        self.assertEqual(len(ch["bars"]["t"]), len(ch["ind"]["rsi"]))
        self.assertIsNone(ch["ind"]["sma200"][0])
        self.assertEqual(self.call("/api/backtest?symbol=SPY&strategy=donchian")[0], 200)
        self.assertEqual(self.call("/api/chart?symbol=%3Cscript%3E")[0], 400)
        self.assertEqual(self.call("/api/chart")[0], 400)
        self.assertEqual(self.call("/api/nope")[0], 404)

    def test_orders_and_csrf(self):
        self.assertEqual(self.call("/api/orders", {"symbol": "SPY", "side": "buy", "qty": 2})[0], 200)
        st, body = self.call("/api/orders", {"symbol": "SPY", "side": "buy", "qty": 1e7})
        self.assertEqual(st, 422)
        self.assertTrue(body["reasons"])
        self.assertEqual(self.call("/api/orders", {"symbol": "SPY", "side": "buy", "qty": 1}, {"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.call("/api/orders", {"symbol": "SPY", "side": "buy", "qty": 1}, {"Content-Type": "text/plain"})[0], 403)
        self.assertEqual(self.call("/api/killswitch", {"on": True})[1]["kill_switch"], True)
        self.assertEqual(self.call("/api/orders", {"symbol": "SPY", "side": "sell", "qty": 1})[0], 422)
        self.call("/api/killswitch", {"on": False})
        self.assertIn("TSLA", self.call("/api/watchlist", {"add": "tsla"})[1])
        self.assertNotIn("TSLA", self.call("/api/watchlist", {"remove": "TSLA"})[1])


if __name__ == "__main__":
    unittest.main()
