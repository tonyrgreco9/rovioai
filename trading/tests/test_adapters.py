import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from rovio_trader import backtest, config, replay, server
from rovio_trader.brokers import AlpacaBroker, BrokerError, BrokerTimeout, BrokerUnavailable, OandaBroker, connect
from rovio_trader.brokers.unavailable import FIDELITY_STATUS, IBKR_STATUS
from rovio_trader.store import Store

KEY = {"ALPACA_API_KEY_ID": "unit-test-key-id", "ALPACA_API_SECRET_KEY": "unit-test-secret-value"}
OKEY = {"OANDA_API_TOKEN": "unit-test-oanda-token", "OANDA_ACCOUNT_ID": "101-001-0000000-001"}


class FakeHttp:
    """Stands in for the broker's servers. Proves our requests are well-formed and our parsing is
    self-consistent; it cannot prove the real API behaves this way."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, req, timeout=0):
        body = json.loads(req.data) if req.data else None
        self.calls.append((req.get_method(), req.full_url, body, dict(req.header_items())))
        for (method, frag), resp in self.routes.items():
            if method == req.get_method() and frag in req.full_url:
                if isinstance(resp, Exception):
                    raise resp
                if callable(resp):
                    resp = resp(body)
                if isinstance(resp, int):
                    raise urllib.error.HTTPError(req.full_url, resp, "err", {}, io.BytesIO(b'{"message":"nope"}'))
                return io.BytesIO(json.dumps(resp).encode())
        raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, io.BytesIO(b"{}"))


ORDER = {"id": "o-1", "client_order_id": "rve-1", "symbol": "AAPL", "side": "buy", "type": "market", "qty": "10", "filled_qty": "4",
         "filled_avg_price": "201.5", "status": "partially_filled", "stop_price": None}


class Alpaca(unittest.TestCase):
    def broker(self, routes, live=False, env=KEY):
        self.http = FakeHttp(routes)
        return AlpacaBroker(live=live, env=env, opener=self.http)

    def test_credentials_and_host_separation(self):
        with self.assertRaises(BrokerUnavailable):
            AlpacaBroker(env={})
        with self.assertRaises(BrokerUnavailable):
            AlpacaBroker(live=True, env=KEY)                      # paper keys never unlock the live host
        b = self.broker({})
        self.assertEqual((b.mode, b.base), ("PAPER", "https://paper-api.alpaca.markets"))
        live = AlpacaBroker(live=True, env={"ALPACA_LIVE_API_KEY_ID": "a-live-id", "ALPACA_LIVE_API_SECRET_KEY": "a-live-secret"}, opener=FakeHttp({}))
        self.assertEqual((live.mode, live.base), ("LIVE", "https://api.alpaca.markets"))
        self.assertNotIn("unit-test", repr(b))
        self.assertFalse(b.connection_verified)

    def test_account_marks_connection_verified_and_masks_account(self):
        b = self.broker({("GET", "/v2/account"): {"equity": "1000.5", "cash": "400", "account_number": "PA123456789"}})
        self.assertEqual(b.account(), {"equity": 1000.5, "cash": 400.0, "margin_available": 400.0})
        self.assertTrue(b.connection_verified)
        self.assertEqual(b.account_alias, "...6789")
        self.assertEqual(self.http.calls[0][3]["Apca-api-key-id"], "unit-test-key-id")

    def test_stock_orders(self):
        b = self.broker({("GET", "/v2/assets/AAPL"): {"tradable": True, "status": "active", "class": "us_equity", "exchange": "NASDAQ"},
                         ("POST", "/v2/orders"): ORDER, ("GET", "/v2/positions"): [{"symbol": "AAPL", "qty": "10", "avg_entry_price": "200"}]})
        o = b.submit({"cid": "rve-1", "symbol": "AAPL", "side": "buy", "type": "market", "qty": 10})
        self.assertEqual((o["status"], o["filled_qty"], o["avg_price"], o["fees"]), ("partially_filled", 4.0, 201.5, None))
        self.assertEqual(self.http.calls[-1][2], {"symbol": "AAPL", "qty": "10", "side": "buy", "client_order_id": "rve-1", "type": "market", "time_in_force": "day"})
        b.submit({"cid": "rvs-1", "symbol": "AAPL", "side": "sell", "type": "stop", "qty": 10, "stop_price": 194.126})
        self.assertEqual(self.http.calls[-1][2], {"symbol": "AAPL", "qty": "10", "side": "sell", "client_order_id": "rvs-1", "type": "stop", "time_in_force": "gtc", "stop_price": "194.13"})
        with self.assertRaises(BrokerError):                      # an exit larger than the holding would open a short: refused locally
            b.submit({"cid": "rvx-1", "symbol": "AAPL", "side": "sell", "type": "market", "qty": 11})

    def test_crypto_uses_stop_limit_and_broker_increments(self):
        b = self.broker({("GET", "/v2/assets/BTC%2FUSD"): {"tradable": True, "status": "active", "class": "crypto", "min_order_size": "0.0001",
                                                            "min_trade_increment": "0.0001", "price_increment": "0.01"},
                         ("GET", "/v2/positions"): [{"symbol": "BTCUSD", "asset_class": "crypto", "qty": "0.25", "avg_entry_price": "90000"}],
                         ("POST", "/v2/orders"): {**ORDER, "symbol": "BTC/USD", "status": "accepted", "filled_qty": "0", "filled_avg_price": None}})
        inst = b.instrument("BTC/USD")
        self.assertEqual((inst.market, inst.qty_step, inst.min_qty, inst.tick_size), ("crypto", 0.0001, 0.0001, 0.01))
        self.assertEqual(b.positions(), {"BTC/USD": {"qty": 0.25, "avg_price": 90000.0}})
        o = b.submit({"cid": "rvs-2", "symbol": "BTC/USD", "side": "sell", "type": "stop", "qty": 0.25, "stop_price": 88000.0})
        body = self.http.calls[-1][2]
        self.assertEqual((body["type"], body["time_in_force"], body["stop_price"], body["limit_price"], body["qty"]), ("stop_limit", "gtc", "88000.0", "87560.0", "0.25"))
        self.assertEqual(o["status"], "new")

    def test_untradable_asset_and_lookup_by_client_id(self):
        b = self.broker({("GET", "/v2/assets/ZZZZ"): {"tradable": False, "status": "active"}, ("GET", "orders:by_client_order_id"): ORDER})
        self.assertIsNone(b.instrument("ZZZZ"))
        self.assertIsNone(b.instrument("NOPE"))                   # 404
        self.assertEqual(b.get_by_cid("rve-1")["cid"], "rve-1")
        self.assertIn("client_order_id=rve-1", self.http.calls[-1][1])
        self.assertIsNone(self.broker({}).get_by_cid("missing"))

    def test_errors_distinguish_refused_from_unknown(self):
        asset = {("GET", "/v2/assets/AAPL"): {"tradable": True, "status": "active", "class": "us_equity"}}
        order = {"cid": "c", "symbol": "AAPL", "side": "buy", "type": "market", "qty": 1}
        with self.assertRaises(BrokerError) as cm:
            self.broker({**asset, ("POST", "/v2/orders"): 403}).submit(order)
        self.assertIn("nope", str(cm.exception))
        for failure in (500, 429, urllib.error.URLError("reset"), TimeoutError()):
            with self.assertRaises(BrokerTimeout) as cm:
                self.broker({**asset, ("POST", "/v2/orders"): failure}).submit(order)
            self.assertNotIn("unit-test", str(cm.exception))      # credentials never appear in errors

    def test_quotes_and_daily_bars(self):
        b = self.broker({("GET", "/v2/stocks/quotes/latest"): {"quotes": {"AAPL": {"ap": 200.1, "bp": 200.0, "t": "2026-01-05T15:00:00Z"}}},
                         ("GET", "/v2/assets/AAPL"): {"tradable": True, "status": "active", "class": "us_equity"},
                         ("GET", "/v2/stocks/bars"): {"bars": {"AAPL": [{"t": "2026-01-02T05:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10}]}, "next_page_token": None}})
        inst = b.instrument("AAPL")
        q = b.quote(inst)
        self.assertEqual((q["bid"], q["ask"]), (200.0, 200.1))
        df = b.bars(inst, "1d", 5)
        self.assertEqual(str(df.index[0]), "2026-01-02 21:00:00+00:00")       # the bar is usable only after the 16:00 ET close
        self.assertEqual(str(df["start"].iloc[0]), "2026-01-02 14:30:00+00:00")
        self.assertIsNone(self.broker({("GET", "/v2/stocks/quotes/latest"): {"quotes": {"AAPL": {"ap": 0, "bp": 0, "t": "2026-01-05T15:00:00Z"}}}}).quote(inst))


class Oanda(unittest.TestCase):
    def broker(self, routes):
        self.http = FakeHttp(routes)
        return OandaBroker(env=OKEY, opener=self.http)

    def test_hosts_and_account(self):
        with self.assertRaises(BrokerUnavailable):
            OandaBroker(env={})
        b = self.broker({("GET", "/summary"): {"account": {"NAV": "50010", "balance": "50000", "marginAvailable": "49000", "currency": "USD"}}})
        self.assertIn("api-fxpractice.oanda.com", b._acct)
        self.assertEqual(b.account()["margin_available"], 49000.0)
        self.assertEqual(b.account_alias, "...-001")
        self.assertNotIn("unit-test", repr(b))
        eur = self.broker({("GET", "/summary"): {"account": {"NAV": "1", "balance": "1", "marginAvailable": "1", "currency": "EUR"}}})
        with self.assertRaises(BrokerUnavailable):
            eur.account()

    def test_instrument_precision_comes_from_the_broker(self):
        b = self.broker({("GET", "/instruments"): {"instruments": [{"name": "USD_JPY", "pipLocation": -2, "displayPrecision": 3, "tradeUnitsPrecision": 0,
                                                                     "minimumTradeSize": "1", "marginRate": "0.04"}]}})
        i = b.instrument("USD/JPY")
        self.assertEqual((i.pip, i.tick_size, i.qty_step, i.margin_rate, i.quote_ccy, i.shortable), (0.01, 0.001, 1.0, 0.04, "JPY", True))

    def test_orders_are_reduce_only_for_exits_and_stops(self):
        fill = {"orderCreateTransaction": {"id": "77"}, "orderFillTransaction": {"units": "-10000", "price": "150.812", "commission": "0"}}
        b = self.broker({("POST", "/orders"): fill, ("GET", "/instruments"): {"instruments": [{"pipLocation": -2, "displayPrecision": 3}]}})
        o = b.submit({"cid": "rvx-9", "symbol": "USD/JPY", "side": "sell", "type": "market", "qty": 10000, "reduce_only": True})
        body = self.http.calls[-1][2]["order"]
        self.assertEqual((body["units"], body["type"], body["timeInForce"], body["positionFill"], body["clientExtensions"]), ("-10000", "MARKET", "FOK", "REDUCE_ONLY", {"id": "rvx-9"}))
        self.assertEqual((o["status"], o["filled_qty"], o["avg_price"]), ("filled", 10000.0, 150.812))
        b.submit({"cid": "rvs-9", "symbol": "USD/JPY", "side": "sell", "type": "stop", "qty": 10000, "stop_price": 150.5, "reduce_only": True})
        body = self.http.calls[-1][2]["order"]
        self.assertEqual((body["type"], body["price"], body["positionFill"], body["timeInForce"]), ("STOP", "150.500", "REDUCE_ONLY", "GTC"))
        entry = self.broker({("POST", "/orders"): fill})
        entry.submit({"cid": "rve-9", "symbol": "USD/JPY", "side": "buy", "type": "market", "qty": 10000})
        self.assertEqual(self.http.calls[-1][2]["order"]["positionFill"], "DEFAULT")
        rej = self.broker({("POST", "/orders"): {"orderRejectTransaction": {"rejectReason": "INSUFFICIENT_MARGIN"}}})
        with self.assertRaises(BrokerError):
            rej.submit({"cid": "c", "symbol": "USD/JPY", "side": "buy", "type": "market", "qty": 1})

    def test_quote_and_tradeable_flag(self):
        b = self.broker({("GET", "/pricing"): {"prices": [{"tradeable": False, "time": "2026-01-03T10:00:00Z", "bids": [{"price": "150.10"}], "asks": [{"price": "150.12"}]}]}})
        inst = type("I", (), {"symbol": "USD/JPY"})()
        self.assertFalse(b.market_open(inst))
        self.assertEqual(b.quote(inst)["ask"], 150.12)


class Wiring(unittest.TestCase):
    def test_missing_credentials_block_only_that_integration(self):
        cfg = config.load()
        brokers, status = connect(cfg, env=KEY)
        self.assertEqual(list(brokers), ["alpaca"])
        self.assertFalse(status["oanda"]["available"])
        self.assertIn("OANDA_API_TOKEN", status["oanda"]["reason"])
        self.assertFalse(status["ibkr"]["available"])
        self.assertEqual(connect(cfg, env={})[0], {})

    def test_unavailable_integrations_are_reported_not_faked(self):
        self.assertFalse(FIDELITY_STATUS["order_entry_available"])
        self.assertFalse(IBKR_STATUS["paper_adapter_implemented"])

    def test_config_validation_and_redaction(self):
        d = Path(tempfile.mkdtemp())
        (d / "c.toml").write_text('[risk]\nmax_positons = 3\n')
        with self.assertRaises(ValueError):
            config.load(d / "c.toml")
        (d / "c.toml").write_text('[risk]\nmax_positions = "three"\n')
        with self.assertRaises(ValueError):
            config.load(d / "c.toml")
        (d / "c.toml").write_text('[risk]\nmax_positions = 3\n[risk.groups.energy]\nsymbols = ["XOM", "CVX"]\nmax_pct = 15.0\n')
        cfg = config.load(d / "c.toml")
        self.assertEqual((cfg.risk.max_positions, cfg.risk.groups.energy.max_pct), (3, 15.0))
        self.assertNotEqual(cfg.strategy_hash("breakout"), config.load().strategy_hash("breakout"))    # limits changed -> evidence/activation invalid
        self.assertEqual(config.redact("key=unit-test-secret-value!", env=KEY), "key=[REDACTED]!")
        self.assertFalse(cfg.risk.set_by_user)


class Backtest(unittest.TestCase):
    def test_synthetic_run_is_never_evidence(self):
        cfg = config.load(None, {"markets": {"stock": {"watchlist": ["AAA", "BBB", "CCC"]}}})
        r = backtest.evaluate(cfg, "breakout", "synthetic")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["verdict"], "research_only")
        self.assertIn("synthetic data", r["verdict_reasons"])
        self.assertTrue(any("Survivorship" in f for f in r["flags"]))
        res = r["results"]
        self.assertLess(res["train"]["end"], res["out_of_sample"]["end"])
        self.assertEqual(res["train"]["end"], res["validation"]["start"])                 # chronological, no overlap
        self.assertLessEqual(res["out_of_sample_costs_x2"]["net_pnl"], res["out_of_sample"]["net_pnl"] + 1e-6)
        for k in ("trades", "net_pnl", "avg_net_per_trade", "win_rate", "avg_win", "avg_loss", "max_drawdown_pct", "exposure_pct", "turnover_x", "benchmark_buy_hold_pct"):
            self.assertIn(k, res["out_of_sample"])
        self.assertIn("VERDICT: RESEARCH ONLY", backtest.format_report(r))
        self.assertEqual(r["strategy_hash"], cfg.strategy_hash("breakout"))


class Dashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = str(Path(tempfile.mkdtemp()) / "replay.db")
        replay.run(cls.db, verbose=False)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(cls.db, config.load()))
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
                raw = r.read()
                return r.status, json.loads(raw) if "json" in r.headers["Content-Type"] else raw
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_state_answers_the_four_questions(self):
        st, s = self.call("/api/state")
        self.assertEqual(st, 200)
        self.assertEqual(s["mode"], "REPLAY")
        self.assertIn("SIMULATED", s["data_label"])
        self.assertTrue(all(b["mode"] == "REPLAY" and b["account"] for b in s["brokers"].values()))
        self.assertGreater(len(s["scanner"]), 10)
        traded = next(x for x in s["signals"] if x["status"] == "traded" and x["symbol"] == "USD/JPY")
        self.assertTrue(traded["why"] and traded["decision"] and traded["sizing"]["steps"] and traded["inputs"])
        self.assertIsNotNone(s["portfolio"]["equity"])
        self.assertEqual(len(s["strategies"]), 5)
        self.assertEqual({x["name"]: x["closed"] for x in s["strategies"]}["orb"], 2)
        self.assertFalse(s["integrations"]["fidelity"]["order_entry_available"])
        j = s["journal"][0]
        self.assertTrue({"expected_entry", "entry_avg", "expected_exit", "exit_avg", "fees", "strategy_version", "mode"} <= j.keys())
        self.assertEqual(self.call("/")[0], 200)
        text = json.dumps(s)
        self.assertNotIn("unit-test", text)

    def test_chart_has_bars_and_markers_and_empty_state(self):
        st, c = self.call("/api/chart?symbol=RPLA")
        self.assertEqual(st, 200)
        self.assertGreater(len(c["bars"]["t"]), 50)
        self.assertEqual({m["kind"] for m in c["marks"]}, {"signal", "entry", "exit"})
        self.assertEqual(c["timezone"], "UTC")
        self.assertTrue(self.call("/api/chart?symbol=NOPE")[1]["empty"])
        self.assertEqual(self.call("/api/chart?symbol=%3Cscript%3E")[0], 400)

    def test_controls_are_recorded_and_protected(self):
        self.assertEqual(self.call("/api/control", {"action": "kill", "on": True})[0], 200)
        self.assertTrue(self.call("/api/state")[1]["controls"]["kill"])
        self.call("/api/control", {"action": "kill", "on": False})
        self.assertEqual(self.call("/api/control", {"action": "flatten", "trade_ids": ["x"]})[0], 400)
        self.assertEqual(self.call("/api/control", {"action": "bogus"})[0], 400)
        self.assertEqual(self.call("/api/control", {"action": "kill", "on": True}, {"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.call("/api/control", {"action": "kill", "on": True}, {"Content-Type": "text/plain"})[0], 403)
        s = Store(self.db)
        self.assertFalse(s.get("kill"))
        s.close()


if __name__ == "__main__":
    unittest.main()
