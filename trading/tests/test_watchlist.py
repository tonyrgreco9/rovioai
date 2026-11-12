import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from rovio_trader import config, server
from rovio_trader.store import Store
from rovio_trader.watchlist import Watchlist, normalize
from tests.helpers import U, run, world_for


def tmp_cfg(extra=None):
    return config.load(None, {"state_dir": tempfile.mkdtemp(), **(extra or {})})


class Rules(unittest.TestCase):
    def test_symbol_formats(self):
        self.assertEqual(normalize("stock", " tsla "), "TSLA")
        self.assertEqual(normalize("stock", "brk.b"), "BRK.B")
        self.assertEqual(normalize("forex", "audusd"), "AUD/USD")
        self.assertEqual(normalize("forex", "usd-cad"), "USD/CAD")
        self.assertEqual(normalize("crypto", "eth"), "ETH/USD")
        self.assertEqual(normalize("futures", "mes"), "MES")
        for market, bad in [("stock", "TOOLONGX"), ("stock", "<script>"), ("stock", ""), ("forex", "EUR/GBP"), ("forex", "EUR/XYZ"), ("crypto", "BTC/EUR"),
                            ("futures", "ES"), ("bonds", "TLT")]:
            with self.assertRaises(ValueError, msg=(market, bad)):
                normalize(market, bad)

    def test_add_remove_persist_and_override_config(self):
        cfg = tmp_cfg()
        w = Watchlist(cfg)
        self.assertEqual(w.get()["stock"], list(cfg.markets.stock.watchlist))      # starts from the config
        w.add("stock", "tsla")
        w.add("stock", "tsla")                                                       # no duplicates
        w.add("stock", "iwm", etf=True)
        w.remove("stock", "AAPL")
        d = Watchlist(tmp_cfg({"state_dir": cfg.state_dir})).get()                   # a different process sees the same file
        self.assertEqual(d["stock"].count("TSLA"), 1)
        self.assertNotIn("AAPL", d["stock"])
        self.assertIn("IWM", d["etfs"])
        before = cfg.strategy_hash("breakout")
        self.assertTrue(w.apply())
        self.assertFalse(w.apply())
        self.assertIn("TSLA", cfg.markets.stock.watchlist)
        self.assertNotEqual(cfg.strategy_hash("breakout"), before)                   # a live activation must be re-confirmed
        self.assertEqual(cfg.strategy_hash("fx_pullback"), tmp_cfg().strategy_hash("fx_pullback"))


class WorkerPicksItUp(unittest.TestCase):
    def test_added_symbol_is_scanned_and_removed_symbol_keeps_its_exit(self):
        cfg, w, brokers, store, eng, cal = world_for(["RPLD", "RPLG"], {"state_dir": tempfile.mkdtemp()})
        self.addCleanup(store.close)
        wl = Watchlist(cfg)
        wl.save({"stock": ["RPLD"], "etfs": [], "forex": [], "crypto": [], "futures": []})
        eng.watchlist = wl
        run(w, eng, "2025-12-03 14:30", "2025-12-05 21:00", cal)                     # RPLD bought and protected
        self.assertEqual(eng.watching, {"RPLD"})
        self.assertEqual([t["symbol"] for t in store.open_trades()], ["RPLD"])
        wl.add("stock", "RPLG")
        wl.add("stock", "ZZZZ")                                                      # unknown to the broker
        wl.remove("stock", "RPLD")
        run(w, eng, "2025-12-08 14:30", "2025-12-09 14:30", cal)
        self.assertEqual(eng.watching, {"RPLG"})
        self.assertIn("ZZZZ", store.get("instrument_problems"))
        self.assertEqual(store.one("SELECT status FROM signals WHERE symbol='RPLG'")["status"], "traded")   # the new symbol was scanned and traded
        self.assertEqual(len(store.active_orders(symbol="RPLD", role="stop")), 1)    # removed symbol: still protected
        self.assertNotIn("breakout|RPLD", store.get("scanner"))
        eng.flatten()
        run(w, eng, "2025-12-09 21:00", "2025-12-10 14:30", cal)
        self.assertEqual(store.one("SELECT status, exit_reason FROM trades WHERE symbol='RPLD'"), {"status": "closed", "exit_reason": "manual flatten"})
        self.assertTrue(any("watchlist changed" in e["msg"] for e in store.q("SELECT msg FROM events WHERE kind='watchlist'")))

    def test_etf_flag_reaches_the_earnings_strategy(self):
        cfg, w, brokers, store, eng, cal = world_for(["RPLE"], {"state_dir": tempfile.mkdtemp()})
        self.addCleanup(store.close)
        wl = Watchlist(cfg)
        wl.save({"stock": ["RPLE"], "etfs": ["RPLE"], "forex": [], "crypto": [], "futures": []})
        eng.watchlist = wl
        run(w, eng, "2025-11-17 14:30", "2025-11-19 14:30", cal)
        self.assertEqual(store.q("SELECT * FROM signals WHERE strategy='earnings'"), [])     # flagged as an ETF: earnings never evaluated


class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = tmp_cfg()
        cls.db = str(cls.cfg.state_path / "paper.db")
        s = Store(cls.db)
        s.set("mode", "PAPER")
        s.close()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(cls.db, cls.cfg))
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
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_watchlist_endpoint_and_empty_journal_state(self):
        st, s = self.call("/api/state")
        self.assertEqual((st, s["mode"]), (200, "PAPER"))
        self.assertIsNone(s["portfolio"]["equity"])
        self.assertEqual(s["signals"], [])
        self.assertTrue(all(r["status"] == "pending" for r in s["watchlist"]))
        self.assertEqual(self.call("/api/watchlist", {"action": "add", "market": "stock", "symbol": "tsla"}), (200, {"ok": True, "symbol": "TSLA"}))
        self.assertEqual(self.call("/api/watchlist", {"action": "add", "market": "forex", "symbol": "EUR/GBP"})[0], 400)
        self.assertEqual(self.call("/api/watchlist", {"action": "add", "market": "stock", "symbol": "<b>"})[0], 400)
        self.assertEqual(self.call("/api/watchlist", {"action": "nuke", "market": "stock", "symbol": "X"})[0], 400)
        self.assertEqual(self.call("/api/watchlist", {"action": "add", "market": "stock", "symbol": "X"}, {"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.call("/api/watchlist", {"action": "remove", "market": "stock", "symbol": "SPY"})[0], 200)
        syms = [r["symbol"] for r in self.call("/api/state")[1]["watchlist"] if r["market"] == "stock"]
        self.assertIn("TSLA", syms)
        self.assertNotIn("SPY", syms)
        self.assertEqual(Watchlist(self.cfg).get()["stock"], syms)


if __name__ == "__main__":
    unittest.main()
