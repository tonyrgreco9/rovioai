"""Fidelity ticket mode: no order leaves the machine; fills are what the account holder records."""
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pandas as pd

from rovio_trader import config, fixtures as F, server
from rovio_trader.__main__ import manual_cfg
from rovio_trader.brokers.manual import ManualBroker, instruction, ticket_action
from rovio_trader.brokers.yahoo import YahooFeed, parse, yahoo_symbol
from rovio_trader.data import BrokerData, frame, session_times
from rovio_trader.engine import Engine
from rovio_trader.store import Store

U = lambda s: pd.Timestamp(s, tz="UTC")
EP = lambda t: int(pd.Timestamp(t).timestamp())


class FakeFeed:
    """Serves the fictional RPLA fixture the way the real feed would look at a given moment."""
    verified = True

    def __init__(self, daily, clock):
        self.daily, self.clock, self.calls, self.fail = daily, clock, 0, False

    def chart(self, symbol, tf):
        self.calls += 1
        if self.fail:
            from rovio_trader.brokers.base import BrokerTimeout
            raise BrokerTimeout("no response from price feed")
        d = self.daily.get(symbol)
        if d is None:
            return None
        now = self.clock()
        seen = d[d["start"] <= now]                       # includes a session in progress
        last = seen.iloc[-1]
        live = last.name > now
        price = float(last["open"] if live else last["close"])
        meta = {"instrumentType": "EQUITY", "currency": "USD", "exchangeTimezoneName": "America/New_York", "regularMarketPrice": price,
                "regularMarketTime": EP(now if live else last.name), "currentTradingPeriod": {"regular": {"start": EP(last["start"]), "end": EP(last.name)}}}
        if tf == "1d":
            return {"meta": meta, "bars": seen}
        t = now.floor("5min")
        return {"meta": meta, "bars": frame([t], [t + pd.Timedelta(minutes=5)], [price], [price], [price], [price], [1000.0])}


def setup(cash=100_000.0):
    cfg = manual_cfg(config.load(None, {"state_dir": tempfile.mkdtemp(), "markets": {"stock": {"watchlist": ["RPLA"], "etfs": []}}}))
    _, _, data, _, _ = F.build(config.load())
    clock = {"t": U("2025-11-10 21:30")}
    store = Store()
    store.set("manual_cash", cash)
    b = ManualBroker(store, FakeFeed({"RPLA": data.frames[("RPLA", "1d")]}, lambda: clock["t"]), clock=lambda: clock["t"])
    eng = Engine(cfg, store, {"fidelity": b}, BrokerData({"fidelity": b}, ttl=0), "MANUAL", only=["breakout"], live_view=True)

    def step(t):
        clock["t"] = U(t)
        eng.step(clock["t"])
    return cfg, store, b, eng, step


def tickets(store, **w):
    sql = "SELECT * FROM tickets WHERE 1=1" + "".join(f" AND {k}=?" for k in w)
    return store.q(sql + " ORDER BY created, cid", list(w.values()))


class TicketLifecycle(unittest.TestCase):
    def test_signal_to_entry_stop_and_exit_all_by_hand(self):
        cfg, store, b, eng, step = setup()
        step("2025-11-10 21:30")                                   # after the close: the breakout bar is complete
        sig = store.one("SELECT * FROM signals")
        self.assertEqual((sig["symbol"], sig["status"]), ("RPLA", "pending"))
        self.assertEqual(tickets(store), [])                       # market closed: no ticket yet

        step("2025-11-11 14:40")                                   # ten minutes after the open
        (t,) = tickets(store)
        self.assertEqual((t["role"], t["side"], t["type"], t["status"], t["placed"]), ("entry", "buy", "market", "new", 0))
        self.assertIn("BUY", instruction(t))
        self.assertIsNone(store.one("SELECT * FROM trades"))       # a ticket is not a position
        acct0 = b.account()
        self.assertEqual(acct0["cash"], 100_000.0)

        with self.assertRaises(ValueError):
            ticket_action(store, t["cid"], "fill", t["qty"] + 1, 104.0)          # cannot record more than the ticket
        with self.assertRaises(ValueError):
            ticket_action(store, t["cid"], "fill", "abc", 104.0)
        ticket_action(store, t["cid"], "placed")
        ticket_action(store, t["cid"], "fill", t["qty"], 104.25)
        step("2025-11-11 14:41")
        tr = store.one("SELECT * FROM trades")
        self.assertEqual((tr["status"], tr["qty"], tr["entry_avg"], tr["mode"]), ("open", t["qty"], 104.25, "MANUAL"))
        self.assertTrue(any("fill recorded by you" in e["msg"] for e in store.q("SELECT msg FROM events WHERE kind='fill'")))
        (stop,) = tickets(store, role="stop")
        self.assertEqual((stop["side"], stop["type"], stop["qty"], stop["stop_price"], stop["placed"]), ("sell", "stop", t["qty"], tr["stop"], 0))
        self.assertIn("NO protective stop", instruction(stop))
        self.assertAlmostEqual(b.account()["cash"], 100_000.0 - t["qty"] * 104.25)
        self.assertEqual(store.get("external_positions"), {})      # recorded fills are never mistaken for a foreign holding
        self.assertEqual(store.get("paused_instruments"), {})

        state = server.build_state(store, cfg)
        self.assertTrue(state["positions"][0]["stop_unplaced"])
        self.assertEqual([x["needs_action"] for x in state["tickets"] if x["active"]], [True])
        ticket_action(store, stop["cid"], "placed")
        self.assertFalse(server.build_state(store, cfg)["positions"][0]["stop_unplaced"])

        eng.flatten()
        step("2025-11-12 15:00")
        stop = tickets(store, role="stop")[0]
        self.assertEqual((stop["status"], stop["cancel_requested"]), ("new", 1))   # resting at Fidelity: only the owner can cancel it
        self.assertEqual(tickets(store, role="exit"), [])                           # never ask to sell shares a stop still holds
        self.assertIn("CANCEL", instruction(stop))
        ticket_action(store, stop["cid"], "cancel")
        step("2025-11-12 15:01")
        (ex,) = tickets(store, role="exit")
        self.assertEqual((ex["side"], ex["type"], ex["qty"]), ("sell", "market", t["qty"]))
        ticket_action(store, ex["cid"], "fill", t["qty"] / 2 if t["qty"] % 2 == 0 else t["qty"], 106.0)
        if t["qty"] % 2 == 0:
            ticket_action(store, ex["cid"], "fill", t["qty"] / 2, 107.0)             # two partial fills average
        step("2025-11-12 15:02")
        tr = store.one("SELECT * FROM trades")
        self.assertEqual((tr["status"], tr["exit_reason"]), ("closed", "manual flatten"))
        px = 106.5 if t["qty"] % 2 == 0 else 106.0
        self.assertAlmostEqual(tr["realized"], (px - 104.25) * t["qty"])
        self.assertAlmostEqual(b.account()["equity"], 100_000.0 + tr["realized"])
        self.assertEqual(b.positions(), {})
        self.assertEqual(tickets(store, status="new"), [])

    def test_unacted_entry_ticket_is_withdrawn_when_the_signal_lapses(self):
        cfg, store, b, eng, step = setup()
        step("2025-11-10 21:30")
        step("2025-11-11 14:40")
        (t,) = tickets(store)
        step("2025-11-17 15:00")                                   # the signal is valid for five days
        (t,) = tickets(store)
        self.assertEqual((t["status"], t["note"]), ("canceled", "withdrawn before you placed it"))
        self.assertEqual(store.one("SELECT status FROM signals")["status"], "skipped")
        self.assertIsNone(store.one("SELECT * FROM trades"))

    def test_declined_entry_and_cancelled_stop(self):
        cfg, store, b, eng, step = setup()
        step("2025-11-10 21:30")
        step("2025-11-11 14:40")
        (t,) = tickets(store)
        ticket_action(store, t["cid"], "fill", 10, 104.0)          # partial, then the owner stops
        ticket_action(store, t["cid"], "cancel")
        step("2025-11-11 14:41")
        tr = store.one("SELECT * FROM trades")
        self.assertEqual((tr["qty"], tr["status"]), (10, "open"))
        (stop,) = tickets(store, role="stop")
        self.assertEqual(stop["qty"], 10)                           # protection matches what actually filled
        ticket_action(store, stop["cid"], "cancel")                 # owner refuses the stop: the bot asks again
        step("2025-11-11 14:42")
        live = [x for x in tickets(store, role="stop") if x["status"] == "new"]
        self.assertEqual(len(live), 1)
        self.assertNotEqual(live[0]["cid"], stop["cid"])
        with self.assertRaises(ValueError):
            ticket_action(store, stop["cid"], "placed")             # finished tickets cannot be changed

    def test_feed_outage_blocks_entries_and_live_view_records_quotes(self):
        cfg, store, b, eng, step = setup()
        step("2025-11-10 21:30")
        q = store.get("quotes")["RPLA"]
        self.assertEqual((q["bid"], q["ask"]), (None, None))        # last trade only: never presented as a bid/ask
        self.assertGreater(q["price"], 0)
        self.assertEqual({r["tf"] for r in store.q("SELECT tf FROM barcache WHERE symbol='RPLA'")}, {"5m", "1d"})
        b.feed.fail = True
        step("2025-11-11 14:40")
        self.assertEqual(tickets(store), [])                        # no price, no ticket
        b.feed.fail = False
        step("2025-11-11 14:45")
        self.assertEqual(len(tickets(store)), 1)

    def test_account_needs_a_cash_figure_and_non_stocks_are_refused(self):
        from rovio_trader.brokers.base import BrokerError
        b = ManualBroker(Store(), FakeFeed({}, lambda: U("2025-11-11 15:00")))
        with self.assertRaises(BrokerError):
            b.account()
        self.assertIsNone(b.instrument("NOPE"))
        cfg = manual_cfg(config.load())
        self.assertEqual([m for m in cfg.markets if cfg.markets[m].enabled], ["stock"])
        self.assertEqual(cfg.markets.stock.broker, "fidelity")


class Feed(unittest.TestCase):
    PAYLOAD = {"chart": {"error": None, "result": [{
        "meta": {"currency": "USD", "symbol": "XYZ", "instrumentType": "EQUITY", "regularMarketPrice": 12.5, "regularMarketTime": 1762873200,
                 "exchangeTimezoneName": "America/New_York", "currentTradingPeriod": {"regular": {"start": 1762871400, "end": 1762894800}}},
        "timestamp": [1762525800, 1762785000, 1762873200],        # Fri 7 Nov 09:30 ET, Mon 10 Nov 09:30 ET, Tue 11 Nov 10:00 ET (in progress)
        "indicators": {"quote": [{"open": [10.0, 11.0, 12.0], "high": [10.5, None, 12.6], "low": [9.5, 10.8, 11.9], "close": [10.2, 11.5, 12.5],
                                  "volume": [1000, 2000, None]}]}}]}}

    def test_daily_rows_become_sessions_and_gaps_are_dropped(self):
        out = parse(self.PAYLOAD, "1d")
        d = out["bars"]
        self.assertEqual(list(d["close"]), [10.2, 12.5])            # the row with a missing high is dropped, not invented
        op, cl = session_times([pd.Timestamp("2025-11-11")])
        self.assertEqual((d["start"].iloc[-1], d.index[-1]), (op[0], cl[0]))       # in-progress row keyed to its session, ends at the close
        self.assertEqual(d["volume"].iloc[-1], 0.0)
        self.assertIsNone(parse({"chart": {"error": {"code": "Not Found"}, "result": None}}, "1d"))
        self.assertEqual(yahoo_symbol("BRK.B"), "BRK-B")

    def test_requests_carry_only_the_symbol_and_errors_are_never_trading_decisions(self):
        from rovio_trader.brokers.base import BrokerError, BrokerTimeout
        seen = []

        class H:
            def call(self, method, url, params=None, ok_404=False, body=None):
                seen.append((method, url, params))
                if "BAD" in url:
                    raise BrokerError("HTTP 429: slow down")
                return Feed.PAYLOAD
        f = YahooFeed(http=H())
        self.assertFalse(f.verified)
        self.assertEqual(len(f.chart("BRK.B", "5m")["bars"]), 2)
        self.assertTrue(f.verified)
        f.chart("BRK.B", "5m")
        self.assertEqual(seen, [("GET", "https://query1.finance.yahoo.com/v8/finance/chart/BRK-B", {"interval": "5m", "range": "5d"})])   # second call cached
        with self.assertRaises(BrokerTimeout):
            f.chart("BAD", "1d")


class Api(unittest.TestCase):
    def test_ticket_and_quote_endpoints(self):
        cfg, store0, b, eng, step = setup()
        path = tempfile.mkdtemp() + "/manual.db"
        store = Store(path)
        store.set("manual_cash", 50_000.0)
        b.store = eng.store = store
        store.set("mode", "MANUAL")
        step("2025-11-10 21:30")
        step("2025-11-11 14:40")
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(path, cfg))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def call(p, body=None):
            req = urllib.request.Request(base + p, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status, json.load(r)
            except urllib.error.HTTPError as e:
                return e.code, json.load(e)
        try:
            st = call("/api/state")[1]
            (t,) = [x for x in st["tickets"] if x["active"]]
            self.assertEqual((st["mode"], st["manual_cash"], t["role"]), ("MANUAL", 50_000.0, "entry"))
            self.assertIn("RPLA", call("/api/quotes")[1]["quotes"])
            self.assertEqual(call("/api/ticket", {"cid": t["cid"], "action": "fill", "qty": 0, "price": 5})[0], 400)
            self.assertEqual(call("/api/ticket", {"cid": "nope", "action": "placed"})[0], 400)
            self.assertEqual(call("/api/ticket", {"cid": t["cid"], "action": "fill", "qty": t["qty"], "price": 104.0}), (200, {"ok": True, "status": "filled", "note": "Recorded. The worker picks it up on its next cycle."}))
            step("2025-11-11 14:41")
            st = call("/api/state")[1]
            self.assertEqual(len(st["positions"]), 1)
            ch = call("/api/chart?symbol=RPLA&tf=5m")[1]
            self.assertEqual((ch["tf"], ch["tfs"]), ("5m", ["5m", "1d"]))
            self.assertEqual(call("/api/chart?symbol=RPLA")[1]["tf"], "1d")
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
