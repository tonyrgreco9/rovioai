import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rovio_trader import config, instruments as I, replay, risk as R
from rovio_trader.brokers.base import BrokerError
from rovio_trader.engine import Engine
from rovio_trader.store import ACTIVE, Store
from tests.helpers import U, run, world_for

CFG = config.load()


class Sizing(unittest.TestCase):
    def size(self, inst, entry, stop, eq=200_000.0, cfg=CFG, view=None, acct=None, side=1, rates=None):
        view = view or R.portfolio_view(cfg, eq, [], [], {}, {}, {})
        acct = acct or {"equity": eq, "cash": eq, "margin_available": eq}
        return R.size_position(cfg, inst, side, entry, stop, acct, view, rates or {"USD/JPY": 150.0})

    def test_stock_by_stop_distance_and_costs(self):
        z = self.size(I.stock("X", "alpaca"), 200.0, 194.0)
        # $1,000 budget / ($6 stop + $0.30 est. round-trip costs) = 158.7 -> 158 shares
        self.assertEqual((z["qty"], z["binding"]), (158, "planned loss per trade"))
        self.assertLessEqual(z["risk_usd"], 1000.0)

    def test_mes_uses_ticks_and_tick_value(self):
        mes = I.future("MES", "2026-03-20", "ibkr")
        z = self.size(mes, 6850.0, 6841.0)
        self.assertAlmostEqual(z["risk_per_unit_usd"], 36 * 1.25 + R.est_cost_per_unit(CFG, mes, 6850.0, 1.0))
        self.assertEqual(z["qty"], 1)                              # notional 34,250 per contract vs a 25% instrument limit of 50,000
        self.assertEqual(z["margin_usd"], 2500.0)                  # margin is tracked separately from notional
        self.assertAlmostEqual(z["notional_usd"], 34250.0)
        self.assertIn("36 ticks x $1.25", z["steps"][0]["detail"])

    def test_usdjpy_converted_to_usd(self):
        z = self.size(I.forex("USD/JPY", "oanda"), 150.0, 149.7)
        self.assertAlmostEqual(z["fx"], 1 / 150)
        self.assertAlmostEqual(z["risk_per_unit_usd"], 0.3 / 150 + R.est_cost_per_unit(CFG, I.forex("USD/JPY", "oanda"), 150.0, 1 / 150))
        self.assertEqual(z["qty"], 50_000)                         # 25% of 200k in USD notional
        self.assertAlmostEqual(z["margin_usd"], 50_000 * 0.05)

    def test_bitcoin_fractional_with_fees_both_ways(self):
        btc = I.crypto("BTC/USD", "alpaca")
        z = self.size(btc, 90_000.0, 87_750.0)
        cost = R.est_cost_per_unit(CFG, btc, 90_000.0, 1.0)
        self.assertAlmostEqual(cost, 2 * (90_000 * 10 / 1e4 + 90_000 * 25 / 1e4))    # spread+slippage and the 25 bps fee, entry and exit
        self.assertEqual(z["qty"], btc.round_qty(1000 / (2250 + cost)))
        self.assertNotEqual(z["qty"] % 1, 0)

    def test_smallest_legal_position_too_large_is_skipped_with_reason(self):
        tight = config.load(None, {"risk": {"max_loss_per_trade_usd": 100.0}})
        z = self.size(I.future("MES", "2026-03-20", "ibkr"), 6850.0, 6790.0, cfg=tight)
        self.assertEqual(z["qty"], 0)
        self.assertIn("smallest legal position (1 contract) risks $", z["reason"])

    def test_correlated_group_counts_spy_against_mes(self):
        spy, mes = I.stock("SPY", "alpaca"), I.future("MES", "2026-03-20", "ibkr")
        held = [{"symbol": "SPY", "qty": 80, "entry_avg": 600.0, "stop": 590.0, "side": 1, "status": "open"}]      # $48,000 of S&P exposure
        view = R.portfolio_view(CFG, 200_000.0, held, [], {"SPY": spy, "MES": mes}, {"SPY": 600.0}, {})
        z = self.size(mes, 6850.0, 6841.0, view=view)
        self.assertEqual(z["qty"], 0)                              # 30% group limit = 60,000; only 12,000 left, one MES is 34,250
        self.assertIn("correlated group 'sp500'", z["reason"])

    def test_cash_at_one_broker_cannot_fund_another(self):
        z = self.size(I.stock("X", "alpaca"), 200.0, 194.0, acct={"equity": 5_000.0, "cash": 5_000.0, "margin_available": 5_000.0})
        self.assertEqual(z["binding"], "cash at alpaca")           # portfolio equity is 200k, but only 5k sits at this broker
        self.assertEqual(z["qty"], 24)

    def test_pending_orders_reserve_risk_and_exposure(self):
        x = I.stock("X", "alpaca")
        pend = [{"symbol": "X", "qty": 100, "filled_qty": 0, "reserved_risk": 5900.0, "reserved_notional": 20_000.0}]
        view = R.portfolio_view(CFG, 200_000.0, [], pend, {"X": x}, {}, {})
        self.assertEqual((view["open_risk"], view["gross"], view["positions"]), (5900.0, 20_000.0, 1))
        z = self.size(I.stock("Y", "alpaca"), 200.0, 194.0, view=view)
        self.assertEqual(z["binding"], "total open portfolio risk")   # 3% = 6,000 total; 5,900 already reserved
        self.assertEqual(z["qty"], 15)

    def test_currency_exposure_across_pairs(self):
        e, g = I.forex("EUR/USD", "oanda"), I.forex("EUR/GBP", "oanda")
        t = [{"symbol": "EUR/USD", "qty": 100_000, "entry_avg": 1.1, "stop": 1.09, "side": 1, "status": "open"}]
        v = R.portfolio_view(CFG, 200_000.0, t, [], {"EUR/USD": e, "EUR/GBP": g}, {"EUR/USD": 1.1}, {"EUR/USD": 1.1})
        self.assertAlmostEqual(v["currency"]["EUR"], 110_000.0)
        self.assertAlmostEqual(v["currency"]["USD"], -110_000.0)


class Simulator(unittest.TestCase):
    def setUp(self):
        self.cfg, self.w, self.b, self.store, self.eng, self.cal = world_for(["RPLC", "RPLD", "USD/JPY", "MESH6", "BTC/USD"])
        self.addCleanup(self.store.close)

    def test_gap_through_stop_fills_at_the_open_not_the_stop(self):
        alp = self.b["alpaca"]
        self.w.advance(U("2025-12-08 14:30"))
        alp.submit({"cid": "e1", "symbol": "RPLC", "side": "buy", "type": "market", "qty": 100})
        alp.submit({"cid": "s1", "symbol": "RPLC", "side": "sell", "type": "stop", "qty": 100, "stop_price": 101.0, "reduce_only": True})
        for t in self.w.timeline(U("2025-12-08 21:00"), U("2025-12-09 21:00")):
            self.w.advance(t)
        s = alp.get_by_cid("s1")
        self.assertEqual(s["status"], "filled")
        self.assertLess(s["avg_price"], 92.0)                      # the bar opened ~12% lower: nowhere near 101
        self.assertEqual(alp.positions(), {})

    def test_reduce_only_duplicate_and_short_rules(self):
        alp = self.b["alpaca"]
        self.w.advance(U("2025-12-08 14:30"))
        alp.submit({"cid": "e1", "symbol": "RPLD", "side": "buy", "type": "market", "qty": 10})
        with self.assertRaises(BrokerError):
            alp.submit({"cid": "e1", "symbol": "RPLD", "side": "buy", "type": "market", "qty": 10})                       # duplicate client id
        with self.assertRaises(BrokerError):
            alp.submit({"cid": "x1", "symbol": "RPLD", "side": "sell", "type": "market", "qty": 11, "reduce_only": True})   # would reverse
        with self.assertRaises(BrokerError):
            alp.submit({"cid": "x2", "symbol": "RPLD", "side": "sell", "type": "market", "qty": 11})                       # stock short not allowed
        alp.submit({"cid": "x3", "symbol": "RPLD", "side": "sell", "type": "market", "qty": 10, "reduce_only": True})
        with self.assertRaises(BrokerError):
            alp.submit({"cid": "x4", "symbol": "RPLD", "side": "sell", "type": "market", "qty": 1, "reduce_only": True})   # nothing left to reduce
        with self.assertRaises(BrokerError):
            alp.submit({"cid": "x5", "symbol": "BTC/USD", "side": "buy", "type": "market", "qty": 0.00001})                # below the minimum

    def test_short_futures_and_jpy_pnl_in_usd(self):
        ib, oa = self.b["ibkr"], self.b["oanda"]
        self.w.advance(U("2025-12-12 15:10"))
        ib.submit({"cid": "f1", "symbol": "MESH6", "side": "sell", "type": "market", "qty": 2})
        px = ib.get_by_cid("f1")["avg_price"]
        self.assertEqual(ib.positions()["MESH6"]["qty"], -2)
        self.w.advance(U("2025-12-12 15:15"))
        m = self.w.mark("MESH6")
        self.assertAlmostEqual(ib.account()["equity"], 50_000 - 2 * 0.62 + 2 * 5 * (px - m), places=6)
        self.assertAlmostEqual(ib.account()["margin_used"], 5000.0)
        self.w.advance(U("2025-12-02 14:00"))
        oa.submit({"cid": "j1", "symbol": "USD/JPY", "side": "buy", "type": "market", "qty": 10_000})
        e = oa.get_by_cid("j1")["avg_price"]
        self.w.advance(U("2025-12-02 20:00"))
        oa.submit({"cid": "j2", "symbol": "USD/JPY", "side": "sell", "type": "market", "qty": 10_000, "reduce_only": True})
        x = oa.get_by_cid("j2")["avg_price"]
        self.assertAlmostEqual(oa.cash, 50_000 + 10_000 * (x - e) / self.w.mark("USD/JPY"), places=4)

    def test_market_order_waits_for_the_open(self):
        alp = self.b["alpaca"]
        self.w.advance(U("2025-12-06 12:00"))                      # Saturday
        o = alp.submit({"cid": "w1", "symbol": "RPLD", "side": "buy", "type": "market", "qty": 5})
        self.assertEqual(o["status"], "new")
        self.assertFalse(self.w.is_open("RPLD"))
        self.assertTrue(self.w.is_open("BTC/USD"))
        self.w.advance(U("2025-12-08 14:30"))
        self.assertEqual(alp.get_by_cid("w1")["status"], "filled")


class EngineBehaviour(unittest.TestCase):
    def start(self, symbols=("RPLD",), overrides=None, mode="REPLAY", store=None):
        self.cfg, self.w, self.b, self.store, self.eng, self.cal = world_for(list(symbols), overrides, mode, store)
        self.addCleanup(self.store.close)
        return self

    def go(self, a, b, each=None):
        run(self.w, self.eng, a, b, self.cal, each)

    def sig(self, like):
        return self.store.one("SELECT * FROM signals WHERE id LIKE ?", (like + "%",))

    def test_intent_is_saved_before_the_broker_is_contacted(self):
        self.start()
        seen = []
        orig = self.b["alpaca"].submit
        self.b["alpaca"].submit = lambda o: (seen.append(self.store.one("SELECT status FROM orders WHERE cid=?", (o["cid"],))["status"]), orig(o))[1]
        self.go("2025-12-03 14:30", "2025-12-04 14:30")
        self.assertEqual(seen[0], "created")

    def test_fill_then_protection_matches_filled_quantity(self):
        self.start()
        self.go("2025-12-03 14:30", "2025-12-04 14:30")
        t = self.store.one("SELECT * FROM trades")
        stop = self.store.active_orders(role="stop")
        self.assertEqual(len(stop), 1)
        self.assertEqual((stop[0]["qty"], stop[0]["side"], stop[0]["stop_price"]), (t["qty"], "sell", t["stop"]))
        self.assertEqual(self.b["alpaca"].get_by_cid(stop[0]["cid"])["status"], "new")
        self.assertEqual(self.store.get("unprotected"), [])

    def test_kill_switch_cancels_entries_and_keeps_stops(self):
        self.start(["RPLD", "RPLG"])
        self.go("2025-12-03 14:30", "2025-12-08 21:00")            # RPLD held; RPLG signal pending for the next open
        self.assertEqual(self.sig("breakout:RPLG")["status"], "pending")
        self.eng.kill(True)
        self.go("2025-12-09 14:30", "2025-12-09 14:30")
        self.assertEqual(self.sig("breakout:RPLG")["status"], "skipped")
        self.assertIn("Kill switch", self.sig("breakout:RPLG")["decision"])
        stop = self.store.active_orders(role="stop")
        self.assertEqual(len(stop), 1)
        self.assertEqual(self.b["alpaca"].get_by_cid(stop[0]["cid"])["status"], "new")

    def test_kill_switch_cancels_a_working_entry_order(self):
        self.start()
        self.b["alpaca"].faults.append({"symbol": "RPLD", "role": "entry", "kind": "partial", "fraction": 0.4})
        self.go("2025-12-03 14:30", "2025-12-04 14:30")
        e = self.store.one("SELECT * FROM orders WHERE role='entry'")
        self.assertEqual(e["status"], "partially_filled")
        self.eng.kill(True)
        self.eng.step(self.eng.now)
        e = self.store.one("SELECT * FROM orders WHERE role='entry'")
        self.assertEqual(e["status"], "canceled")
        t = self.store.one("SELECT * FROM trades")
        self.assertEqual(sum(s["qty"] for s in self.store.active_orders(role="stop")), t["qty"])    # protection for the filled part stays

    def test_pause_entries_still_manages_exits(self):
        self.start(["RPLA"], {"risk": {"max_instrument_pct": 20.0}})
        self.go("2025-11-10 14:30", "2025-11-12 21:00")
        self.eng.pause_entries(True)
        self.go("2025-11-13 14:30", "2025-12-01 14:30")
        t = self.store.one("SELECT * FROM trades")
        self.assertEqual(t["status"], "closed")
        self.assertIn("exit rule", t["exit_reason"])

    def test_stale_quote_waits_then_expires(self):
        self.start(["BTC/USD"])
        self.w.stale.add("BTC/USD")
        self.go("2025-12-05 20:00", "2025-12-06 00:00")
        s = self.sig("btc_breakout:BTC/USD:2025-12-06T00:00")
        self.assertEqual(s["status"], "pending")
        self.assertIn("stale", s["decision"])
        self.go("2025-12-06 04:00", "2025-12-06 04:00")
        s = self.sig("btc_breakout:BTC/USD:2025-12-06T00:00")
        self.assertEqual(s["status"], "expired")
        self.assertEqual(self.store.q("SELECT * FROM orders"), [])

    def test_suspended_venue_blocks_entries(self):
        self.start(["BTC/USD"])
        self.w.suspended.add("BTC/USD")
        self.go("2025-12-05 20:00", "2025-12-06 08:00")
        self.assertEqual(self.store.q("SELECT * FROM orders"), [])
        self.assertTrue(all(s["status"] in ("expired", "pending") for s in self.store.q("SELECT status FROM signals")))

    def test_protection_failure_halts_entries_and_flattens(self):
        self.start()
        self.b["alpaca"].faults.append({"symbol": "RPLD", "role": "stop", "kind": "reject"})
        self.go("2025-12-03 14:30", "2025-12-04 14:30")
        self.assertIn("could not be established", self.store.get("halt_protection"))
        kinds = [e["kind"] for e in self.store.q("SELECT kind FROM events WHERE level='ALERT'")]
        self.assertIn("protection_failed", kinds)
        self.go("2025-12-04 21:00", "2025-12-05 14:30")
        t = self.store.one("SELECT * FROM trades")
        self.assertEqual((t["status"], t["exit_reason"]), ("closed", "emergency exit: protective stop could not be established"))
        self.assertEqual(self.b["alpaca"].positions(), {})
        self.eng.resume()
        self.assertIsNone(self.store.get("halt_protection"))

    def test_hold_and_alert_policy_keeps_retrying_the_stop(self):
        self.start(overrides={"risk": {"emergency_exit_policy": "hold_and_alert"}})
        self.b["alpaca"].faults.append({"symbol": "RPLD", "role": "stop", "kind": "reject"})
        self.go("2025-12-03 14:30", "2025-12-04 21:00")
        t = self.store.one("SELECT * FROM trades")
        self.assertEqual(t["status"], "open")
        self.assertEqual(len(self.store.active_orders(role="stop")), 1)           # the retry succeeded
        self.assertTrue(self.store.get("halt_protection"))                         # still halted until the operator resumes

    def test_timeout_before_acceptance_is_resent_exactly_once(self):
        self.start()
        self.b["alpaca"].faults.append({"symbol": "RPLD", "role": "entry", "kind": "timeout_before_accept"})
        self.go("2025-12-03 14:30", "2025-12-04 14:30")
        e = self.store.one("SELECT * FROM orders WHERE role='entry'")
        self.assertEqual(e["status"], "uncertain")
        self.assertEqual(len(self.b["alpaca"].orders), 0)
        self.eng.step(self.eng.now)                                # reconcile: the broker has no such client id -> safe to send
        e = self.store.one("SELECT * FROM orders WHERE role='entry'")
        self.assertEqual((e["status"], e["attempts"]), ("filled", 2))
        self.assertEqual(sum(1 for o in self.b["alpaca"].orders.values() if o["side"] == "buy"), 1)

    def test_broker_rejection_is_alerted_and_never_rerouted(self):
        self.start()
        self.b["alpaca"].faults.append({"symbol": "RPLD", "role": "entry", "kind": "reject"})
        self.go("2025-12-03 14:30", "2025-12-05 21:00")
        s = self.sig("breakout:RPLD")
        self.assertEqual(s["status"], "skipped")
        self.assertIn("Broker rejected", s["decision"])
        self.assertEqual(len(self.store.q("SELECT * FROM orders")), 1)
        self.assertTrue(all(not b.orders for b in self.b.values()))
        self.assertIn("order_rejected", [e["kind"] for e in self.store.q("SELECT kind FROM events WHERE level='ALERT'")])

    def test_broker_outage_blocks_entries_until_reconciled(self):
        self.start()
        self.go("2025-12-03 14:30", "2025-12-03 21:00")
        alp = self.b["alpaca"]
        orig = alp.positions
        alp.positions = lambda: (_ for _ in ()).throw(BrokerError("down"))
        self.go("2025-12-04 14:30", "2025-12-04 14:30")
        self.assertEqual(self.store.q("SELECT * FROM orders"), [])
        self.assertFalse(self.store.get("reconcile")["alpaca"]["ok"])
        self.assertIn("broker_disconnected", [e["kind"] for e in self.store.q("SELECT kind FROM events WHERE level='ALERT'")])
        alp.positions = orig
        self.eng.step(self.eng.now)
        self.assertEqual(len(self.store.q("SELECT * FROM orders WHERE role='entry' AND status='filled'")), 1)

    def test_exit_is_never_duplicated(self):
        self.start()
        self.go("2025-12-03 14:30", "2025-12-05 21:00")
        self.eng.flatten()
        self.eng.step(self.eng.now)                                # market closed: queued
        self.eng.flatten()
        self.go("2025-12-08 14:30", "2025-12-08 14:30")
        self.eng.step(self.eng.now)
        self.eng.step(self.eng.now)
        exits = self.store.q("SELECT * FROM orders WHERE role='exit'")
        self.assertEqual(len(exits), 1)
        self.assertEqual(self.b["alpaca"].positions(), {})
        self.assertEqual(self.store.active_orders(), [])

    def test_two_simultaneous_signals_cannot_spend_the_same_risk_budget(self):
        # open-risk budget of 0.5% = one trade's worth: the second simultaneous signal must be cut to what is left
        self.start(["RPLG", "RPLH"], {"risk": {"max_open_risk_pct": 0.5, "max_loss_per_trade_pct": 0.5, "max_instrument_pct": 50.0, "max_order_notional_usd": 200000.0}})
        self.go("2025-12-08 14:30", "2025-12-09 14:30")
        risk = sum(t["planned_risk"] for t in self.store.q("SELECT * FROM trades"))
        eq = self.store.get("snapshot")["equity"]
        self.assertLessEqual(risk, eq * 0.005 * 1.05)
        st = sorted(s["status"] for s in self.store.q("SELECT status FROM signals"))
        self.assertIn("skipped", st)
        self.assertIn("traded", st)
        why = [s["decision"] for s in self.store.q("SELECT decision FROM signals WHERE status='skipped'")]
        self.assertTrue(any("total open portfolio risk" in w for w in why), why)

    def test_drawdown_halt_is_sticky_until_resume(self):
        self.start(["RPLC", "RPLG"], {"risk": {"max_drawdown_pct": 1.0, "daily_loss_limit_pct": 50.0}})
        self.go("2025-12-02 14:30", "2025-12-09 14:30")
        self.assertIn("drawdown limit", self.store.get("halt_drawdown"))
        self.assertIn("drawdown limit", self.sig("breakout:RPLG")["decision"])
        self.go("2025-12-09 21:00", "2025-12-11 21:00")           # days later: still halted
        self.assertTrue(self.store.get("halt_drawdown"))
        self.eng.resume()
        self.assertIsNone(self.store.get("halt_drawdown"))

    def test_daily_reset_time_and_persistence(self):
        self.start(["BTC/USD"])
        self.go("2025-12-06 20:00", "2025-12-07 00:00")           # a weekend
        day = self.store.get("day")
        self.assertEqual(day["period"], "2025-12-06T22:00:00Z")   # 17:00 New York on Saturday
        self.assertEqual(day["next_reset"], "2025-12-07T22:00:00Z")

    def test_live_mode_requires_activation_per_broker_market_strategy(self):
        self.start(mode="LIVE")
        self.go("2025-12-03 14:30", "2025-12-04 14:30")
        s = self.sig("breakout:RPLD")
        self.assertEqual(s["status"], "skipped")
        self.assertIn("not activated", s["decision"])
        self.assertEqual(self.store.q("SELECT * FROM orders"), [])
        # activation for a different market does not unlock stocks; a settings change invalidates an activation
        self.start(mode="LIVE")
        alias = self.b["alpaca"].account_alias
        self.store.db.execute("INSERT INTO activations VALUES('alpaca',?,'stock','breakout','stale-hash','{}','t')", (alias,))
        self.go("2025-12-03 14:30", "2025-12-04 14:30")
        self.assertIn("re-activate", self.sig("breakout:RPLD")["decision"])
        self.start(mode="LIVE")
        self.store.db.execute("INSERT INTO activations VALUES('alpaca',?,'stock','breakout',?,'{}','t')", (alias, self.cfg.strategy_hash("breakout")))
        self.go("2025-12-03 14:30", "2025-12-04 14:30")
        self.assertEqual(self.sig("breakout:RPLD")["status"], "traded")

    def test_entry_window_is_the_next_session_only(self):
        self.start()
        self.go("2025-12-03 14:30", "2025-12-03 21:00")
        self.eng.pause_entries(False)
        self.w.suspended.add("RPLD")                               # no tradable quote on the next session
        self.go("2025-12-04 14:30", "2025-12-04 21:00")
        self.w.suspended.clear()
        self.go("2025-12-05 14:30", "2025-12-05 14:30")
        s = self.sig("breakout:RPLD")
        self.assertIn(s["status"], ("skipped", "expired"))
        self.assertEqual(self.store.q("SELECT * FROM orders"), [])

    def test_alert_persists_when_external_notification_fails(self):
        self.start()
        self.eng.notify = lambda kind, msg: (_ for _ in ()).throw(OSError("webhook down"))
        self.eng.now = U("2025-12-03 14:30")
        self.eng.alert("test_alert", "something happened")
        self.assertEqual(len(self.store.q("SELECT * FROM events WHERE kind='test_alert' AND level='ALERT'")), 1)
        self.assertEqual(len(self.store.q("SELECT * FROM events WHERE kind='notify_failed'")), 1)


class ReplayAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.res = replay.run(Path(cls.dir) / "replay.db", verbose=False)

    def test_every_acceptance_check_passes(self):
        failed = [c for c in self.res["checks"] if not c["passed"]]
        self.assertEqual(failed, [])
        self.assertGreaterEqual(len(self.res["checks"]), 23)

    def test_journal_is_internally_consistent(self):
        s = Store(self.res["db"])
        for t in s.q("SELECT * FROM trades"):
            self.assertEqual(t["status"], "closed")
            self.assertAlmostEqual(t["exit_qty"], t["qty_total"], places=8)       # every unit entered was exited exactly once
            fills = s.q("SELECT SUM(filled_qty) q FROM orders WHERE trade_id=? AND role IN ('stop','exit')", (t["id"],))[0]["q"]
            self.assertAlmostEqual(fills, t["qty_total"], places=8)
        self.assertEqual(s.q("SELECT cid, COUNT(*) n FROM orders GROUP BY cid HAVING n>1"), [])
        self.assertEqual(s.get("mode"), "REPLAY")
        self.assertTrue(all("simulated" in e["msg"] for e in s.q("SELECT msg FROM events WHERE kind='fill'")))
        s.close()


if __name__ == "__main__":
    unittest.main()
