"""The trading engine. One `step(now)` does, in order:

  reconcile with every broker -> snapshot equity, apply loss limits -> apply controls (kill / flatten)
  -> manage exits -> scan for signals -> gate, size and submit entries

The same code runs in replay, backtest, paper and live; only the broker adapters and data source differ.
Nothing here asks for approval: once a mode is running, qualifying orders are sent automatically.
"""
from __future__ import annotations

import json
import uuid

import pandas as pd

from . import risk as R
from .brokers.base import TERMINAL, BrokerError, BrokerTimeout
from .config import STRATEGY_MARKET, STRATEGY_VERSION, redact
from .instruments import ET, active_contract, usd_per_quote
from .store import ACTIVE
from .strategies import enabled as enabled_strategies, iso, upto

EPS = 1e-9
TF = {"5m": pd.Timedelta(minutes=5), "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}


def ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


class Engine:
    def __init__(self, cfg, store, brokers: dict, data, mode: str, earnings=None, calendar=None, notify=None, only=None, record_scanner=True):
        self.cfg, self.store, self.brokers, self.data, self.mode = cfg, store, brokers, data, mode
        self.earnings, self.calendar, self.notify = earnings, calendar, notify
        self.strategies = enabled_strategies(cfg, only)
        self.record_scanner = record_scanner
        self.now: pd.Timestamp | None = None
        self.insts: dict = {}
        self.broker_ok: dict[str, bool] = {}
        self.marks: dict[str, float] = {}
        self.rates: dict[str, float] = {}
        self.accounts: dict[str, dict] = {}
        self._bar_count: dict = {}
        self._inst_day = None
        store.set("mode", mode)
        store.set("brokers", {k: b.describe() for k, b in brokers.items()})
        if earnings is not None:
            store.set("earnings_status", earnings.describe())
        if calendar is not None:
            store.set("calendar_status", calendar.describe())

    # ------------------------------------------------------------------ helpers
    def log(self, kind: str, msg: str, level: str = "INFO"):
        self.store.event(iso(self.now) if self.now is not None else "", level, kind, redact(msg))

    def alert(self, kind: str, msg: str):
        """Persist first; external notification is best-effort and can fail without losing the alert."""
        self.log(kind, msg, "ALERT")
        if self.notify:
            try:
                self.notify(kind, redact(msg))
            except Exception as e:
                self.log("notify_failed", f"external notification failed: {type(e).__name__}")

    def _get(self, k, d=None):
        return self.store.get(k, d)

    def _strategy(self, name):
        return next((s for s in self.strategies if s.name == name), None)

    def fx(self, inst) -> float:
        return usd_per_quote(inst.quote_ccy, self.rates)

    # ------------------------------------------------------------------ step
    def step(self, now: pd.Timestamp):
        self.now = ts(now)
        self._refresh_instruments()
        self.reconcile()
        self._snapshot()
        self._controls()
        self._manage_exits()
        self._scan()
        self._process_signals()
        self._ensure_protection()
        self.store.set("heartbeat", iso(self.now))

    # ------------------------------------------------------------------ instruments
    def _refresh_instruments(self):
        day = self.now.normalize()
        if self._inst_day == day and self.insts:
            return
        self._inst_day = day
        m, problems, insts = self.cfg.markets, {}, {}

        def add(market, symbols, **kw):
            b = self.brokers.get(m[market].broker)
            if not m[market].enabled:
                return
            if b is None:
                problems[market] = f"broker '{m[market].broker}' is not connected; {market} is disabled"
                return
            for s in symbols:
                try:
                    inst = b.instrument(s)
                except (BrokerError, BrokerTimeout) as e:
                    problems[s] = f"instrument lookup failed: {redact(e)}"
                    continue
                if inst is None:
                    problems[s] = f"not tradable at {b.key}"
                else:
                    insts[inst.symbol] = inst
        add("stock", m.stock.watchlist)
        add("forex", m.forex.pairs)
        add("crypto", m.crypto.symbols)
        active = dict(self._get("active_contracts", {}))
        if m.futures.enabled:
            b = self.brokers.get(m.futures.broker)
            if b is None:
                problems["futures"] = f"broker '{m.futures.broker}' is not connected; futures are disabled"
            else:
                for root in m.futures.roots:
                    try:
                        contracts = b.contracts(root)
                    except (BrokerError, BrokerTimeout) as e:
                        problems[root] = f"contract lookup failed: {redact(e)}"
                        continue
                    for c in contracts:
                        insts[c.symbol] = c          # keep every dated contract resolvable (open positions may sit in an older one)
                    cur = active_contract(contracts, self.now, m.futures.expiry_exclusion_days)
                    if cur is None:
                        problems[root] = "no contract outside the expiry exclusion window"
                        continue
                    if active.get(root) and active[root] != cur.symbol:
                        old = insts.get(active[root])
                        self.log("rollover", f"ROLLOVER {root}: {active[root]} (expires {old.expiry if old else '?'}) is inside the {m.futures.expiry_exclusion_days}-day "
                                             f"exclusion window; new entries now use {cur.symbol} (expires {cur.expiry}, multiplier {cur.multiplier:g}, "
                                             f"tick {cur.tick_size:g} = ${cur.tick_value:g}, {cur.exchange})")
                    active[root] = cur.symbol
        self.insts = insts
        self.store.set("active_contracts", active)
        self.store.set("instrument_problems", problems)
        self.store.set("instruments", {s: i.to_dict() for s, i in insts.items()})

    def _tradable(self, inst) -> bool:
        return inst.market != "futures" or self._get("active_contracts", {}).get(inst.root) == inst.symbol

    # ------------------------------------------------------------------ orders
    def _place(self, role, inst, side, typ, qty, trade_id=None, signal_id=None, stop_price=None, reduce_only=False, expected=None,
               reserved_risk=0.0, reserved_notional=0.0, send=True) -> dict:
        cid = f"rv{role[0]}-{uuid.uuid4().hex[:16]}"
        self.store.insert("orders", cid=cid, broker=inst.broker, signal_id=signal_id, trade_id=trade_id, symbol=inst.symbol, side=side, type=typ,
                          qty=qty, stop_price=stop_price, role=role, status="created", reduce_only=int(reduce_only), expected_price=expected,
                          reserved_risk=reserved_risk, reserved_notional=reserved_notional, created=iso(self.now), updated=iso(self.now))
        o = self.store.one("SELECT * FROM orders WHERE cid=?", (cid,))
        return self._send(o) if send else o

    def _send(self, o: dict) -> dict:
        """Send a saved intent. The intent row already exists, so a crash or timeout here is recoverable."""
        b = self.brokers[o["broker"]]
        self.store.update("orders", "cid", o["cid"], attempts=o["attempts"] + 1, updated=iso(self.now))
        try:
            res = b.submit({"cid": o["cid"], "symbol": o["symbol"], "side": o["side"], "type": o["type"], "qty": o["qty"],
                            "stop_price": o["stop_price"], "reduce_only": bool(o["reduce_only"]), "role": o["role"]})
        except BrokerTimeout as e:
            self.store.update("orders", "cid", o["cid"], status="uncertain", note=redact(e))
            self.log("order_uncertain", f"{o['role']} order {o['cid']} for {o['symbol']}: no confirmation ({redact(e)}). Will reconcile before any retry.")
        except BrokerError as e:
            self.store.update("orders", "cid", o["cid"], status="rejected", note=redact(e))
            self._on_reject(self.store.one("SELECT * FROM orders WHERE cid=?", (o["cid"],)))
        else:
            self._apply(self.store.one("SELECT * FROM orders WHERE cid=?", (o["cid"],)), res)
        return self.store.one("SELECT * FROM orders WHERE cid=?", (o["cid"],))

    def _on_reject(self, o: dict):
        msg = f"{o['broker']} rejected the {o['role']} order for {o['symbol']} ({o['side']} {o['qty']:g}): {o['note']}"
        self.alert("order_rejected", msg + ". It was not routed anywhere else.")
        if o["role"] == "entry" and o["filled_qty"] <= 0:
            self.store.update("signals", "id", o["signal_id"], status="skipped", decision=f"Broker rejected the order: {o['note']}")
        elif o["role"] == "stop":
            t = self.store.one("SELECT * FROM trades WHERE id=?", (o["trade_id"],))
            if t and t["status"] == "open":
                self._protection_failed(t, o["note"])
        elif o["role"] == "exit":
            t = self.store.one("SELECT * FROM trades WHERE id=?", (o["trade_id"],))
            if t:
                meta = {**t["meta"], "exit_rejects": t["meta"].get("exit_rejects", 0) + 1}
                self.store.update("trades", "id", t["id"], meta=meta)
                if meta["exit_rejects"] >= 3:
                    self._pause_instrument(t["symbol"], f"exit order rejected {meta['exit_rejects']} times; position remains OPEN and needs attention")

    def _pause_instrument(self, symbol: str, reason: str):
        paused = dict(self._get("paused_instruments", {}))
        if paused.get(symbol) != reason:
            paused[symbol] = reason
            self.store.set("paused_instruments", paused)
            self.alert("instrument_paused", f"{symbol} paused: {reason}")

    def _apply(self, o: dict, b: dict):
        """Fold a broker order report into the journal. Idempotent: only the fill DELTA is booked."""
        delta = round((b.get("filled_qty") or 0.0) - o["filled_qty"], 10)
        status = b["status"] if not (o["status"] == "pending_cancel" and b["status"] in ("new", "partially_filled")) else "pending_cancel"
        inst = self.insts.get(o["symbol"])
        fees = b.get("fees")
        if fees is None and inst is not None and b.get("avg_price"):       # broker does not report fees per order: estimate
            c = self.cfg.costs[inst.market]
            fees = (b["filled_qty"] or 0) * (c.fee_per_unit + b["avg_price"] * inst.multiplier * self.fx(inst) * c.fee_bps / 1e4)
        fee_delta = (fees or 0.0) - (o["fees"] or 0.0)
        self.store.update("orders", "cid", o["cid"], broker_id=b.get("id"), status=status, filled_qty=b.get("filled_qty") or 0.0,
                          avg_price=b.get("avg_price"), fees=fees or 0.0, updated=iso(self.now))
        if delta > EPS and inst is not None:
            # price of just this increment, from the cumulative averages
            px = ((b["avg_price"] * b["filled_qty"]) - ((o["avg_price"] or 0.0) * o["filled_qty"])) / delta
            if o["role"] == "entry":
                self._book_entry(o, inst, delta, px, fee_delta)
            else:
                self._book_exit(o, inst, delta, px, fee_delta)
        if status in TERMINAL and o["status"] not in TERMINAL:
            filled = b.get("filled_qty") or 0.0
            if o["role"] == "entry":
                if filled <= EPS and status != "rejected":
                    self.store.update("signals", "id", o["signal_id"], status="skipped",
                                      decision=f"Entry order {status} without a fill" + (f" ({o['note']})" if o.get("note") else ""))
                elif filled < o["qty"] - EPS:
                    s = self.store.one("SELECT decision FROM signals WHERE id=?", (o["signal_id"],))
                    self.store.update("signals", "id", o["signal_id"], decision=(s["decision"] or "") + f" Partial fill {filled:g} of {o['qty']:g}; remainder {status}.")
                    self.log("partial_fill", f"{o['symbol']} entry filled {filled:g} of {o['qty']:g}; remainder {status}. Protection will match the filled quantity.")
            elif status == "rejected":
                self.store.update("orders", "cid", o["cid"], note=b.get("note") or o.get("note") or "rejected by broker")
                self._on_reject(self.store.one("SELECT * FROM orders WHERE cid=?", (o["cid"],)))

    def _book_entry(self, o, inst, qty, px, fee):
        sig = self.store.one("SELECT * FROM signals WHERE id=?", (o["signal_id"],))
        t = self.store.one("SELECT * FROM trades WHERE signal_id=? AND status!='closed'", (o["signal_id"],))
        fx = self.fx(inst)
        if t is None:
            tid = self.store.insert("trades", signal_id=sig["id"], strategy=sig["strategy"], strategy_version=STRATEGY_VERSION[sig["strategy"]], mode=self.mode,
                                    symbol=inst.symbol, market=inst.market, broker=inst.broker, side=sig["side"], qty=qty, qty_total=qty, entry_avg=px,
                                    expected_entry=o["expected_price"], stop=sig["stop"], target=sig["target"],
                                    planned_risk=abs(px - sig["stop"]) * qty * inst.multiplier * fx, opened=iso(self.now), hold_until=sig["hold_until"],
                                    flatten_at=sig["flatten_at"], status="open", fees=fee, meta=sig["meta"] or {})
            self.store.update("orders", "cid", o["cid"], trade_id=tid)
        else:
            tot = t["qty_total"] + qty
            avg = (t["entry_avg"] * t["qty_total"] + px * qty) / tot
            self.store.update("trades", "id", t["id"], qty=t["qty"] + qty, qty_total=tot, entry_avg=avg, fees=t["fees"] + fee,
                              planned_risk=abs(avg - t["stop"]) * tot * inst.multiplier * fx)
        self.store.update("signals", "id", sig["id"], status="traded")
        label = "simulated fill" if self.mode in ("REPLAY", "BACKTEST") else "fill"
        self.log("fill", f"{label}: {'bought' if o['side'] == 'buy' else 'sold'} {qty:g} {inst.symbol} @ {px:g} ({sig['strategy']} entry; expected {o['expected_price']:g})")

    def _book_exit(self, o, inst, qty, px, fee):
        t = self.store.one("SELECT * FROM trades WHERE id=?", (o["trade_id"],))
        if t is None:
            return
        qty = min(qty, t["qty"])
        pnl = (px - t["entry_avg"]) * t["side"] * qty * inst.multiplier * self.fx(inst)
        done = t["exit_qty"] + qty
        left = round(t["qty"] - qty, 10)
        fields = dict(qty=max(left, 0.0), exit_qty=done, exit_avg=((t["exit_avg"] or 0.0) * t["exit_qty"] + px * qty) / done,
                      realized=t["realized"] + pnl, fees=t["fees"] + fee)
        if o["role"] == "stop" and not t["exit_reason"]:
            beyond = (t["stop"] - px) * t["side"]
            gap = beyond > 0.25 * abs(t["entry_avg"] - t["stop"])
            fields["exit_reason"] = "protective stop" + (f" (GAP: filled {beyond:.4g} beyond the stop, so the loss exceeds the planned risk)" if gap else "")
            fields["expected_exit"] = t["stop"]
        if left <= EPS:
            fields.update(status="closed", closed=iso(self.now), qty=0.0)
        self.store.update("trades", "id", t["id"], **fields)
        label = "simulated fill" if self.mode in ("REPLAY", "BACKTEST") else "fill"
        self.log("fill", f"{label}: {'bought' if o['side'] == 'buy' else 'sold'} {qty:g} {inst.symbol} @ {px:g} ({o['role']}; trade #{t['id']} "
                         f"{'closed' if left <= EPS else f'{left:g} left'}, realized ${fields['realized'] - fields['fees']:,.2f} net)")
        if left <= EPS:
            for s in self.store.active_orders(trade_id=t["id"]):
                if s["cid"] != o["cid"]:
                    self._cancel(s)

    def _cancel(self, o: dict) -> dict:
        """Ask the broker to cancel, then read back what actually happened."""
        b = self.brokers[o["broker"]]
        try:
            if o["status"] in ("created",):
                self.store.update("orders", "cid", o["cid"], status="canceled", note="cancelled before it was sent")
                return self.store.one("SELECT * FROM orders WHERE cid=?", (o["cid"],))
            cur = b.get_by_cid(o["cid"])
            if cur is None:
                if o["status"] == "uncertain":
                    return o                          # unknown: leave it for reconcile
                self.store.update("orders", "cid", o["cid"], status="canceled", note="not found at broker")
                return self.store.one("SELECT * FROM orders WHERE cid=?", (o["cid"],))
            if cur["status"] not in TERMINAL:
                b.cancel(cur)
                self.store.update("orders", "cid", o["cid"], status="pending_cancel")
                o = self.store.one("SELECT * FROM orders WHERE cid=?", (o["cid"],))
                cur = b.get_by_cid(o["cid"]) or cur
            self._apply(o, cur)
        except (BrokerTimeout, BrokerError) as e:
            self.broker_ok[o["broker"]] = False
            self.log("cancel_failed", f"cancel of {o['cid']} ({o['symbol']}) not confirmed: {redact(e)}")
        return self.store.one("SELECT * FROM orders WHERE cid=?", (o["cid"],))

    # ------------------------------------------------------------------ reconcile
    def reconcile(self):
        """Bring the journal in line with each broker. Until a broker answers, it accepts no new signals."""
        external, manual_orders, status = {}, {}, {}
        for key, b in self.brokers.items():
            try:
                for o in self.store.active_orders(broker=key):
                    cur = b.get_by_cid(o["cid"])
                    if cur is not None:
                        self._apply(o, cur)
                    elif o["status"] in ("created", "uncertain"):
                        self._resolve_unsent(o)
                    else:
                        self.store.update("orders", "cid", o["cid"], status="canceled", note="order no longer exists at the broker")
                        self.alert("order_missing", f"{o['role']} order {o['cid']} for {o['symbol']} is no longer at {key}; treated as cancelled")
                pos = b.positions()
                mine: dict[str, float] = {}
                for t in self.store.q("SELECT symbol, side, qty FROM trades WHERE status!='closed' AND broker=?", (key,)):
                    mine[t["symbol"]] = mine.get(t["symbol"], 0.0) + t["side"] * t["qty"]
                for sym in set(pos) | set(mine):
                    diff = round(pos.get(sym, {"qty": 0.0})["qty"] - mine.get(sym, 0.0), 8)
                    if abs(diff) > 1e-8:
                        external[f"{key}:{sym}"] = {"broker": key, "symbol": sym, "qty": diff, "avg_price": pos.get(sym, {}).get("avg_price"),
                                                    "bot_qty": mine.get(sym, 0.0)}
                manual_orders[key] = [o for o in b.open_orders() if not str(o.get("cid") or "").startswith("rv")]
                self.broker_ok[key] = True
                status[key] = {"ok": True, "at": iso(self.now)}
            except (BrokerTimeout, BrokerError) as e:
                if self.broker_ok.get(key, True):
                    self.alert("broker_disconnected", f"{key}: reconciliation failed ({redact(e)}). No new entries at this broker until it succeeds.")
                self.broker_ok[key] = False
                status[key] = {"ok": False, "at": iso(self.now), "error": redact(e)}
        if all(s["ok"] for s in status.values()):
            self.store.set("external_positions", external)
            self.store.set("manual_orders", manual_orders)
            # A manual holding in an instrument the bot trades cannot be told apart from bot units at a netting
            # broker, so automated trading in that instrument is blocked until the manual holding is gone.
            paused = dict(self._get("paused_instruments", {}))
            for sym, why in list(paused.items()):
                if why.startswith("manual holding") and not any(e["symbol"] == sym for e in external.values()):
                    del paused[sym]
                    self.log("instrument_resumed", f"{sym}: manual holding conflict cleared")
            self.store.set("paused_instruments", paused)
            for e in external.values():
                if e["symbol"] in self.insts and self.brokers[e["broker"]].nets_positions:
                    self._pause_instrument(e["symbol"], f"manual holding of {e['qty']:g} {e['symbol']} at {e['broker']} that the bot did not open. The broker nets it with bot "
                                                        f"trades, so ownership cannot be separated safely: automated entries in {e['symbol']} are blocked. "
                                                        f"The bot will never sell or flatten this holding.")
        prev = self._get("reconcile", {})
        self.store.set("reconcile", {**prev, **status})

    def _resolve_unsent(self, o: dict):
        """The broker has no record of this client id: the request never arrived, so sending it now cannot duplicate."""
        if o["role"] == "entry":
            sig = self.store.one("SELECT * FROM signals WHERE id=?", (o["signal_id"],))
            if sig is None or ts(sig["expires"]) < self.now or self._halt_reason():
                self.store.update("orders", "cid", o["cid"], status="canceled", note="not at broker; signal no longer valid, not re-sent")
                if sig:
                    self.store.update("signals", "id", sig["id"], status="skipped", decision="Order never reached the broker and the signal expired before a retry.")
                return
        if o["attempts"] >= 3:
            self.store.update("orders", "cid", o["cid"], status="canceled", note="outcome still ambiguous after 3 attempts")
            self._pause_instrument(o["symbol"], f"{o['role']} order outcome ambiguous after {o['attempts']} attempts")
            return
        self.log("order_resend", f"{o['role']} order {o['cid']} for {o['symbol']} was never received by {o['broker']} (confirmed by client-id lookup); sending it now")
        self._send(o)

    # ------------------------------------------------------------------ protection
    def _ensure_protection(self):
        unprotected = []
        for t in self.store.q("SELECT * FROM trades WHERE status='open' AND qty>0"):
            inst = self.insts.get(t["symbol"])
            if inst is None or not self.broker_ok.get(t["broker"]):
                unprotected.append(t["id"])
                continue
            stops = self.store.active_orders(trade_id=t["id"], role="stop")
            remaining = sum(s["qty"] - s["filled_qty"] for s in stops)
            if stops and abs(remaining - t["qty"]) > 1e-8:
                # quantity drifted (partial entry fill, partial exit): replace so protection equals actual units
                stops = [s for s in (self._cancel(s) for s in stops) if s["status"] in ACTIVE]
                if stops:
                    continue                          # cancel still pending; re-place next step
            if not stops:
                o = self._place("stop", inst, "sell" if t["side"] > 0 else "buy", "stop", t["qty"], trade_id=t["id"], signal_id=t["signal_id"],
                                stop_price=t["stop"], reduce_only=True, expected=t["stop"])
                if o["status"] not in ACTIVE and o["status"] != "filled":
                    unprotected.append(t["id"])
        self.store.set("unprotected", unprotected)

    def _protection_failed(self, t: dict, why: str):
        self.store.set("halt_protection", f"protective stop for {t['symbol']} could not be established: {why}")
        for o in self.store.active_orders(signal_id=t["signal_id"], role="entry"):
            self._cancel(o)                           # cancel the unfilled entry remainder
        policy = self.cfg.risk.emergency_exit_policy
        if policy == "flatten":
            self.store.update("trades", "id", t["id"], status="exit_pending", exit_reason="emergency exit: protective stop could not be established")
        self.alert("protection_failed", f"UNPROTECTED POSITION: {t['qty']:g} {t['symbol']} has no protective stop ({why}). New entries are halted. "
                                        f"Policy '{policy}': " + ("closing the position at market." if policy == "flatten" else "holding; will keep retrying the stop."))

    # ------------------------------------------------------------------ snapshot & limits
    def _snapshot(self):
        accounts = {}
        for key, b in self.brokers.items():
            if not self.broker_ok.get(key):
                accounts[key] = self.accounts.get(key)
                continue
            try:
                accounts[key] = b.account()
            except (BrokerTimeout, BrokerError) as e:
                self.broker_ok[key] = False
                accounts[key] = self.accounts.get(key)
                self.log("account_failed", f"{key}: account read failed: {redact(e)}")
        self.accounts = {k: v for k, v in accounts.items() if v}
        self.store.set("brokers", {k: {**b.describe(), "ok": bool(self.broker_ok.get(k))} for k, b in self.brokers.items()})
        quotes = {}
        for sym, inst in self.insts.items():
            b = self.brokers.get(inst.broker)
            need = inst.market == "forex" or self.store.one("SELECT 1 FROM trades WHERE symbol=? AND status!='closed'", (sym,))
            if not need or b is None or not self.broker_ok.get(inst.broker):
                continue
            try:
                q = b.quote(inst)
            except (BrokerTimeout, BrokerError):
                q = None
            if q and q["bid"] > 0 and q["ask"] > 0:
                self.marks[sym] = (q["bid"] + q["ask"]) / 2
                quotes[sym] = iso(q["ts"])
        self.rates = {s: m for s, m in self.marks.items() if s in self.insts and self.insts[s].market == "forex"}
        if not self.accounts:
            return
        equity = sum(a["equity"] for a in self.accounts.values())
        r, p = self.cfg.risk, self.cfg.portfolio
        # daily reset: a fixed wall-clock time in the portfolio timezone, every day including weekends
        local = self.now.tz_convert(p.daily_reset_tz)
        h, m = map(int, p.daily_reset_time.split(":"))
        start = local.normalize() + pd.Timedelta(hours=h, minutes=m)
        if start > local:
            start -= pd.Timedelta(days=1)
        day = self._get("day")
        if not day or day["period"] != iso(start):
            day = {"period": iso(start), "equity": equity, "next_reset": iso(start + pd.Timedelta(days=1))}
            self.store.set("day", day)
            if self._get("halt_daily"):
                self.store.set("halt_daily", None)
                self.log("risk_limit", "daily loss halt cleared at the daily reset")
        day_pnl = equity - day["equity"]
        if not self._get("halt_daily") and day_pnl <= -day["equity"] * r.daily_loss_limit_pct / 100:
            why = f"daily loss limit reached: ${day_pnl:,.2f} ({day_pnl / day['equity'] * 100:.2f}%) since the reset, limit {r.daily_loss_limit_pct:g}%"
            self.store.set("halt_daily", why)
            self.alert("risk_limit", f"{why}. New entries are halted until {day['next_reset']}; existing positions and their protective exits remain managed.")
        peak = max(self._get("peak_equity", equity), equity)
        self.store.set("peak_equity", peak)
        dd = equity / peak - 1
        if not self._get("halt_drawdown") and dd <= -r.max_drawdown_pct / 100:
            why = f"drawdown limit reached: {dd * 100:.2f}% from the peak of ${peak:,.2f}, limit {r.max_drawdown_pct:g}%"
            self.store.set("halt_drawdown", why)
            self.alert("risk_limit", f"{why}. New entries are halted until you run `resume`; protective exits remain managed.")
        snap = {"equity": equity, "day_pnl": day_pnl, "drawdown": dd, "peak": peak, "accounts": self.accounts, "marks": self.marks, "rates": self.rates,
                "quote_times": {**self._get("snapshot", {}).get("quote_times", {}), **quotes}, "at": iso(self.now)}
        self.store.set("snapshot", snap)
        last = self._get("equity_last")
        if not last or abs(last["equity"] - equity) > 0.005 or (self.now - ts(last["ts"])) >= pd.Timedelta(hours=4):
            self.store.insert("equity", ts=iso(self.now), equity=equity, detail={k: a["equity"] for k, a in self.accounts.items()})
            self.store.set("equity_last", {"ts": iso(self.now), "equity": equity})

    def _halt_reason(self) -> str | None:
        if self._get("kill"):
            return "kill switch is ON"
        if self._get("pause_entries"):
            return "entries are paused"
        return self._get("halt_daily") or self._get("halt_drawdown") or self._get("halt_protection")

    # ------------------------------------------------------------------ controls
    def _controls(self):
        if self._get("kill"):
            for o in self.store.active_orders(role="entry"):
                self._cancel(o)
                self.log("kill_switch", f"kill switch: cancelled working entry {o['cid']} ({o['symbol']}); protective exits untouched")
            for s in self.store.q("SELECT id FROM signals WHERE status='pending'"):
                self.store.update("signals", "id", s["id"], status="skipped", decision="Kill switch is ON: entry not sent.")
        req = self._get("flatten_request")
        if req:
            self.store.set("flatten_request", None)
            ids = None if req == "all" else set(req)
            n = 0
            for t in self.store.q("SELECT * FROM trades WHERE status='open'"):
                if ids is None or t["id"] in ids:
                    self.store.update("trades", "id", t["id"], status="exit_pending", exit_reason="manual flatten")
                    n += 1
            self.log("flatten", f"flatten requested for {'all bot-managed positions' if ids is None else f'trades {sorted(ids)}'}: {n} position(s) queued. "
                                f"Holdings the bot did not open are never touched.")

    # ------------------------------------------------------------------ exits
    def _manage_exits(self):
        report = {}
        for t in self.store.open_trades():
            inst = self.insts.get(t["symbol"])
            if inst is None or not self.broker_ok.get(t["broker"]):
                continue
            if t["status"] == "open":
                why = self._exit_reason(t, inst)
                if why:
                    self.store.update("trades", "id", t["id"], status="exit_pending", exit_reason=why, expected_exit=self.marks.get(t["symbol"]))
                    t = self.store.one("SELECT * FROM trades WHERE id=?", (t["id"],))
            if t["status"] in ("exit_pending", "closing"):
                left = self._progress_exit(t, inst)
                if left:
                    report[str(t["id"])] = f"{t['symbol']}: {left}"
        prev = self._get("exit_blocked", {})
        for k, v in report.items():
            if prev.get(k) != v:
                self.log("exit_waiting", f"position remains open: {v}")
        self.store.set("exit_blocked", report)

    def _exit_reason(self, t, inst) -> str | None:
        if t["flatten_at"] and self.now >= ts(t["flatten_at"]):
            return f"session cutoff ({ts(t['flatten_at']).tz_convert(ET):%H:%M} ET)"
        if t["hold_until"] and self.now >= ts(t["hold_until"]):
            return "maximum holding period reached"
        strat = self._strategy(t["strategy"])
        df = self._bars(inst, strat.timeframe) if strat else None
        if t["target"] is not None:
            m = self.marks.get(t["symbol"])
            hit = m is not None and (m - t["target"]) * t["side"] >= 0
            if not hit and df is not None:
                d = upto(df, self.now)
                d = d[d["start"] >= ts(t["opened"])]
                if len(d):
                    hit = (float(d["high"].max()) >= t["target"]) if t["side"] > 0 else (float(d["low"].min()) <= t["target"])
            if hit:
                return f"profit target {t['target']:g} reached"
        return strat.exit_check(t, inst, df, self.now) if strat and df is not None else None

    def _progress_exit(self, t, inst) -> str | None:
        """Cancel protection, then close with ONE reduce-only market order. Returns why it is still open, if it is."""
        b = self.brokers[t["broker"]]
        if self.store.active_orders(trade_id=t["id"], role="exit"):
            return None                                # an exit is already working: never send a second one
        try:
            if not b.market_open(inst):
                return f"exit queued ({t['exit_reason']}) but the market is closed; the protective stop stays in place"
        except (BrokerTimeout, BrokerError) as e:
            return f"cannot check market state: {redact(e)}"
        for s in self.store.active_orders(trade_id=t["id"], role="stop"):
            if self._cancel(s)["status"] in ACTIVE:
                return "waiting for the protective stop to cancel before sending the exit"
        t = self.store.one("SELECT * FROM trades WHERE id=?", (t["id"],))
        if t["status"] == "closed" or t["qty"] <= EPS:
            return None
        q = None
        try:
            q = b.quote(inst)
        except (BrokerTimeout, BrokerError):
            pass
        expected = (q["bid"] if t["side"] > 0 else q["ask"]) if q else self.marks.get(t["symbol"])
        self.store.update("trades", "id", t["id"], status="closing", expected_exit=expected)
        o = self._place("exit", inst, "sell" if t["side"] > 0 else "buy", "market", t["qty"], trade_id=t["id"], signal_id=t["signal_id"],
                        reduce_only=True, expected=expected)
        return None if o["status"] in ACTIVE or o["status"] == "filled" else f"exit order {o['status']}: {o.get('note') or ''}"

    # ------------------------------------------------------------------ scan
    def _bars(self, inst, tf):
        try:
            return self.data.bars(inst, tf)
        except Exception as e:
            self._feed_errors[f"{inst.symbol}:{tf}"] = redact(e)
            return None

    def _scan(self):
        self._feed_errors = {}
        feeds = dict(self._get("feeds", {}))
        scanner = dict(self._get("scanner", {})) if self.record_scanner else None
        last_scan = dict(self._get("last_scan", {}))
        dirty = False
        for strat in self.strategies:
            for inst in list(self.insts.values()):
                if not strat.applies(inst) or not self._tradable(inst):
                    continue
                df = self._bars(inst, strat.timeframe)
                if df is None or not len(df):
                    continue
                key = f"{strat.name}|{inst.symbol}"
                n = int(df.index.searchsorted(self.now, side="right"))
                if n == 0 or self._bar_count.get(key) == n:
                    continue                           # no newly completed bar for this pair
                self._bar_count[key] = n
                dirty = True
                last_bar = df.index[n - 1]
                feeds[f"{inst.symbol}|{strat.timeframe}"] = {"last_bar": iso(last_bar), "checked": iso(self.now)}
                since = ts(last_scan[key]) if key in last_scan else last_bar - TF[strat.timeframe] * (4 if strat.timeframe == "1d" else 1) - pd.Timedelta(seconds=1)
                extra = self.earnings.events(inst.symbol) if strat.name == "earnings" and self.earnings is not None else None
                for s in strat.signals(inst, df, since, self.now, extra):
                    self.store.insert("signals", ignore=True, id=s["id"], strategy=s["strategy"], symbol=s["symbol"], market=inst.market, broker=inst.broker,
                                      side=s["side"], time=s["time"], expires=s["expires"], status="pending", ref_price=s["ref_price"], stop=s["stop"],
                                      target=s["target"], atr=s["atr"], hold_until=s["hold_until"], flatten_at=s["flatten_at"], why=s["why"],
                                      exit_rule=s["exit_rule"], inputs=s["inputs"], meta=s["meta"], created=iso(self.now))
                for k in strat.skips(inst, df, since, self.now, extra):
                    self.store.insert("signals", ignore=True, id=k["id"], strategy=k["strategy"], symbol=k["symbol"], market=inst.market, broker=inst.broker,
                                      side=k["side"], time=k["time"], status="skipped", why="Earnings release evaluated.", decision="Skipped: " + k["reason"],
                                      inputs=[], meta=k["meta"], created=iso(self.now))
                last_scan[key] = iso(last_bar)
                if scanner is not None:
                    scanner[key] = {"strategy": strat.name, "symbol": inst.symbol, "market": inst.market, "last_bar": iso(last_bar),
                                    "checks": strat.snapshot(inst, df, self.now)}
                    d = df.iloc[max(n - 300, 0):n]
                    payload = json.dumps({"t": [int(x.timestamp()) for x in d["start"]], "e": [int(x.timestamp()) for x in d.index],
                                          **{k[0]: [round(float(x), 6) for x in d[k]] for k in ("open", "high", "low", "close", "volume")}})
                    with self.store.lock:
                        self.store.db.execute("INSERT INTO barcache(symbol,tf,updated,payload) VALUES(?,?,?,?) ON CONFLICT(symbol,tf) DO UPDATE SET "
                                              "updated=excluded.updated, payload=excluded.payload", (inst.symbol, strat.timeframe, iso(self.now), payload))
        if dirty or self._feed_errors:
            self.store.set("last_scan", last_scan)
            self.store.set("feeds", feeds)
            self.store.set("feed_errors", self._feed_errors)
            if scanner is not None:
                self.store.set("scanner", scanner)
            self.store.set("last_scan_at", iso(self.now))

    # ------------------------------------------------------------------ entries
    def _process_signals(self):
        for sig in self.store.q("SELECT * FROM signals WHERE status='pending' ORDER BY time, id"):
            inst, strat = self.insts.get(sig["symbol"]), self._strategy(sig["strategy"])
            if inst is None or strat is None:
                self.store.update("signals", "id", sig["id"], status="skipped", decision="Instrument or strategy is no longer configured.")
                continue
            kind, why, checks = self._decide(sig, inst, strat)
            if kind == "wait":
                if self.now >= ts(sig["expires"]):
                    self.store.update("signals", "id", sig["id"], status="expired", decision=f"Expired without a tradable quote ({why}).", checks=checks)
                elif sig["decision"] != f"Waiting: {why}":
                    self.store.update("signals", "id", sig["id"], decision=f"Waiting: {why}", checks=checks)
            elif kind == "skip":
                self.store.update("signals", "id", sig["id"], status="skipped", decision="Skipped: " + why, checks=checks)
                self.log("skip", f"{sig['strategy']} {sig['symbol']}: skipped - {why}")

    def _decide(self, sig, inst, strat):
        checks: list[dict] = []
        halt = self._halt_reason()
        if halt:
            return "skip", f"new entries are halted ({halt})", checks
        paused = self._get("paused_instruments", {}).get(inst.symbol)
        if paused:
            return "skip", paused, checks
        if not self.broker_ok.get(inst.broker):
            return "wait", f"{inst.broker} not reconciled", checks
        if self.mode == "LIVE":
            b = self.brokers[inst.broker]
            act = self.store.one("SELECT * FROM activations WHERE broker=? AND account=? AND market=? AND strategy=?", (b.key, b.account_alias, inst.market, strat.name))
            if not act:
                return "skip", f"live trading is not activated for {strat.name} / {inst.market} on {b.key} account {b.account_alias}", checks
            if act["strategy_hash"] != self.cfg.strategy_hash(strat.name):
                return "skip", "strategy or risk settings changed since live activation; re-activate to confirm the new settings", checks
        if not self._tradable(inst):
            return "skip", f"{inst.symbol} is not the active contract", checks
        if self.store.one("SELECT 1 FROM trades WHERE symbol=? AND strategy=? AND status!='closed'", (inst.symbol, strat.name)) or \
                self.store.active_orders(symbol=inst.symbol, role="entry"):
            return "skip", f"already in {inst.symbol} or an entry is working (one position per strategy per instrument)", checks
        opp = self.store.one("SELECT strategy FROM trades WHERE symbol=? AND status!='closed' AND side!=?", (inst.symbol, sig["side"]))
        if opp:
            return "skip", f"conflicts with the open {opp['strategy']} position in the opposite direction (the broker nets positions)", checks
        b = self.brokers[inst.broker]
        if strat.timeframe == "1d":
            df = self._bars(inst, "1d")
            d = upto(df, self.now) if df is not None else None
            if d is None or not len(d) or d.index[-1] != ts(sig["time"]):
                return "skip", "entry window passed: a newer session has completed since the signal", checks
        try:
            if not b.market_open(inst):
                return "wait", "market closed", checks
            mins = b.minutes_since_open(inst)
            q = b.quote(inst)
        except (BrokerTimeout, BrokerError) as e:
            return "wait", f"broker unavailable: {redact(e)}", checks
        p = strat.p
        if mins is not None and "entry_delay_min" in p:
            if mins < p.entry_delay_min:
                return "wait", f"waiting {p.entry_delay_min:g} minutes after the open", checks
            if mins > p.entry_cutoff_min:
                return "skip", f"entry cutoff passed ({mins:.0f} minutes after the open, limit {p.entry_cutoff_min:g})", checks
        if not q or q["bid"] <= 0 or q["ask"] <= 0 or q["ask"] < q["bid"]:
            return "wait", "no valid two-sided quote (trading may be suspended)", checks
        age = (self.now - ts(q["ts"])).total_seconds()
        checks.append({"name": "quote age (seconds)", "value": round(age, 1), "threshold": self.cfg.risk.max_quote_age_sec, "passed": age <= self.cfg.risk.max_quote_age_sec})
        if age > self.cfg.risk.max_quote_age_sec:
            return "wait", f"quote is stale ({age:.0f}s old, limit {self.cfg.risk.max_quote_age_sec:g}s)", checks
        checks += strat.entry_checks(sig, inst, q, self.now, {"calendar": self.calendar})
        failed = [c for c in checks if not c["passed"]]
        if failed:
            f = lambda c: c["name"] + (f" = {c['value']:g} (limit {c['threshold']:g})" if isinstance(c["value"], (int, float)) and isinstance(c["threshold"], (int, float)) else "")
            return "skip", "; ".join(f(c) for c in failed), checks
        entry = q["ask"] if sig["side"] > 0 else q["bid"]
        if (entry - sig["stop"]) * sig["side"] <= 0:
            return "skip", f"price {entry:g} is already beyond the stop {sig['stop']:g}", checks
        if sig["side"] < 0 and not inst.shortable:
            return "skip", "short selling is not available for this instrument at this broker", checks
        try:
            fx = self.fx(inst)
        except KeyError as e:
            return "wait", f"cannot convert {inst.quote_ccy} to USD: {e}", checks
        liq = None
        if inst.market == "stock":
            df = self._bars(inst, strat.timeframe)
            d = upto(df, self.now) if df is not None else None
            if d is not None and len(d) and float(d["volume"].iloc[-1]) > 0:
                liq = float(d["volume"].iloc[-1]) * self.cfg.risk.max_volume_participation
        # Reserve risk and record the intent in ONE transaction, so two signals in the same step
        # cannot both spend the same budget.
        with self.store.tx():
            trades = self.store.open_trades()
            pending = self.store.active_orders(role="entry")
            equity = sum(a["equity"] for a in self.accounts.values())
            view = R.portfolio_view(self.cfg, equity, trades, pending, self.insts, self.marks, self.rates)
            if view["positions"] >= self.cfg.risk.max_positions:
                return "skip", f"maximum simultaneous positions reached ({view['positions']} of {self.cfg.risk.max_positions}, counting pending entries)", checks
            sizing = R.size_position(self.cfg, inst, sig["side"], entry, sig["stop"], self.accounts[inst.broker], view, self.rates, liq)
            if sizing["qty"] <= 0:
                self.store.update("signals", "id", sig["id"], sizing=sizing)
                return "skip", sizing["reason"], checks
            self.store.update("signals", "id", sig["id"], status="ordered", checks=checks, sizing=sizing, intended_entry=entry, intended_qty=sizing["qty"],
                              est_costs=sizing["est_costs_usd"],
                              decision=f"Traded: all checks passed. Sized {sizing['qty']:g} @ ~{entry:g}, planned loss at stop ${sizing['risk_usd']:,.2f} "
                                       f"(binding limit: {sizing['binding']}).")
            o = self._place("entry", inst, "buy" if sig["side"] > 0 else "sell", "market", sizing["qty"], signal_id=sig["id"], expected=entry,
                            reserved_risk=sizing["risk_usd"], reserved_notional=sizing["notional_usd"], send=False)
        self._send(o)
        return "order", "", checks

    # ------------------------------------------------------------------ operator actions
    def kill(self, on: bool):
        self.store.set("kill", bool(on))

    def pause_entries(self, on: bool):
        self.store.set("pause_entries", bool(on))

    def resume(self):
        """Clear sticky halts (drawdown, protection failure). The drawdown peak restarts from current equity."""
        self.store.set("halt_drawdown", None)
        self.store.set("halt_protection", None)
        snap = self._get("snapshot")
        if snap:
            self.store.set("peak_equity", snap["equity"])

    def flatten(self, trade_ids=None):
        self.store.set("flatten_request", "all" if trade_ids is None else list(trade_ids))
