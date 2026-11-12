"""Simulated venue for replay and backtests. Fills come from bar data, never from a real broker.

Fill model (documented, deliberately conservative):
  * Market orders fill at the open of the bar starting now, crossed by half the spread plus slippage.
    If no bar starts now the market is closed and the order waits for the next open.
  * Resting stops are checked against each completed bar. If the bar OPENS beyond the stop the fill is
    at that open (a gap-through loss larger than planned); otherwise at the stop price less slippage.
  * Profit targets are not resting orders; the engine exits at market after a bar reaches the target,
    so when one bar touches both stop and target, the stop wins.
  * An order fills at most `max_volume_participation` of the bar's volume (where volume exists).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..instruments import Instrument, usd_per_quote
from .base import TERMINAL, Broker, BrokerError, BrokerTimeout

CASH_MARKETS = ("stock", "crypto")


class SimWorld:
    """Shared clock, bars and quotes for every simulated broker."""

    def __init__(self, cfg, instruments: list[Instrument], exec_bars: dict[str, pd.DataFrame]):
        self.cfg, self.bars = cfg, exec_bars
        self.inst = {i.symbol: i for i in instruments}
        self.now: pd.Timestamp | None = None
        self.stale: set[str] = set()        # fault: quotes stop updating
        self.suspended: set[str] = set()    # fault: venue halts trading
        self.spread_mult: dict[str, float] = {}
        self.brokers: list["SimBroker"] = []
        self._a = {s: {"end": df.index.as_unit("ns").asi8, "start": pd.DatetimeIndex(df["start"]).as_unit("ns").asi8,
                       **{k: df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume")}} for s, df in exec_bars.items()}

    def timeline(self, start=None, end=None) -> list[pd.Timestamp]:
        pts = set()
        for df in self.bars.values():
            pts.update(df.index)
            pts.update(df["start"])
        return sorted(t for t in pts if (start is None or t >= start) and (end is None or t <= end))

    def _ns(self, symbol: str = "") -> int:
        return self.now.as_unit("ns").value

    def _cover(self, symbol: str):
        """(index of the bar that covers `now` or None, index of the last completed bar or None)."""
        a = self._a.get(symbol)
        if a is None or self.now is None:
            return None, None
        now = self._ns(symbol)
        i = int(np.searchsorted(a["end"], now, side="right"))
        cur = i if i < len(a["end"]) and a["start"][i] <= now else None
        return cur, (i - 1 if i > 0 else None)

    def bar(self, symbol: str, i: int) -> dict:
        a = self._a[symbol]
        return {"start": pd.Timestamp(a["start"][i], unit="ns", tz="UTC"), "end": pd.Timestamp(a["end"][i], unit="ns", tz="UTC"),
                "open": a["open"][i], "high": a["high"][i], "low": a["low"][i], "close": a["close"][i], "volume": a["volume"][i]}

    def ended_now(self, symbol: str):
        """Index of the bar whose end is exactly `now`, else None."""
        a = self._a[symbol]
        now = self._ns(symbol)
        i = int(np.searchsorted(a["end"], now, side="left"))
        return i if i < len(a["end"]) and a["end"][i] == now else None

    def is_open(self, symbol: str) -> bool:
        return self._cover(symbol)[0] is not None and symbol not in self.suspended

    def half_spread(self, inst: Instrument, px: float) -> float:
        c = self.cfg.costs[inst.market]
        h = c.half_spread_ticks * inst.tick_size if inst.market == "futures" else c.half_spread_pips * inst.pip if inst.market == "forex" else px * c.half_spread_bps / 1e4
        return h * self.spread_mult.get(inst.symbol, 1.0)

    def slippage(self, inst: Instrument, px: float) -> float:
        c = self.cfg.costs[inst.market]
        return c.slippage_ticks * inst.tick_size if inst.market == "futures" else c.slippage_pips * inst.pip if inst.market == "forex" else px * c.slippage_bps / 1e4

    def quote(self, symbol: str) -> dict | None:
        cur, last = self._cover(symbol)
        inst, a = self.inst[symbol], self._a.get(symbol)
        if cur is not None:
            mid, ts = float(a["open"][cur]), pd.Timestamp(a["start"][cur], unit="ns", tz="UTC")
        elif last is not None:
            mid, ts = float(a["close"][last]), pd.Timestamp(a["end"][last], unit="ns", tz="UTC")
        else:
            return None
        if symbol in self.stale:
            ts = ts - pd.Timedelta(hours=1)
        h = self.half_spread(inst, mid)
        return {"bid": mid - h, "ask": mid + h, "ts": ts}

    def mark(self, symbol: str) -> float | None:
        cur, last = self._cover(symbol)
        if cur is not None:
            return float(self._a[symbol]["open"][cur])
        return float(self._a[symbol]["close"][last]) if last is not None else None

    def rates(self) -> dict[str, float]:
        return {s: m for s in self.bars if self.inst[s].market == "forex" and (m := self.mark(s)) is not None}

    def advance(self, now: pd.Timestamp):
        self.now = now
        for b in self.brokers:
            b._on_time()


class SimBroker(Broker):
    mode = "REPLAY"
    data_label = "SIMULATED (bar-based fills; not broker-confirmed)"

    def __init__(self, key: str, world: SimWorld, cash: float, alias: str = ""):
        self.key, self.w, self.cash = key, world, float(cash)
        self.account_alias = alias or f"sim-{key}"
        self.pos: dict[str, dict] = {}
        self.orders: dict[str, dict] = {}
        self.faults: list[dict] = []   # e.g. {"role": "entry", "symbol": "X", "kind": "timeout_after_accept"}
        self._ids = itertools.count(1)
        self.submitted_log: list[str] = []
        world.brokers.append(self)

    # ---- interface
    def now(self):
        return self.w.now

    def instrument(self, symbol):
        i = self.w.inst.get(symbol)
        return i if i is not None and i.broker == self.key else None

    def contracts(self, root):
        return sorted((i for i in self.w.inst.values() if i.root == root and i.broker == self.key), key=lambda i: i.expiry)

    def market_open(self, inst):
        return self.w.is_open(inst.symbol)

    def quote(self, inst):
        return None if inst.symbol in self.w.suspended else self.w.quote(inst.symbol)

    def bars(self, inst, timeframe, limit):
        raise NotImplementedError("simulated data is served by the replay/backtest data source")

    def positions(self):
        return {s: {"qty": p["qty"], "avg_price": p["avg"]} for s, p in self.pos.items() if abs(p["qty"]) > 1e-12}

    def _value(self):
        rates, mv, upl, margin = self.w.rates(), 0.0, 0.0, 0.0
        for s, p in self.pos.items():
            inst, m = self.w.inst[s], self.w.mark(s)
            if m is None or abs(p["qty"]) < 1e-12:
                continue
            fx = usd_per_quote(inst.quote_ccy, rates)
            if inst.market in CASH_MARKETS:
                mv += p["qty"] * m * fx
            else:
                upl += p["qty"] * inst.multiplier * (m - p["avg"]) * fx
                margin += abs(p["qty"]) * (inst.margin_per_unit if inst.market == "futures" else m * inst.multiplier * fx * inst.margin_rate)
        return mv, upl, margin

    def account(self):
        mv, upl, margin = self._value()
        eq = self.cash + mv + upl
        return {"equity": eq, "cash": self.cash, "margin_available": eq - mv - margin, "margin_used": margin}

    def get_by_cid(self, cid):
        o = self.orders.get(cid)
        return dict(o) if o else None

    def open_orders(self):
        return [dict(o) for o in self.orders.values() if o["status"] not in TERMINAL]

    def cancel(self, order):
        o = self.orders.get(order["cid"])
        if o and o["status"] not in TERMINAL:
            o["status"] = "canceled"

    def submit(self, order):
        fault = next((f for f in self.faults if all(order.get(k) == v for k, v in f.items() if k not in ("kind", "fraction"))), None)
        if fault:
            self.faults.remove(fault)
            if fault["kind"] == "timeout_before_accept":
                raise BrokerTimeout("simulated timeout: request never reached the venue")
            if fault["kind"] == "reject":
                raise BrokerError("simulated rejection")
        if order["cid"] in self.orders:
            raise BrokerError("client order id must be unique")
        inst = self.instrument(order["symbol"])
        if inst is None:
            raise BrokerError(f"{order['symbol']} is not tradable at {self.key}")
        qty = float(order["qty"])
        if qty < inst.min_qty - 1e-12 or abs(inst.round_qty(qty) - qty) > 1e-9:
            raise BrokerError(f"quantity {qty:g} violates minimum {inst.min_qty:g} / increment {inst.qty_step:g}")
        held = self.pos.get(inst.symbol, {"qty": 0.0})["qty"]
        sign = 1 if order["side"] == "buy" else -1
        if order["type"] == "market":
            resting = sum((o["qty"] - o["filled_qty"]) * (1 if o["side"] == "buy" else -1) for o in self.orders.values()
                          if o["symbol"] == inst.symbol and o["type"] == "market" and o["status"] not in TERMINAL)
            after = held + resting + sign * qty
            if order.get("reduce_only") and (abs(after) > abs(held) + 1e-9 or after * held < -1e-12 or abs(held) < 1e-12):
                raise BrokerError("reduce-only order would open or reverse a position")
            if after < -1e-9 and not inst.shortable:
                raise BrokerError("short selling is not available for this instrument")
        o = {"id": f"sim-{next(self._ids)}", "cid": order["cid"], "symbol": inst.symbol, "side": order["side"], "type": order["type"], "qty": qty,
             "filled_qty": 0.0, "avg_price": None, "status": "new", "stop_price": order.get("stop_price"), "fees": 0.0,
             "created": self.w.now, "partial": fault.get("fraction") if fault and fault["kind"] == "partial" else None,
             "day": inst.market == "stock" and order["type"] == "market"}
        self.orders[o["cid"]] = o
        self.submitted_log.append(o["cid"])
        if o["type"] == "market":
            self._try_market(o)
        if fault and fault["kind"] == "timeout_after_accept":
            raise BrokerTimeout("simulated timeout: the venue accepted the order but the reply was lost")
        return dict(o)

    # ---- simulation
    def _try_market(self, o):
        inst = self.w.inst[o["symbol"]]
        i, _ = self.w._cover(o["symbol"])
        cur = self.w.bar(o["symbol"], i) if i is not None else None
        if cur is None or cur["start"] != self.w.now or o["symbol"] in self.w.suspended:
            return                                  # market closed or mid-bar: wait for the next open
        mid = float(cur["open"])
        adj = self.w.half_spread(inst, mid) + self.w.slippage(inst, mid)
        px = mid + adj if o["side"] == "buy" else mid - adj
        want = o["qty"] - o["filled_qty"]
        if o["partial"]:
            want = max(inst.round_qty(o["qty"] * o["partial"]), inst.min_qty)
            o["partial"] = None
            o["stuck"] = True                       # the remainder never fills (until cancelled)
        elif o.get("stuck"):
            return
        elif float(cur["volume"]) > 0:
            cap = inst.round_qty(float(cur["volume"]) * self.w.cfg.risk.max_volume_participation)
            want = min(want, max(cap, inst.min_qty))
        self._fill(o, inst, want, px)

    def _fill(self, o, inst, qty, px):
        if qty <= 0:
            return
        px = inst.round_price(px) if inst.market != "stock" else round(px, 4)
        sign = 1 if o["side"] == "buy" else -1
        fx = usd_per_quote(inst.quote_ccy, self.w.rates()) if inst.quote_ccy != "USD" else 1.0
        c = self.w.cfg.costs[inst.market]
        fee = qty * c.fee_per_unit + qty * px * inst.multiplier * fx * c.fee_bps / 1e4
        p = self.pos.setdefault(inst.symbol, {"qty": 0.0, "avg": 0.0})
        if inst.market in CASH_MARKETS:
            if sign > 0 and qty * px * fx + fee > self.cash + 1e-6:
                o["status"] = "rejected"
                return
            self.cash -= sign * qty * px * fx
        new = p["qty"] + sign * qty
        if p["qty"] * sign >= 0:                    # opening or adding
            p["avg"] = (abs(p["qty"]) * p["avg"] + qty * px) / abs(new) if abs(new) > 1e-12 else 0.0
        else:                                        # reducing (reversal is blocked by reduce-only / shortable checks)
            closed = min(qty, abs(p["qty"]))
            if inst.market not in CASH_MARKETS:
                self.cash += closed * inst.multiplier * (px - p["avg"]) * (1 if p["qty"] > 0 else -1) * fx
            if abs(new) > 1e-12 and new * p["qty"] < 0:
                p["avg"] = px
        p["qty"] = 0.0 if abs(new) < 1e-12 else round(new, 10)
        self.cash -= fee
        tot = o["filled_qty"] + qty
        o["avg_price"] = ((o["avg_price"] or 0.0) * o["filled_qty"] + px * qty) / tot
        o["filled_qty"], o["fees"] = round(tot, 10), o["fees"] + fee
        o["status"] = "filled" if tot >= o["qty"] - 1e-9 else "partially_filled"

    def _on_time(self):
        now = self.w.now
        for o in list(self.orders.values()):
            if o["status"] in TERMINAL:
                continue
            inst = self.w.inst[o["symbol"]]
            j = self.w.ended_now(o["symbol"])
            ended = self.w.bar(o["symbol"], j) if j is not None else None
            if o["type"] == "stop" and ended is not None and o["created"] < now and o["symbol"] not in self.w.suspended:
                op, hi, lo, stop = float(ended["open"]), float(ended["high"]), float(ended["low"]), float(o["stop_price"])
                slip = self.w.slippage(inst, stop)
                # A bar that opened at/after the order was placed and opened beyond the stop is a gap: fill at the open.
                gap_ok = ended["start"] >= o["created"]
                if o["side"] == "sell" and (lo <= stop or (gap_ok and op <= stop)):
                    self._fill(o, inst, o["qty"] - o["filled_qty"], (op if gap_ok and op <= stop else stop) - slip - self.w.half_spread(inst, stop))
                elif o["side"] == "buy" and (hi >= stop or (gap_ok and op >= stop)):
                    self._fill(o, inst, o["qty"] - o["filled_qty"], (op if gap_ok and op >= stop else stop) + slip + self.w.half_spread(inst, stop))
            elif o["type"] == "market":
                if o["day"] and ended is not None and o["created"] < now:
                    o["status"] = "canceled" if o["filled_qty"] else "expired"   # unfilled remainder of a day order
                else:
                    self._try_market(o)
        if now.hour == 21 and now.minute == 0:       # forex financing, charged once a day on open positions
            for s, p in self.pos.items():
                inst = self.w.inst[s]
                if inst.market == "forex" and abs(p["qty"]) > 1e-12 and (m := self.w.mark(s)):
                    self.cash -= abs(p["qty"]) * m * usd_per_quote(inst.quote_ccy, self.w.rates()) * self.w.cfg.costs.forex.financing_bps_per_day / 1e4

    # ---- test/replay helpers: things that happen outside the bot
    def seed_position(self, symbol: str, qty: float, avg: float):
        """A holding the bot did not open (a manual position)."""
        p = self.pos.setdefault(symbol, {"qty": 0.0, "avg": 0.0})
        inst = self.w.inst[symbol]
        if inst.market in CASH_MARKETS:
            self.cash -= qty * avg
        p["avg"] = (abs(p["qty"]) * p["avg"] + abs(qty) * avg) / max(abs(p["qty"] + qty), 1e-12)
        p["qty"] += qty
