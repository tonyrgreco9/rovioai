"""Portfolio risk view and position sizing. One USD view across all broker accounts; cash and margin
are checked per broker because capital at one broker cannot fund an order at another."""
from __future__ import annotations

from .instruments import Instrument, group_symbol, usd_per_quote


def est_cost_per_unit(cfg, inst: Instrument, price: float, fx: float) -> float:
    """Estimated round-trip cost in USD for one unit: spread + slippage both ways, plus fees both ways."""
    c = cfg.costs[inst.market]
    if inst.market == "futures":
        px = (c.half_spread_ticks + c.slippage_ticks) * inst.tick_size
    elif inst.market == "forex":
        px = (c.half_spread_pips + c.slippage_pips) * inst.pip
    else:
        px = price * (c.half_spread_bps + c.slippage_bps) / 1e4
    return 2 * (px * inst.multiplier * fx + c.fee_per_unit + price * inst.multiplier * fx * c.fee_bps / 1e4)


def portfolio_view(cfg, equity: float, trades: list[dict], pending: list[dict], insts: dict, marks: dict, rates: dict) -> dict:
    """Exposure and risk from open trades AND pending entry orders (their reservations)."""
    v = {"equity": equity, "open_risk": 0.0, "gross": 0.0, "market": {}, "broker": {}, "instrument": {}, "currency": {}, "positions": 0,
         "cash_reserved": {}, "margin_reserved": {}}
    add = lambda d, k, x: d.__setitem__(k, d.get(k, 0.0) + x)
    for t in trades:
        inst = insts.get(t["symbol"])
        if inst is None or t["qty"] <= 0:
            continue
        fx = usd_per_quote(inst.quote_ccy, rates)
        px = marks.get(t["symbol"]) or t["entry_avg"]
        n = t["qty"] * inst.multiplier * px * fx
        v["gross"] += n
        add(v["market"], inst.market, n); add(v["broker"], inst.broker, n); add(v["instrument"], group_symbol(inst), n)
        if t["status"] == "open":
            v["open_risk"] += abs(t["entry_avg"] - t["stop"]) * t["qty"] * inst.multiplier * fx
        if inst.market == "forex":
            add(v["currency"], inst.base_ccy, t["side"] * t["qty"] * px * fx)
            add(v["currency"], inst.quote_ccy, -t["side"] * t["qty"] * px * fx)
        v["positions"] += 1
    for o in pending:
        inst = insts.get(o["symbol"])
        if inst is None:
            continue
        frac = max(o["qty"] - o["filled_qty"], 0.0) / o["qty"] if o["qty"] else 0.0
        n = (o["reserved_notional"] or 0.0) * frac
        v["open_risk"] += (o["reserved_risk"] or 0.0) * frac
        v["gross"] += n
        add(v["market"], inst.market, n); add(v["broker"], inst.broker, n); add(v["instrument"], group_symbol(inst), n)
        if inst.market in ("stock", "crypto"):
            add(v["cash_reserved"], inst.broker, n)
        else:
            add(v["margin_reserved"], inst.broker, (max(o["qty"] - o["filled_qty"], 0.0) * inst.margin_per_unit) if inst.market == "futures" else n * inst.margin_rate)
        if o["filled_qty"] <= 0:
            v["positions"] += 1
    return v


def size_position(cfg, inst: Instrument, side: int, entry: float, stop: float, account: dict, view: dict, rates: dict,
                  liquidity_units: float | None = None) -> dict:
    """Largest quantity allowed by the risk budget and every limit, rounded DOWN to the broker increment.

    Returns {"qty", "risk_usd", "notional_usd", "est_costs_usd", "steps": [...], "reason"}; qty 0 carries the reason.
    Note: `risk_usd` is the PLANNED loss at the stop. A gap or slippage through the stop can lose more.
    """
    r, eq = cfg.risk, view["equity"]
    fx = usd_per_quote(inst.quote_ccy, rates)
    stop_dist = abs(entry - stop)
    cost = est_cost_per_unit(cfg, inst, entry, fx)
    risk_unit = stop_dist * inst.multiplier * fx + cost
    npu = entry * inst.multiplier * fx
    steps = []

    def cap(name, units, detail):
        steps.append({"limit": name, "max_units": units, "detail": detail})

    budget = eq * r.max_loss_per_trade_pct / 100
    if r.max_loss_per_trade_usd > 0:
        budget = min(budget, r.max_loss_per_trade_usd)
    remaining = eq * r.max_open_risk_pct / 100 - view["open_risk"]
    risk_detail = (f"stop distance {stop_dist:g} x multiplier {inst.multiplier:g} x {fx:.6g} USD/{inst.quote_ccy} = ${stop_dist * inst.multiplier * fx:,.4f} "
                   f"+ est. round-trip costs ${cost:,.4f} = ${risk_unit:,.4f} per unit")
    if inst.market == "futures":
        ticks = stop_dist / inst.tick_size
        risk_detail = f"{ticks:g} ticks x ${inst.tick_value * fx:g} tick value = ${ticks * inst.tick_value * fx:,.2f} + est. costs ${cost:,.2f} = ${risk_unit:,.2f} per contract"
    cap("planned loss per trade", budget / risk_unit, f"budget ${budget:,.2f} / ({risk_detail})")
    cap("total open portfolio risk", remaining / risk_unit, f"remaining ${max(remaining, 0):,.2f} of ${eq * r.max_open_risk_pct / 100:,.2f}")
    gs = group_symbol(inst)
    cap("instrument concentration", (eq * r.max_instrument_pct / 100 - view["instrument"].get(gs, 0.0)) / npu, f"{r.max_instrument_pct:g}% of equity in {gs}")
    for name, g in r.groups.items():
        if gs in g["symbols"]:
            used = sum(view["instrument"].get(s, 0.0) for s in g["symbols"])
            cap(f"correlated group '{name}'", (eq * g["max_pct"] / 100 - used) / npu, f"{g['max_pct']:g}% of equity across {', '.join(g['symbols'])}; ${used:,.0f} in use")
    if inst.market in r.per_market_pct:
        cap(f"market '{inst.market}'", (eq * r.per_market_pct[inst.market] / 100 - view["market"].get(inst.market, 0.0)) / npu, f"{r.per_market_pct[inst.market]:g}% of equity")
    if inst.broker in r.per_broker_pct:
        cap(f"broker '{inst.broker}'", (eq * r.per_broker_pct[inst.broker] / 100 - view["broker"].get(inst.broker, 0.0)) / npu, f"{r.per_broker_pct[inst.broker]:g}% of equity")
    cap("gross exposure / leverage", (eq * r.max_gross_exposure_pct / 100 - view["gross"]) / npu, f"{r.max_gross_exposure_pct:g}% of equity; ${view['gross']:,.0f} in use")
    cap("order size", r.max_order_notional_usd / npu, f"${r.max_order_notional_usd:,.0f} max per order")
    if inst.market == "forex":
        lim = eq * r.max_currency_exposure_pct / 100
        room = min(lim - side * view["currency"].get(inst.base_ccy, 0.0) if inst.base_ccy != "USD" else float("inf"),
                   lim + side * view["currency"].get(inst.quote_ccy, 0.0) if inst.quote_ccy != "USD" else float("inf"))
        cap("currency exposure", room / npu, f"{r.max_currency_exposure_pct:g}% of equity net per non-USD currency")
    if inst.market in ("stock", "crypto"):
        cash = account["cash"] - view["cash_reserved"].get(inst.broker, 0.0)
        cap(f"cash at {inst.broker}", cash * 0.995 / (npu + cost / 2), f"${cash:,.2f} available (capital at other brokers cannot fund this order)")
    else:
        m = account["margin_available"] - view["margin_reserved"].get(inst.broker, 0.0)
        per = inst.margin_per_unit if inst.market == "futures" else npu * inst.margin_rate
        cap(f"margin at {inst.broker}", m / per if per > 0 else float("inf"), f"${m:,.2f} available, ${per:,.2f} per unit")
    if liquidity_units is not None:
        cap("liquidity", liquidity_units, f"{r.max_volume_participation:.2%} of the signal bar's volume")
    binding = min(steps, key=lambda s: s["max_units"])
    qty = inst.round_qty(max(binding["max_units"], 0.0))
    out = {"qty": qty, "entry": entry, "stop": stop, "risk_per_unit_usd": risk_unit, "risk_usd": qty * risk_unit, "notional_usd": qty * npu,
           "est_costs_usd": qty * cost, "margin_usd": qty * (inst.margin_per_unit if inst.market == "futures" else npu * inst.margin_rate if inst.market == "forex" else npu),
           "fx": fx, "steps": steps, "binding": binding["limit"], "reason": None}
    if qty < inst.min_qty:
        out["qty"] = 0.0
        if binding["limit"] in ("planned loss per trade", "total open portfolio risk"):
            out["reason"] = (f"smallest legal position ({inst.min_qty:g} {'contract' if inst.market == 'futures' else 'unit'}) risks ${inst.min_qty * risk_unit:,.2f}, "
                             f"above the {binding['limit']} budget of ${max(binding['max_units'], 0) * risk_unit:,.2f}")
        else:
            out["reason"] = f"no room under the '{binding['limit']}' limit ({binding['detail']})"
    return out
