"""Credential-free REPLAY: the whole signal -> risk -> order -> fill -> protection -> exit flow in all four
markets, driven by scripted fixture prices (see fixtures.py) through the same engine used for paper and
live trading. Every fill here is SIMULATED. It verifies the plumbing, not profitability."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config, fixtures as F, simrun
from .engine import Engine
from .store import ACTIVE, Store

U = lambda s: pd.Timestamp(s, tz="UTC")


def run(db_path: str | Path, verbose: bool = True, extra_overrides: dict | None = None) -> dict:
    db_path = Path(db_path)
    for suffix in ("", "-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)       # the replay database is rebuilt from scratch every run
    cfg = config.load(None, F.OVERRIDES)
    if extra_overrides:
        cfg = config.load(None, {**F.OVERRIDES, **extra_overrides})
    insts, exec_bars, data, earnings, calendar = F.build(cfg)
    world, brokers, store, engine = simrun.make(cfg, insts, exec_bars, data, "REPLAY", earnings, calendar, store=Store(db_path))
    alp = brokers["alpaca"]
    store.set("data_label", F.LABEL)
    store.set("replay", {"start": str(F.START), "end": str(F.END), "note": "All fills simulated from scripted fixture prices."})

    # Things that happen outside the bot: two manual holdings at the same broker ...
    alp.seed_position("MANL", 50, 100.0)       # not on the watchlist
    alp.seed_position("RPLI", 30, 100.0)       # on the watchlist: creates an ownership conflict
    # ... and two scripted broker behaviours.
    alp.faults += [{"symbol": "RPLC", "role": "entry", "kind": "timeout_after_accept"},
                   {"symbol": "RPLD", "role": "entry", "kind": "partial", "fraction": 0.4}]

    seen, checks = {}, []

    def restart(why: str) -> Engine:
        nonlocal store, engine
        store.close()
        store = Store(db_path)                  # a new process would see only what was persisted
        now = engine.now
        engine = Engine(cfg, store, brokers, data, "REPLAY", earnings, calendar)
        engine.now = now
        engine.log("restart", f"SIMULATED WORKER RESTART ({why}): in-memory state discarded, journal reopened")
        return engine

    for t in world.timeline(F.START, F.END):
        if t == U("2025-12-11 21:00"):
            engine.flatten()                    # explicit user action, requested while the stock market is closed
        world.advance(t)
        calendar.asof = t                       # the fixture calendar counts as freshly downloaded
        engine.step(t)
        if t == U("2025-12-03 14:30"):          # RPLC entry: venue accepted it, reply lost
            o = store.one("SELECT * FROM orders WHERE symbol='RPLC' AND role='entry'")
            seen["timeout_status"] = o["status"]
            seen["timeout_broker_has_it"] = alp.get_by_cid(o["cid"]) is not None
            restart("after the lost broker reply")
        if t == U("2025-12-06 00:00"):
            seen["btc_weekday"] = t.day_name()
            seen["stocks_open_at_btc_entry"] = world.is_open("RPLD")
        if t == U("2025-12-09 14:30"):          # loss limit has just tripped
            seen["halt_before_restart"] = store.get("halt_daily")
            restart("while the daily loss halt is active")
            seen["halt_after_restart"] = store.get("halt_daily")
            stops = store.active_orders(symbol="RPLD", role="stop")
            seen["rpld_stop_live"] = bool(stops) and all((alp.get_by_cid(s["cid"]) or {}).get("status") == "new" for s in stops)
        if t == U("2025-12-11 21:00"):
            seen["flatten_blocked"] = dict(store.get("exit_blocked", {}))

    q, one = store.q, store.one
    closed = q("SELECT * FROM trades WHERE status='closed'")
    sig = lambda i: one("SELECT * FROM signals WHERE id LIKE ?", (i + "%",))

    def check(name, ok, evidence):
        checks.append({"name": name, "passed": bool(ok), "evidence": evidence})

    for m in ("stock", "futures", "forex", "crypto"):
        t = next((x for x in closed if x["market"] == m and x["exit_avg"]), None)
        check(f"complete trade lifecycle: {m}", t is not None,
              t and f"{t['strategy']} {t['symbol']} {'long' if t['side'] > 0 else 'short'} {t['qty_total']:g} in @ {t['entry_avg']:g}, out @ {t['exit_avg']:g} "
                    f"({t['exit_reason']}), net ${t['realized'] - t['fees']:,.2f}")
    e = sig("earnings:RPLE")
    check("earnings example traded, with its evidence", e["status"] == "traded" and e["meta"].get("eps_actual") == 1.32, e["why"])
    check("skipped trade with exact reason (opening gap)", sig("breakout:RPLB")["status"] == "skipped", sig("breakout:RPLB")["decision"])
    check("earnings skip when the release time is unverified", sig("earnings:RPLF")["status"] == "skipped", sig("earnings:RPLF")["decision"])
    check("forex entry blocked by calendar blackout", sig("fx_pullback:EUR/USD")["status"] == "skipped", sig("fx_pullback:EUR/USD")["decision"])
    roll = q("SELECT msg FROM events WHERE kind='rollover' AND msg LIKE '%MES:%'")
    check("futures rollover is explicit", bool(roll) and one("SELECT 1 FROM trades WHERE symbol='MESH6'") is not None, roll[0]["msg"] if roll else None)
    j = sig("fx_pullback:USD/JPY")
    step0 = j["sizing"]["steps"][0]["detail"] if j["sizing"] else ""
    check("USD/JPY sized in a USD account", j["status"] == "traded" and "USD/JPY" in step0, f"{j['sizing']['qty']:g} units; {step0}; binding: {j['sizing']['binding']}")
    b = one("SELECT * FROM trades WHERE symbol='BTC/USD'")
    check("fractional Bitcoin fill", b and b["qty_total"] % 1 != 0, b and f"{b['qty_total']:g} BTC @ {b['entry_avg']:,.2f}")
    check("Bitcoin traded on a weekend while equities were closed", seen.get("btc_weekday") == "Saturday" and seen.get("stocks_open_at_btc_entry") is False,
          f"entry {b['opened']} ({seen.get('btc_weekday')} UTC); stock market open: {seen.get('stocks_open_at_btc_entry')}")
    n_broker = sum(1 for o in alp.orders.values() if o["symbol"] == "RPLC" and o["side"] == "buy")
    n_journal = len(q("SELECT 1 FROM orders WHERE symbol='RPLC' AND role='entry'"))
    check("timeout after acceptance + restart: exactly one entry", seen.get("timeout_status") == "uncertain" and seen.get("timeout_broker_has_it") and n_broker == 1 and n_journal == 1,
          f"journal status after the lost reply: {seen.get('timeout_status')}; venue had the order: {seen.get('timeout_broker_has_it')}; "
          f"entry orders at venue: {n_broker}; in journal: {n_journal}")
    d_entry = one("SELECT * FROM orders WHERE symbol='RPLD' AND role='entry'")
    d_trade = one("SELECT * FROM trades WHERE symbol='RPLD'")
    d_stops = q("SELECT * FROM orders WHERE symbol='RPLD' AND role='stop' ORDER BY created")
    check("partial fill then cancellation: protection equals filled units",
          d_entry["status"] == "canceled" and 0 < d_entry["filled_qty"] < d_entry["qty"] and d_stops and d_stops[0]["qty"] == d_entry["filled_qty"] == d_trade["qty_total"],
          f"entry {d_entry['filled_qty']:g} of {d_entry['qty']:g} filled, remainder {d_entry['status']}; protective stop for {d_stops[0]['qty']:g}")
    g, h = sig("breakout:RPLG"), sig("earnings:RPLH")
    check("loss limit halts two simultaneous strategy signals", g["status"] == h["status"] == "skipped" and "daily loss limit" in g["decision"] and "daily loss limit" in h["decision"],
          g["decision"])
    check("protective exit stays active during the halt", seen.get("rpld_stop_live"), "RPLD stop order still working at the venue after the halt")
    check("halt survives a restart", bool(seen.get("halt_before_restart")) and seen.get("halt_before_restart") == seen.get("halt_after_restart"), seen.get("halt_after_restart"))
    c = one("SELECT * FROM trades WHERE symbol='RPLC'")
    check("gap produced a loss larger than the planned stop risk", -(c["realized"] - c["fees"]) > c["planned_risk"] * 1.5,
          f"planned ${c['planned_risk']:,.2f}, actual ${c['realized'] - c['fees']:,.2f} ({c['exit_reason']})")
    pos = alp.positions()
    check("flatten leaves manual holdings alone", pos.get("MANL", {}).get("qty") == 50 and pos.get("RPLI", {}).get("qty") == 30 and "RPLD" not in pos
          and d_trade["exit_reason"] == "manual flatten", f"after flatten-all: MANL {pos.get('MANL', {}).get('qty')}, RPLI {pos.get('RPLI', {}).get('qty')} (manual), RPLD closed")
    check("flatten reports positions it cannot close yet", any("market is closed" in v for v in seen.get("flatten_blocked", {}).values()),
          "; ".join(seen.get("flatten_blocked", {}).values()))
    check("manual holding in a watched symbol blocks automation there", sig("breakout:RPLI")["status"] == "skipped" and "manual holding" in sig("breakout:RPLI")["decision"],
          sig("breakout:RPLI")["decision"])
    both = q("SELECT strategy, status, sizing FROM signals WHERE symbol='RPLJ' ORDER BY strategy")
    second = both[1]["sizing"] if len(both) == 2 else None
    check("same stock flagged by both strategies: both recorded, one symbol limit",
          len(both) == 2 and second and second["binding"] == "instrument concentration" and second["qty"] < both[0]["sizing"]["qty"],
          f"signals: {[(x['strategy'], x['status']) for x in both]}; breakout took {both[0]['sizing']['qty']:g} shares, earnings was cut to "
          f"{second['qty']:g} by the shared '{second['binding']}' limit")
    stray = {s: p["qty"] for s, p in {**alp.positions(), **brokers["oanda"].positions(), **brokers["ibkr"].positions()}.items() if s not in ("MANL", "RPLI")}
    check("no leftover or reversed positions", not stray and not store.active_orders(), f"bot positions at venues: {stray or 'none'}; working orders: {len(store.active_orders())}")
    store.set("replay_checks", checks)

    if verbose:
        print("=" * 100)
        print("REPLAY MODE - scripted fixture prices, SIMULATED fills. No credentials, no broker, no market data.")
        print("=" * 100)
        print("\nDECISIONS")
        for s in q("SELECT * FROM signals ORDER BY time, id"):
            print(f"  {s['time'][:16].replace('T', ' ')}  {s['strategy']:<12} {s['symbol']:<8} {'LONG ' if s['side'] > 0 else 'SHORT'} {s['status']:<8} {(s['decision'] or '')[:170]}")
        print("\nTRADES (simulated)")
        for t in q("SELECT * FROM trades ORDER BY id"):
            print(f"  #{t['id']:<2} {t['strategy']:<12} {t['symbol']:<8} {'long ' if t['side'] > 0 else 'short'} {t['qty_total']:>10g} in {t['entry_avg']:>10g} "
                  f"out {t['exit_avg'] or 0:>10g}  net ${t['realized'] - t['fees']:>9,.2f}  {t['exit_reason']}")
        print("\nKEY EVENTS")
        for ev in q("SELECT * FROM events WHERE level='ALERT' OR kind IN ('rollover','flatten','exit_waiting','order_uncertain','order_resend','partial_fill','restart','risk_limit')"):
            print(f"  {ev['ts'][:16].replace('T', ' ')}  {ev['kind']:<18} {ev['msg'][:210]}")
        print("\nRESULTS BY STRATEGY (fixture outcomes; not evidence of an edge)")
        for r in q("SELECT strategy, market, COUNT(*) n, SUM(realized-fees) net FROM trades WHERE status='closed' GROUP BY strategy, market"):
            print(f"  {r['strategy']:<12} {r['market']:<8} trades {r['n']}  net ${r['net']:,.2f}")
        print("\nACCEPTANCE CHECKS")
        for c in checks:
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}\n         {c['evidence']}")
        print(f"\n{sum(c['passed'] for c in checks)}/{len(checks)} checks passed. Journal: {db_path}")
    store.close()
    return {"checks": checks, "ok": all(c["passed"] for c in checks), "db": str(db_path)}
