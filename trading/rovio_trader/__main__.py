"""Rovio Trader command line.

  python -m rovio_trader replay              credential-free replay of all four markets (simulated fills)
  python -m rovio_trader serve --mode replay dashboard for the replay / paper / live journal
  python -m rovio_trader worker              background worker: scan, trade (PAPER), manage exits, reconcile
  python -m rovio_trader backtest breakout --data csv:mydata    train / validation / out-of-sample report
  python -m rovio_trader check alpaca        prove a broker connection with real responses (no orders)
  python -m rovio_trader status              per-market status table
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

from . import backtest, config as C, instruments as I, risk as R
from .brokers import FIDELITY_STATUS, IBKR_STATUS, BrokerError, BrokerTimeout, BrokerUnavailable, connect
from .data import BrokerData, CsvData
from .engine import Engine
from .events import EarningsSource, EconCalendar
from .store import Store
from .watchlist import MARKETS, Watchlist


def _db(cfg, mode: str) -> Path:
    return cfg.state_path / f"{mode.lower()}.db"      # replay, paper and live never share a journal


def _notify_fn(cfg):
    url = os.environ.get(cfg.alert_webhook_env or "")
    if not url:
        return None

    def send(kind, msg):
        req = urllib.request.Request(url, data=json.dumps({"text": f"[Rovio Trader] {kind}: {msg}"}).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=8).read()
    return send


def _calendar(cfg) -> EconCalendar:
    p = cfg.markets.forex.calendar_csv
    return EconCalendar.from_csv(p) if p else EconCalendar(status="not_configured", detail="markets.forex.calendar_csv is not set: forex entries stay paused")


def _connections(cfg) -> dict:
    p = cfg.state_path / "connections.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _record_connection(cfg, key: str, mode: str, alias: str):
    d = _connections(cfg)
    d[f"{key}:{mode}"] = {"verified_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), "account": alias}
    cfg.state_path.mkdir(parents=True, exist_ok=True)
    (cfg.state_path / "connections.json").write_text(json.dumps(d, indent=1))


# ------------------------------------------------------------------ commands
def cmd_replay(cfg, a):
    from . import replay
    res = replay.run(_db(cfg, "replay"))
    if a.serve:
        from .server import serve
        serve(str(_db(cfg, "replay")), cfg, "127.0.0.1", a.port)
    return 0 if res["ok"] else 1


def cmd_serve(cfg, a):
    from .server import serve
    db = _db(cfg, a.mode)
    if a.mode == "replay" and not db.exists():
        from . import replay
        replay.run(db, verbose=False)
    if not db.exists():
        Store(db).close()
        print(f"No {a.mode} journal yet; created an empty one. Start the worker to fill it.")
    st = Store(db)
    if st.get("mode") is None:
        st.set("mode", a.mode.upper())          # journal exists but no worker has written to it yet
    st.close()
    serve(str(db), cfg, a.host, a.port)
    return 0


def cmd_init(cfg, a):
    cfg.state_path.mkdir(parents=True, exist_ok=True)
    for mode in ("paper", "live"):
        Store(_db(cfg, mode)).close()
    print(f"State directory ready: {cfg.state_path}\n  paper.db, live.db created (replay.db is rebuilt by `replay`).")
    return 0


def cmd_worker(cfg, a):
    live = a.live
    mode = "LIVE" if live else "PAPER"
    store = Store(_db(cfg, mode))
    if live:
        if not cfg.risk.set_by_user:
            print("Refusing to start LIVE: risk.set_by_user is false. Set your own limits in the config first.", file=sys.stderr)
            return 2
        acts = store.q("SELECT * FROM activations")
        if not acts:
            print("Refusing to start LIVE: nothing is activated. Run `activate-live` for each broker / market / strategy you want.", file=sys.stderr)
            return 2
    brokers, status = connect(cfg, live=live)
    for k, s in status.items():
        print(f"  {k:<8} {'connected (' + s['mode'] + ')' if s['available'] else 'UNAVAILABLE: ' + s['reason']}")
    if not brokers:
        print("No broker is available, so there is nothing to run. Add credentials to .env (see README) or use `replay`.", file=sys.stderr)
        return 2
    engine = Engine(cfg, store, brokers, BrokerData(brokers), mode, EarningsSource.from_config(cfg), _calendar(cfg), notify=_notify_fn(cfg), watchlist=Watchlist(cfg))
    store.set("data_label", None)
    print(f"{mode} worker running. Orders that qualify are sent automatically, with no per-trade approval. Ctrl+C to stop.")
    if live:
        print("LIVE activations:", ", ".join(f"{x['broker']}/{x['account']}/{x['market']}/{x['strategy']}" for x in acts))
    seen = set()
    cal_loaded = time.time()
    while True:
        try:
            if time.time() - cal_loaded > 600:
                engine.calendar, cal_loaded = _calendar(cfg), time.time()
                store.set("calendar_status", engine.calendar.describe())
            engine.step(pd.Timestamp.now(tz="UTC"))
            for k, b in brokers.items():
                if b.connection_verified and k not in seen:
                    seen.add(k)
                    _record_connection(cfg, k, b.mode, b.account_alias)
        except KeyboardInterrupt:
            print("\nWorker stopped. Protective stop orders already at the broker remain there.")
            return 0
        except Exception as e:   # keep running: one bad cycle must not stop exit management
            try:
                engine.alert("worker_error", f"cycle failed: {type(e).__name__}: {C.redact(e)}")
            except Exception:
                pass
        if a.once:
            return 0
        try:
            time.sleep(max(cfg.poll_seconds, 5))
        except KeyboardInterrupt:
            print("\nWorker stopped. Protective stop orders already at the broker remain there.")
            return 0


def cmd_backtest(cfg, a):
    Watchlist(cfg).apply()
    names = list(C.STRATEGY_MARKET) if a.strategy == "all" else [a.strategy]
    source = "synthetic" if a.data == "synthetic" else CsvData(a.data.split(":", 1)[1])
    earnings = EarningsSource.from_config(cfg)
    cal = _calendar(cfg)
    for n in names:
        if n == "earnings" and earnings.status != "configured":
            print(f"\nearnings: SKIPPED - {earnings.detail}")
            continue
        calendar = cal
        note = None
        if n == "fx_pullback" and cal.status != "configured":
            calendar = EconCalendar([], status="configured", detail="empty")
            note = "No economic calendar supplied: the event-blackout rule was NOT applied in this backtest."
        rep = backtest.evaluate(cfg, n, source, tune=a.tune, earnings=earnings, calendar=calendar, progress=print if a.tune else (lambda *_: None))
        if note and rep.get("status") == "ok":
            rep["flags"].append(note)
        print(backtest.format_report(rep))
        if a.save and rep.get("status") == "ok":
            print(f"saved to {backtest.save_evidence(cfg, rep)}")
    return 0


def cmd_check(cfg, a):
    brokers, status = connect(cfg, live=False)
    s = status.get(a.broker)
    if a.broker == "ibkr":
        print(json.dumps(IBKR_STATUS, indent=1))
        return 1
    if a.broker == "fidelity":
        print(json.dumps(FIDELITY_STATUS, indent=1))
        return 1
    if not s or not s["available"]:
        print(f"{a.broker}: NOT CONNECTED - {s['reason'] if s else 'no enabled market uses this broker'}")
        return 1
    b = brokers[a.broker]
    try:
        acct = b.account()
        print(f"{a.broker} {b.mode}: account {b.account_alias}  equity ${acct['equity']:,.2f}  cash ${acct['cash']:,.2f}   <- real broker response")
        syms = (cfg.markets.stock.watchlist[:2] + cfg.markets.crypto.symbols) if a.broker == "alpaca" else cfg.markets.forex.pairs
        for sym in syms:
            inst = b.instrument(sym)
            if inst is None:
                print(f"  {sym}: not tradable at this broker")
                continue
            q = b.quote(inst)
            print(f"  {sym}: increment {inst.qty_step:g}, min {inst.min_qty:g}, tick {inst.tick_size:g}; open={b.market_open(inst)}; "
                  + (f"bid {q['bid']:g} ask {q['ask']:g} at {q['ts']}" if q else "no quote"))
        print(f"  open orders: {len(b.open_orders())}; positions: {len(b.positions())}")
        _record_connection(cfg, a.broker, b.mode, b.account_alias)
        print("Connection verified and recorded. No order was sent.")
        return 0
    except (BrokerError, BrokerTimeout, BrokerUnavailable) as e:
        print(f"{a.broker}: connection FAILED - {C.redact(e)}")
        return 1


def cmd_sizing(cfg, a):
    eq = 200_000.0
    view = R.portfolio_view(cfg, eq, [], [], {}, {}, {})
    acct = {"equity": eq, "cash": eq, "margin_available": eq}
    rates = {"USD/JPY": 150.0}
    cases = [("Stock (AAPL-like), long", I.stock("STOCK", "alpaca"), 1, 200.00, 194.00, None),
             ("One Micro E-mini S&P 500 (MES), long", I.future("MES", "2026-03-20", "ibkr"), 1, 6850.00, 6841.00, None),
             ("USD/JPY in a USD account, long", I.forex("USD/JPY", "oanda"), 1, 150.000, 149.700, None),
             ("Bitcoin, fractional, long", I.crypto("BTC/USD", "alpaca"), 1, 90_000.00, 87_750.00, None)]
    print(f"Sizing demo on ${eq:,.0f} equity with the example limits (max planned loss {cfg.risk.max_loss_per_trade_pct}% = ${eq * cfg.risk.max_loss_per_trade_pct / 100:,.0f} per trade).")
    for title, inst, side, entry, stop, liq in cases:
        z = R.size_position(cfg, inst, side, entry, stop, acct, view, rates, liq)
        print(f"\n{title}: entry {entry:g}, stop {stop:g}")
        for s in z["steps"]:
            print(f"   {s['limit']:<28} max {s['max_units']:>14,.4f}   {s['detail']}")
        if z["qty"] > 0:
            print(f"   => {z['qty']:g} units (rounded down to the {inst.qty_step:g} increment; binding: {z['binding']})")
            print(f"      planned loss ${z['risk_usd']:,.2f}, notional ${z['notional_usd']:,.2f}, margin/cash needed ${z['margin_usd']:,.2f}, est. costs ${z['est_costs_usd']:,.2f} (fees both ways included)")
            gap = entry - 3 * (entry - stop)
            loss = (entry - gap) * z["qty"] * inst.multiplier * z["fx"]
            print(f"      if price gaps to {gap:g} before the stop can fill: loss ${loss:,.2f} = {loss / max(z['risk_usd'], 1e-9):.1f}x the planned amount")
        else:
            print(f"   => NO TRADE: {z['reason']}")
    tight = C.load(None, {"risk": {"max_loss_per_trade_usd": 100.0}})
    z = R.size_position(tight, cases[1][1], 1, 6850.0, 6790.0, acct, view, rates)
    print(f"\nMES with a 60-point stop when the per-trade loss cap is $100\n   => NO TRADE: {z['reason']}")
    return 0


def cmd_status(cfg, a):
    conn = _connections(cfg)
    checks = {}
    rp = _db(cfg, "replay")
    if rp.exists():
        s = Store(rp)
        checks = {c["name"]: c["passed"] for c in (s.get("replay_checks") or [])}
        s.close()
    rep = lambda m: "not run" if not checks else ("yes" if checks.get(f"complete trade lifecycle: {m}") else "FAILED")
    ver = lambda k: (f"yes ({conn[k + ':PAPER']['verified_at'][:10]})" if f"{k}:PAPER" in conn else "no - never connected")
    env = os.environ
    need = lambda *keys: ", ".join(k for k in keys if not env.get(k)) or "none"
    rows = [
        ("US stocks / ETFs", "alpaca", rep("stock"), "yes", ver("alpaca"), "yes, never exercised", need("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY") + "; SIP data subscription optional; earnings history CSV for the earnings strategy"),
        ("Futures MES / MNQ", "ibkr", rep("futures"), "NO", "no", "NO", "IBKR adapter not built; needs TWS/IB Gateway paper + CME data subscription"),
        ("Spot forex", "oanda", rep("forex"), "yes", ver("oanda"), "yes, never exercised", need("OANDA_API_TOKEN", "OANDA_ACCOUNT_ID") + "; economic calendar CSV (entries pause without it)"),
        ("Bitcoin BTC/USD", "alpaca", rep("crypto"), "yes", ver("alpaca"), "yes, never exercised", need("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY") + "; crypto enabled on the Alpaca account"),
        ("Fidelity", "-", "n/a", "NO", "no", "NO", "no official retail order-entry API found; see `check fidelity`"),
    ]
    hdr = ("Market", "Broker", "Replay tested", "Paper adapter", "Paper connection verified", "Live adapter", "Remaining credentials / subscriptions")
    w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(6)]
    for r in [hdr] + rows:
        print("  ".join(str(r[i]).ljust(w[i]) for i in range(6)) + "  " + r[6])
    print("\n'Connection verified' means a real broker response was received by `check` or the worker. Simulated fills never count.")
    return 0


def cmd_control(cfg, a):
    s = Store(_db(cfg, a.mode))
    if a.cmd == "kill":
        s.set("kill", a.state == "on")
    elif a.cmd == "pause":
        s.set("pause_entries", a.state == "on")
    elif a.cmd == "resume":
        s.set("halt_drawdown", None)
        s.set("halt_protection", None)
        snap = s.get("snapshot")
        if snap:
            s.set("peak_equity", snap["equity"])
    elif a.cmd == "flatten":
        s.set("flatten_request", "all" if not a.ids else [int(i) for i in a.ids])
    s.event(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), "INFO", "operator", f"command line: {a.cmd} {getattr(a, 'state', '') or ' '.join(getattr(a, 'ids', []) or [])}")
    s.close()
    print(f"{a.cmd}: recorded in the {a.mode} journal. The worker applies it on its next cycle.")
    return 0


def cmd_watchlist(cfg, a):
    w = Watchlist(cfg)
    try:
        if a.action == "add":
            print(f"added {w.add(a.market, a.symbol, a.etf)} to {a.market}")
        elif a.action == "remove":
            print(f"removed {w.remove(a.market, a.symbol)} from {a.market}")
    except (ValueError, TypeError) as e:
        print(f"Not changed: {e}")
        return 2
    d = w.get()
    for m in MARKETS:
        print(f"  {m:<8} {', '.join(s + (' (ETF)' if m == 'stock' and s in d['etfs'] else '') for s in d[m]) or '(none)'}")
    print("The paper/live worker applies changes on its next cycle. Open positions in a removed symbol keep being managed.")
    return 0


def cmd_activate(cfg, a):
    Watchlist(cfg).apply()
    if not cfg.risk.set_by_user:
        print("Refused: set your own limits under [risk] and set risk.set_by_user = true first.")
        return 2
    if C.STRATEGY_MARKET.get(a.strategy) != a.market:
        print(f"Refused: strategy {a.strategy} trades {C.STRATEGY_MARKET.get(a.strategy)}, not {a.market}.")
        return 2
    ev = backtest.load_evidence(cfg).get(a.strategy)
    if not ev or ev["verdict"] != "eligible_for_live" or ev["strategy_hash"] != cfg.strategy_hash(a.strategy):
        why = "no backtest evidence saved" if not ev else "; ".join(ev["verdict_reasons"]) or "settings changed since the backtest"
        print(f"Refused: {a.strategy} is research-only ({why}). Run `backtest {a.strategy} --data csv:<dir> --save` on real data first.")
        return 2
    brokers, status = connect(cfg, live=True)
    b = brokers.get(a.broker)
    if b is None:
        print(f"Refused: {a.broker} live is unavailable - {status.get(a.broker, {}).get('reason', 'not used by any enabled market')}")
        return 2
    acct = b.account()
    print(f"\nLIVE ACTIVATION\n  broker   {a.broker}\n  account  {b.account_alias}  (equity ${acct['equity']:,.2f})\n  market   {a.market}\n  strategy {a.strategy} v{C.STRATEGY_VERSION[a.strategy]}")
    print("  limits   " + ", ".join(f"{k}={v}" for k, v in cfg.risk.items() if not isinstance(v, dict)))
    print("\nOnce activated, the LIVE worker sends qualifying orders with REAL MONEY automatically, without asking per trade.")
    phrase = f"ACTIVATE {a.broker} {b.account_alias} {a.market} {a.strategy}"
    if input(f"Type exactly:  {phrase}\n> ").strip() != phrase:
        print("Not activated.")
        return 1
    s = Store(_db(cfg, "live"))
    with s.lock:
        s.db.execute("INSERT OR REPLACE INTO activations VALUES(?,?,?,?,?,?,?)", (a.broker, b.account_alias, a.market, a.strategy, cfg.strategy_hash(a.strategy),
                                                                              json.dumps({k: v for k, v in cfg.risk.items()}), pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")))
    s.close()
    print("Activated. Start it with:  python -m rovio_trader worker --live")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rovio_trader", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="TOML settings file (default: config.toml if present)")
    ap.add_argument("--env-file", default=str(C.ROOT / ".env"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("replay"); p.add_argument("--serve", action="store_true"); p.add_argument("--port", type=int, default=8787)
    p = sub.add_parser("serve"); p.add_argument("--mode", choices=["replay", "paper", "live"], default="paper"); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8787)
    sub.add_parser("init")
    p = sub.add_parser("worker"); p.add_argument("--live", action="store_true"); p.add_argument("--once", action="store_true")
    p = sub.add_parser("backtest"); p.add_argument("strategy", choices=list(C.STRATEGY_MARKET) + ["all"]); p.add_argument("--data", default="synthetic", help="synthetic | csv:<directory>")
    p.add_argument("--tune", action="store_true"); p.add_argument("--save", action="store_true", help="record the result as this strategy's evidence")
    p = sub.add_parser("check"); p.add_argument("broker", choices=["alpaca", "oanda", "ibkr", "fidelity"])
    sub.add_parser("sizing-demo")
    sub.add_parser("status")
    for name in ("kill", "pause"):
        p = sub.add_parser(name); p.add_argument("state", choices=["on", "off"]); p.add_argument("--mode", choices=["paper", "live"], default="paper")
    p = sub.add_parser("resume"); p.add_argument("--mode", choices=["paper", "live"], default="paper")
    p = sub.add_parser("flatten"); p.add_argument("ids", nargs="*"); p.add_argument("--mode", choices=["paper", "live"], default="paper")
    p = sub.add_parser("watchlist"); p.add_argument("action", choices=["list", "add", "remove"]); p.add_argument("market", nargs="?", choices=list(MARKETS))
    p.add_argument("symbol", nargs="?"); p.add_argument("--etf", action="store_true", help="stock is an ETF (excluded from the earnings strategy)")
    p = sub.add_parser("activate-live"); p.add_argument("--broker", required=True); p.add_argument("--market", required=True); p.add_argument("--strategy", required=True)
    a = ap.parse_args(argv)
    C.load_env_file(Path(a.env_file))
    path = a.config or (str(C.ROOT / "config.toml") if (C.ROOT / "config.toml").exists() else None)
    try:
        cfg = C.load(path)
    except (ValueError, OSError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    fn = {"replay": cmd_replay, "serve": cmd_serve, "init": cmd_init, "worker": cmd_worker, "backtest": cmd_backtest, "check": cmd_check,
          "sizing-demo": cmd_sizing, "status": cmd_status, "kill": cmd_control, "pause": cmd_control, "resume": cmd_control, "flatten": cmd_control,
          "activate-live": cmd_activate, "watchlist": cmd_watchlist}[a.cmd]
    return fn(cfg, a)


if __name__ == "__main__":
    sys.exit(main())
