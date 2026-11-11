"""Command line: python -m rovio_trader [--demo] <serve|scan|analyze|backtest|propose|run|portfolio>"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import backtest
from .brokers import BrokerError
from .config import ROOT, Settings, load_env_file
from .data import DataError
from .engine import build


def _pct(x):
    return "   n/a" if x is None else f"{x * 100:6.1f}%"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rovio_trader", description=__doc__)
    ap.add_argument("--demo", action="store_true", help="use synthetic (fake) prices; no network, no real broker")
    ap.add_argument("--broker", choices=["paper", "manual", "alpaca"])
    ap.add_argument("--env-file", default=str(ROOT / ".env"))
    ap.add_argument("--state-dir")
    sub = ap.add_subparsers(dest="cmd")
    sv = sub.add_parser("serve", help="run the web dashboard (default)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8787)
    sub.add_parser("scan", help="print signals for the watchlist")
    an = sub.add_parser("analyze")
    an.add_argument("symbol")
    bt = sub.add_parser("backtest")
    bt.add_argument("symbol")
    bt.add_argument("--strategy", default="all")
    bt.add_argument("--range", default="5y")
    sub.add_parser("propose", help="generate order proposals for review")
    rn = sub.add_parser("run", help="scan on a timer; with --auto also execute proposals")
    rn.add_argument("--every", type=int, default=900, help="seconds between scans")
    rn.add_argument("--auto", action="store_true", help="execute proposals without asking")
    rn.add_argument("--confirm-live", action="store_true", help="required for --auto when real money is at risk")
    rn.add_argument("--once", action="store_true")
    sub.add_parser("portfolio")
    a = ap.parse_args(argv)

    load_env_file(Path(a.env_file))
    try:
        eng = build(Settings.from_env(demo=a.demo or None, broker=a.broker, state_dir=Path(a.state_dir) if a.state_dir else None))
    except (ValueError, BrokerError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    if a.demo:
        print("DEMO MODE: prices are synthetic. Nothing here reflects a real market.\n")
    cmd = a.cmd or "serve"
    try:
        if cmd == "serve":
            from .server import serve
            serve(eng, getattr(a, "host", "127.0.0.1"), getattr(a, "port", 8787))
        elif cmd == "scan":
            for r in eng.scan():
                print(f"{r['symbol']:10s} ERROR {r['error']}" if "error" in r else
                      f"{r['symbol']:10s} {r['action']:12s} score {r['score']:+6.1f}  price {r['price']:<12g} RSI {r['indicators']['rsi'] or 0:4.0f}  stop {r['levels']['stop']:g}")
        elif cmd == "analyze":
            r = eng.analyze(a.symbol)
            print(f"{r['symbol']} ({r['asset_class']}): {r['action']}  score {r['score']:+.1f}  price {r['price']:g}")
            for c in r["components"]:
                print(f"  {c['points']:+6.1f}  {c['name']:10s} {c['reason']}")
            for w in r["warnings"] + ([r["note"]] if r.get("note") else []):
                print(f"  ! {w}")
        elif cmd == "backtest":
            df = eng.provider.history(a.symbol, a.range, "1d")
            names = list(backtest.STRATEGIES) if a.strategy == "all" else [a.strategy]
            print(f"{a.symbol} {a.range}, fills at next open, costs included\n{'strategy':15s} {'return':>7s} {'CAGR':>7s} {'maxDD':>7s} {'Sharpe':>6s} {'trades':>6s} {'win%':>7s} {'OOS ret':>7s}")
            for n in names:
                r = backtest.run(df, n)
                s = r["stats"]
                print(f"{n:15s} {_pct(s['total_return'])} {_pct(s['cagr'])} {_pct(s['max_drawdown'])} {s['sharpe'] or 0:6.2f} {s['trades']:6d} {_pct(s['win_rate'])} {_pct(r['out_of_sample'].get('total_return'))}")
        elif cmd == "propose":
            for p in eng.generate_proposals() or [None]:
                print("No trades proposed." if p is None else f"{p['id']}  {p['side'].upper():4s} {p['qty']:g} {p['symbol']} ~${p['est_value']:,.2f}  ({p['action']} {p['score']:+g})  {p['reason']}")
        elif cmd == "run":
            if a.auto and eng.broker.live and not a.confirm_live:
                print("Refusing --auto against a LIVE account without --confirm-live.", file=sys.stderr)
                return 2
            if a.auto and not eng.broker.executes:
                print("The manual broker cannot execute; proposals become tickets for you to place.")
            while True:
                out = eng.run_cycle(auto=a.auto)
                print(time.strftime("%Y-%m-%d %H:%M:%S"), f"{len(out['proposals'])} proposal(s)", *[f"\n  {'OK ' if x['ok'] else 'ERR'} {x['side']} {x['qty']:g} {x['symbol']} {x.get('error', '')}" for x in out["executed"]])
                if a.once:
                    break
                time.sleep(max(a.every, 60))
        elif cmd == "portfolio":
            p = eng.portfolio()
            print(f"Broker {eng.broker.name}: equity ${p['account']['equity']:,.2f}  cash ${p['account']['cash']:,.2f}")
            for x in p["positions"]:
                print(f"  {x['symbol']:10s} {x['qty']:<12g} avg {x['avg_price']:<12g} last {x['price']:<12g} P&L ${x['pnl']:,.2f}")
    except (DataError, BrokerError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
