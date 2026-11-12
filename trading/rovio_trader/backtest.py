"""Backtests run the SAME engine, strategies, sizing and risk code as replay/paper/live, against the
simulated venue. Results are split chronologically into train / validation / out-of-sample.

What this can and cannot tell you:
  * Costs modelled: spread, slippage, fees (per market), forex financing, gap-through-stop fills.
  * Not modelled: taxes, borrow costs, market impact beyond the participation cap, intrabar order of
    stop vs target (the stop is assumed to win), exchange outages.
  * A fixed watchlist chosen today has SURVIVORSHIP BIAS: it excludes names that failed or were
    delisted, which flatters every long strategy. Results are flagged accordingly.
  * Synthetic data is for checking that the pipeline runs. It is never evidence.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path

import pandas as pd

from . import config as C, instruments as I, simrun
from .data import FrameData
from .strategies import ALL

GRID = {  # small, pre-declared grids; tuned on TRAIN, chosen on VALIDATION, reported on untouched OOS
    "breakout": {"lookback": [20, 55], "stop_atr_mult": [2.0, 3.0]},
    "earnings": {"min_surprise_pct": [0.0, 5.0]},
    "orb": {"range_minutes": [15, 30], "target_r": [1.5, 2.0]},
    "fx_pullback": {"stop_atr_mult": [1.5, 2.0], "target_r": [1.5, 2.5]},
    "btc_breakout": {"lookback": [20, 40], "stop_atr_mult": [2.0, 3.0]},
}


def dataset(cfg, strategy: str, source, synthetic_end: str | None = None):
    """Instruments + bars for one strategy's market. `source` is a data object, or 'synthetic'."""
    from .data import synthetic
    market, tf = C.STRATEGY_MARKET[strategy], ALL[strategy].timeframe
    m = cfg.markets[market]
    if market == "stock":
        insts = [I.stock(s, m.broker, s in m.etfs) for s in m.watchlist]
    elif market == "forex":
        insts = [I.forex(s, m.broker) for s in m.pairs]
    elif market == "crypto":
        insts = [I.crypto(s, m.broker) for s in m.symbols]
    else:   # research only: one continuous series per root, modelled as a far-dated contract
        insts = []
        for root in m.roots:
            c = I.future(root, "2099-12-18", m.broker)
            c.symbol = root
            insts.append(c)
    frames, kept = {}, []
    periods = {"1d": 1500, "1h": 9000, "4h": 4000, "5m": 78 * 250}[tf]
    for inst in insts:
        df = synthetic(inst.symbol, market, tf, synthetic_end or "2025-12-12", periods) if source == "synthetic" else source.bars(inst, tf)
        if df is not None and len(df) > 100:
            frames[(inst.symbol, tf)] = df
            kept.append(inst)
    label = "SYNTHETIC random walk (pipeline check only, not evidence)" if source == "synthetic" else source.label
    return kept, frames, label, [i.symbol for i in insts if i not in kept]


def _scale_costs(cfg, mult: float):
    c = copy.deepcopy(dict(cfg))
    for mk, d in c["costs"].items():
        c["costs"][mk] = {k: v * mult for k, v in d.items()}
    return C._wrap(json.loads(json.dumps(c)))


def simulate(cfg, strategy: str, insts, frames, start, end, earnings=None, calendar=None) -> dict:
    """One run over [start, end] with a fresh account. History before `start` is visible for warm-up only."""
    tf = ALL[strategy].timeframe
    exec_bars = {s: df for (s, _), df in frames.items()}
    world, brokers, store, engine = simrun.make(cfg, insts, exec_bars, FrameData(frames, "backtest"), "BACKTEST", earnings, calendar,
                                                only=[strategy], record_scanner=False)
    if calendar is not None:
        calendar.sim_always_fresh = True
    simrun.drive(world, engine, world.timeline(start, end))
    trades = store.q("SELECT * FROM trades WHERE status='closed' ORDER BY id")
    eq = pd.Series({pd.Timestamp(r["ts"]): r["equity"] for r in store.q("SELECT ts, equity FROM equity ORDER BY ts")})
    start_eq = sum(cfg.portfolio.sim_starting_cash.get(k, 0.0) for k in brokers)
    net = [t["realized"] - t["fees"] for t in trades]
    wins, losses = [x for x in net if x > 0], [x for x in net if x <= 0]
    # per-trade return on the trade's own notional, for the t-statistic
    rets = [(t["realized"] - t["fees"]) / max(t["qty_total"] * t["entry_avg"] * next(i for i in insts if i.symbol == t["symbol"]).multiplier, 1e-9) for t in trades]
    mean = sum(rets) / len(rets) if rets else 0.0
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)) if len(rets) > 1 else 0.0
    held = sum((pd.Timestamp(t["closed"]) - pd.Timestamp(t["opened"])).total_seconds() for t in trades)
    span = max((end - start).total_seconds(), 1.0)
    by = lambda key: {k: {"trades": len(v), "net": round(sum(v), 2)} for k, v in
                      {k: [t["realized"] - t["fees"] for t in trades if key(t) == k] for k in sorted({key(t) for t in trades})}.items()}
    bh = []
    for inst in insts:
        df = frames[(inst.symbol, tf)]
        d = df[(df.index >= start) & (df.index <= end)]
        if len(d) > 1:
            bh.append(float(d["close"].iloc[-1] / d["open"].iloc[0] - 1))
    out = {
        "start": str(start), "end": str(end), "trades": len(trades), "net_pnl": round(sum(net), 2),
        "avg_net_per_trade": round(sum(net) / len(net), 2) if net else None, "win_rate": round(len(wins) / len(net), 3) if net else None,
        "avg_win": round(sum(wins) / len(wins), 2) if wins else None, "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
        "return_pct": round(sum(net) / start_eq * 100, 2) if start_eq else None,
        "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 2) if len(eq) else 0.0,
        "exposure_pct": round(held / span * 100, 1), "turnover_x": round(sum(2 * t["qty_total"] * t["entry_avg"] for t in trades) / start_eq, 2) if start_eq else None,
        "mean_trade_return_pct": round(mean * 100, 4), "t_stat": round(mean / (sd / math.sqrt(len(rets))), 2) if sd > 0 else None,
        "by_symbol": by(lambda t: t["symbol"]), "by_direction": by(lambda t: "long" if t["side"] > 0 else "short"),
        "benchmark_buy_hold_pct": round(sum(bh) / len(bh) * 100, 2) if bh else None,
        "signals": {r["status"]: r["n"] for r in store.q("SELECT status, COUNT(*) n FROM signals GROUP BY status")},
    }
    store.close()
    return out


def evaluate(cfg, strategy: str, source="synthetic", tune: bool = False, earnings=None, calendar=None, progress=print) -> dict:
    insts, frames, label, missing = dataset(cfg, strategy, source)
    if not insts:
        return {"strategy": strategy, "status": "no_data", "detail": f"No bars available for {strategy} from {label}.", "missing": missing}
    tf = ALL[strategy].timeframe
    t0 = max(df.index[min(len(df) - 1, 260)] for df in frames.values())     # leave warm-up history before the first test bar
    t1 = min(df.index[-1] for df in frames.values())
    ev = cfg.evidence
    a = t0 + (t1 - t0) * ev.train_fraction
    b = a + (t1 - t0) * ev.validation_fraction
    run = lambda c, s, e: simulate(c, strategy, insts, frames, s, e, earnings, calendar)
    chosen, tuning = cfg, None
    if tune:
        keys = list(GRID[strategy])
        cands = []
        for combo in itertools.product(*GRID[strategy].values()):
            c = C.load(None, _plain_overrides(cfg, {"strategies": {strategy: dict(zip(keys, combo))}}))
            tr = run(c, t0, a)
            cands.append((dict(zip(keys, combo)), c, tr))
            progress(f"  train {dict(zip(keys, combo))}: {tr['trades']} trades, avg ${tr['avg_net_per_trade']}")
        top = sorted([x for x in cands if x[2]["trades"] >= 10], key=lambda x: -(x[2]["avg_net_per_trade"] or -1e18))[:3] or cands[:1]
        scored = [(p, c, tr, run(c, a, b)) for p, c, tr in top]
        best = max(scored, key=lambda x: (x[3]["avg_net_per_trade"] if x[3]["trades"] else -1e18) or -1e18)
        chosen = best[1]
        tuning = {"grid": GRID[strategy], "chosen": best[0], "candidates_tried": len(cands),
                  "note": "Parameters were fitted on train and selected on validation only. Out-of-sample was not looked at during selection."}
    res = {"train": run(chosen, t0, a), "validation": run(chosen, a, b), "out_of_sample": run(chosen, b, t1)}
    res["out_of_sample_costs_x2"] = simulate(_scale_costs(chosen, 2.0), strategy, insts, frames, b, t1, earnings, calendar)
    oos = res["out_of_sample"]
    flags = []
    synthetic_data = source == "synthetic"
    if synthetic_data:
        flags.append("SYNTHETIC DATA: this run only proves the pipeline executes. It is not evidence about any market.")
    if C.STRATEGY_MARKET[strategy] == "stock":
        flags.append("Survivorship bias: the watchlist is today's list of names; companies that failed or were delisted are absent, which flatters long strategies.")
    if C.STRATEGY_MARKET[strategy] == "futures":
        flags.append("Continuous futures series (research only): roll gaps and per-contract liquidity are not represented.")
    if missing:
        flags.append(f"Incomplete history: no usable data for {', '.join(missing)}.")
    if oos["trades"] < ev.min_oos_trades:
        flags.append(f"Insufficient sample: {oos['trades']} out-of-sample trades (minimum {ev.min_oos_trades}).")
    reasons = []
    if synthetic_data:
        reasons.append("synthetic data")
    if oos["trades"] < ev.min_oos_trades:
        reasons.append(f"only {oos['trades']} out-of-sample trades")
    if not oos["avg_net_per_trade"] or oos["avg_net_per_trade"] <= 0:
        reasons.append("average net result per trade is not positive out-of-sample")
    if oos["t_stat"] is None or oos["t_stat"] < ev.min_t_stat:
        reasons.append(f"t-statistic {oos['t_stat']} below {ev.min_t_stat:g}")
    if (res["out_of_sample_costs_x2"]["avg_net_per_trade"] or 0) <= 0:
        reasons.append("not profitable if trading costs are doubled")
    return {
        "strategy": strategy, "strategy_version": C.STRATEGY_VERSION[strategy], "strategy_hash": chosen.strategy_hash(strategy), "status": "ok",
        "data_source": label, "timeframe": tf, "instruments": [i.symbol for i in insts], "parameters": dict(chosen.strategies[strategy]),
        "costs": dict(chosen.costs[C.STRATEGY_MARKET[strategy]]), "tuning": tuning, "results": res, "flags": flags,
        "verdict": "eligible_for_live" if not reasons else "research_only", "verdict_reasons": reasons,
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _plain_overrides(cfg, extra: dict) -> dict:
    """Current settings expressed as overrides of the defaults (so a variant can be loaded with validation)."""
    base = json.loads(json.dumps({k: v for k, v in cfg.items()}))
    for k, v in extra.items():
        for k2, v2 in v.items():
            base[k][k2].update(v2)
    return base


def save_evidence(cfg, report: dict) -> Path:
    p = cfg.state_path / "evidence.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    allr = json.loads(p.read_text()) if p.exists() else {}
    allr[report["strategy"]] = report
    p.write_text(json.dumps(allr, indent=1))
    return p


def load_evidence(cfg) -> dict:
    p = cfg.state_path / "evidence.json"
    return json.loads(p.read_text()) if p.exists() else {}


def format_report(r: dict) -> str:
    if r["status"] != "ok":
        return f"{r['strategy']}: {r['detail']}"
    L = [f"\n{r['strategy']} v{r['strategy_version']} ({r['timeframe']})   data: {r['data_source']}", f"instruments: {', '.join(r['instruments'])}",
         f"parameters: {r['parameters']}", f"costs: {r['costs']}"]
    if r["tuning"]:
        L.append(f"tuned: {r['tuning']['chosen']} from {r['tuning']['candidates_tried']} candidates (train -> validation)")
    L.append(f"{'period':<24}{'trades':>7}{'net P&L':>12}{'avg/trade':>11}{'win%':>7}{'avg win':>10}{'avg loss':>10}{'maxDD%':>8}{'expo%':>7}{'turn x':>8}{'B&H%':>8}")
    for k in ("train", "validation", "out_of_sample", "out_of_sample_costs_x2"):
        x = r["results"][k]
        f = lambda v, w, d=2: f"{v:>{w},.{d}f}" if v is not None else " " * (w - 3) + "n/a"
        L.append(f"{k:<24}{x['trades']:>7}{f(x['net_pnl'], 12)}{f(x['avg_net_per_trade'], 11)}{f(x['win_rate'] and x['win_rate'] * 100, 7, 1)}{f(x['avg_win'], 10)}"
                 f"{f(x['avg_loss'], 10)}{f(x['max_drawdown_pct'], 8)}{f(x['exposure_pct'], 7, 1)}{f(x['turnover_x'], 8)}{f(x['benchmark_buy_hold_pct'], 8)}")
    o = r["results"]["out_of_sample"]
    L.append(f"out-of-sample {o['start'][:10]} to {o['end'][:10]}: by direction {o['by_direction']}; signals {o['signals']}; t-stat {o['t_stat']}")
    L.append(f"by symbol (OOS): {o['by_symbol']}")
    L += [f"  ! {x}" for x in r["flags"]]
    L.append(f"VERDICT: {r['verdict'].upper().replace('_', ' ')}" + (f" - {'; '.join(r['verdict_reasons'])}" if r["verdict_reasons"] else ""))
    return "\n".join(L)
