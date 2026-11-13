"""Dashboard API. Reads the journal database written by the worker (or by a replay run); it never talks
to a broker and never sees credentials. Controls are written as flags that the worker acts on.
Binds to 127.0.0.1 and has no login: do not expose it to a network."""
from __future__ import annotations

import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from . import backtest
from .brokers.manual import instruction, ticket_action
from .brokers.unavailable import FIDELITY_STATUS, IBKR_STATUS
from .config import STRATEGY_MARKET, STRATEGY_VERSION
from .instruments import usd_per_quote
from .store import Store
from .watchlist import MARKETS, Watchlist

WEB = Path(__file__).resolve().parent.parent / "web"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def build_state(store: Store, cfg) -> dict:
    g = store.get
    mode, snap = g("mode", "?"), g("snapshot", {}) or {}
    insts, rates, marks = g("instruments", {}), snap.get("rates", {}), snap.get("marks", {})
    hb = g("heartbeat")
    # In replay the clock is the replay's own; otherwise it is the wall clock.
    now = pd.Timestamp(hb) if mode in ("REPLAY", "BACKTEST") and hb else pd.Timestamp.now(tz="UTC")
    age = lambda t: None if not t else round((now - pd.Timestamp(t)).total_seconds(), 1)

    def fx(sym):
        try:
            return usd_per_quote(insts[sym]["quote_ccy"], rates)
        except Exception:
            return None

    positions, unreal_by, open_risk, gross = [], {}, 0.0, 0.0
    unprotected = set(g("unprotected", []) or [])
    for t in store.q("SELECT * FROM trades WHERE status!='closed' ORDER BY id"):
        i, f, m = insts.get(t["symbol"]), fx(t["symbol"]), marks.get(t["symbol"])
        mult = i["multiplier"] if i else 1.0
        u = (m - t["entry_avg"]) * t["side"] * t["qty"] * mult * f if m is not None and f else None
        risk = abs(t["entry_avg"] - t["stop"]) * t["qty"] * mult * (f or 0)
        notional = (m or t["entry_avg"]) * t["qty"] * mult * (f or 0)
        open_risk += risk
        gross += notional
        unreal_by[t["strategy"]] = unreal_by.get(t["strategy"], 0.0) + (u or 0.0)
        stops = store.q("SELECT cid, status, qty, filled_qty FROM orders WHERE trade_id=? AND role='stop' AND status IN ('created','uncertain','new','partially_filled','pending_cancel')", (t["id"],))
        positions.append({**t, "mark": m, "unrealized": u, "open_risk": risk, "notional": notional,
                          "protected": t["id"] not in unprotected and bool(stops), "stop_qty": sum(s["qty"] - s["filled_qty"] for s in stops),
                          "stop_unplaced": any(store.one("SELECT 1 FROM tickets WHERE cid=? AND placed=0", (s["cid"],)) for s in stops)})
    strategies = []
    evidence = backtest.load_evidence(cfg)
    acts = store.q("SELECT * FROM activations")
    for name, market in STRATEGY_MARKET.items():
        r = store.one("SELECT COUNT(*) n, COALESCE(SUM(realized-fees),0) net, COALESCE(SUM(fees),0) fees, COALESCE(SUM(CASE WHEN realized-fees>0 THEN 1 ELSE 0 END),0) wins "
                      "FROM trades WHERE status='closed' AND strategy=?", (name,))
        sg = {x["status"]: x["n"] for x in store.q("SELECT status, COUNT(*) n FROM signals WHERE strategy=? GROUP BY status", (name,))}
        ev = evidence.get(name)
        strategies.append({"name": name, "market": market, "version": STRATEGY_VERSION[name], "enabled": cfg.strategies[name].enabled, "closed": r["n"],
                           "wins": r["wins"], "realized_net": r["net"], "fees": r["fees"], "unrealized": unreal_by.get(name, 0.0), "signals": sg,
                           "evidence": {"verdict": ev["verdict"], "reasons": ev["verdict_reasons"], "data": ev["data_source"], "generated": ev["generated"],
                                        "current": ev["strategy_hash"] == cfg.strategy_hash(name)} if ev else None,
                           "live_activated": [a for a in acts if a["strategy"] == name]})
    realized = store.one("SELECT COALESCE(SUM(realized-fees),0) v FROM trades")["v"]
    feeds = {k: {**v, "age_sec": age(v["last_bar"])} for k, v in (g("feeds", {}) or {}).items()}
    wl = Watchlist(cfg).get()
    problems, known, scanned, active = g("instrument_problems", {}) or {}, insts, g("scanner", {}) or {}, g("active_contracts", {}) or {}
    last_bar = {}
    for v in scanned.values():
        last_bar[v["symbol"]] = max(last_bar.get(v["symbol"], ""), v["last_bar"])
    rows = []
    for m in MARKETS:
        mcfg = cfg.markets[m]
        for sym in wl[m]:
            real = active.get(sym, sym) if m == "futures" else sym
            if mode == "REPLAY":
                st, detail = "saved", "Used by the paper/live worker. The replay is a fixed scripted run and does not use this list."
            elif problems.get(sym) or problems.get(m):
                st, detail = "problem", problems.get(sym) or problems.get(m)
            elif real in known:
                st = "watching"
                detail = (f"contract {real}; " if m == "futures" else "") + (f"last completed bar {last_bar[real][:16].replace('T', ' ')} UTC" if real in last_bar else "no completed bar seen yet")
            else:
                st, detail = "pending", "Saved. The worker applies it on its next cycle" + ("" if hb else " (the worker is not running)") + "."
            rows.append({"market": m, "symbol": sym, "broker": mcfg["broker"], "etf": m == "stock" and sym in wl["etfs"], "status": st, "detail": detail,
                         "held": bool(store.one("SELECT 1 FROM trades WHERE status!='closed' AND symbol=?", (real,)))})
    tickets = []
    for t in store.q("SELECT * FROM tickets ORDER BY (status IN ('new','partially_filled')) DESC, created DESC LIMIT 200"):
        o = store.one("SELECT signal_id, trade_id, expected_price FROM orders WHERE cid=?", (t["cid"],)) or {}
        sig = store.one("SELECT strategy, stop, target, expires, why FROM signals WHERE id=?", (o.get("signal_id"),)) or {}
        tr = store.one("SELECT exit_reason FROM trades WHERE id=?", (o.get("trade_id"),)) or {}
        active = t["status"] in ("new", "partially_filled")
        tickets.append({**t, "active": active, "needs_action": active and (not t["placed"] or bool(t["cancel_requested"])), "instruction": instruction(t) if active else "",
                        "expected_price": o.get("expected_price"), "strategy": sig.get("strategy"), "plan_stop": sig.get("stop"), "plan_target": sig.get("target"),
                        "expires": sig.get("expires") if t["role"] == "entry" else None, "why": sig.get("why") if t["role"] == "entry" else tr.get("exit_reason")})
    return _clean({
        "tickets": tickets, "manual_cash": g("manual_cash"), "quotes": g("quotes", {}) or {},
        "watchlist": rows, "watchlist_markets": {m: {"broker": cfg.markets[m]["broker"], "enabled": cfg.markets[m]["enabled"]} for m in MARKETS},
        "mode": mode, "now": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "data_label": g("data_label") or "; ".join(f"{k}: {b['data']}" for k, b in (g("brokers", {}) or {}).items()),
        "replay": g("replay"), "brokers": g("brokers", {}), "reconcile": g("reconcile", {}),
        "controls": {"kill": bool(g("kill")), "pause_entries": bool(g("pause_entries")), "halt_daily": g("halt_daily"), "halt_drawdown": g("halt_drawdown"),
                     "halt_protection": g("halt_protection"), "paused_instruments": g("paused_instruments", {}), "day": g("day")},
        "portfolio": {"equity": snap.get("equity"), "day_pnl": snap.get("day_pnl"), "drawdown": snap.get("drawdown"), "peak": snap.get("peak"),
                      "accounts": snap.get("accounts", {}), "realized": realized, "unrealized": sum(unreal_by.values()), "open_risk": open_risk, "gross": gross,
                      "quote_times": snap.get("quote_times", {}), "snapshot_at": snap.get("at"),
                      "limits": {k: cfg.risk[k] for k in ("max_loss_per_trade_pct", "max_open_risk_pct", "max_gross_exposure_pct", "max_instrument_pct", "max_positions",
                                                           "daily_loss_limit_pct", "max_drawdown_pct")},
                      "limits_set_by_user": cfg.risk.set_by_user,
                      "daily_reset": f"{cfg.portfolio.daily_reset_time} {cfg.portfolio.daily_reset_tz}, every day"},
        "strategies": strategies,
        "scanner": sorted((g("scanner", {}) or {}).values(), key=lambda x: (x["market"], x["symbol"], x["strategy"])),
        "signals": store.q("SELECT * FROM signals ORDER BY time DESC, id LIMIT 300"),
        "positions": positions,
        "orders": store.q("SELECT * FROM orders ORDER BY created DESC, cid LIMIT 150"),
        "journal": store.q("SELECT * FROM trades WHERE status='closed' ORDER BY closed DESC LIMIT 300"),
        "alerts": store.q("SELECT * FROM events WHERE level='ALERT' ORDER BY id DESC LIMIT 100"),
        "log": store.q("SELECT * FROM events ORDER BY id DESC LIMIT 200"),
        "health": {"heartbeat": hb, "heartbeat_age_sec": None if mode == "REPLAY" else age(hb), "last_scan": g("last_scan_at"), "feeds": feeds,
                   "feed_errors": g("feed_errors", {}), "unprotected": sorted(unprotected), "instrument_problems": g("instrument_problems", {}),
                   "external_positions": list((g("external_positions", {}) or {}).values()), "manual_orders": g("manual_orders", {}),
                   "exit_blocked": g("exit_blocked", {}), "active_contracts": g("active_contracts", {})},
        "integrations": {"earnings": g("earnings_status"), "calendar": g("calendar_status"), "fidelity": FIDELITY_STATUS, "ibkr": IBKR_STATUS},
        "equity": store.q("SELECT ts, equity FROM (SELECT * FROM equity ORDER BY ts DESC LIMIT 600) ORDER BY ts"),
        "replay_checks": g("replay_checks"),
    })


def build_chart(store: Store, symbol: str, tf: str | None) -> dict:
    rows = store.q("SELECT * FROM barcache WHERE symbol=?" + (" AND tf=?" if tf else ""), (symbol, tf) if tf else (symbol,))
    if not rows:
        return {"symbol": symbol, "empty": True, "message": "No bars cached for this symbol yet. The worker stores bars when it scans."}
    rows.sort(key=lambda x: ["1d", "4h", "1h", "5m"].index(x["tf"]) if x["tf"] in ("1d", "4h", "1h", "5m") else 9)
    r = rows[0]
    ep = lambda x: int(pd.Timestamp(x).timestamp()) if x else None
    marks = []
    for s in store.q("SELECT * FROM signals WHERE symbol=?", (symbol,)):
        marks.append({"t": ep(s["time"]), "kind": "signal", "side": s["side"], "label": f"{s['strategy']} signal: {s['status']}", "price": s["ref_price"]})
    for o in store.q("SELECT * FROM orders WHERE symbol=? AND filled_qty>0", (symbol,)):
        marks.append({"t": ep(o["updated"]), "kind": o["role"], "side": 1 if o["side"] == "buy" else -1, "price": o["avg_price"],
                      "label": f"{o['role']} fill {o['side']} {o['filled_qty']:g} @ {o['avg_price']:g}"})
    lines = [{"from": ep(t["opened"]), "to": ep(t["closed"]), "stop": t["stop"], "target": t["target"], "entry": t["entry_avg"]}
             for t in store.q("SELECT * FROM trades WHERE symbol=?", (symbol,))]
    return _clean({"symbol": symbol, "tf": r["tf"], "updated": r["updated"], "bars": json.loads(r["payload"]), "marks": marks, "lines": lines,
                   "timezone": "UTC", "quote": (store.get("quotes", {}) or {}).get(symbol),
                   "tfs": sorted((x["tf"] for x in store.q("SELECT tf FROM barcache WHERE symbol=?", (symbol,))), key=["5m", "1h", "4h", "1d"].index)})


def make_handler(db_path: str, cfg):
    def store() -> Store:
        return Store(db_path)

    class H(BaseHTTPRequestHandler):
        server_version = "RovioTrader"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _guard(self, fn):
            s = None
            try:
                s = store()
                self._send(200, fn(s))
            except (ValueError, KeyError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:
                self._send(500, {"error": f"Internal error: {type(e).__name__}"})
            finally:
                if s is not None:
                    s.close()

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self._guard(lambda s: build_state(s, cfg))
            if u.path == "/api/quotes":
                return self._guard(lambda s: _clean({"quotes": s.get("quotes", {}) or {}, "heartbeat": s.get("heartbeat"),
                                                     "now": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")}))
            if u.path == "/api/chart":
                if not re.fullmatch(r"[A-Za-z0-9./\-=]{1,20}", q.get("symbol", "")):
                    return self._send(400, {"error": "bad symbol"})
                return self._guard(lambda s: build_chart(s, q["symbol"], q.get("tf")))
            self._send(404, {"error": "Not found"})

        def do_POST(self):
            origin, host = self.headers.get("Origin"), self.headers.get("Host", "")
            if (origin and urlparse(origin).netloc != host) or "application/json" not in self.headers.get("Content-Type", ""):
                return self._send(403, {"error": "Forbidden"})
            try:
                body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 65536)) or b"{}")
            except ValueError:
                return self._send(400, {"error": "Invalid JSON"})
            if urlparse(self.path).path == "/api/watchlist":
                def edit(s: Store):
                    w = Watchlist(cfg)
                    market, action = body.get("market"), body.get("action")
                    if market not in MARKETS or action not in ("add", "remove"):
                        raise ValueError("market must be stock, forex, crypto or futures; action must be add or remove")
                    sym = w.add(market, str(body.get("symbol", "")), bool(body.get("etf"))) if action == "add" else w.remove(market, str(body.get("symbol", "")))
                    s.event(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), "INFO", "operator", f"watchlist: {action} {market} {sym}")
                    return {"ok": True, "symbol": sym}
                return self._guard(edit)
            if urlparse(self.path).path == "/api/ticket":
                def tick(s: Store):
                    cid, action = str(body.get("cid", "")), body.get("action")
                    t = ticket_action(s, cid, action, body.get("qty"), body.get("price"), body.get("fees") or 0.0)
                    s.event(pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), "INFO", "operator",
                            f"ticket {t['symbol']} {t['role']}: {action}" + (f" {float(body['qty']):g} @ {float(body['price']):g}" if action == "fill" else ""))
                    return {"ok": True, "status": t["status"], "note": "Recorded. The worker picks it up on its next cycle."}
                return self._guard(tick)
            if urlparse(self.path).path != "/api/control":
                return self._send(404, {"error": "Not found"})

            def act(s: Store):
                a, ts = body.get("action"), pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
                if a == "kill":
                    s.set("kill", bool(body.get("on")))
                elif a == "pause":
                    s.set("pause_entries", bool(body.get("on")))
                elif a == "resume":
                    s.set("halt_drawdown", None)
                    s.set("halt_protection", None)
                    snap = s.get("snapshot")
                    if snap:
                        s.set("peak_equity", snap["equity"])
                elif a == "flatten":
                    ids = body.get("trade_ids")
                    if ids is not None and not (isinstance(ids, list) and all(isinstance(i, int) for i in ids)):
                        raise ValueError("trade_ids must be a list of integers")
                    s.set("flatten_request", "all" if ids is None else ids)
                else:
                    raise ValueError("unknown action")
                s.event(ts, "INFO", "operator", f"dashboard action: {a} {json.dumps({k: v for k, v in body.items() if k != 'action'})}")
                return {"ok": True, "note": "The worker applies this on its next cycle." if s.get("mode") != "REPLAY" else
                        "Recorded. Replay is a finished run with no worker, so nothing will act on it."}
            self._guard(act)

    return H


def serve(db_path: str, cfg, host: str, port: int):
    httpd = ThreadingHTTPServer((host, port), make_handler(db_path, cfg))
    print(f"Rovio Trader dashboard: http://{host}:{httpd.server_address[1]}   (journal: {db_path})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
