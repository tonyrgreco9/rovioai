"""Local JSON API + dashboard. Binds to 127.0.0.1 by default; there is no authentication, so do not expose it."""
from __future__ import annotations

import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import backtest
from .brokers import BrokerError
from .data import DataError
from .engine import Engine, RiskRejected
from .indicators import compute_all

WEB = Path(__file__).resolve().parent.parent / "web"
CHART_COLS = ["sma50", "sma200", "bb_up", "bb_lo", "rsi", "macd", "macd_signal", "macd_hist"]


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def chart_payload(engine: Engine, symbol: str, range_: str, interval: str) -> dict:
    df = engine.provider.history(symbol, range_, interval)
    ind = compute_all(df).tail(1500)
    col = lambda k, nd=6: [None if not math.isfinite(x) else round(float(x), nd) for x in ind[k].to_numpy()]
    out = engine.analyze(symbol, range_, interval)
    out["bars"] = {"t": [int(t.timestamp()) for t in ind.index], "o": col("open"), "h": col("high"), "l": col("low"), "c": col("close"), "v": col("volume", 0)}
    out["ind"] = {k: col(k) for k in CHART_COLS}
    out.update(range=range_, interval=interval)
    return out


def make_handler(engine: Engine):
    class H(BaseHTTPRequestHandler):
        server_version = "RovioTrader"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(_clean(body)).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _guard(self, fn):
            try:
                self._send(200, fn())
            except RiskRejected as e:
                self._send(422, {"error": "Blocked by risk checks", "reasons": e.reasons})
            except (DataError, BrokerError, ValueError, KeyError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # keep the server alive; do not leak internals
                self._send(500, {"error": f"Internal error: {type(e).__name__}"})

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            routes = {
                "/api/info": engine.info,
                "/api/scan": engine.scan,
                "/api/portfolio": engine.portfolio,
                "/api/proposals": engine.proposals,
                "/api/strategies": lambda: {k: v[0] for k, v in backtest.STRATEGIES.items()},
                "/api/chart": lambda: chart_payload(engine, q["symbol"], q.get("range", "2y"), q.get("interval", "1d")),
                "/api/backtest": lambda: backtest.run(engine.provider.history(q["symbol"], q.get("range", "5y"), q.get("interval", "1d")), q.get("strategy", "composite")),
            }
            if u.path in routes:
                return self._guard(routes[u.path])
            self._send(404, {"error": "Not found"})

        def do_POST(self):
            # Cross-site request protection: browsers must be on this origin and send JSON.
            origin, host = self.headers.get("Origin"), self.headers.get("Host", "")
            if (origin and urlparse(origin).netloc != host) or "application/json" not in self.headers.get("Content-Type", ""):
                return self._send(403, {"error": "Forbidden"})
            try:
                body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 65536)) or b"{}")
            except ValueError:
                return self._send(400, {"error": "Invalid JSON"})
            p = urlparse(self.path).path
            m = re.fullmatch(r"/api/proposals/(\w+)/(approve|reject)", p)
            f = re.fullmatch(r"/api/orders/(\w+)/(fill|cancel)", p)
            if p == "/api/proposals/generate":
                return self._guard(engine.generate_proposals)
            if m:
                return self._guard(lambda: (engine.approve if m[2] == "approve" else engine.reject)(m[1]))
            if p == "/api/orders":
                return self._guard(lambda: engine.submit_order(body["symbol"], body["side"], float(body["qty"]), note="Manual order"))
            if f:
                if not hasattr(engine.broker, "confirm_fill"):
                    return self._send(400, {"error": "Only the manual broker has tickets to fill"})
                if f[2] == "cancel":
                    return self._guard(lambda: engine.broker.cancel(f[1]))
                return self._guard(lambda: engine.broker.confirm_fill(f[1], float(body["price"]), body.get("qty")))
            if p == "/api/watchlist":
                return self._guard(lambda: engine.edit_watchlist(body.get("add"), body.get("remove")))
            if p == "/api/killswitch":
                return self._guard(lambda: (engine.set_kill_switch(bool(body.get("on"))), engine.info())[1])
            self._send(404, {"error": "Not found"})

    return H


def serve(engine: Engine, host: str, port: int):
    httpd = ThreadingHTTPServer((host, port), make_handler(engine))
    print(f"Rovio Trader dashboard: http://{host}:{httpd.server_address[1]}  (broker={engine.broker.name}, data={engine.provider.name})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
