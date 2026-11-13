"""SQLite journal: signals, order intents, trades, alerts, equity history, controls.

An order intent is committed here BEFORE the broker is contacted, so after a crash or timeout the
engine can ask the broker about that exact client order id instead of guessing or re-sending.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS signals(
  id TEXT PRIMARY KEY, strategy TEXT, symbol TEXT, market TEXT, broker TEXT, side INTEGER, time TEXT, expires TEXT,
  status TEXT, ref_price REAL, stop REAL, target REAL, atr REAL, hold_until TEXT, flatten_at TEXT, why TEXT, exit_rule TEXT,
  inputs TEXT, meta TEXT, decision TEXT, checks TEXT, sizing TEXT, intended_entry REAL, intended_qty REAL, est_costs REAL, created TEXT);
CREATE TABLE IF NOT EXISTS orders(
  cid TEXT PRIMARY KEY, broker TEXT, broker_id TEXT, signal_id TEXT, trade_id INTEGER, symbol TEXT, side TEXT, type TEXT, qty REAL,
  stop_price REAL, role TEXT, status TEXT, filled_qty REAL DEFAULT 0, avg_price REAL, fees REAL DEFAULT 0, reduce_only INTEGER DEFAULT 0,
  reserved_risk REAL DEFAULT 0, reserved_notional REAL DEFAULT 0, expected_price REAL, attempts INTEGER DEFAULT 0, note TEXT, created TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS trades(
  id INTEGER PRIMARY KEY AUTOINCREMENT, signal_id TEXT, strategy TEXT, strategy_version TEXT, mode TEXT, symbol TEXT, market TEXT, broker TEXT,
  side INTEGER, qty REAL, qty_total REAL, entry_avg REAL, expected_entry REAL, stop REAL, target REAL, planned_risk REAL, opened TEXT,
  hold_until TEXT, flatten_at TEXT, status TEXT, exit_reason TEXT, exit_avg REAL, expected_exit REAL, closed TEXT, realized REAL DEFAULT 0,
  fees REAL DEFAULT 0, exit_qty REAL DEFAULT 0, meta TEXT);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, level TEXT, kind TEXT, msg TEXT);
CREATE TABLE IF NOT EXISTS equity(ts TEXT, equity REAL, detail TEXT);
CREATE TABLE IF NOT EXISTS barcache(symbol TEXT, tf TEXT, updated TEXT, payload TEXT, PRIMARY KEY(symbol, tf));
CREATE TABLE IF NOT EXISTS activations(broker TEXT, account TEXT, market TEXT, strategy TEXT, strategy_hash TEXT, limits TEXT, activated TEXT,
  PRIMARY KEY(broker, account, market, strategy));
CREATE TABLE IF NOT EXISTS tickets(
  cid TEXT PRIMARY KEY, symbol TEXT, side TEXT, type TEXT, qty REAL, stop_price REAL, role TEXT, status TEXT, filled_qty REAL DEFAULT 0,
  avg_price REAL, fees REAL DEFAULT 0, placed INTEGER DEFAULT 0, cancel_requested INTEGER DEFAULT 0, note TEXT, created TEXT, updated TEXT);
"""
JSON_COLS = {"inputs", "meta", "checks", "sizing", "detail", "limits"}
ACTIVE = ("created", "uncertain", "new", "partially_filled", "pending_cancel")


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=15)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        if self.path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        for k in JSON_COLS & d.keys():
            if d[k] is not None:
                d[k] = json.loads(d[k])
        return d

    @staticmethod
    def _enc(fields: dict) -> dict:
        return {k: (json.dumps(v) if k in JSON_COLS and v is not None else v) for k, v in fields.items()}

    def q(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [self._row(r) for r in self.db.execute(sql, args).fetchall()]

    def one(self, sql: str, args=()) -> dict | None:
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def insert(self, table: str, ignore: bool = False, **fields) -> int:
        f = self._enc(fields)
        with self.lock:
            cur = self.db.execute(f"INSERT {'OR IGNORE ' if ignore else ''}INTO {table}({','.join(f)}) VALUES({','.join('?' * len(f))})", list(f.values()))
            return cur.lastrowid if cur.rowcount else 0

    def update(self, table: str, key: str, value, **fields):
        f = self._enc(fields)
        with self.lock:
            self.db.execute(f"UPDATE {table} SET {','.join(k + '=?' for k in f)} WHERE {key}=?", [*f.values(), value])

    def tx(self):
        """Exclusive write transaction (used to reserve risk and record the intent as one atomic step)."""
        store = self

        class _Tx:
            def __enter__(self):
                store.lock.acquire()
                store.db.execute("BEGIN IMMEDIATE")

            def __exit__(self, et, ev, tb):
                store.db.execute("ROLLBACK" if et else "COMMIT")
                store.lock.release()
        return _Tx()

    def get(self, k: str, default=None):
        r = self.one("SELECT v FROM meta WHERE k=?", (k,))
        return json.loads(r["v"]) if r else default

    def set(self, k: str, v):
        with self.lock:
            self.db.execute("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))

    def event(self, ts: str, level: str, kind: str, msg: str):
        self.insert("events", ts=ts, level=level, kind=kind, msg=msg)

    def active_orders(self, **where) -> list[dict]:
        sql = f"SELECT * FROM orders WHERE status IN ({','.join('?' * len(ACTIVE))})"
        args = list(ACTIVE)
        for k, v in where.items():
            sql += f" AND {k}=?"
            args.append(v)
        return self.q(sql + " ORDER BY created, cid", args)

    def open_trades(self) -> list[dict]:
        return self.q("SELECT * FROM trades WHERE status!='closed' ORDER BY id")
