"""Tiny JSON-over-HTTPS helper shared by the REST adapters. Credentials travel only in request headers
and never appear in an exception message, a log line or a return value."""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request

from .base import BrokerError, BrokerTimeout


class Http:
    def __init__(self, headers: dict, opener=None, timeout: float = 15.0):
        self._headers, self._open, self.timeout = headers, opener or urllib.request.urlopen, timeout

    def call(self, method: str, url: str, body: dict | None = None, params: dict | None = None, ok_404: bool = False):
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={**self._headers, "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self._open(req, timeout=self.timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            if e.code == 404 and ok_404:
                return None
            try:
                payload = json.loads(e.read() or b"{}")
                msg = payload.get("message") or payload.get("errorMessage") or ""
            except Exception:
                msg = ""
            if e.code >= 500 or e.code == 429:
                # server-side failure: the request may or may not have been processed
                raise BrokerTimeout(f"HTTP {e.code} from broker") from None
            raise BrokerError(f"HTTP {e.code}: {msg}"[:300]) from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            raise BrokerTimeout(f"no response from broker ({type(e).__name__})") from None
        except json.JSONDecodeError:
            raise BrokerTimeout("unreadable response from broker") from None
