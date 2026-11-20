"""HTTP client that saves every response to the sweep's cache.

The saved file is the evidence behind a price or a catalogue match: URL, parameters, retrieval time and
the raw body. A replay of the same sweep reads the cache, so results are reproducible and free.
Secrets (API keys) are stripped from what is saved.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from claimagent.capture.store import utc_now

log = logging.getLogger(__name__)
SECRET_PARAMS = {"api_key", "key", "client_secret", "access_token"}
# Requests per second a source allows (Google Books: 100 queries per minute per user).
RATE_LIMITS = {"googlebooks": 1.5}
PER_MINUTE_WAIT_S = 20.0  # a per-minute quota hit: wait for the window, do not give up on the source
# Catalogue lookups are many and optional: a slow site must not hold the packet (due 5 min after the sweep).
FAST_SOURCES = {"openlibrary": (8.0, 2), "googlebooks": (8.0, 2)}  # timeout s, attempts
DEFAULT_TIMEOUT = (20.0, 3)
MAX_TIMEOUTS_IN_A_ROW = 3  # then the source is skipped for the rest of this sweep


@dataclass
class Response:
    url: str  # public URL without secrets, safe to put in the packet
    status: int
    data: Any
    retrieved_at: str
    cache_file: str
    from_cache: bool


class CachedHttp:
    def __init__(self, cache_dir: Path, offline: bool = False, timeout: float = 20.0) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": "claimagent/0.1"}, follow_redirects=True)
        self.calls = 0
        self.cache_hits = 0
        self.paid_calls: dict[str, int] = {}
        self.disabled: dict[str, str] = {}  # source -> reason (quota exhausted, auth failed)
        self._next_slot: dict[str, float] = {}
        self._rate_lock = threading.Lock()
        self._count_lock = threading.Lock()
        self._timeouts_in_a_row: dict[str, int] = {}

    def _pace(self, source: str) -> None:
        """Spaces requests to a rate-limited source; called from several worker threads."""
        rate = RATE_LIMITS.get(source)
        if not rate:
            return
        with self._rate_lock:
            now = time.monotonic()
            slot = max(now, self._next_slot.get(source, 0.0))
            self._next_slot[source] = slot + 1.0 / rate
        if slot > now:
            time.sleep(slot - now)

    def _key(self, method: str, url: str, params: dict[str, Any], body: Any) -> str:
        clean = {k: v for k, v in sorted(params.items()) if k not in SECRET_PARAMS}
        raw = json.dumps([method, url, clean, body], sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        source: str = "",
        paid: bool = False,
        vary: str = "",
    ) -> Response:
        """vary: anything outside the URL that changes the answer (for example a marketplace header)."""
        return self.request("GET", url, params or {}, headers, None, source, paid, vary)

    @staticmethod
    def _cached(path: Path) -> dict[str, Any] | None:
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError):
            return None  # missing or unreadable: fetched again
        return saved if isinstance(saved, dict) and "data" in saved else None

    def request(
        self,
        method: str,
        url: str,
        params: dict[str, Any],
        headers: dict[str, str] | None,
        body: Any,
        source: str,
        paid: bool,
        vary: str = "",
    ) -> Response:
        key = self._key(method, url, params, [body, vary])
        path = self.cache_dir / f"{source or 'http'}-{key}.json"
        public = str(httpx.URL(url, params={k: v for k, v in params.items() if k not in SECRET_PARAMS}))
        saved = self._cached(path)
        if saved is not None:
            with self._count_lock:
                self.cache_hits += 1
            return Response(public, saved["status"], saved["data"], saved["retrieved_at"], path.name, True)
        if self.offline:
            raise LookupError(f"offline and not cached: {public}")
        if source in self.disabled:
            raise RuntimeError(f"{source} disabled: {self.disabled[source]}")
        last_exc: Exception | None = None
        timeout, attempts = FAST_SOURCES.get(source, DEFAULT_TIMEOUT)
        for attempt in range(attempts):
            try:
                self._pace(source)
                r = self.client.request(method, url, params=params, headers=headers, json=body, timeout=timeout)
                self._timeouts_in_a_row[source] = 0
                per_minute = r.status_code == 429 and "per minute" in r.text.lower()
                if per_minute and attempt < 2:
                    log.info("%s per-minute quota reached; waiting %.0f s", source, PER_MINUTE_WAIT_S)
                    time.sleep(PER_MINUTE_WAIT_S)
                    continue
                if r.status_code in (401, 403) or (r.status_code == 429 and "quota" in r.text.lower()):
                    self.disabled[source] = f"HTTP {r.status_code}: {r.text[:160]}"
                    raise RuntimeError(f"{source} disabled: HTTP {r.status_code}")
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(1.5 * (attempt + 1))
                    last_exc = httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                    continue
                break
            except httpx.HTTPError as exc:
                last_exc = exc
                if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
                    n = self._timeouts_in_a_row.get(source, 0) + 1
                    self._timeouts_in_a_row[source] = n
                    if n >= MAX_TIMEOUTS_IN_A_ROW and source:
                        self.disabled[source] = f"{n} timeouts in a row ({exc.__class__.__name__})"
                        raise RuntimeError(f"{source} disabled: not responding") from exc
                time.sleep(1.0 * (attempt + 1))
        else:
            raise RuntimeError(f"request failed: {public}: {last_exc}")
        with self._count_lock:
            self.calls += 1
            if paid:
                self.paid_calls[source] = self.paid_calls.get(source, 0) + 1
        try:
            data: Any = r.json()
        except ValueError:
            data = {"text": r.text[:20000]}
        retrieved = utc_now()
        if r.status_code < 400:
            # Written whole or not at all: another thread may read the same key at the same time.
            tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
            tmp.write_text(
                json.dumps({"url": public, "status": r.status_code, "retrieved_at": retrieved, "data": data})
            )
            os.replace(tmp, path)
        else:
            log.warning("%s -> %s", public, r.status_code)
        return Response(public, r.status_code, data, retrieved, path.name, False)
