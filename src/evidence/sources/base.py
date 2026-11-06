"""
P3 — Evidence source base.

Every source client returns a list of `RawDoc` (a normalized record) so the
chunker / gate / TEG downstream are source-agnostic. Clients are resilient:
network failure yields an empty list and a logged warning, never a crash — one
dead source must not sink a run.

Resilient is not the same as silent, though. A 429 or a timeout used to come back as
[] exactly like a query with no hits, so the query ladder broadened (or fell through to
CrossRef) on what was really a transport failure, and nothing outside stdout recorded
it. Transient failures are now retried with backoff, requests to one host are spaced
out, and every request that still fails is counted per source so P-MER can put the
count into retrieval telemetry. A host that stops resolving altogether (this machine
offline) is waited out rather than retried into; see src/services/network.py.
"""
from __future__ import annotations

import html
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests

from config.settings import get_settings
from src.services import network


# JATS (<jats:p>), HTML (<h4>, <i>, <sup>) and stray entities show up inside abstract
# fields from Europe PMC and CrossRef. They are not text, and they reach the embedder
# and NLI as if they were.
_TAG = re.compile(r"<[^>]+>")


def clean_markup(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(html.unescape(_TAG.sub(" ", text)).split())


@dataclass
class RawDoc:
    title: str
    abstract: str
    source: str
    doi: Optional[str] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    citation_count: int = 0
    ext_id: Optional[str] = None  # PMID / OpenAlex id / S2 id

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.abstract}".strip()

    def has_content(self) -> bool:
        return bool(self.abstract and len(self.abstract) > 40)


# Requests that failed after every retry, per source name. Read and reset by P-MER
# around each retrieval call (see failure_snapshot).
_FAILURES: Counter = Counter()
_FAIL_LOCK = threading.Lock()

# Minimum spacing between two requests to the same host. NCBI allows 3 requests/s
# without a key (10 with one) and a single premise can issue six E-utilities calls,
# so bursts were the likeliest cause of the 429s; the others get a polite floor.
_LAST_REQUEST: dict[str, float] = {}
_HOST_LOCK = threading.Lock()


def _min_interval(host: str) -> float:
    if "ncbi.nlm.nih.gov" in host:
        return 0.11 if get_settings().ncbi_api_key else 0.35
    return 0.1


def failure_snapshot() -> dict[str, int]:
    with _FAIL_LOCK:
        return dict(_FAILURES)


def _record_failure(name: str) -> None:
    with _FAIL_LOCK:
        _FAILURES[name] += 1


_RETRY_STATUS = {429, 500, 502, 503, 504}

# A 429 whose Retry-After runs past this many seconds is a spent quota, not a burst.
# OpenAlex now meters keyless requests against a small daily budget shared by the IP
# (about 100 searches) and answers 429 with Retry-After until midnight UTC. Retried and
# then counted as an ordinary failure, it starved 15 of case 005's 18 retrievals in the
# first ablation run while the batch reported every case done.
_QUOTA_RETRY_AFTER_S = 120


class SourceBudgetExhausted(RuntimeError):
    """A source refused for a spent quota. Raised, not swallowed: the case must fail
    (run_batch rolls its corpus back and stops) instead of finishing without evidence."""


class SourceClient:
    name: str = "base"
    timeout: int = 20
    retries: int = 2          # extra attempts after the first, for transient failures

    def fetch(self, query: str, max_results: int = 5) -> list[RawDoc]:
        raise NotImplementedError

    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        with _HOST_LOCK:
            wait = _LAST_REQUEST.get(host, 0.0) + _min_interval(host) - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _LAST_REQUEST[host] = time.monotonic()

    # shared HTTP helper with polite headers, spacing, retry on transient errors
    def _get(self, url: str, params: dict | None = None, headers: dict | None = None):
        email = get_settings().contact_email
        h = {"User-Agent": f"Polyphony/1.0 (mailto:{email})"}
        if headers:
            h.update(headers)
        last: Exception | None = None
        host = urlparse(url).hostname
        waited_out = False
        attempt = 0
        while attempt <= self.retries:
            self._throttle(url)
            try:
                r = requests.get(url, params=params, headers=h, timeout=self.timeout)
                retry_after = r.headers.get("Retry-After", "")
                if (r.status_code == 429 and retry_after.isdigit()
                        and int(retry_after) > _QUOTA_RETRY_AFTER_S):
                    _record_failure(self.name)
                    raise SourceBudgetExhausted(
                        f"[{self.name}] quota spent: 429 with Retry-After {retry_after}s "
                        f"from {host}. Set the source's API key in .env (OpenAlex: "
                        f"OPENALEX_API_KEY) or resume after the reset.")
                if r.status_code in _RETRY_STATUS and attempt < self.retries:
                    delay = float(retry_after) if retry_after.isdigit() else 1.5 * (2 ** attempt)
                    time.sleep(min(delay, 10.0))
                    attempt += 1
                    continue
                r.raise_for_status()
                return r
            except SourceBudgetExhausted:
                raise
            except (requests.Timeout, requests.ConnectionError) as e:
                last = e
                # A host that no longer resolves means this machine is offline, not the
                # server slow: wait for the network (once per request) and repeat the
                # same attempt, rather than spend the retries inside the outage.
                if not waited_out and not network.resolves(host):
                    waited_out = True
                    if network.wait_for_network(host, self.name):
                        continue
                    break
                if attempt < self.retries:
                    time.sleep(1.5 * (2 ** attempt))
                    attempt += 1
                    continue
                break
            except Exception as e:  # noqa: BLE001
                last = e
                break
        print(f"[{self.name}] request failed: {last}")
        _record_failure(self.name)
        return None
