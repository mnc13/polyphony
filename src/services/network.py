"""
Waiting out a local network outage.

A failed request is retried a couple of times with a few seconds of backoff. That is
enough for a slow or overloaded server, but not for this machine losing DNS. On the
case-001 pilot a ~4 minute outage failed 31 evidence requests for decision d5 (every
host, NameResolutionError), and the retrieval went on as if the literature had nothing.
An LLM call in that window would have used up its retries and failed the case.

So a connection failure is checked first: if the host no longer resolves, the problem is
local, and the caller blocks until it resolves again (up to hp.network_outage_wait_s)
instead of spending its retries. Each wait is recorded as a `network_outage` event, so
an affected run can be identified afterwards.
"""
from __future__ import annotations

import socket
import threading
import time

from config.settings import get_settings

POLL_S = 15.0
_LOCK = threading.Lock()
# When a wait for a host has run out, later callers for that host do not each wait the
# full limit again (a long outage would otherwise stall every remaining request in turn);
# they fail fast until it resolves or another full limit has passed. Kept per host, so a
# give-up on one API cannot cancel the wait for another.
_GAVE_UP_AT: dict[str, float] = {}
# Hosts the pipeline talks to anyway. If any of them still resolves, the machine is
# online and only the one host is failing (provider DNS, a filter): waiting would stall
# the run for nothing, so the request fails at once and is counted as a source error.
REFERENCE_HOSTS = ("openrouter.ai", "api.openalex.org", "www.ebi.ac.uk",
                   "eutils.ncbi.nlm.nih.gov")


def resolves(host: str | None) -> bool:
    if not host:
        return True
    try:
        socket.getaddrinfo(host, 443)
        return True
    except OSError:
        return False


def _event(**fields) -> None:
    try:
        from src.services import telemetry
        telemetry.record_event("network_outage", **fields)
    except Exception:
        pass  # telemetry must never break a run


def wait_for_network(host: str | None, caller: str) -> bool:
    """
    Block while `host` does not resolve, up to the configured limit. Returns True if it
    resolves (at once or after waiting), False if the wait ran out.
    """
    if resolves(host):
        return True
    if any(resolves(h) for h in REFERENCE_HOSTS if h != host):
        print(f"[network] {host} does not resolve but other hosts do; not waiting")
        return False
    max_wait = get_settings().hp.network_outage_wait_s
    t0 = time.monotonic()
    gave_up = _GAVE_UP_AT.get(host or "")
    if gave_up is not None and t0 - gave_up < max_wait:
        return False
    # One waiter polls at a time; the others queue here and find the host back.
    with _LOCK:
        ok = resolves(host)
        while not ok and time.monotonic() - t0 < max_wait:
            time.sleep(POLL_S)
            ok = resolves(host)
        if ok:
            _GAVE_UP_AT.pop(host or "", None)
        else:
            _GAVE_UP_AT[host or ""] = time.monotonic()
    waited = round(time.monotonic() - t0, 1)
    print(f"[network] {host} unreachable; waited {waited}s for {caller}: "
          f"{'recovered' if ok else 'gave up'}")
    _event(host=host, caller=caller, waited_s=waited, recovered=ok)
    return ok
