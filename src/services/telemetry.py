"""
Per-call telemetry: tokens (prompt / completion / total), latency, role, model —
tagged with the current variant + sample + run so it can be aggregated per agent, per
sample, per run, and per condition (main / each ablation / each baseline).

Every LLM call routes through `record(...)`. A small context (variant, sample_id,
run_id) is set by the batch/baseline runners so each call knows which sample, condition
and run it belongs to. Records are appended to:
  outputs/<variant>/telemetry/calls.jsonl

The files accumulate across runs, so the run_id matters: the main variant's calls.jsonl
held the Oct 14-16 attempts and the Oct 20-21 run side by side, and summing it counted
both. Filter on run_id (src/eval/tokens.py does) to get one run.

Retrieval telemetry (whether P-MER served a premise from the local corpus or had to
go live) is recorded by `record_retrieval(...)` into:
  outputs/<variant>/telemetry/retrieval.jsonl

Events that change how a run behaves without failing it -- a model's request
parameters relaxed after a routing error, a truncated output retried with a bigger
budget -- go to `record_event(...)`:
  outputs/<variant>/telemetry/events.jsonl

Aggregation + the report live in src/eval/tokens.py (scripts/token_report.py).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from config.settings import OUTPUTS_DIR

NL = chr(10)
_lock = threading.Lock()
_ctx = {"variant": "_adhoc", "sample_id": "_none", "run_id": None}


def set_context(variant: str, sample_id: str, run_id: str | None = None) -> None:
    _ctx["variant"] = variant or "_adhoc"
    _ctx["sample_id"] = sample_id or "_none"
    if run_id is not None:
        _ctx["run_id"] = run_id


def clear_context() -> None:
    _ctx["variant"] = "_adhoc"
    _ctx["sample_id"] = "_none"
    _ctx["run_id"] = None


def _path(variant: str, name: str):
    d = OUTPUTS_DIR / variant / "telemetry"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def _calls_path(variant: str):
    return _path(variant, "calls.jsonl")


def _retrieval_path(variant: str):
    return _path(variant, "retrieval.jsonl")


def _append(path, rec: dict) -> None:
    try:
        with _lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + NL)
    except Exception:
        pass  # telemetry must never break a run


def _base() -> dict:
    return {"ts": datetime.now(timezone.utc).isoformat(), "run_id": _ctx["run_id"],
            "variant": _ctx["variant"], "sample_id": _ctx["sample_id"]}


def record_retrieval(premise_id: str, dimension: str, local_hits: int,
                     local_relevant: int, used_live: bool, candidates: int,
                     admitted: int, rejected: dict, mean_novelty: float,
                     decision_id: str | None = None, local_admitted: int | None = None,
                     source_errors: dict | None = None, rung_used=None,
                     forced: bool = False, hinted: bool = False,
                     rung_query: str | None = None, transport_failed: bool = False) -> None:
    """
    One record per P-MER call. `local_hits` is the raw local count; `local_relevant` is
    how many of those clear the relevance bar for this premise, which is what decides
    whether a live fetch happens; `local_admitted` is how many then survived the whole
    admission gate. `source_errors` counts requests that failed after retries, so an
    empty result caused by a 429 is distinguishable from one caused by no hits.
    `premise_id` repeats across decisions; (decision_id, premise_id) is the key.
    """
    rec = {
        **_base(),
        "decision_id": decision_id,
        "premise_id": premise_id, "dimension": dimension,
        "local_hits": int(local_hits), "local_relevant": int(local_relevant),
        "local_admitted": None if local_admitted is None else int(local_admitted),
        "used_live": bool(used_live), "forced_live": bool(forced), "hinted": bool(hinted),
        "rung_used": rung_used, "rung_query": rung_query,
        "transport_failed": bool(transport_failed), "candidates": int(candidates),
        "admitted": int(admitted), "rejected": rejected,
        "source_errors": source_errors or {},
        "mean_novelty": round(float(mean_novelty or 0.0), 3),
    }
    _append(_retrieval_path(_ctx["variant"]), rec)


def record(role: str, model: str, prompt_tokens: int, completion_tokens: int,
           total_tokens: int, latency_s: float, ok: bool = True, note: str | None = None) -> None:
    """`ok=False` marks a call that was billed but whose output was unusable (e.g.
    truncated by the length limit); it still costs tokens and is still recorded."""
    rec = {
        **_base(),
        "role": role, "model": model,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "latency_s": round(float(latency_s or 0.0), 2),
        "ok": bool(ok),
    }
    if note:
        rec["note"] = note
    _append(_calls_path(_ctx["variant"]), rec)


def record_event(kind: str, **fields) -> None:
    _append(_path(_ctx["variant"], "events.jsonl"), {**_base(), "kind": kind, **fields})
