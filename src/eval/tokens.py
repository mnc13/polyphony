"""
Token + time aggregation from telemetry, per variant (main / each ablation / each
baseline). Answers: how many tokens does each agent/role use on average, how many per
sample, how many for a whole run, and how long each run takes — plus a projection to
the full dataset so a budget can be set before buying the real (OpenRouter) API access.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from config.settings import OUTPUTS_DIR


def _load(variant: str) -> list[dict]:
    p = OUTPUTS_DIR / variant / "telemetry" / "calls.jsonl"
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def run_ids(variant: str) -> list[str]:
    """Run ids present in a variant's telemetry, in first-seen order."""
    seen: list[str] = []
    for r in _load(variant):
        rid = r.get("run_id")
        if rid and rid not in seen:
            seen.append(rid)
    return seen


def aggregate_tokens(variant: str, project_to: int | None = None,
                     run_id: str | None = None) -> dict:
    """
    run_id: None = every record (the file accumulates across runs, so this sums all of
    them); "latest" = the most recent run id present; anything else = that run only.
    Records written before run ids were recorded carry none and only count under None.

    Billed calls whose output was unusable (ok=False, e.g. truncated at the length
    limit) count toward tokens and time -- they were paid for -- and are reported
    separately as failed_calls.
    """
    recs = _load(variant)
    if run_id == "latest":
        ids = run_ids(variant)
        # Telemetry from before run ids were recorded has none: report all of it.
        run_id = ids[-1] if ids else None
    if run_id is not None:
        recs = [r for r in recs if r.get("run_id") == run_id]
    failed_calls = sum(1 for r in recs if not r.get("ok", True))
    samples = {}
    by_role = defaultdict(lambda: {"total": 0, "prompt": 0, "completion": 0,
                                   "calls": 0, "time": 0.0})
    for r in recs:
        sid = r.get("sample_id", "_none")
        s = samples.setdefault(sid, {"total": 0, "prompt": 0, "completion": 0,
                                     "time": 0.0, "calls": 0, "by_role": defaultdict(int)})
        tot = r.get("total_tokens", 0)
        s["total"] += tot
        s["prompt"] += r.get("prompt_tokens", 0)
        s["completion"] += r.get("completion_tokens", 0)
        s["time"] += r.get("latency_s", 0.0)
        s["calls"] += 1
        s["by_role"][r.get("role", "?")] += tot
        br = by_role[r.get("role", "?")]
        br["total"] += tot
        br["prompt"] += r.get("prompt_tokens", 0)
        br["completion"] += r.get("completion_tokens", 0)
        br["calls"] += 1
        br["time"] += r.get("latency_s", 0.0)

    n = len(samples)
    total_tokens = sum(s["total"] for s in samples.values())
    total_time = sum(s["time"] for s in samples.values())

    per_role = {}
    for role, d in sorted(by_role.items(), key=lambda kv: -kv[1]["total"]):
        per_role[role] = {
            "total_tokens": d["total"],
            "avg_tokens_per_call": round(d["total"] / d["calls"], 1) if d["calls"] else 0,
            "avg_tokens_per_sample": round(d["total"] / n, 1) if n else 0,
            "calls": d["calls"],
            "avg_calls_per_sample": round(d["calls"] / n, 2) if n else 0,
            "avg_latency_s": round(d["time"] / d["calls"], 2) if d["calls"] else 0,
        }

    avg_tokens_per_sample = round(total_tokens / n, 1) if n else 0

    out = {
        "variant": variant,
        "run_id": run_id,
        "failed_calls": failed_calls,
        "n_samples": n,                    # how many samples were ACTUALLY run
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": avg_tokens_per_sample,   # average over those n samples
        "total_time_s": round(total_time, 1),
        "avg_time_per_sample_s": round(total_time / n, 1) if n else 0,
        "per_role": per_role,              # per-agent averages over the n samples run
        "per_sample": {sid: {"total_tokens": s["total"], "prompt": s["prompt"],
                             "completion": s["completion"], "calls": s["calls"],
                             "time_s": round(s["time"], 1),
                             "by_role": dict(s["by_role"])}
                       for sid, s in sorted(samples.items())},
    }
    # Projection is OPTIONAL — only when explicitly requested (e.g. to budget 62).
    if project_to:
        out["projection"] = {
            "target_samples": project_to,
            "projected_total_tokens": round(avg_tokens_per_sample * project_to),
            "projected_per_role_tokens": {
                role: round(pr["avg_tokens_per_sample"] * project_to)
                for role, pr in per_role.items()},
        }
    return out


def report_md(agg: dict) -> str:
    v, n = agg["variant"], agg["n_samples"]
    lines = [f"# Token & Time Report — variant `{v}`", "",
             f"Samples measured: **{n}** (averages below are over these {n} run(s) only)", "",
             "## Overall", "",
             "| metric | value |", "|---|---|",
             f"| total tokens | {agg['total_tokens']:,} |",
             f"| avg tokens / sample | {agg['avg_tokens_per_sample']:,} |",
             f"| total time (s) | {agg['total_time_s']:,} |",
             f"| avg time / sample (s) | {agg['avg_time_per_sample_s']:,} |",
             f"| run id | {agg.get('run_id') or 'all runs'} |",
             f"| billed calls with unusable output | {agg.get('failed_calls', 0)} |", "",
             "## Per agent / role", "",
             "| role | total tokens | avg/call | avg/sample | calls | avg/sample calls | avg latency (s) |",
             "|---|---|---|---|---|---|---|"]
    for role, d in agg["per_role"].items():
        lines.append(f"| {role} | {d['total_tokens']:,} | {d['avg_tokens_per_call']:,} | "
                     f"{d['avg_tokens_per_sample']:,} | {d['calls']} | "
                     f"{d['avg_calls_per_sample']} | {d['avg_latency_s']} |")
    if agg.get("projection"):
        p = agg["projection"]
        lines += ["", f"## Projection to {p['target_samples']} samples "
                  "(optional — only if you asked for it)", "",
                  f"- projected TOTAL tokens: **{p['projected_total_tokens']:,}**", "",
                  "| role | projected tokens |", "|---|---|"]
        for role, t in p["projected_per_role_tokens"].items():
            lines.append(f"| {role} | {t:,} |")
    return "\n".join(lines)
