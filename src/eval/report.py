"""
P11 — Evaluation report + human-eval export. Reads the finalized per-sample audit
bundles from outputs/09_arbiter/<sid>/audit_bundle.json, computes the Tier-1 metric
suite, aggregates by stratum, and writes:
  outputs/11_eval/eval_report.json
  outputs/11_eval/eval_report.md      (readable table)
  outputs/11_eval/human_eval/<sid>.md (central scripts for the 3 expert raters)
"""
from __future__ import annotations

import json
from pathlib import Path

from config.settings import OUTPUTS_DIR
from src.eval.metrics import aggregate, compute_metrics

# Overridable so the report can target any variant (main / an ablation / a baseline).
VARIANT = "main"
_ARB_DIR = OUTPUTS_DIR / VARIANT / "09_arbiter"
_EVAL_DIR = OUTPUTS_DIR / VARIANT / "11_eval"
_INGEST_DIR = OUTPUTS_DIR / VARIANT / "01_ingestion"


def _set_variant(variant: str) -> None:
    global VARIANT, _ARB_DIR, _EVAL_DIR, _INGEST_DIR
    VARIANT = variant
    _ARB_DIR = OUTPUTS_DIR / variant / "09_arbiter"
    _EVAL_DIR = OUTPUTS_DIR / variant / "11_eval"
    _INGEST_DIR = OUTPUTS_DIR / variant / "01_ingestion"


def _stratum_for(sid: str) -> str:
    f = _INGEST_DIR / sid / "normalised.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8")).get("stratum", "unknown")
        except Exception:
            return "unknown"
    return "unknown"


def _ledger() -> set[str] | None:
    f = OUTPUTS_DIR / VARIANT / "processed_samples.txt"
    if not f.exists():
        return None
    return {ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()}


def collect_bundles() -> list[tuple[str, str, dict]]:
    """
    Return (sample_id, stratum, audit_bundle) for every FINALIZED sample.

    Finalized = listed in the variant's ledger. A case that crashed after its arbiter
    step leaves an audit bundle behind without ever reaching the ledger, and used to be
    reported alongside the completed ones. Without a ledger every bundle is used.
    """
    out = []
    if not _ARB_DIR.exists():
        return out
    done = _ledger()
    for sid_dir in sorted(_ARB_DIR.iterdir()):
        if done is not None and sid_dir.name not in done:
            continue
        ab = sid_dir / "audit_bundle.json"
        if ab.exists():
            try:
                bundle = json.loads(ab.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append((sid_dir.name, _stratum_for(sid_dir.name), bundle))
    return out


def _md_table(agg: dict) -> str:
    from src.eval.metrics import REPORT_KEYS
    keys = REPORT_KEYS
    lines = ["# Evaluation Report (Tier-1, non-LLM metrics)", "",
             f"Samples: {agg['n_samples']}", "", "## Overall", "",
             "| metric | value |", "|---|---|"]
    for k in keys:
        n = agg.get("n_measured", {}).get(k)
        lines.append(f"| {k} | {_fmt(agg['overall'][k])}"
                     + (f" (n={n})" if n is not None and n != agg["n_samples"] else "") + " |")
    lines += ["", "## By stratum", "",
              "| stratum | n | " + " | ".join(keys) + " |",
              "|" + "---|" * (len(keys) + 2)]
    for strat, row in agg["by_stratum"].items():
        lines.append(f"| {strat} | {row['n']} | " + " | ".join(_fmt(row[k]) for k in keys) + " |")
    lines += ["", "Metrics with no measured value for a sample (e.g. faithfulness not yet "
              "recomputed, or recomputed under an older definition) are excluded from that "
              "mean; n= shows how many samples it covers."]
    c, pooled = agg.get("counts") or {}, agg.get("pooled") or {}
    if c:
        lines += ["", "## Pooled integrity (all cases together)", "",
                  "| metric | pooled | counts |", "|---|---|---|",
                  f"| citation_integrity | {_fmt(pooled.get('citation_integrity'))} | "
                  f"{c['sit_cited_supported']}/{c['sit_cited']} cited SIT rows supported |",
                  f"| testimony_citation_integrity | "
                  f"{_fmt(pooled.get('testimony_citation_integrity'))} | "
                  f"{c['testimony_cited_supported']}/{c['testimony_cited']} cited claims supported |",
                  f"| testimony_citation_rate | {_fmt(pooled.get('testimony_citation_rate'))} | "
                  f"{c['testimony_cited']}/{c['testimony_claims']} claims cite evidence |"]
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return str(v)


def build_report(variant: str = "main") -> dict:
    _set_variant(variant)
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    bundles = collect_bundles()
    done = _ledger()
    if done and not bundles:
        # A ledger with no bundles (e.g. a baseline run by code that predates
        # save_outputs) used to produce an empty report that looked finished.
        raise SystemExit(f"{variant}: the ledger lists {len(done)} finished samples but no "
                         f"09_arbiter/<sid>/audit_bundle.json was found -- nothing to score")
    agg = aggregate([(strat, b) for _, strat, b in bundles])

    from src.eval.metrics import json_safe
    (_EVAL_DIR / "eval_report.json").write_text(
        json.dumps(json_safe(agg), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    (_EVAL_DIR / "eval_report.md").write_text(_md_table(agg), encoding="utf-8")

    # human-eval export: central scripts for the expert raters
    hdir = _EVAL_DIR / "human_eval"
    hdir.mkdir(parents=True, exist_ok=True)
    for sid, strat, _b in bundles:
        script = _ARB_DIR / sid / "central_script.md"
        if script.exists():
            (hdir / f"{sid}.md").write_text(
                f"<!-- sample {sid} ({strat}) — rate: Analytic Quality, Breadth, "
                f"Depth, Consistency, Schema Transparency (0-100) -->\n\n"
                + script.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[eval] {agg['n_samples']} samples -> {_EVAL_DIR/'eval_report.md'}")
    return agg
