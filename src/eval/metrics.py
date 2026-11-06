"""
P11 — Tier-1 metrics (no LLM scoring). All computed deterministically from a sample's
audit bundle:

  premise_coverage        mean per-decision coverage: premises addressed with a cited node
                          that is topically relevant to the premise (src/debate/coverage.py),
                          RECOMPUTED from the bundle so every run is scored by one rule
  premise_coverage_as_run the coverage the debate engine stored when it ran
  citation_integrity      of SIT claims that cite evidence, fraction NLI-supported
  sit_profile             counts: cited / asserted / rebutted / nli_supported / total
                          (rebutted is derived from row order; see src/storage/sit.py)
  stakeholder_coverage    SR stakeholders with >=1 NLI-SUPPORTED cited SIT row / total
  stakeholder_coverage_cited   the old definition: any cited row, supported or not
  script_faithfulness     the arbiter faithfulness check; NaN until actually measured
  provenance_rate         distinct TEG chunks with a resolvable id (DOI or source id) /
                          distinct chunks -- ~1.0 by construction, since the admission
                          gate requires one; doi_rate is the informative figure
  doi_rate                distinct TEG chunks with a DOI / distinct chunks
  testimony_grounding     testimony claims that are NLI-supported / total claims
  testimony_citation_rate         claims citing any evidence / total claims
  testimony_citation_integrity    of the claims that cite, the fraction NLI-supported

Every number is recomputable from stored bundles alone — full reproducibility.
NaN means "not measurable for this sample" and is excluded from means, never averaged in.
"""
from __future__ import annotations

import math
from statistics import mean

# Debaters sometimes write "S2 (Patients)" for "S2"; ids are reconciled before any
# metric compares them against the register, or the same stakeholder counts twice.
# The helper lives with the SIT store, which needs it for rebuttal matching too.
from src.storage.sit import canonical_stakeholder_id, mark_rebuttals  # noqa: F401

NAN = float("nan")

REPORT_KEYS = ["premise_coverage", "citation_integrity", "stakeholder_coverage",
               "script_faithfulness", "provenance_rate", "doi_rate",
               "testimony_grounding", "testimony_citation_rate",
               "testimony_citation_integrity"]

_AGG_KEYS = REPORT_KEYS + ["premise_coverage_as_run", "stakeholder_coverage_cited"]


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


def script_faithfulness_of(bundle: dict) -> float:
    """
    The measured faithfulness, or NaN if the bundle only holds the placeholder.

    With in-run faithfulness off (the default), the pipeline stored 1.0 for every case
    until the offline recompute replaced it, and build_eval averaged that 1.0 in as if it
    were measured: the main run's mean read 0.97 with four of six cases unmeasured.
    Bundles now say whether the value was measured; older ones are judged by their
    shape (a recompute stamps `recomputed_offline`; the bare placeholder is exactly
    1.0 with no ungrounded sentences).
    """
    fb = bundle.get("faithfulness", {}) or {}
    v = fb.get("fraction_entailed")
    if not _is_num(v):
        return NAN
    # Only figures under the CURRENT definition: a variant part-recomputed after a
    # definition change would otherwise average incomparable numbers (main's two v1
    # cases read 0.909 under a v2 report). Older figures are NaN until recomputed.
    from src.eval.faithfulness import FAITHFULNESS_DEF
    if fb.get("definition") != FAITHFULNESS_DEF:
        return NAN
    if fb.get("recomputed_offline") or fb.get("measured") is True:
        return float(v)
    return NAN


def _premise_coverage(decisions: list[dict]) -> float:
    from src.debate.coverage import coverage_fraction
    covs = []
    for d in decisions:
        premises = (d.get("tps") or {}).get("premises", [])
        turns = (d.get("debate") or {}).get("turns", [])
        nodes = (d.get("teg") or {}).get("nodes", [])
        if not premises:
            continue
        node_text = {n.get("node_id"): n.get("text", "") for n in nodes}
        covs.append(coverage_fraction(premises, turns, node_text))
    return round(mean(covs), 3) if covs else NAN


def _unique_nodes(decisions: list[dict]) -> list[dict]:
    """TEG chunks counted once each, however many decisions retrieved them."""
    seen, out = set(), []
    for d in decisions:
        for n in (d.get("teg") or {}).get("nodes", []):
            key = (n.get("doi") or n.get("ext_id") or "", (n.get("text") or "")[:200])
            if key not in seen:
                seen.add(key)
                out.append(n)
    return out


def compute_metrics(bundle: dict) -> dict:
    decisions = bundle.get("decisions", [])
    sit = mark_rebuttals(bundle.get("sit_rows", []))
    sr = bundle.get("stakeholder_register", {}).get("stakeholders", [])

    # premise coverage — recomputed under the current rule, plus the stored value
    stored = [d.get("debate", {}).get("coverage") for d in decisions]
    stored = [c for c in stored if _is_num(c)]
    premise_coverage_as_run = round(mean(stored), 3) if stored else NAN
    premise_coverage = _premise_coverage(decisions)

    # SIT profile
    cited = [r for r in sit if r.get("evidence_node")]
    asserted = [r for r in sit if not r.get("evidence_node")]
    rebutted = [r for r in sit if r.get("rebutted")]
    nli_ok = [r for r in sit if r.get("nli_supported")]
    citation_integrity = (round(len([r for r in cited if r.get("nli_supported")]) / len(cited), 3)
                          if cited else NAN)

    # stakeholder coverage. Counting any cited row put this at 1.0 on all six cases of
    # the post-fix run while citation_integrity was 0.15-0.36: a stakeholder "covered"
    # by a citation that NLI rejected is not covered by evidence.
    total_sr = len(sr)
    register_ids = {canonical_stakeholder_id(x.get("id", "")) for x in sr}
    cited_ids = {canonical_stakeholder_id(r["stakeholder_id"]) for r in cited} & register_ids
    supported_ids = {canonical_stakeholder_id(r["stakeholder_id"])
                     for r in cited if r.get("nli_supported")} & register_ids
    stakeholder_coverage = round(len(supported_ids) / total_sr, 3) if total_sr else NAN
    stakeholder_coverage_cited = round(len(cited_ids) / total_sr, 3) if total_sr else NAN

    # provenance over DISTINCT chunks (node instances repeat across decisions)
    nodes = _unique_nodes(decisions)
    n_instances = sum(len((d.get("teg") or {}).get("nodes", [])) for d in decisions)
    with_id = [n for n in nodes if n.get("doi") or n.get("ext_id")]
    with_doi = [n for n in nodes if n.get("doi")]
    provenance_rate = round(len(with_id) / len(nodes), 3) if nodes else NAN
    doi_rate = round(len(with_doi) / len(nodes), 3) if nodes else NAN

    # testimony grounding
    #
    # Reported as three numbers, because the single ratio is actively misleading. It
    # divides NLI-supported claims by ALL claims, and most claims cite nothing at all --
    # they report that the literature does not settle their premise, so they are
    # unsupported by construction rather than by any failure of grounding. On case 002
    # the ratio FELL from 0.164 to 0.086 across a run in which every cited claim became
    # supported (5/5 against 9/10), purely because fewer claims cited and more claims
    # existed. Splitting it separates "did the scientist find citable evidence" from
    # "when they cited it, did it hold up"; the product of the two is the old figure,
    # which is kept so earlier reports stay comparable. Support itself is stamped by
    # src/testimony/testify.py:claim_supported (hedges are not support).
    tclaims = [c for d in decisions for t in d.get("testimony", []) for c in t.get("claims", [])]
    tcited = [c for c in tclaims if c.get("cited_nodes")]
    tgrounded = [c for c in tclaims if c.get("nli_supported")]
    testimony_grounding = round(len(tgrounded) / len(tclaims), 3) if tclaims else NAN
    testimony_citation_rate = round(len(tcited) / len(tclaims), 3) if tclaims else NAN
    testimony_citation_integrity = (
        round(len([c for c in tcited if c.get("nli_supported")]) / len(tcited), 3)
        if tcited else NAN
    )

    return {
        "premise_coverage": premise_coverage,
        "premise_coverage_as_run": premise_coverage_as_run,
        "citation_integrity": citation_integrity,
        "stakeholder_coverage": stakeholder_coverage,
        "stakeholder_coverage_cited": stakeholder_coverage_cited,
        "script_faithfulness": script_faithfulness_of(bundle),
        "provenance_rate": provenance_rate,
        "doi_rate": doi_rate,
        "testimony_grounding": testimony_grounding,
        "testimony_citation_rate": testimony_citation_rate,
        "testimony_citation_integrity": testimony_citation_integrity,
        "sit_profile": {
            "total": len(sit), "cited": len(cited), "asserted": len(asserted),
            "rebutted": len(rebutted), "nli_supported": len(nli_ok),
        },
        # numerators/denominators of the two integrity ratios, for pooled figures
        "counts": {
            "sit_cited": len(cited),
            "sit_cited_supported": len([r for r in cited if r.get("nli_supported")]),
            "testimony_claims": len(tclaims),
            "testimony_cited": len(tcited),
            "testimony_cited_supported": len([c for c in tcited if c.get("nli_supported")]),
        },
        "n_decisions": len(decisions),
        "n_teg_nodes": n_instances,
        "n_teg_distinct_chunks": len(nodes),
    }


def aggregate(bundles: list[tuple[str, dict]]) -> dict:
    """
    Aggregate metrics across finalized samples, broken down by stratum.
    `bundles` is a list of (stratum, bundle). Returns overall + per-stratum means
    (means of per-sample values, NaN samples excluded) and, per metric, how many
    samples the mean covers.
    """
    per_sample = []
    for stratum, b in bundles:
        m = compute_metrics(b)
        m["stratum"] = stratum
        per_sample.append(m)

    def _vals(rows, key):
        return [r[key] for r in rows if _is_num(r.get(key))]

    def _avg(rows, key):
        vals = _vals(rows, key)
        return round(mean(vals), 3) if vals else NAN

    overall = {k: _avg(per_sample, k) for k in _AGG_KEYS}
    n_measured = {k: len(_vals(per_sample, k)) for k in _AGG_KEYS}
    by_stratum: dict = {}
    for strat in sorted({r["stratum"] for r in per_sample}):
        rows = [r for r in per_sample if r["stratum"] == strat]
        by_stratum[strat] = {"n": len(rows), **{k: _avg(rows, k) for k in _AGG_KEYS}}

    # Pooled (micro) integrity: the per-case means above weight a case with one cited
    # claim like a case with forty (main's testimony integrity read 0.958 as a mean of
    # cases, 0.917 pooled: 22 of 24), so both are reported, with their counts.
    tot = {k: sum(r["counts"][k] for r in per_sample) for k in per_sample[0]["counts"]}         if per_sample else {}
    pooled = {
        "citation_integrity": (round(tot["sit_cited_supported"] / tot["sit_cited"], 3)
                               if tot.get("sit_cited") else NAN),
        "testimony_citation_integrity": (
            round(tot["testimony_cited_supported"] / tot["testimony_cited"], 3)
            if tot.get("testimony_cited") else NAN),
        "testimony_citation_rate": (round(tot["testimony_cited"] / tot["testimony_claims"], 3)
                                    if tot.get("testimony_claims") else NAN),
    }
    return {"n_samples": len(per_sample), "overall": overall, "n_measured": n_measured,
            "pooled": pooled, "counts": tot,
            "by_stratum": by_stratum, "per_sample": per_sample}


def json_safe(obj):
    """NaN -> None recursively, so reports are valid JSON (json.dumps writes bare NaN)."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj
