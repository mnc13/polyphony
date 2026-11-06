"""
P11 — Anchor sensitivity (no LLM). Run the SAME case under different anchors and
compare the structured records: do the premises / dimensions / stakeholder-claim
directions change with the anchor, or only the vocabulary? Computed on the audit
bundles — no LLM judgment.
"""
from __future__ import annotations

from src.eval.metrics import canonical_stakeholder_id


def _claim_signatures(bundle: dict) -> set[tuple]:
    """
    (stakeholder, dimension, polarity) signatures over all SIT rows.

    Ids are canonicalised first: a debater that writes "S2 (Patients)" where another
    wrote "S2" would otherwise contribute a second, distinct signature, depressing the
    Jaccard overlap between anchors for a purely cosmetic difference and making anchors
    look more divergent than they are.
    """
    return {(canonical_stakeholder_id(r["stakeholder_id"]), r["dimension"], r["polarity"])
            for r in bundle.get("sit_rows", [])}


def _dimension_mix(bundle: dict) -> dict:
    mix: dict[str, int] = {}
    for d in bundle.get("decisions", []):
        for p in d.get("tps", {}).get("premises", []):
            mix[p["dimension"]] = mix.get(p["dimension"], 0) + 1
    return mix


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return round(len(a & b) / len(a | b), 3) if (a | b) else 0.0


def compare_anchors(bundles_by_anchor: dict[str, dict]) -> dict:
    """
    bundles_by_anchor: {anchor_label: audit_bundle} for the SAME case.
    Returns pairwise claim-signature overlap + dimension mixes.
    """
    anchors = list(bundles_by_anchor)
    sigs = {a: _claim_signatures(bundles_by_anchor[a]) for a in anchors}
    mixes = {a: _dimension_mix(bundles_by_anchor[a]) for a in anchors}

    pairwise = {}
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            a, b = anchors[i], anchors[j]
            pairwise[f"{a}|{b}"] = _jaccard(sigs[a], sigs[b])

    return {
        "anchors": anchors,
        "claim_signature_overlap": pairwise,
        "dimension_mix": mixes,
        "interpretation": (
            "Low overlap => the anchor materially changes which considerations surface "
            "(bounded relativism is doing real work). High overlap => anchors converge "
            "empirically."
        ),
    }
