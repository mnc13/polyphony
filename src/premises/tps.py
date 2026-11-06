"""
P6 — Premise typing, dedup, and the Typed Premise Set (Step 5b).

- Dedup near-identical premises by embedding cosine (0.85), keeping the MOST SPECIFIC
  (longest) of a duplicate cluster -- but only between premises with the same
  polarity, dimension and stakeholders that NLI does not read as contradicting.
- Re-id the survivors p1..pn.
- Build a supports/conflicts graph. To avoid an O(n^2) NLI sweep, relations are only
  computed between premises that share a stakeholder AND a dimension (the pairs where
  support/conflict is actually meaningful): opposite polarity -> test NLI contradiction;
  same polarity -> test NLI entailment.

The TypedPremiseSet (premise list) is the debate's checklist; the relations are an
auxiliary graph exported alongside for later analysis.
"""
from __future__ import annotations

import numpy as np

from config.settings import get_settings
from src.schemas.core import Premise, PremiseRelation, TypedPremiseSet
from src.services import nli
from src.services.embedder import embed

_REL_THRESHOLD = 0.55


# Two premises above the cosine threshold are still NOT duplicates if NLI reads them as
# contradicting each other.
_DEDUP_CONTRADICTION = 0.50


def _is_duplicate(p: Premise, v: np.ndarray, q: Premise, qv: np.ndarray, threshold: float) -> bool:
    """
    Near-duplicate = same claim restated, not merely similar wording.

    Embedding cosine cannot see a flipped verb: "X reduces S1's risk" and "X increases
    S1's risk" score 0.94-0.99, so the opposite claim was silently dropped at 0.85 and
    only one side of the question reached the debate. A pair is a duplicate only if it
    is that similar AND has the same polarity, dimension and stakeholders AND NLI does
    not read it as a contradiction.
    """
    if float(np.dot(v, qv)) < threshold:
        return False
    if p.polarity != q.polarity or p.dimension != q.dimension:
        return False
    if set(p.stakeholder_ids) != set(q.stakeholder_ids):
        return False
    return nli.contradicts(q.text, p.text) < _DEDUP_CONTRADICTION


def _dedup_keep_specific(premises: list[Premise], threshold: float) -> list[Premise]:
    # Longest text first, so the most specific survives a near-duplicate cluster.
    ordered = sorted(premises, key=lambda p: len(p.text), reverse=True)
    kept: list[Premise] = []
    kept_vecs: list[np.ndarray] = []
    vecs = {id(p): v for p, v in zip(ordered, embed([p.text for p in ordered]))}
    for p in ordered:
        v = vecs[id(p)]
        if any(_is_duplicate(p, v, q, qv, threshold) for q, qv in zip(kept, kept_vecs)):
            continue
        kept.append(p)
        kept_vecs.append(v)
    return kept


def _relations(premises: list[Premise]) -> list[PremiseRelation]:
    rels: list[PremiseRelation] = []
    for i in range(len(premises)):
        for j in range(i + 1, len(premises)):
            a, b = premises[i], premises[j]
            if a.dimension != b.dimension:
                continue
            if not set(a.stakeholder_ids) & set(b.stakeholder_ids):
                continue
            if a.polarity != 0 and b.polarity != 0 and a.polarity != b.polarity:
                c = nli.contradicts(a.text, b.text)
                if c >= _REL_THRESHOLD:
                    rels.append(PremiseRelation(src=a.id, dst=b.id, type="conflicts", score=round(c, 3)))
            else:
                e = nli.entails(a.text, b.text)
                if e >= _REL_THRESHOLD:
                    rels.append(PremiseRelation(src=a.id, dst=b.id, type="supports", score=round(e, 3)))
    return rels


def build_tps(
    case_id: str,
    decision_id: str,
    premises: list[Premise],
    compute_relations: bool = True,
) -> tuple[TypedPremiseSet, list[PremiseRelation]]:
    hp = get_settings().hp
    deduped = _dedup_keep_specific(premises, hp.dedup_cosine_threshold)
    # Re-id p1..pn in stable order (by original discipline then text).
    deduped.sort(key=lambda p: (p.discipline.value, p.text))
    reid: list[Premise] = []
    remap: dict[str, str] = {}
    for i, p in enumerate(deduped, start=1):
        new = f"p{i}"
        remap[p.id] = new
        reid.append(p.model_copy(update={"id": new}))
    rels = _relations(reid) if compute_relations else []
    return TypedPremiseSet(case_id=case_id, decision_id=decision_id, premises=reid), rels
