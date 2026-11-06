"""
Premise coverage — the debate's stop signal and the premise_coverage metric.

A premise counts as covered when some debate CLAIM made about that premise (its
premise_id) cites at least one evidence node that is topically relevant to the premise
(embedding cosine >= relevance_cosine_threshold, the admission gate's own relevance bar).

Until 2026-10-22 the pairing was per TURN: every node cited anywhere in a turn was paired
with every premise the turn listed as addressed. Debaters list the whole checklist (16
of 18 pilot turns), so on the case-001 pilot two trolley-philosophy papers "covered" all
six covered d1 premises, PTSD and prefrontal stress among them. Transcripts written
before claims carried a premise_id (no claim has the key at all) are still scored by
that per-turn rule, so old bundles stay readable; new runs use the per-claim rule only.

The previous rule credited every premise a turn LISTED as soon as the turn cited any
node at all. Every round-1 turn cites something and lists most of the checklist, so
coverage reached 1.0 after the opening exchange: 32 of 35 debates on the post-fix run
stopped after round 1, progressive retrieval almost never ran, and premise_coverage
read 0.997 while measuring nothing. Tying the citation to the premise by topic is
deterministic (no LLM judges coverage) and uses only stored artifacts, so the metric
can be recomputed from any audit bundle.

Turns may be DebateTurn objects or the dicts stored in transcripts/bundles.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np

from config.settings import get_settings
from src.services.embedder import embed


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cited_nodes(turn: Any) -> list[str]:
    nodes = list(_get(turn, "evidence_cited", []) or [])
    for c in _get(turn, "claims", []) or []:
        nodes.extend(_get(c, "evidence_nodes", []) or [])
    return list(dict.fromkeys(nodes))


_PID = re.compile(r"\bp\d+\b", re.I)


def claim_premise_id(claim: Any) -> str:
    """The claim's premise id normalised ("P3", "p3 (wages)" -> "p3"); "" if none."""
    m = _PID.search(str(_get(claim, "premise_id", "") or ""))
    return m.group(0).lower() if m else ""


def _legacy(turns: list[Any]) -> bool:
    """Stored transcripts from before claims carried premise_id (the key is absent)."""
    claims = [c for t in turns for c in (_get(t, "claims", []) or [])]
    return bool(claims) and all(isinstance(c, dict) and "premise_id" not in c for c in claims)


def covered_premises(premises: Iterable[Any], turns: Iterable[Any],
                     node_text: dict[str, str], threshold: float | None = None) -> set[str]:
    """Ids of premises some claim about them supports with a topically relevant node."""
    premises = list(premises)
    turns = list(turns)
    if threshold is None:
        threshold = get_settings().hp.relevance_cosine_threshold
    pid_text = {_get(p, "id"): _get(p, "text", "") for p in premises}
    pairs: list[tuple[str, str]] = []           # (premise id, cited node id)
    legacy = _legacy(turns)
    for t in turns:
        if legacy:
            nodes = [n for n in _cited_nodes(t) if node_text.get(n)]
            for pid in (_get(t, "premises_addressed", []) or []) if nodes else []:
                if pid in pid_text:
                    pairs.extend((pid, n) for n in nodes)
            continue
        for c in _get(t, "claims", []) or []:
            pid = claim_premise_id(c)
            if pid in pid_text:
                pairs.extend((pid, n) for n in (_get(c, "evidence_nodes", []) or [])
                             if node_text.get(n))
    if not pairs:
        return set()
    pids = sorted({p for p, _ in pairs})
    nids = sorted({n for _, n in pairs})
    pv = dict(zip(pids, embed([pid_text[p] for p in pids])))
    nv = dict(zip(nids, embed([node_text[n] for n in nids])))
    return {p for p, n in pairs if float(np.dot(pv[p], nv[n])) >= threshold}


def coverage_fraction(premises: Iterable[Any], turns: Iterable[Any],
                      node_text: dict[str, str], threshold: float | None = None) -> float:
    premises = list(premises)
    if not premises:
        return 0.0
    return len(covered_premises(premises, turns, node_text, threshold)) / len(premises)
