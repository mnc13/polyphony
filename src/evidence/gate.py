"""
P3 — Evidence admission gate (three rule-based filters, no LLM).

Replaces PROCLAIM's LLM-scored admissibility (w = relevance × credibility, both
LLM-given) — the one parent component Design Law 1 forbids. A candidate chunk is
admitted only if it clears all three:

  1. Novelty     : novelty = 1 - max cosine(chunk, current pool) >= 0.20
  2. Relevance   : TOPICAL cosine(chunk, premise) >= 0.35   (embedding math;
                   hp.relevance_cosine_threshold, raised from 0.28 after the pilot)
  3. Credibility : a resolvable identifier (DOI, or the source's own PMID / PMC /
                   OpenAlex / S2 id) OR citation_count > 0

Relevance is deliberately cosine, NOT NLI entailment: an abstract seldom *entails*
a specific premise, so an entailment bar rejects on-topic evidence wholesale
(observed in P3 smoke). NLI entailment is used later, at citation time, where a
debater cites a specific chunk for a specific claim — that is where entailment is
the right test. Both are deterministic/model-based, never LLM-scored.

Rejections are returned with a reason so the gate itself is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.settings import get_settings
from src.schemas.core import Premise
from src.services.embedder import embed, embed_one, max_cosine


@dataclass
class Candidate:
    text: str
    doi: str | None
    citation_count: int
    source: str
    venue: str | None
    year: int | None
    ext_id: str | None = None


@dataclass
class GateResult:
    admitted: bool
    reason: str
    novelty: float
    relevance: float


class AdmissionGate:
    def __init__(self):
        hp = get_settings().hp
        self.novelty_threshold = hp.novelty_threshold
        self.relevance_threshold = hp.relevance_cosine_threshold

    def evaluate(
        self,
        cand: Candidate,
        premise: Premise,
        pool_vecs: np.ndarray,
        premise_vec: np.ndarray | None = None,
    ) -> GateResult:
        v = embed([cand.text])[0]

        # 1. Novelty (vs current pool)
        novelty = 1.0 - max_cosine(v, pool_vecs)
        if novelty < self.novelty_threshold:
            return GateResult(False, f"low_novelty({novelty:.2f})", novelty, 0.0)

        # 3. Credibility. A PMID or OpenAlex id resolves a record as surely as a DOI
        # does; requiring a DOI rejected DOI-less PubMed/Europe PMC records outright
        # (PubMed never reports citation counts), 44 of them on the post-fix run.
        credible = bool(cand.doi) or bool(cand.ext_id) or cand.citation_count > 0
        if not credible:
            return GateResult(False, "no_provenance", novelty, 0.0)

        # 2. Relevance — topical cosine to the premise (embedding math, not NLI)
        pv = premise_vec if premise_vec is not None else embed_one(premise.text)
        rel = float(np.dot(v, pv))
        if rel < self.relevance_threshold:
            return GateResult(False, f"off_topic({rel:.2f})", novelty, rel)

        return GateResult(True, "admitted", novelty, rel)
