"""
P2 — NLI service (the no-LLM-scoring backbone).

Every quality judgment in the pipeline that could tempt an LLM-as-judge routes
through this DeBERTa-v3 entailment model instead (Design Law 1). One core call,
`entails(premise, hypothesis) -> P(entailment)`, plus four task wrappers:

  (a) relevance         — does an evidence chunk pertain to a premise?      (gate)
  (b) citation_integrity— does a cited chunk entail the claim it supports?  (SIT/audit)
  (c) testimony_fidelity— does a debater's use entail what a scientist said? (audit)
  (d) faithfulness      — SummaC-style: is each summary sentence entailed by
                          some source span?                                 (arbiter)

The model is loaded lazily so importing this module is cheap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from config.settings import get_settings

_pipe = None
_ENTAIL_LABELS = {"entailment", "entail"}


def _get_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        hp = get_settings().hp
        _pipe = pipeline("text-classification", model=hp.nli_model,
                         top_k=None, truncation=True,
                         # Without an explicit max_length, truncation=True is a no-op:
                         # the tokenizer reports model_max_length=1e19 while the model
                         # only has 512 positions. See Hyperparams.nli_max_length.
                         max_length=hp.nli_max_length)
    return _pipe


def entails(premise: str, hypothesis: str) -> float:
    """P(premise entails hypothesis) in [0,1]. 'premise' is the source/evidence text."""
    if not premise.strip() or not hypothesis.strip():
        return 0.0
    # DeBERTa NLI expects "premise [SEP] hypothesis"; the pipeline takes a pair via text/text_pair.
    out = _get_pipe()({"text": premise, "text_pair": hypothesis})
    scores = out[0] if isinstance(out[0], list) else out
    for item in scores:
        if item["label"].lower() in _ENTAIL_LABELS:
            return float(item["score"])
    return 0.0


def _passes(score: float) -> bool:
    return score >= get_settings().hp.nli_entailment_threshold


# Note: admission-gate *relevance* is topical embedding cosine (see evidence/gate.py),
# not entailment — an abstract rarely entails a specific premise. NLI entailment is
# used below at citation time, where a specific claim cites specific evidence.


# --- (b) citation integrity --------------------------------------------------
def citation_integrity(evidence_text: str, claim_text: str) -> tuple[bool, float]:
    s = entails(evidence_text, claim_text)
    return _passes(s), s


# --- (c) testimony fidelity --------------------------------------------------
def testimony_fidelity(testimony_text: str, debater_use_text: str) -> tuple[bool, float]:
    """Did the debater's representation stay faithful to what the scientist testified?"""
    s = entails(testimony_text, debater_use_text)
    return _passes(s), s


# --- (d) faithfulness (SummaC-style) -----------------------------------------
@dataclass
class SentenceVerdict:
    sentence: str
    entailed: bool
    best_score: float


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def entails_batch(pairs: list[tuple[str, str]]) -> list[float]:
    """
    P(premise entails hypothesis) for many (premise, hypothesis) pairs in one forward
    pass. One-at-a-time calls are dominated by per-call overhead on CPU — a 14-token
    forward measured 1,272 ms on this machine against ~1,075 ms for the bare model
    call — so batching is worth far more here than any input-size tuning.
    """
    if not pairs:
        return []
    pipe = _get_pipe()
    inputs = [{"text": a, "text_pair": b} for a, b in pairs]
    out = pipe(inputs, batch_size=min(16, len(inputs)))
    scores: list[float] = []
    for item in out:
        row = item if isinstance(item, list) else [item]
        s = 0.0
        for lab in row:
            if lab["label"].lower() in _ENTAIL_LABELS:
                s = float(lab["score"])
                break
        scores.append(s)
    return scores


def faithfulness(summary: str, source_spans: list[str],
                 top_k: int | None = None) -> tuple[float, list[SentenceVerdict]]:
    """
    Fraction of summary sentences entailed by at least one source span.

    Two departures from the naive all-pairs loop, both necessary and both improvements:

    1. CANDIDATE PREFILTER. Scoring every sentence against every span is
       len(sents) x len(spans) NLI calls — 12,240 for a six-decision case, which at this
       machine's measured ~4 s per call is over fourteen hours. Spans are instead ranked
       by embedding cosine first (MiniLM, milliseconds for the whole set) and only the
       top `top_k` go to NLI. A span with no topical overlap essentially never entails,
       so little is lost.

    2. BATCHED SCORING. The surviving pairs are scored in one batched call per sentence
       rather than one call each.

    The prefilter also makes the metric stricter, not looser. The previous max over ~69
    candidates at a 0.60 cutoff made a pass near-certain for any sentence, which is why
    every case scored exactly 1.000; taking the max over the few topically relevant spans
    is what the metric was meant to express.
    """
    from src.services.embedder import embed

    hp = get_settings().hp
    k = top_k if top_k is not None else getattr(hp, "nli_faithfulness_top_k", 5)
    sents = _split_sentences(summary)
    if not sents or not source_spans:
        return 1.0, []

    span_vecs = embed(list(source_spans))
    sent_vecs = embed(sents)
    sims = sent_vecs @ span_vecs.T          # both L2-normalised

    verdicts: list[SentenceVerdict] = []
    for i, sent in enumerate(sents):
        order = sims[i].argsort()[::-1][:k]
        cands = [source_spans[j] for j in order]
        scores = entails_batch([(c, sent) for c in cands])
        best = max(scores) if scores else 0.0
        verdicts.append(SentenceVerdict(sent, _passes(best), best))
    frac = sum(v.entailed for v in verdicts) / len(verdicts)
    return frac, verdicts


def contradicts(text_a: str, text_b: str) -> float:
    """P(contradiction) — used to draw TEG contradiction edges."""
    out = _get_pipe()({"text": text_a, "text_pair": text_b})
    scores = out[0] if isinstance(out[0], list) else out
    for item in scores:
        if item["label"].lower().startswith("contradict"):
            return float(item["score"])
    return 0.0
