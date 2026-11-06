"""
P7 — Scientist testimony generation + NLI citation gate (Step 7b).

For one selected discipline:
  1. gather its premises (those tagged to it) and the candidate TEG nodes for its
     dimension;
  2. the scientist_pool model, wearing that discipline's persona, testifies —
     structured claims citing node ids;
  3. deterministically gate each claim: validate cited nodes exist, and run NLI
     citation-integrity. `nli_supported` is stamped by code, never by the LLM (see
     `claim_supported` for exactly what has to hold).

Testimony is a FLOOR, not a ceiling: P-MER stays live during the debate (P8), so
debaters can retrieve past what the scientists supplied.
"""
from __future__ import annotations

from config.settings import get_settings
from src.ingestion.anchor import AnchorConfig
from src.prompts.testimony import testimony_system, testimony_user
from src.schemas.core import (
    Discipline, Premise, ScientistTestimony, ScientistTestimonyDraft,
    StakeholderRegister, TestimonyClaim,
)
from src.services import nli
from src.services.llm_client import get_client
from src.services.vacuity import committed_sentences, is_vacuous, sentences
from src.evidence.teg import TypedEvidenceGraph

# Committed sentences of a finding checked against each cited node. Findings run to a
# handful of sentences; the cap only bounds the NLI cost of an unusually long one.
_MAX_SENTENCES_CHECKED = 6


def claim_supported(finding: str, evidence_strength: str, cited_texts: list[str],
                    premise_text: str | None = None) -> bool:
    """
    True iff a cited node entails something the finding actually ASSERTS about the
    premise.

    The finding used to be scored whole against each node. Most findings open with a
    hedge ("The provided evidence does not settle whether ...") and go on to describe
    what the cited paper studied, and DeBERTa rates a paper as entailing a hedge about
    itself (0.979 on 001/d1/p7). So 22 of 24 cited claims came out "supported" on the
    post-fix run, 19 of them hedges -- the 91.7% testimony integrity figure measured
    nothing. Now:
      * a claim the scientist rates 'insufficient' is not evidence-supported, whatever
        it cites -- the witness has said the evidence does not carry it;
      * only committed (non-hedge) sentences are hypotheses; a finding that is nothing
        but hedges has nothing for evidence to support;
      * with the premise given, a hypothesis must also be ABOUT it (embedding cosine to
        the premise >= the admission gate's relevance bar). A sentence that only
        describes what the cited paper studied ("TEG-d4-6 explores uncertainty in
        breast cancer survivors") is entailed by that paper whatever the premise says;
        it is an accurate description, not support;
      * a 'weak' claim whose verdict (its first sentence) is itself a hedge is not
        supported. On the case-001 pilot, three of the four claims stamped as
        supported opened "The provided evidence does not establish that ..." and were
        carried by a later sentence describing the cited paper. A witness that rates
        the evidence weak and opens by disclaiming the premise has not testified that
        the evidence carries it. 'moderate'/'strong' claims are left to the NLI check.
    """
    if not cited_texts or evidence_strength == "insufficient":
        return False
    sents = sentences(finding)
    if not sents or (evidence_strength == "weak" and is_vacuous(sents[0])):
        return False
    hyps = committed_sentences(finding)[:_MAX_SENTENCES_CHECKED]
    if hyps and premise_text:
        from src.services.embedder import embed
        vecs = embed(hyps + [premise_text])
        bar = get_settings().hp.relevance_cosine_threshold
        hyps = [h for h, v in zip(hyps, vecs[:-1]) if float(v @ vecs[-1]) >= bar]
    if not hyps:
        return False
    pairs = [(ev, h) for ev in cited_texts for h in hyps]
    return any(s >= get_settings().hp.nli_entailment_threshold
               for s in nli.entails_batch(pairs))


def clamp_strength(strength: str, cited: bool, supported: bool) -> str:
    """
    Strength describes the evidence actually in hand, so it is clamped by code rather
    than trusted from the model: no surviving citation cannot be 'moderate'/'strong',
    and a citation that no NLI check supports cannot be 'strong'. Direction is left
    alone — a field leaning one way with thin evidence is a real (and reportable)
    state; `nli_supported` still marks what is actually grounded.
    """
    if not cited and strength in ("strong", "moderate"):
        return "weak"
    if cited and not supported and strength == "strong":
        return "moderate"
    return strength


def _nodes_for(teg: TypedEvidenceGraph, premises: list[Premise]) -> list[dict]:
    dims = {p.dimension.value for p in premises}
    return [data for _, data in teg.g.nodes(data=True) if data.get("dimension") in dims]


def testify(
    discipline: Discipline,
    premises: list[Premise],
    sr: StakeholderRegister,
    teg: TypedEvidenceGraph,
    anchor: AnchorConfig,
) -> ScientistTestimony:
    nodes = _nodes_for(teg, premises)
    node_ids = {n["node_id"] for n in nodes}
    node_text = {n["node_id"]: n["text"] for n in nodes}

    client = get_client("scientist_pool")
    draft: ScientistTestimonyDraft = client.generate_structured(
        user=testimony_user(premises, nodes, sr),
        schema=ScientistTestimonyDraft,
        system=testimony_system(discipline, anchor),
        max_tokens=2200,
    )

    premise_text = {p.id: p.text for p in premises}
    gated: list[TestimonyClaim] = []
    for c in draft.claims:
        if c.premise_id not in premise_text:
            continue
        cited = [nid for nid in c.cited_nodes if nid in node_ids]  # drop hallucinated ids
        supported = claim_supported(c.finding, c.evidence_strength,
                                    [node_text[nid] for nid in cited],
                                    premise_text=premise_text[c.premise_id])
        strength = clamp_strength(c.evidence_strength, bool(cited), supported)
        gated.append(TestimonyClaim(
            premise_id=c.premise_id,
            finding=c.finding.strip(),
            direction=c.direction,
            evidence_strength=strength,
            cited_nodes=cited,
            nli_supported=supported,
        ))
    return ScientistTestimony(discipline=discipline, claims=gated)
