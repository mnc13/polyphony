"""
P18 — evidence queries must be indexable, and citation integrity must test the claim.

Both behaviours regressed silently on the first six-case run and only surfaced when the
metrics were questioned, so they are pinned here.

The first run sent whole premise sentences to the literature APIs. PubMed, EuropePMC and
OpenAlex are term matchers and returned nothing, so every premise fell through to the
CrossRef fallback: 184 of 195 evidence nodes came from CrossRef and 163 of 195 were
title-only. Titles cannot ground specific claims, which is what drove citation integrity
to 0.024 and left most testimony reporting that the evidence did not settle its premise.
"""
from __future__ import annotations

from src.evidence.pmer import PMER, keywords
from src.schemas.core import Dimension, Premise

PREMISE_TEXT = (
    "Undisclosed use of an imperfect memory-implant protocol raises the probability that "
    "S1 suffers licensure loss, income collapse, and stress-related health decline if the "
    "concealment is exposed."
)


def _premise() -> Premise:
    return Premise(
        id="p1", text=PREMISE_TEXT, discipline="sociology", dimension=Dimension.SOCIO,
        polarity=-1, stakeholder_ids=["S1"],
    )


def _pmer() -> PMER:
    return PMER.__new__(PMER)  # no vector db needed: query building is pure


def test_keywords_drop_stopwords_ids_and_short_tokens():
    kw = keywords(PREMISE_TEXT)
    assert "the" not in kw and "that" not in kw and "raises" not in kw
    assert "s1" not in kw, "stakeholder placeholders mean nothing to a literature index"
    assert all(len(w) >= 4 for w in kw)
    assert "memory-implant" in kw, "hyphenated compounds are kept; indexes tokenise them"


def test_keywords_are_deduplicated_and_ordered_by_appearance():
    kw = keywords("Memory and memory and cohesion of memory", limit=6)
    assert kw == ["memory", "cohesion"]


def test_query_is_terms_not_the_sentence():
    q = _pmer().build_query(_premise(), "the clinician who performed it; licensed")
    assert PREMISE_TEXT not in q, "sending the sentence is what broke retrieval"
    assert "probability" not in q and " that " not in f" {q} "
    # Field keywords are NOT on the first rung: PubMed/Europe PMC AND every term, and
    # the "3 terms -> 8 docs" measurement behind the ladder was taken without them.
    assert "sociology" not in q
    assert q.split()[:3] == keywords(PREMISE_TEXT)[:3]


def test_ladder_relaxes_specificity_then_falls_back_to_the_field():
    rungs = _pmer().build_query_ladder(_premise(), "the clinician; licensed")
    assert len(rungs) >= 2, "a single rung cannot recover from a zero-hit query"
    terms = keywords(PREMISE_TEXT)
    # Premise-derived rungs only ever drop premise terms.
    premise_counts = [sum(1 for w in r.split() if w in terms) for r in rungs]
    assert premise_counts == sorted(premise_counts, reverse=True), premise_counts
    assert rungs[-2].split()[:2] == terms[:2] and rungs[-2].split()[-1] == "sociology"
    assert rungs[-1] == "sociology social cohesion institutions", (
        "the final rung is the field alone, which always matches something topical"
    )
    assert len(set(rungs)) == len(rungs), "duplicate rungs would waste a round trip"


def test_discipline_sets_the_field_vocabulary():
    p = _premise()
    p.discipline = "economics"
    rungs = _pmer().build_query_ladder(p, "x")
    assert rungs[-1] == "economics incentives welfare"
    assert "sociology" not in " ".join(rungs)


def test_debate_gap_hint_leads_the_ladder():
    hint = "longitudinal data on licensure revocation after malpractice disclosure"
    rungs = _pmer().build_query_ladder(_premise(), "x", hint=hint)
    assert rungs[0].split()[0] in keywords(hint)
    plain = _pmer().build_query_ladder(_premise(), "x")
    assert rungs[0] not in plain, "a hinted round must not replay the round-1 query"


def test_every_rung_stays_within_the_length_cap():
    long_premise = _premise()
    long_premise.text = " ".join(f"consideration{i}" for i in range(200))
    for r in _pmer().build_query_ladder(long_premise, "x"):
        assert len(r) <= PMER.MAX_QUERY_CHARS


def test_citation_hypothesis_is_the_matching_sentence_not_the_opening():
    from src.debate.engine import _closest_sentence

    argument = (
        "I will argue that the decision is sound with respect to Survivability. "
        "Caregiver psychological states before loss predict complicated grief outcomes. "
        "Therefore the panel should weigh institutional trust heavily."
    )
    evidence = "Pre-loss symptoms related to risk of complicated grief in caregivers"
    picked = _closest_sentence(evidence, argument)
    assert "complicated grief" in picked
    assert not picked.startswith("I will argue"), (
        "scoring the turn's opening framing is what produced citation_integrity 0.024"
    )


def test_closest_sentence_falls_back_when_there_is_nothing_to_score():
    from src.debate.engine import _closest_sentence

    assert _closest_sentence("evidence", "") == ""
    assert _closest_sentence("evidence", "Short.") == "Short."[:400]
