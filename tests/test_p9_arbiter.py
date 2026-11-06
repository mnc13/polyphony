"""
P9 — arbiter compile wiring + faithfulness gate + audit bundle. Arbiter LLM mocked;
the NLI faithfulness check runs for real on a tiny input.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.anchor import get_anchor_config
from src.schemas.core import (
    Anchor, DebateResult, DebateTurn, Dimension, Discipline, Premise,
    ScientistTestimony, Stakeholder, StakeholderRegister, TestimonyClaim,
    TurnClaim, TypedPremiseSet,
)
from src.schemas.core import TestimonySet as _TestimonySet


def _materials():
    from src.arbiter.compile import DecisionMaterial
    tps = TypedPremiseSet(case_id="c1", decision_id="d1", premises=[
        Premise(id="p1", text="Radium extends survival.", dimension=Dimension.NEURO,
                discipline=Discipline.health_science, stakeholder_ids=["S1"], polarity=1)])
    tset = _TestimonySet(case_id="c1", decision_id="d1", testimonies=[
        ScientistTestimony(discipline=Discipline.health_science, claims=[
            TestimonyClaim(premise_id="p1", finding="Radium treatment improves cancer survival.",
                           direction=1, evidence_strength="moderate", cited_nodes=["TEG-d1-1"],
                           nli_supported=True)])])
    turns = [DebateTurn(round=1, speaker="proponent", premises_addressed=["p1"],
                        evidence_cited=["TEG-d1-1"],
                        claims=[TurnClaim(stakeholder_id="S1", dimension=Dimension.NEURO,
                                          polarity=1, evidence_nodes=["TEG-d1-1"])],
                        evidential_gap="", argument_text="Radium treatment improves survival for the patient.")]
    result = DebateResult(case_id="c1", decision_id="d1", rounds_run=1,
                          stop_reason="coverage", coverage=1.0, turns=turns)
    return [DecisionMaterial(decision_id="d1", statement="Heinz should steal the drug.",
                             tps=tps, testimony=tset, result=result)]


def test_arbiter_compile_and_faithfulness(monkeypatch):
    import src.arbiter.compile as arb

    class _Fake:
        def generate(self, user, system=None, max_tokens=0, **kw):
            # One grounded sentence + one hallucinated sentence.
            return ("Radium treatment improves survival for the patient. "
                    "The moon is made of cheese.")

    monkeypatch.setattr(arb, "get_client", lambda role: _Fake())
    out = arb.compile_central_script("Heinz case", get_anchor_config(Anchor.survivability), _materials())
    assert 0.0 <= out.faithfulness <= 1.0
    # the cheese sentence must be flagged ungrounded
    assert any("cheese" in s for s in out.ungrounded)
    # the grounded sentence should not be flagged
    assert not any("survival" in s for s in out.ungrounded)


def test_audit_bundle_shape():
    import src.arbiter.compile as arb
    sr = StakeholderRegister(case_id="c1", stakeholders=[
        Stakeholder(id="S1", description="wife", type="individual", role="patient")])
    mats = _materials()
    out = arb.ArbiterOutput(script_md="x", faithfulness=0.9, ungrounded=[])
    bundle = arb.build_audit_bundle(sr, mats, out, sit_rows=[{"row_id": 1}])
    assert "stakeholder_register" in bundle and "decisions" in bundle
    assert bundle["decisions"][0]["decision_id"] == "d1"
    assert bundle["faithfulness"]["fraction_entailed"] == 0.9
