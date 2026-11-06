"""
P6 — premise generation wiring (dimension derivation, orphan drop) + TPS dedup/re-id,
mocked LLM, relations disabled (NLI covered by the live smoke).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.anchor import get_anchor_config
from src.premises.tps import build_tps
from src.schemas.core import (
    Anchor, Decision, Dimension, Discipline, Premise, PremiseDraft, PremiseDraftSet,
    Stakeholder, StakeholderRegister,
)


def _sr():
    return StakeholderRegister(case_id="c1", stakeholders=[
        Stakeholder(id="S1", description="the wife", type="individual", role="patient"),
        Stakeholder(id="S3", description="Heinz", type="individual", role="agent"),
    ])


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def generate_structured(self, user, schema, system=None, max_tokens=0):
        return self.payload


def test_dimension_derivation_and_orphan_drop(monkeypatch):
    import src.premises.generate as gen
    payload = PremiseDraftSet(premises=[
        PremiseDraft(text="Radium treatment can extend S1's survival.",
                     discipline=Discipline.health_science, stakeholder_ids=["S1"], polarity=1),
        PremiseDraft(text="A theft conviction reduces S3's employment prospects.",
                     discipline=Discipline.sociology, stakeholder_ids=["S3"], polarity=-1),
        PremiseDraft(text="Orphan premise bound to nobody real.",
                     discipline=Discipline.economics, stakeholder_ids=["S9"], polarity=0),
    ])
    monkeypatch.setattr(gen, "get_client", lambda role: _FakeClient(payload))

    prem = gen.generate_premises(
        Decision(id="d1", statement="Heinz should steal the drug."),
        "case text", _sr(), get_anchor_config(Anchor.survivability))
    # orphan (S9) dropped
    assert len(prem) == 2
    # health_science -> NEURO, sociology -> SOCIO (derived, not model-set)
    by_disc = {p.discipline: p.dimension for p in prem}
    assert by_disc[Discipline.health_science] == Dimension.NEURO
    assert by_disc[Discipline.sociology] == Dimension.SOCIO


def test_tps_dedup_keeps_specific_and_reids():
    premises = [
        # near-identical pair (cosine ~0.91) -> should collapse, keeping the longer
        Premise(id="x1", text="A criminal record from theft lowers S3 long-term job prospects.",
                dimension=Dimension.SOCIO, discipline=Discipline.sociology,
                stakeholder_ids=["S3"], polarity=-1),
        Premise(id="x2", text="A theft conviction reduces S3's long-term employment prospects considerably.",
                dimension=Dimension.SOCIO, discipline=Discipline.sociology,
                stakeholder_ids=["S3"], polarity=-1),
        # distinct premise -> stays
        Premise(id="x3", text="Radium extends S1's survival by treating the cancer.",
                dimension=Dimension.NEURO, discipline=Discipline.health_science,
                stakeholder_ids=["S1"], polarity=1),
    ]
    tps, rels = build_tps("c1", "d1", premises, compute_relations=False)
    ids = [p.id for p in tps.premises]
    assert ids == [f"p{i}" for i in range(1, len(ids) + 1)]
    # near-identical theft pair collapses to one; the longer (more specific) survives
    socio = [p for p in tps.premises if p.dimension == Dimension.SOCIO]
    assert len(socio) == 1 and "employment prospects" in socio[0].text
    assert len(tps.premises) == 2
    assert tps.decision_id == "d1"
