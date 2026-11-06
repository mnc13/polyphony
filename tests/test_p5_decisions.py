"""
P5 — decision dedup + id assignment, tested without a live LLM by mocking the
client. The live path runs in scripts/run_full.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas.core import (
    ActionInventory, DecisionGeneration, StakeholderRegister, Stakeholder,
)


def _sr():
    return StakeholderRegister(case_id="c1", stakeholders=[
        Stakeholder(id="S1", description="the beggar", type="individual", role="agent"),
    ])


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def generate_structured(self, user, schema, system=None, max_tokens=0):
        return self.payload


def test_dedup_and_ids(monkeypatch):
    import src.decisions.generate as gen

    inv = ActionInventory(performed=["begged"], omitted=[], alternatives=["steal", "work"])
    # Two near-identical 'steal' phrasings + distinct others -> dedup should drop one.
    payload = DecisionGeneration(decisions=[
        "Stealing the money is the right course for the beggar.",
        "Taking the money by theft is the right course for the beggar.",
        "Earning the money through work is the right course for the beggar.",
        "Begging for the money is the right course for the beggar.",
    ])
    monkeypatch.setattr(gen, "get_client", lambda role: _FakeClient(payload))

    inv_out, dset = gen.generate_decisions("c1", "a beggar needs medicine ...", _sr(), inventory=inv)
    ids = [d.id for d in dset.decisions]
    assert ids[0] == "d1" and ids == [f"d{i}" for i in range(1, len(ids) + 1)]
    # the two near-identical steal statements collapse to one
    steal_like = [d for d in dset.decisions if "the" in d.statement.lower()
                  and ("steal" in d.statement.lower() or "theft" in d.statement.lower())]
    assert len(steal_like) == 1, f"expected dedup of steal variants, got {steal_like}"
    assert len(dset.decisions) == 3


def test_cap(monkeypatch):
    import src.decisions.generate as gen
    from config.settings import get_settings

    payload = DecisionGeneration(decisions=[f"Distinct option number {i} is right." for i in range(10)])
    monkeypatch.setattr(gen, "get_client", lambda role: _FakeClient(payload))
    _, dset = gen.generate_decisions("c1", "case", _sr(),
                                     inventory=ActionInventory(performed=[], omitted=[], alternatives=[]))
    assert len(dset.decisions) <= get_settings().hp.max_decisions
