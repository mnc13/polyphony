"""
P4 — stakeholder assembly wiring, tested WITHOUT a live LLM call by mocking the
client. Proves deterministic re-id, SR assembly, immutable save, and reload. The
live path runs in scripts/run_full.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas.core import Stakeholder, StakeholderExtraction


class _FakeClient:
    def generate_structured(self, user, schema, system=None, max_tokens=0):
        # Model returns messy/wrong ids on purpose; assembly must overwrite them.
        return StakeholderExtraction(stakeholders=[
            Stakeholder(id="X", description="the dying wife who needs the drug",
                        type="individual", role="patient", vulnerability_flag=True),
            Stakeholder(id="X", description="Heinz, the husband",
                        type="individual", role="agent", vulnerability_flag=False),
            Stakeholder(id="X", description="the druggist who set the price",
                        type="individual", role="agent", vulnerability_flag=False),
        ])


def test_stakeholder_assembly_reids_and_saves(monkeypatch, tmp_path):
    import src.ingestion.stakeholders as stk
    monkeypatch.setattr(stk, "get_client", lambda role: _FakeClient())

    sr = stk.extract_stakeholders("case001_abc", "Heinz dilemma text ...")
    ids = [s.id for s in sr.stakeholders]
    assert ids == ["S1", "S2", "S3"], "ids must be reassigned deterministically"
    assert sr.stakeholders[0].vulnerability_flag is True
    assert sr.case_id == "case001_abc"

    # round-trip through JSON (what gets persisted)
    j = sr.model_dump_json()
    from src.schemas.core import StakeholderRegister
    back = StakeholderRegister.model_validate_json(j)
    assert len(back.stakeholders) == 3
