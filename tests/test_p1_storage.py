import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.storage.cases import load_cases, repair_text
from src.storage.sit import SITRow, SITStore


def test_mojibake_repair():
    assert repair_text("druggistâ€™s wife") == "druggist's wife"
    assert repair_text("  clean  ") == "clean"


def test_load_cases():
    csv = Path(__file__).resolve().parent.parent.parent / "all_combined_moral_cases.csv"
    if not csv.exists():
        pytest.skip("dataset not present next to repo")
    cases = load_cases(csv)
    assert len(cases) >= 50
    strata = {c.stratum for c in cases}
    assert "ordinary" in strata


def test_sit_append_and_query(tmp_path):
    store = SITStore(tmp_path / "sit.db")
    rid = store.append(SITRow(
        run_id="r1", case_id="c1", decision_id="d1", round=1,
        speaker="proponent", stakeholder_id="S1", dimension="PSYCH",
        polarity=-1, evidence_node="TEG-d1-1", rebutted=False, nli_supported=True,
    ))
    assert rid == 1
    rows = store.rows_for("c1", "d1")
    assert len(rows) == 1 and rows[0]["polarity"] == -1
    store.close()


def test_sit_rejects_mutation(tmp_path):
    store = SITStore(tmp_path / "sit.db")
    with pytest.raises(PermissionError):
        store.query("DELETE FROM sit")
    with pytest.raises(PermissionError):
        store.query("UPDATE sit SET polarity=0")
    store.close()
