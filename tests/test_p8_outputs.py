"""P8 — OutputStore step-wise saving + resume ledger (no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_output_store_saves_and_ledger(monkeypatch, tmp_path):
    # Redirect outputs to a temp dir (variant-namespaced under it).
    import src.storage.outputs as out
    monkeypatch.setattr(out, "OUTPUTS_DIR", tmp_path)

    store = out.OutputStore("run_test", variant="main")
    base = tmp_path / "main"
    store.save("stakeholders", "001", "sr", {"stakeholders": [{"id": "S1"}]}, echo=False)
    store.save("decisions", "001", "decisions", {"decisions": [{"id": "d1"}]}, echo=False)

    # step-wise files exist under outputs/<variant>/
    assert (base / "02_stakeholders" / "001" / "sr.json").exists()
    assert (base / "04_decisions" / "001" / "decisions.json").exists()

    # not yet in ledger
    assert not store.is_processed("001")

    store.finalize_sample("001")
    assert store.is_processed("001")
    assert (base / "full_pipeline" / "001.json").exists()
    assert "001" in store.processed_set()

    # idempotent
    store.mark_processed("001")
    assert list(store.processed_set()) == ["001"]


def test_variants_isolated(monkeypatch, tmp_path):
    import src.storage.outputs as out
    monkeypatch.setattr(out, "OUTPUTS_DIR", tmp_path)
    main = out.OutputStore("r", variant="main")
    abl = out.OutputStore("r", variant="abl_no_pmer")
    main.save("decisions", "001", "d", {"x": 1}, echo=False)
    main.finalize_sample("001")
    # ablation ledger must NOT see the main run's sample -> it will re-run it
    assert main.is_processed("001")
    assert not abl.is_processed("001")
