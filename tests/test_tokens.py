"""Token telemetry recording + aggregation (offline, synthetic calls)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_telemetry_record_and_aggregate(monkeypatch, tmp_path):
    import src.services.telemetry as tel
    import src.eval.tokens as tok
    monkeypatch.setattr(tel, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(tok, "OUTPUTS_DIR", tmp_path)

    # two samples, a couple of roles each
    tel.set_context("main", "001")
    tel.record("proponent", "gpt-4.1", 100, 50, 150, 2.0)
    tel.record("opponent", "gpt-4-turbo", 80, 40, 120, 1.5)
    tel.set_context("main", "002")
    tel.record("proponent", "gpt-4.1", 200, 100, 300, 3.0)
    tel.record("arbiter", "o1", 500, 300, 800, 10.0)

    # default: averages over the samples actually run, NO projection
    agg = tok.aggregate_tokens("main")
    assert agg["n_samples"] == 2
    assert agg["total_tokens"] == 150 + 120 + 300 + 800  # 1370
    assert agg["avg_tokens_per_sample"] == 685.0
    assert "projection" not in agg
    # proponent appears in both samples: 150 + 300 = 450 total, avg/sample 225
    assert agg["per_role"]["proponent"]["total_tokens"] == 450
    assert agg["per_role"]["proponent"]["avg_tokens_per_sample"] == 225.0
    # time tracked
    assert agg["total_time_s"] == 16.5

    # projection only when explicitly requested
    projd = tok.aggregate_tokens("main", project_to=62)
    assert projd["projection"]["projected_total_tokens"] == round(685.0 * 62)

    md = tok.report_md(agg)
    assert "Token & Time Report" in md and "proponent" in md


def test_variant_isolation(monkeypatch, tmp_path):
    import src.services.telemetry as tel
    import src.eval.tokens as tok
    monkeypatch.setattr(tel, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(tok, "OUTPUTS_DIR", tmp_path)
    tel.set_context("baseline_b1", "001")
    tel.record("action_decomposition", "gpt-4o-mini", 50, 25, 75, 1.0)
    tel.set_context("main", "001")
    tel.record("proponent", "gpt-4.1", 100, 50, 150, 2.0)
    # each variant's telemetry is separate
    assert tok.aggregate_tokens("baseline_b1")["total_tokens"] == 75
    assert tok.aggregate_tokens("main")["total_tokens"] == 150
