"""
Baselines B1-B5 wiring, with mocked LLM, NLI and embedder. No network.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _Fake:
    """Records every call; replies from a script (cycled) or a fixed analysis."""

    def __init__(self, replies=None):
        self.replies = list(replies or ["A step-by-step moral analysis of the options."])
        self.calls = []

    def generate(self, user, system=None, max_tokens=0, retry_truncated=False):
        self.calls.append({"user": user, "system": system, "max_tokens": max_tokens,
                           "retry_truncated": retry_truncated})
        return self.replies[(len(self.calls) - 1) % len(self.replies)]


def _same_if_equal(monkeypatch):
    """Vote-judge stub: two answers recommend the same course iff they are the same string."""
    import src.eval.baselines as b
    monkeypatch.setattr(b, "vote_judge", lambda: None)
    monkeypatch.setattr(b, "same_answer", lambda a, c, judge: a == c)
    import src.services.embedder as emb
    monkeypatch.setattr(emb, "embed",
                        lambda texts: np.eye(len(texts), dtype="float32"))


def test_b1_votes_on_final_positions_and_returns_the_majority_analysis(monkeypatch):
    import src.eval.baselines as b
    _same_if_equal(monkeypatch)
    fake = _Fake([f"analysis {i}\nFINAL POSITION: {pos}" for i, pos in
                  enumerate(["Divert", "Stay", "Stay", "Divert", "Stay"])])
    monkeypatch.setattr(b, "_client", lambda *a, **k: fake)
    out = b.run_b1_monolithic("Trolley case", k=5)
    assert out["baseline"] == "B1_monolithic_cot_selfconsistency"
    assert out["final_positions"] == ["Divert", "Stay", "Stay", "Divert", "Stay"]
    assert out["majority"] == [1, 2, 4] and out["vote_share"] == 0.6
    assert "FINAL POSITION: Stay" in out["script"]
    assert out["sources"] == []                       # nothing to be faithful to
    # anchored like the pipeline, and never cut off
    assert "ANCHOR — Survivability" in fake.calls[0]["system"]
    assert all(c["retry_truncated"] for c in fake.calls)


def test_b1_samples_at_a_diverse_temperature(monkeypatch):
    import src.eval.baselines as b
    seen = {}

    def _client(base, temperature=None, name=None):
        seen.update(base=base, temperature=temperature)
        return _Fake(["x\nFINAL POSITION: A"])
    monkeypatch.setattr(b, "_client", _client)
    _same_if_equal(monkeypatch)
    b.run_b1_monolithic("case", k=2)
    assert seen == {"base": "action_decomposition", "temperature": 0.7}


def test_majority_vote_tie_and_unparsed(monkeypatch):
    import src.eval.baselines as b
    _same_if_equal(monkeypatch)
    # a 1-1 tie goes to the group formed first; an answer with no line never wins
    v = b.majority_vote(["", "B", "A"], ["t0", "t1", "t2"])
    assert v["majority"] == [1] and v["selected"] == 1
    v = b.majority_vote(["", ""], ["t0", "t1"])
    assert v["vote_share"] == 0.0 and v["majority"] == [0, 1]


def test_same_answer_asks_the_judge_at_temperature_zero(monkeypatch):
    import src.eval.baselines as b
    seen = {}

    def _client(base, temperature=None, name=None):
        seen.update(base=base, temperature=temperature, name=name)
        return _Fake(["**Yes.**"])
    monkeypatch.setattr(b, "_client", _client)
    judge = b.vote_judge()
    assert seen == {"base": "decision_generation", "temperature": 0.0, "name": "baseline_vote"}
    assert b.same_answer("Divert the trolley.", "Turn onto the spur.", judge) is True
    call = judge.calls[0]
    assert "A: Divert the trolley." in call["user"] and "B: Turn onto the spur." in call["user"]
    assert call["retry_truncated"]
    assert b.same_answer("x", "y", _Fake(["NO"])) is False
    assert b.same_answer("x", "y", _Fake([""])) is False


def _teg(tmp_path, variant="main_x", sid="001", ledgered=True):
    d = tmp_path / variant / "06_evidence_teg" / sid
    d.mkdir(parents=True)
    node = {"node_id": "TEG-d1-1", "text": "Chronic stress impairs memory.",
            "doi": "10.1/x", "source": "pubmed", "year": 2020}
    (d / "TEG_d1.json").write_text(json.dumps({"nodes": [node]}), encoding="utf-8")
    (d / "TEG_d2.json").write_text(json.dumps({"nodes": [
        dict(node, node_id="TEG-d2-1"),                       # same chunk, other decision
        {"node_id": "TEG-d2-2", "text": "Trust falls after theft.", "source": "openalex"}]}),
        encoding="utf-8")
    if ledgered:
        (tmp_path / variant / "processed_samples.txt").write_text(sid + "\n", encoding="utf-8")


def test_b2_reads_the_pipeline_teg_deduped(monkeypatch, tmp_path):
    import src.eval.baselines as b
    monkeypatch.setattr(b, "OUTPUTS_DIR", tmp_path)
    _teg(tmp_path)
    fake = _Fake()
    monkeypatch.setattr(b, "get_client", lambda role: fake)
    out = b.run_b2_rag_only("Heinz case", "001", "main_x")
    assert out["n_evidence"] == 2
    assert out["sources"] == ["Chronic stress impairs memory.", "Trust falls after theft."]
    prompt = fake.calls[0]["user"]
    assert "[E1] (pubmed, 2020, DOI 10.1/x) Chronic stress impairs memory." in prompt
    assert "ANCHOR — Survivability" in prompt and fake.calls[0]["retry_truncated"]


def test_b2_refuses_an_unfinished_or_missing_teg(monkeypatch, tmp_path):
    import src.eval.baselines as b
    monkeypatch.setattr(b, "OUTPUTS_DIR", tmp_path)
    _teg(tmp_path, ledgered=False)
    with pytest.raises(FileNotFoundError, match="has not finished"):
        b.load_teg_evidence("main_x", "001")
    (tmp_path / "main_x" / "processed_samples.txt").write_text("002\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="has not finished"):
        b.load_teg_evidence("main_x", "001")
    with pytest.raises(FileNotFoundError):
        b.load_teg_evidence("main_x", "002")


def test_b3_follows_du_et_al(monkeypatch):
    import src.eval.baselines as b
    _same_if_equal(monkeypatch)
    replies = ["r1-a1\nFINAL POSITION: X", "r1-a2\nFINAL POSITION: Y",
               "r1-a3\nFINAL POSITION: Y", "r2-a1\nFINAL POSITION: Y",
               "r2-a2\nFINAL POSITION: X", "r2-a3\nFINAL POSITION: Y"]
    fake = _Fake(replies)
    monkeypatch.setattr(b, "_client", lambda *a, **k: fake)
    out = b.run_b3_standard_mad("Heinz case", n_agents=3, rounds=2)
    assert out["baseline"] == "B3_standard_mad" and len(out["transcript"]) == 6
    r1 = [c["user"] for c in fake.calls[:3]]
    assert not any("r1-" in u for u in r1)            # round 1 is independent
    a1_r2 = fake.calls[3]["user"]
    assert "YOUR PREVIOUS ANSWER:\nr1-a1" in a1_r2    # sees its own answer
    assert replies[1] in a1_r2 and replies[2] in a1_r2  # others in full
    a3_r2 = fake.calls[5]["user"]
    assert "r2-" not in a3_r2                         # never this round's answers
    # majority of the last round is Y (agents 1 and 3); the script is not its own source
    assert out["majority"] == [0, 2] and out["script"] in (replies[3], replies[5])
    assert out["script"] not in out["sources"] and len(out["sources"]) == 5


def test_b5_cognialign_faithful(monkeypatch):
    import src.eval.baselines as b

    class _FakeLLM(_Fake):
        def __init__(self, spec):
            super().__init__(["survivability-based analysis"])
            self.spec = spec

    monkeypatch.setattr(b, "LLMClient", _FakeLLM)
    out = b.run_b5_cognialign("Heinz case")
    assert out["baseline"] == "B5_cognialign_faithful"
    assert out["model"] == "gpt-4o"                       # single-model, their config
    assert set(out["arguments"]) == {"neuroscience", "psychology", "sociology",
                                     "evolutionary_biology"}
    assert len(out["rebuttals"]) == 4 and out["final_conclusion"]
    assert out["script"] == out["final_conclusion"] and len(out["sources"]) == 8


def test_b5_uses_verbatim_prompts():
    # sanity: the ported prompts match distinctive lines from the original prompt.yml
    from src.eval import cognialign_prompts as P
    assert "Actions that enhance survivability are generally considered moral" in P.PERSONA["neuroscience"]
    assert "You must not introduce any new arguments" in P.ARBITER
    assert "{node_prompt}" in P.PERSONA["sociology"]


def test_single_model_spec_follows_the_active_roster(monkeypatch):
    import src.services.llm_client as lc

    class _Reg:
        mode = "target_family"
    monkeypatch.setattr(lc, "get_registry", lambda: _Reg())
    s = lc.single_model_spec("b5", "gpt-4o", 0.2)
    assert (s.provider, s.model) == ("openrouter", "openai/gpt-4o")
    _Reg.mode = "interim_openai"
    s = lc.single_model_spec("b5", "openai/gpt-4o", 0.2)
    assert (s.provider, s.model) == ("openai", "gpt-4o")


def test_b4_force_model_keeps_each_roles_temperature(monkeypatch):
    import src.services.llm_client as lc
    from src.services.model_registry import get_model
    monkeypatch.setattr(lc, "OpenAI", lambda **kw: object())        # no real client
    monkeypatch.setattr(lc, "get_settings", lambda: type(
        "S", (), {"openai_api_key": "sk-x", "openrouter_api_key": "sk-or"})())
    lc.set_force_model("gpt-4o")
    try:
        for role in ("proponent", "arbiter", "stakeholder_extraction"):
            spec = lc.get_client(role).spec
            assert spec.model.split("/")[-1] == "gpt-4o"
            assert spec.temperature == get_model(role).temperature
        assert lc.forced_model() == "gpt-4o"
    finally:
        lc.set_force_model(None)
    assert lc.forced_model() is None


def test_manifest_records_the_forced_model(monkeypatch, tmp_path):
    import json as _json
    import src.pipeline.run as run
    import src.services.llm_client as lc
    monkeypatch.setattr(run, "manifest_path", lambda rid: tmp_path / "m.json")
    lc.set_force_model("gpt-4o")
    try:
        run.RunContext.create("survivability", run_id="r")
    finally:
        lc.set_force_model(None)
    m = _json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert m["forced_model"] == "gpt-4o"
    assert {v["model"].split("/")[-1] for v in m["models"].values()} == {"gpt-4o"}


def test_output_cap_bounds_gpt4o_requests(monkeypatch):
    import src.services.llm_client as lc
    from src.services.model_registry import ModelSpec
    monkeypatch.setattr(lc, "OpenAI", lambda **kw: object())
    monkeypatch.setattr(lc, "get_settings", lambda: type(
        "S", (), {"openai_api_key": "sk-x", "openrouter_api_key": "sk-or"})())
    c = lc.LLMClient(ModelSpec(role="b5", provider="openai", model="gpt-4o", temperature=0.2))
    assert c._ceiling(12000) == 16384                 # not 24000: gpt-4o would 400
    assert c._base_params(20000)["max_tokens"] == 16384
    other = lc.LLMClient(ModelSpec(role="x", provider="openai", model="gpt-4.1", temperature=0.2))
    assert other._ceiling(12000) == 24000


def test_baseline_config_change_is_refused_and_no_corpus_needed(monkeypatch, tmp_path):
    import src.pipeline.orchestrate as orch
    import src.storage.outputs as out
    import src.storage.vectordb as vdb
    from src.storage.outputs import OutputStore
    monkeypatch.setattr(out, "OUTPUTS_DIR", tmp_path)
    warm = tmp_path / "corpus"
    warm.mkdir()
    (warm / "NEURO.jsonl").write_text('{"text": "t"}\n' * 9, encoding="utf-8")
    monkeypatch.setattr(vdb, "VECTORDB_DIR", warm)
    s = OutputStore("r", "baseline_b1")
    orch._check_variant_state(s, uses_corpus=False, config={"baseline": "b1", "k": 5})
    with pytest.raises(SystemExit, match="same options"):
        orch._check_variant_state(s, uses_corpus=False, config={"baseline": "b1", "k": 3})


def test_save_outputs_is_scorable(monkeypatch, tmp_path):
    """A baseline sample must reach build_eval and the faithfulness recompute."""
    import src.eval.report as rep
    import src.storage.outputs as out
    from src.eval.baselines import save_outputs
    from src.ingestion.anchor import select_anchor
    from src.storage.outputs import OutputStore
    monkeypatch.setattr(out, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(rep, "OUTPUTS_DIR", tmp_path)
    case = type("C", (), {"sample_id": "001", "case_id": "case001_x", "stratum": "trolley",
                          "text": "A trolley's brakes fail."})()
    store = OutputStore("r", "baseline_b3")
    save_outputs(store, case, select_anchor("survivability"), "b3",
                 {"baseline": "B3_standard_mad", "script": "Divert the trolley.",
                  "sources": ["Agent2 argued diverting saves five."], "model": "m"})
    store.finalize_sample("001")
    agg = rep.build_report("baseline_b3")
    assert agg["n_samples"] == 1
    assert (tmp_path / "baseline_b3" / "11_eval" / "human_eval" / "001.md").exists()
    import scripts.recompute_faithfulness as rf
    monkeypatch.setattr(rf, "OUTPUTS_DIR", tmp_path)
    assert rf.load_materials("baseline_b3", "001") == ([], ["Agent2 argued diverting saves five."])
    ctx = rf.load_context("baseline_b3", "001")
    assert "A trolley's brakes fail." in ctx[0] and "Survivability" in ctx[1]
