"""
P14 — regressions for the defects found auditing sample 001.

  1. stakeholder extraction admitted objects/concepts ("the movie", "the fast")
  2. json_schema-incapable roles were invisible to the token report
  3. the arbiter dropped a whole decision from the central script
  4. testimony claimed 'moderate' evidence while citing nothing
  5. the TEG scored contradictions between bare paper titles / a paper and itself
  6. a shorter re-run left the previous run's decision-keyed files behind

All offline: LLM and NLI are mocked wherever a real call would be needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas.core import Dimension, Discipline, EvidenceChunk


# --------------------------------------------------------------- 1. stakeholders
def test_stakeholder_prompt_excludes_non_agents():
    from src.prompts.ingestion import STAKEHOLDER_SYSTEM
    s = STAKEHOLDER_SYSTEM.lower()
    assert "never list inanimate objects" in s
    # the two shapes actually observed in the bad run
    assert "movie" in s and ("ritual" in s or "practice" in s)
    assert "must be able to have interests" in s


# ------------------------------------------------------- 2. fallback telemetry
def test_gpt4_turbo_flagged_as_no_json_schema():
    from src.services.llm_client import _supports_json_schema
    from src.services.model_registry import ModelSpec

    def spec(model):
        return ModelSpec(role="opponent", provider="openai", model=model, temperature=0.5)

    assert _supports_json_schema(spec("gpt-4o")) is True
    assert _supports_json_schema(spec("gpt-4.1")) is True
    assert _supports_json_schema(spec("o4-mini")) is True
    assert _supports_json_schema(spec("gpt-4-turbo")) is False
    assert _supports_json_schema(spec("gpt-4")) is False


def test_json_mode_fallback_records_telemetry(monkeypatch, tmp_path):
    """The opponent bug: the fallback produced output but logged zero tokens."""
    import src.services.llm_client as lc
    import src.services.telemetry as tel
    import src.eval.tokens as tok
    from src.services.model_registry import ModelSpec
    from pydantic import BaseModel

    monkeypatch.setattr(tel, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(tok, "OUTPUTS_DIR", tmp_path)

    class Out(BaseModel):
        answer: str

    class _Usage:
        prompt_tokens, completion_tokens, total_tokens = 700, 300, 1000

    class _Msg:
        content = '{"answer": "rebuttal"}'

    class _Resp:
        usage = _Usage()
        choices = [type("C", (), {"message": _Msg()})()]

    class _Completions:
        def create(self, **kw):
            return _Resp()

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr(lc, "OpenAI", lambda **kw: _Client())
    monkeypatch.setattr(lc, "get_settings",
                        lambda: type("S", (), {"openai_api_key": "k"})())

    client = lc.LLMClient(ModelSpec(role="opponent", provider="openai",
                                    model="gpt-4-turbo", temperature=0.5))
    tel.set_context("main", "001")
    got = client.generate_structured(user="argue", schema=Out, system="be adversarial")

    assert got.answer == "rebuttal"
    agg = tok.aggregate_tokens("main")
    assert "opponent" in agg["per_role"], "fallback path must reach the token report"
    assert agg["per_role"]["opponent"]["total_tokens"] == 1000


# ------------------------------------------------------------------ 3. arbiter
def test_missing_decisions_detected():
    from src.arbiter.compile import _missing_decisions, DecisionMaterial

    mats = [DecisionMaterial(decision_id=f"d{i}", statement=f"stmt {i}",
                             tps=None, testimony=None, result=None) for i in range(1, 6)]
    # the real failure: d5 named in the roster but given no block of its own
    script = (
        "# 3. The Decisions Examined\n- **d5**: Watching a different movie.\n\n"
        "# 4. Findings\n"
        "## Decision d1: A\n## Decision d2: B\n## Decision d3: C\n## Decision d4: D\n"
    )
    assert _missing_decisions(script, mats) == ["d5"]

    complete = script + "## Decision d5: E\n"
    assert _missing_decisions(complete, mats) == []


def test_arbiter_retries_when_a_decision_is_dropped(monkeypatch):
    """First draft omits d2; the corrective retry must be taken."""
    import src.arbiter.compile as ac
    from src.arbiter.compile import DecisionMaterial

    mats = [DecisionMaterial(decision_id="d1", statement="A", tps=None,
                             testimony=None, result=None),
            DecisionMaterial(decision_id="d2", statement="B", tps=None,
                             testimony=None, result=None)]

    calls = []

    class _Client:
        def generate(self, user, system=None, max_tokens=0, **kw):
            calls.append(user)
            if len(calls) == 1:
                return "## Decision d1: A\n"
            return "## Decision d1: A\n## Decision d2: B\n"

    monkeypatch.setattr(ac, "get_client", lambda role: _Client())
    monkeypatch.setattr(ac, "_decision_block", lambda m: f"block {m.decision_id}")
    monkeypatch.setattr(ac, "_source_spans", lambda mats_: [])

    out = ac.compile_central_script("case", type("A", (), {"label": "Survivability"})(),
                                    mats, check_faithfulness=False)

    assert len(calls) == 2, "omission must trigger exactly one corrective retry"
    assert "OMITTED" in calls[1] and "d2" in calls[1]
    assert out.missing_decisions == []
    assert "Decision d2" in out.script_md


# ---------------------------------------------------------------- 4. testimony
def test_uncited_testimony_strength_is_clamped(monkeypatch):
    import src.testimony.testify as t
    from src.schemas.core import (
        Premise, ScientistTestimonyDraft, TestimonyClaim,
        Stakeholder, StakeholderRegister,
    )
    from src.evidence.teg import TypedEvidenceGraph

    prem = Premise(id="p1", text="Shared fasting strengthens social bonds between peers.",
                   dimension=Dimension.SOCIO, discipline=Discipline.sociology,
                   stakeholder_ids=["S1"], polarity=1)
    sr = StakeholderRegister(case_id="c1", stakeholders=[
        Stakeholder(id="S1", type="individual", role="agent",
                    description="the person fasting", vulnerability_flag=False)])
    teg = TypedEvidenceGraph("c1", "d1")

    draft = ScientistTestimonyDraft(claims=[
        # no citations at all -> cannot stay 'moderate'
        TestimonyClaim(premise_id="p1", finding="The literature suggests bonds strengthen.",
                            direction=1, evidence_strength="moderate", cited_nodes=[]),
        # cites a node that does not exist -> dropped, so also uncited
        TestimonyClaim(premise_id="p1", finding="Strong effects are well established.",
                            direction=1, evidence_strength="strong",
                            cited_nodes=["TEG-d1-999"]),
    ])

    class _Client:
        def generate_structured(self, user, schema, system=None, max_tokens=0):
            return draft

    monkeypatch.setattr(t, "get_client", lambda role: _Client())
    out = t.testify(Discipline.sociology, [prem], sr, teg,
                    type("A", (), {"label": "Survivability", "definition": "",
                                   "positive_direction": ""})())

    assert len(out.claims) == 2
    for c in out.claims:
        assert c.cited_nodes == []
        assert c.evidence_strength not in ("strong", "moderate")
        assert c.nli_supported is False
    # direction is preserved — an unconfirmed lean is still reportable
    assert out.claims[0].direction == 1


# ---------------------------------------------------------------------- 5. TEG
def _chunk(text, doi, nid=""):
    return EvidenceChunk(node_id=nid, text=text, doi=doi, source="openalex",
                         dimension=Dimension.SOCIO, stakeholder_ids=["S1"],
                         citation_count=1)


def test_teg_skips_title_length_nodes(monkeypatch):
    """The 0.945 false positive: two bare titles scored as contradictory."""
    from src.evidence import teg as teg_mod

    monkeypatch.setattr(teg_mod.nli, "contradicts", lambda a, b: 0.99)
    g = teg_mod.TypedEvidenceGraph("c1", "d1")
    g.add(_chunk("Individual, Social, and Evolutionary Adaptation in Collective Systems",
                 "10.1201/b14908-16"))
    g.add(_chunk("Chapter 1. Cooperation, Coordination, and Collective Action",
                 "10.1515/9781400845484-002"))
    assert g.g.number_of_edges() == 0, "titles carry no proposition to contradict"


def test_teg_skips_same_doi_and_needs_both_directions(monkeypatch):
    from src.evidence import teg as teg_mod

    long_a = ("Chronic psychosocial stress markedly elevates circulating cortisol and "
              "measurably impairs working memory in healthy adults over time.")
    long_b = ("Chronic psychosocial stress has no detectable effect on circulating "
              "cortisol and leaves working memory entirely unchanged in adults.")

    # same paper retrieved twice -> never an edge, whatever NLI says
    monkeypatch.setattr(teg_mod.nli, "contradicts", lambda a, b: 0.99)
    same = teg_mod.TypedEvidenceGraph("c1", "d1")
    same.add(_chunk(long_a, "10.1/dup"))
    same.add(_chunk(long_b, "10.1/dup"))
    assert same.g.number_of_edges() == 0

    # one-way hit only -> rejected as asymmetric specificity
    monkeypatch.setattr(teg_mod.nli, "contradicts",
                        lambda a, b: 0.99 if a == long_a else 0.10)
    oneway = teg_mod.TypedEvidenceGraph("c1", "d1")
    oneway.add(_chunk(long_a, "10.1/a"))
    oneway.add(_chunk(long_b, "10.1/b"))
    assert oneway.g.number_of_edges() == 0

    # genuine both-way contradiction -> edge, scored at the weaker direction
    monkeypatch.setattr(teg_mod.nli, "contradicts",
                        lambda a, b: 0.99 if a == long_a else 0.80)
    both = teg_mod.TypedEvidenceGraph("c1", "d1")
    both.add(_chunk(long_a, "10.1/a"))
    both.add(_chunk(long_b, "10.1/b"))
    assert both.g.number_of_edges() == 1
    assert list(both.g.edges(data=True))[0][2]["score"] == 0.8


# ------------------------------------------------------- 6. stale sample artifacts
def test_reset_sample_clears_stale_decision_files(monkeypatch, tmp_path):
    """A 5-decision run followed by a 4-decision one must not leave d5 behind."""
    import src.storage.outputs as so

    monkeypatch.setattr(so, "OUTPUTS_DIR", tmp_path)
    store = so.OutputStore("run1", variant="main")

    # first run: five decisions, plus a second sample that must survive untouched
    for d in range(1, 6):
        store.save("premises", "001", f"TPS_d{d}", {"decision_id": f"d{d}"}, echo=False)
        store.save("testimony", "001", f"testimony_d{d}", {"d": d}, echo=False)
    store.save("premises", "002", "TPS_d1", {"decision_id": "d1"}, echo=False)
    store.finalize_sample("001")
    assert store.is_processed("001")

    # second run of 001: only four decisions
    removed = store.reset_sample("001")
    assert removed, "reset must report what it cleared"
    for d in range(1, 5):
        store.save("premises", "001", f"TPS_d{d}", {"decision_id": f"d{d}"}, echo=False)

    prem = sorted(p.name for p in (tmp_path / "main" / "05_premises" / "001").iterdir())
    assert prem == ["TPS_d1.json", "TPS_d2.json", "TPS_d3.json", "TPS_d4.json"]
    # the whole stale step folder is gone, not just the files we happened to rewrite
    assert not (tmp_path / "main" / "07_testimony" / "001").exists()
    # other samples are untouched
    assert (tmp_path / "main" / "05_premises" / "002" / "TPS_d1.json").exists()
    # resume state is the caller's business, so the ledger survives
    assert store.is_processed("001")


def test_process_sample_resets_before_writing(monkeypatch):
    """The reset must be wired into the pipeline, not just available on the store."""
    import inspect
    from src.pipeline import orchestrate

    src = inspect.getsource(orchestrate.process_sample)
    body = src.split("def process_sample", 1)[1]
    assert "store.reset_sample(sid)" in body
    # ...and before the first artifact is written, or it would delete this run's work
    assert body.index("store.reset_sample(sid)") < body.index('store.save("ingestion"')
