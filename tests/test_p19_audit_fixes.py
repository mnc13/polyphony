"""
P19 — fixes from the 2026-10-21 audit of the old and post-fix main runs.

Each test pins one defect that produced wrong or misleading results while the run
itself looked normal. Offline: LLMs, APIs and (where the judgement is not the point)
NLI are faked; embeddings run for real.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas.core import (  # noqa: E402
    Dimension, Discipline, EvidenceChunk, Premise, ScientistSelection, TypedPremiseSet,
)


def _premise(pid="p1", text="Chronic stress impairs working memory in caregivers.",
             disc=Discipline.neuroscience, dim=Dimension.NEURO, pol=-1, sids=("S1",)):
    return Premise(id=pid, text=text, dimension=dim, discipline=disc,
                   stakeholder_ids=list(sids), polarity=pol)


# ------------------------------------------------------------ retrieval gating
class _FakeVDB:
    def __init__(self, hits):
        self.hits = hits
        self.added = []

    def search(self, query, dim, top_k=5):
        return self.hits[:top_k]

    def add_chunks(self, chunks):
        self.added.extend(chunks)


class _FakeSource:
    def __init__(self):
        self.queries = []

    def fetch(self, q, max_results=5):
        from src.evidence.sources.base import RawDoc
        self.queries.append(q)
        return [RawDoc(title="Stress and working memory in family caregivers",
                       abstract="Chronic caregiving stress was associated with poorer "
                                "working memory performance in a cohort of caregivers.",
                       source="pubmed", ext_id="PMID1")]


def _pmer(monkeypatch, hits):
    import src.evidence.pmer as pm
    src = _FakeSource()
    monkeypatch.setattr(pm, "_sources_for", lambda dim, discipline=None: [src])
    p = pm.PMER(vectordb=_FakeVDB(hits))
    return p, src


def _chunk(text, doi="10.1/x"):
    return EvidenceChunk(node_id="", text=text, doi=doi, source="openalex",
                         dimension=Dimension.NEURO, citation_count=1)


def test_irrelevant_local_hits_do_not_suppress_live_fetch(monkeypatch):
    """Five off-topic chunks in the corpus used to stop every later live fetch."""
    from src.evidence.teg import TypedEvidenceGraph
    junk = [_chunk(f"Bonobos respond aversively to unequal reward distributions {i}.")
            for i in range(5)]
    p, src = _pmer(monkeypatch, junk)
    res = p.retrieve(_premise(), "the caregiver", TypedEvidenceGraph("c", "d1"))
    assert res.used_live and src.queries, "raw local hit count must not gate the fetch"


def test_relevant_local_hits_are_enough(monkeypatch):
    from src.evidence.teg import TypedEvidenceGraph
    good = [_chunk(f"Chronic stress impairs working memory in caregivers: study {i}.",
                   doi=f"10.1/{i}") for i in range(5)]
    p, src = _pmer(monkeypatch, good)
    res = p.retrieve(_premise(), "the caregiver", TypedEvidenceGraph("c", "d1"))
    assert not res.used_live and not src.queries
    forced = p.retrieve(_premise(), "x", TypedEvidenceGraph("c", "d2"), force_live=True)
    assert forced.used_live


def test_mean_novelty_is_zero_when_nothing_is_admitted(monkeypatch):
    from src.evidence.pmer import PMERResult
    from src.evidence.teg import TypedEvidenceGraph
    assert PMERResult().mean_novelty == 0.0
    junk = [_chunk("Completely unrelated text about volcanic soil chemistry.")]
    p, src = _pmer(monkeypatch, junk)
    monkeypatch.setattr(src, "fetch", lambda q, max_results=5: [])
    import src.evidence.pmer as pm
    monkeypatch.setattr(pm.CrossRefClient, "fetch", lambda self, q, max_results=5: [])
    res = p.retrieve(_premise(), "x", TypedEvidenceGraph("c", "d1"))
    assert res.admitted == [] and res.mean_novelty == 0.0


def test_discipline_routes_sources():
    from src.evidence.pmer import _sources_for
    econ = _sources_for(Dimension.SOCIO, Discipline.economics)
    assert econ[0].name == "openalex" and econ[0].concept_filter == "economics"
    eco = _sources_for(Dimension.EVOBIO, Discipline.ecology)
    assert eco[0].concept_filter == "ecology"
    assert [c.name for c in _sources_for(Dimension.NEURO)] == ["pubmed", "europepmc"]


# ------------------------------------------------------------------ sources
_PUBMED_XML = """<PubmedArticleSet><PubmedArticle>
<MedlineCitation><PMID>111</PMID><Article>
<ArticleTitle>Effects of <i>Homo sapiens</i> stress</ArticleTitle>
<Abstract><AbstractText Label="RESULTS">Scores fell (F<sub>2,51</sub> = 5.5, p &lt; .01) and recovered later.</AbstractText></Abstract>
<ELocationID EIdType="doi">10.9/own</ELocationID>
</Article><CommentsCorrectionsList><CommentsCorrections><PMID>999</PMID></CommentsCorrections></CommentsCorrectionsList></MedlineCitation>
<PubmedData><ArticleIdList><ArticleId IdType="doi">10.9/own</ArticleId></ArticleIdList>
<ReferenceList><Reference><ArticleIdList><ArticleId IdType="doi">10.9/cited</ArticleId></ArticleIdList></Reference></ReferenceList>
</PubmedData></PubmedArticle></PubmedArticleSet>"""


def test_pubmed_keeps_text_after_inline_tags_and_uses_own_doi():
    from src.evidence.sources.pubmed import PubMedClient
    doc = PubMedClient()._parse(_PUBMED_XML)[0]
    assert "recovered later" in doc.abstract, "text after <sub> must survive"
    assert doc.abstract.startswith("RESULTS:")
    assert doc.title == "Effects of Homo sapiens stress"
    assert doc.doi == "10.9/own", "a cited reference's DOI must never be stored"
    assert doc.ext_id == "111"


def test_markup_is_stripped():
    from src.evidence.sources.base import clean_markup
    assert clean_markup("<jats:p>Stress &amp; memory</jats:p> <h4> Abstract </h4>") == \
        "Stress & memory Abstract"


def test_failed_requests_are_retried_and_counted(monkeypatch):
    import src.evidence.sources.base as b
    calls = []

    class _R:
        status_code = 429
        headers = {}

        def raise_for_status(self):
            raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(b.requests, "get", lambda *a, **k: calls.append(1) or _R())
    monkeypatch.setattr(b.time, "sleep", lambda s: None)
    before = b.failure_snapshot().get("base", 0)
    assert b.SourceClient()._get("https://example.org/x") is None
    assert len(calls) == b.SourceClient.retries + 1
    assert b.failure_snapshot().get("base", 0) == before + 1


def test_chunk_tail_is_folded_and_titled():
    from src.evidence.chunker import chunk_doc, chunk_text
    from src.evidence.sources.base import RawDoc
    words = " ".join(f"w{i}" for i in range(150))
    chunks = chunk_text(words, size=140, overlap=32)
    assert len(chunks) == 1, "a 10-word tail is folded, not admitted as its own chunk"
    long_words = " ".join(f"w{i}" for i in range(400))
    for c in chunk_doc(RawDoc(title="A Title", abstract=long_words, source="x")):
        assert c.startswith("A Title")
        assert len(c.split()) <= 140 + 35 + 2


def test_gate_accepts_pmid_as_provenance():
    import numpy as np
    from src.evidence.gate import AdmissionGate, Candidate
    from src.services.embedder import embed
    p = _premise()
    cand = Candidate(text="Chronic stress impairs working memory in caregivers.",
                     doi=None, citation_count=0, source="pubmed", venue=None, year=None,
                     ext_id="PMID1")
    r = AdmissionGate().evaluate(cand, p, np.zeros((0, 384), "float32"),
                                 premise_vec=embed([p.text])[0])
    assert r.admitted, r.reason


def test_teg_skips_contradiction_between_unrelated_abstracts(monkeypatch):
    from src.evidence import teg as teg_mod
    calls = []
    monkeypatch.setattr(teg_mod.nli, "contradicts", lambda a, b: calls.append(1) or 0.99)
    g = teg_mod.TypedEvidenceGraph("c1", "d1")
    g.add(_chunk("Glucocorticoid treatment reduced panic attacks in a randomised trial of "
                 "adults with panic disorder over twelve weeks.", "10.1/a"))
    g.add(_chunk("This editorial argues that brain ecology should study how neural "
                 "systems adapt to their environments across species.", "10.1/b"))
    assert g.g.number_of_edges() == 0 and calls == []


# ------------------------------------------------------------- SIT / rebuttal
def test_rows_for_filters_by_run(tmp_path):
    from src.storage.sit import SITRow, SITStore
    s = SITStore(tmp_path / "sit.db")
    for run in ("dead", "live"):
        s.append(SITRow(run_id=run, case_id="c", decision_id="d1", round=1,
                        speaker="proponent", stakeholder_id="S1", dimension="NEURO",
                        polarity=1))
    assert len(s.rows_for("c")) == 2
    assert [r["run_id"] for r in s.rows_for("c", run_id="live")] == ["live"]


def test_rebuttal_is_derived_from_order():
    from src.storage.sit import mark_rebuttals
    rows = [
        dict(decision_id="d1", speaker="proponent", stakeholder_id="S1", dimension="NEURO", polarity=1),
        dict(decision_id="d1", speaker="opponent", stakeholder_id="S1 (wife)", dimension="NEURO", polarity=-1),
        dict(decision_id="d1", speaker="opponent", stakeholder_id="S2", dimension="NEURO", polarity=-1),
        dict(decision_id="d2", speaker="proponent", stakeholder_id="S1", dimension="NEURO", polarity=1),
    ]
    assert [r["rebutted"] for r in mark_rebuttals(rows)] == [True, False, False, False]


# ------------------------------------------------------------------ testimony
def test_hedge_is_not_support(monkeypatch):
    import src.testimony.testify as t
    monkeypatch.setattr(t.nli, "entails_batch", lambda pairs: [0.99] * len(pairs))
    hedge = ("The provided evidence does not settle whether diversion harms S1's mental "
             "health. The cited node examines stress in a different setting.")
    assert not t.claim_supported(hedge.split(". ")[0] + ".", "weak", ["paper"])
    assert not t.claim_supported(hedge, "insufficient", ["paper"]), \
        "an 'insufficient' rating is the witness saying it is not supported"
    assert not t.claim_supported(hedge, "weak", ["paper"]), \
        "a weak claim whose verdict is a hedge is not carried by a later sentence"
    assert t.claim_supported(hedge, "moderate", ["paper"]), \
        "at moderate strength the committed second sentence can still be supported"
    assert not t.claim_supported("Stress impairs memory in caregivers.", "moderate", [])


def test_pilot_hedge_verdicts_are_not_support(monkeypatch):
    # The three false positives from the case-001 pilot, shortened.
    import src.testimony.testify as t
    monkeypatch.setattr(t.nli, "entails_batch", lambda pairs: [0.99] * len(pairs))
    for verdict in [
        "The provided evidence does not directly establish that turning the trolley kills S3.",
        "The provided evidence does not empirically establish that turning saves the five.",
        "The provided evidence does not establish that S1 reports less guilt after an omission.",
    ]:
        finding = verdict + " TEG-d1-3 frames the trolley problem as a choice between deaths."
        assert not t.claim_supported(finding, "weak", ["paper"])


def test_abbreviations_do_not_split_sentences():
    from src.services.vacuity import sentences
    s = sentences("The evidence does not directly address the consequences for Mrs. Foot (S4) "
                  "of S3 surviving. Studies, e.g. cohort designs, are rare.")
    assert len(s) == 2 and s[0].endswith("surviving.")


@pytest.mark.parametrize("hedge", [
    "The provided evidence does not directly test trolley outcomes or survival effects.",
    "The provided evidence offers only indirect support for the claim that S1 is harmed.",
    "The premise as stated is therefore not settled by the retrieved literature.",
    "The provided evidence partially supports the premise.",
    "The premise's claim of unresolvedness is supported by the absence of direct evidence.",
    "The premise asserts that the effect on S3 is empirically unresolved.",
    "From a sociological perspective, one would expect that trust could be undermined.",
    "The provided evidence does not directly establish the physical outcome for S3.",
    "The provided evidence does not empirically establish that turning saves the five.",
    "The field has competing expectations: action-based harms carry more guilt.",
    "The effect has not been established by the retrieved studies.",
])
def test_real_hedges_from_the_run_are_vacuous(hedge):
    from src.services.vacuity import is_vacuous
    assert is_vacuous(hedge)


@pytest.mark.parametrize("claim", [
    "The trial does not show a mortality benefit.",
    "S4 was not directly affected by the collision.",
    "The policy does not address S2's housing.",
    "No data breaches occurred.",
    "Diversion establishes a precedent that S4 is expendable.",
    "The driver does not establish contact with S3.",
    "Mortality was reported higher in five-earner households.",
])
def test_substantive_claims_are_not_vacuous(claim):
    from src.services.vacuity import is_vacuous
    assert not is_vacuous(claim)


def test_description_of_the_paper_is_not_support_for_the_premise(monkeypatch):
    import src.testimony.testify as t
    monkeypatch.setattr(t.nli, "entails_batch", lambda pairs: [0.99] * len(pairs))
    describes = "TEG-d4-6 explores how uncertainty affects breast cancer survivors and partners."
    premise = "Fast-tracking the drug raises the CEO's risk of criminal prosecution."
    assert not t.claim_supported(describes, "weak", ["paper"], premise_text=premise)
    on_topic = "Executives who conceal trial data face a higher risk of criminal prosecution."
    assert t.claim_supported(on_topic, "weak", ["paper"], premise_text=premise)


def test_floor_witness_gets_premises_and_fixed_panel_covers_everything():
    from src.testimony.select import assign_premises, select_disciplines
    lone = TypedPremiseSet(case_id="c", decision_id="d1", premises=[
        _premise("p1", disc=Discipline.economics, dim=Dimension.SOCIO)])
    sel = select_disciplines(lone)
    assert sel.floor_applied and len(sel.selected) == 2
    by = assign_premises(lone, sel)
    assert all(by.get(d) for d in sel.selected), "the floor's second witness must testify"

    mixed = TypedPremiseSet(case_id="c", decision_id="d1", premises=[
        _premise("p1", disc=Discipline.economics, dim=Dimension.SOCIO),
        _premise("p2", disc=Discipline.health_science, dim=Dimension.NEURO),
        _premise("p3", disc=Discipline.ecology, dim=Dimension.EVOBIO)])
    fsel = select_disciplines(mixed, fixed_panel=True)
    fby = assign_premises(mixed, fsel, fixed_panel=True)
    assigned = {p.id for ps in fby.values() for p in ps}
    assert assigned == {"p1", "p2", "p3"}
    assert set(fby) <= set(fsel.selected)


# ------------------------------------------------------------------- premises
def test_dedup_keeps_opposite_claims(monkeypatch):
    import src.premises.tps as tps
    monkeypatch.setattr(tps.nli, "contradicts", lambda a, b: 0.0)
    up = _premise("p1", text="Diverting the trolley reduces S1's risk of fatal injury.", pol=1)
    down = _premise("p2", text="Diverting the trolley increases S1's risk of fatal injury.", pol=-1)
    kept = tps._dedup_keep_specific([up, down], 0.85)
    assert len(kept) == 2
    same = _premise("p3", text="Diverting the trolley reduces the risk of fatal injury to S1.", pol=1)
    assert len(tps._dedup_keep_specific([up, same], 0.85)) == 1


# ---------------------------------------------------------------- LLM client
def _fake_openai(monkeypatch, responses):
    import src.services.llm_client as lc
    sent = []

    class _Completions:
        def create(self, **kw):
            sent.append(kw)
            return responses.pop(0)

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr(lc, "OpenAI", lambda **kw: _Client())
    monkeypatch.setattr(lc, "get_settings", lambda: type("S", (), {"openai_api_key": "k"})())
    return sent


def _resp(content, finish):
    usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})()
    choice = type("C", (), {"message": type("M", (), {"content": content})(),
                            "finish_reason": finish})()
    return type("R", (), {"usage": usage, "choices": [choice]})()


def test_truncated_text_is_retried_with_a_bigger_budget(monkeypatch, tmp_path):
    import src.services.llm_client as lc
    import src.services.telemetry as tel
    from src.services.model_registry import ModelSpec
    monkeypatch.setattr(tel, "OUTPUTS_DIR", tmp_path)
    sent = _fake_openai(monkeypatch, [_resp("The script stops mid", "length"),
                                      _resp("The full script.", "stop"),
                                      _resp("A capped answer stops mid", "length")])
    c = lc.LLMClient(ModelSpec(role="arbiter", provider="openai", model="gpt-4.1",
                               temperature=0.2))
    assert c.generate("write", max_tokens=3000, retry_truncated=True) == "The full script."
    assert sent[0]["max_tokens"] == 3000 and sent[1]["max_tokens"] == 6000
    # Callers whose cap is part of their specification (baselines, socratic) keep it.
    assert c.generate("write", max_tokens=800) == "A capped answer stops mid"
    assert sent[2]["max_tokens"] == 800 and len(sent) == 3


def test_a_provider_error_mid_generation_is_retried_not_returned(monkeypatch, tmp_path):
    import json
    import src.services.llm_client as lc
    import src.services.telemetry as tel
    from src.services.model_registry import ModelSpec
    monkeypatch.setattr(tel, "OUTPUTS_DIR", tmp_path)
    sent = _fake_openai(monkeypatch, [_resp("Step 3: Communities that uphold and protect", "error"),
                                      _resp("The full analysis.", "stop")])
    c = lc.LLMClient(ModelSpec(role="b1_monolithic", provider="openai", model="gpt-4.1",
                               temperature=0.7))
    monkeypatch.setattr(c, "_backoff", lambda *a: None)
    assert c.generate("write", max_tokens=3000) == "The full analysis."
    assert len(sent) == 2 and sent[1]["max_tokens"] == 3000
    rows = [json.loads(l) for p in tmp_path.rglob("calls.jsonl") for l in open(p, encoding="utf-8")]
    assert [r["ok"] for r in rows] == [False, True]


def test_length_error_from_parse_is_billed_and_escalated(monkeypatch, tmp_path):
    import src.services.llm_client as lc
    import src.services.telemetry as tel
    import src.eval.tokens as tok
    from openai import LengthFinishReasonError
    from pydantic import BaseModel
    from src.services.model_registry import ModelSpec

    monkeypatch.setattr(tel, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(tok, "OUTPUTS_DIR", tmp_path)

    class Out(BaseModel):
        answer: str

    budgets = []

    class _Beta:
        def parse(self, response_format=None, **kw):
            budgets.append(kw.get("max_tokens"))
            if len(budgets) == 1:
                raise LengthFinishReasonError(completion=_resp('{"answer": "cut', "length"))
            r = _resp(None, "stop")
            r.choices[0].message.parsed = Out(answer="ok")
            return r

    class _Client:
        beta = type("B", (), {"chat": type("C", (), {"completions": _Beta()})()})()
        chat = type("Chat", (), {"completions": None})()

    monkeypatch.setattr(lc, "OpenAI", lambda **kw: _Client())
    monkeypatch.setattr(lc, "get_settings", lambda: type("S", (), {"openai_api_key": "k"})())
    tel.set_context("main", "001", "run_test")
    c = lc.LLMClient(ModelSpec(role="opponent", provider="openai", model="gpt-4.1",
                               temperature=0.5))
    assert c.generate_structured("argue", Out, max_tokens=1000).answer == "ok"
    assert budgets == [1000, 4000], "same-budget JSON fallback must be skipped"
    agg = tok.aggregate_tokens("main", run_id="run_test")
    assert agg["failed_calls"] == 1 and agg["total_tokens"] == 60


def test_budget_grows_to_at_most_four_times_the_request():
    import src.services.llm_client as lc
    assert lc._ceiling(2200) == 8800            # scientist
    assert lc._ceiling(6000) == 24000           # arbiter, premises: unchanged
    assert lc._ceiling(12000) == 24000          # debate: unchanged
    c = lc.LLMClient.__new__(lc.LLMClient)
    c.spec = type("S", (), {"role": "scientist_pool", "model": "m"})()
    err = lc._BudgetExhausted("cut")
    assert c._escalate(err, 2200, lc._ceiling(2200)) == ("retry", 8800)
    assert c._escalate(err, 8800, lc._ceiling(2200))[0] == "stop"


def test_runaway_structured_call_stops_at_its_ceiling(monkeypatch, tmp_path):
    # case 004: a scientist answer that normally runs 158-465 tokens looped to 2.2k,
    # 8.8k and 24k. The budget must stop at 4x the request.
    import src.services.llm_client as lc
    import src.services.telemetry as tel
    from openai import LengthFinishReasonError
    from pydantic import BaseModel
    from src.services.model_registry import ModelSpec
    monkeypatch.setattr(tel, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr("time.sleep", lambda s: None)

    class Out(BaseModel):
        answer: str

    budgets = []

    class _Beta:
        def parse(self, response_format=None, **kw):
            budgets.append(kw.get("max_tokens"))
            raise LengthFinishReasonError(completion=_resp('{"answer": "loop loop', "length"))

    class _Chat:
        def create(self, **kw):
            budgets.append(("json", kw.get("max_tokens")))
            return _resp('{"answer": "ok"}', "stop")

    class _Client:
        beta = type("B", (), {"chat": type("C", (), {"completions": _Beta()})()})()
        chat = type("Chat", (), {"completions": _Chat()})()

    monkeypatch.setattr(lc, "OpenAI", lambda **kw: _Client())
    monkeypatch.setattr(lc, "get_settings", lambda: type("S", (), {"openai_api_key": "k"})())
    tel.set_context("main", "001", "run_test")
    c = lc.LLMClient(ModelSpec(role="scientist_pool", provider="openai", model="gpt-4.1",
                               temperature=0.5))
    assert c.generate_structured("testify", Out, max_tokens=2200).answer == "ok"
    assert budgets == [2200, 8800, ("json", 8800)]


# ---------------------------------------------------------------- faithfulness
def test_headings_are_not_sentences_and_spans_fit_the_models():
    from src.eval.faithfulness import WINDOW_MAX_WORDS, script_sentences, windows
    md = ("# Title\n\n## 2.\n### Scientists' key findings\nThe evidence for d1 was "
          "uneven and often indirect.\n- **d1** relies on faster uptake of the drug.\n---\n")
    sents = script_sentences(md)
    assert sents == ["The evidence for d1 was uneven and often indirect.",
                     "d1 relies on faster uptake of the drug."]
    long = " ".join(f"Sentence number {i} says something about stress." for i in range(60))
    assert all(len(w.split()) <= WINDOW_MAX_WORDS for w in windows(long))
    assert len(windows(long)) > 1


def test_vacuous_spans_ground_nothing_and_meta_sentences_are_not_scored(monkeypatch):
    import src.services.nli as nli
    from src.eval.faithfulness import score_script
    seen = []

    def fake(pairs):
        seen.extend(pairs)
        return [0.99 if "dosage" in a else 0.1 for a, _ in pairs]

    monkeypatch.setattr(nli, "entails_batch", fake)
    script = ("The proponent argued that the dosage lowers mortality in elderly patients. "
              "The evidence does not settle whether the effect lasts.")
    res = score_script(script,
                       ["The provided evidence does not settle whether the dosage helps."],
                       ["The dosage lowers mortality in elderly patients, per TEG-d1-1."])
    assert res.n_meta == 1 and res.n_scored == 1 and res.fraction == 1.0
    assert res.n_vacuous_sentences == 1
    assert all("does not settle" not in span for span, _ in seen)


def test_case_text_and_decisions_can_ground_the_script(monkeypatch):
    import src.services.nli as nli
    from src.eval.faithfulness import context_texts, score_script
    monkeypatch.setattr(nli, "entails_batch",
                        lambda pairs: [0.99 if "brakes have failed" in a else 0.1 for a, _ in pairs])
    script = "This case concerns a trolley whose brakes have failed near five workmen."
    without = score_script(script, [], ["Some unrelated debate turn about cooperation norms."])
    ctx = context_texts("A trolley's brakes have failed; five workmen are ahead.",
                        "Survivability", "Persistence of life.", [("d1", "Divert the trolley.")])
    with_ctx = score_script(script, [], ["Some unrelated debate turn about cooperation norms."],
                            context=ctx)
    assert without.fraction == 0.0 and with_ctx.fraction == 1.0


# --------------------------------------------------------------------- metrics
def test_stakeholder_coverage_needs_supported_rows_and_json_is_valid():
    from src.eval.metrics import compute_metrics, json_safe
    b = {"stakeholder_register": {"stakeholders": [{"id": "S1"}, {"id": "S2"}]},
         "decisions": [],
         "sit_rows": [{"stakeholder_id": "S1", "evidence_node": "n", "nli_supported": True},
                      {"stakeholder_id": "S2", "evidence_node": "n", "nli_supported": False}],
         "faithfulness": {}}
    m = compute_metrics(b)
    assert m["stakeholder_coverage"] == 0.5 and m["stakeholder_coverage_cited"] == 1.0
    assert math.isnan(m["premise_coverage"])
    json.dumps(json_safe(m), allow_nan=False)


def test_report_only_counts_ledgered_samples(monkeypatch, tmp_path):
    import src.eval.report as rep
    monkeypatch.setattr(rep, "OUTPUTS_DIR", tmp_path)
    for sid in ("001", "002"):
        d = tmp_path / "main" / "09_arbiter" / sid
        d.mkdir(parents=True)
        (d / "audit_bundle.json").write_text("{}", encoding="utf-8")
    (tmp_path / "main" / "processed_samples.txt").write_text("001\n", encoding="utf-8")
    rep._set_variant("main")
    assert [sid for sid, _, _ in rep.collect_bundles()] == ["001"]


# ---------------------------------------------------------------- orchestration
def test_limit_applies_after_the_ledger(monkeypatch):
    import src.pipeline.orchestrate as orch

    class _Case:
        def __init__(self, sid):
            self.sample_id, self.case_id, self.stratum = sid, f"case{sid}", "x"

    class _Store:
        def __init__(self, run_id, variant="main"):
            pass

        def processed_set(self):
            return {"001", "002"}

        def is_processed(self, sid):
            return sid in self.processed_set()

        def save(self, *a, **k):
            pass

        def finalize_sample(self, *a, **k):
            pass

    ran = []
    monkeypatch.setattr(orch, "_select_cases",
                        lambda *a, **k: [_Case(s) for s in ("001", "002", "003", "004")])
    ctx = type("Ctx", (), {"run_id": "r",
                           "anchor": type("A", (), {"label": "Survivability"})()})()
    monkeypatch.setattr(orch.RunContext, "create", classmethod(lambda cls, a: ctx))
    monkeypatch.setattr(orch, "OutputStore", _Store)
    monkeypatch.setattr(orch, "_check_variant_state", lambda *a, **k: None)
    monkeypatch.setattr(orch, "process_sample",
                        lambda case, ctx, store, ablation=None: ran.append(case.sample_id) or {})
    orch.run_batch("x.csv", limit=1)
    assert ran == ["003"], "--limit must pick the next UNPROCESSED sample"


def test_gap_hints_and_rotation():
    from src.debate.engine import gap_hints, pick_retrieval_targets
    from src.schemas.core import DebateTurn
    t = DebateTurn(round=1, speaker="opponent", premises_addressed=[], evidence_cited=[],
                   claims=[], argument_text="x",
                   evidential_gap="I need cohort data on licensure loss (p3), and data "
                                  "that trust erodes (p5)")
    h = gap_hints([t], {"p3", "p5"})
    assert "licensure" in h["p3"] and "trust" in h["p5"]

    class P:
        def __init__(self, i):
            self.id = i
    U = [P(f"p{i}") for i in range(1, 9)]
    assert pick_retrieval_targets(U, {"p7": "x"}, 2)[0].id == "p7"
    r2 = {p.id for p in pick_retrieval_targets(U, {}, 2)}
    r3 = {p.id for p in pick_retrieval_targets(U, {}, 3)}
    assert r2.isdisjoint(r3), "later premises must get their turn"


def test_prompts_use_the_anchor_not_survivability():
    from src.ingestion.anchor import get_anchor_config
    from src.prompts.arbiter import ARBITER_SYSTEM
    from src.prompts.debate import debate_system
    from src.schemas.core import Anchor
    s = debate_system("proponent", get_anchor_config(Anchor.well_being))
    assert "survivability" not in s.lower()
    assert "survivability" not in ARBITER_SYSTEM.lower()
