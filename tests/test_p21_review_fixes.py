"""
P21 — fixes from the full code review of 2026-10-22 (six reviewers, after the pilot).

The debate-prompt, coverage, novelty and hedge fixes are pinned in test_p8_debate.py;
the run guards and the ablation proposition in test_p13_ablation.py; the faithfulness
definition check and pooled counts in test_p11_eval.py. This file holds the rest.
Offline: LLMs and APIs are faked; embeddings run for real.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas.core import Dimension, Discipline, EvidenceChunk, Premise  # noqa: E402

_PREMISE = Premise(id="p1", text="Chronic stress impairs working memory in caregivers.",
                   dimension=Dimension.NEURO, discipline=Discipline.neuroscience,
                   stakeholder_ids=["S1"], polarity=-1)


def _doc(title, abstract, ext="PMID1"):
    from src.evidence.sources.base import RawDoc
    return RawDoc(title=title, abstract=abstract, source="pubmed", ext_id=ext)


_OFF = _doc("Re-emerging syphilis among women of reproductive age",
            "Temporal dynamics and demographic drivers of syphilis incidence in a registry.",
            ext="PMID9")
_ON = _doc("Stress and working memory in family caregivers",
           "Chronic caregiving stress was associated with poorer working memory "
           "performance in a cohort of caregivers.", ext="PMID1")


class _EmptyVDB:
    def search(self, *a, **k):
        return []

    def add_chunks(self, chunks):
        pass


class _LadderSource:
    """Answers rung by rung from a script; None = a transport failure."""
    name = "fake"

    def __init__(self, script):
        self.script = list(script)
        self.queries = []

    def fetch(self, q, max_results=5):
        from src.evidence.sources import base
        self.queries.append(q)
        out = self.script.pop(0) if self.script else []
        if out is None:
            base._record_failure(self.name)
            return []
        return out


def _pmer(monkeypatch, source):
    import src.evidence.pmer as pm
    monkeypatch.setattr(pm, "_sources_for", lambda dim, discipline=None: [source])
    monkeypatch.setattr(pm.CrossRefClient, "fetch", lambda self, q, max_results=5: [])
    monkeypatch.setattr("time.sleep", lambda s: None)
    return pm.PMER(vectordb=_EmptyVDB())


# ------------------------------------------------------------------ query ladder
def test_ladder_moves_past_a_rung_whose_hits_are_all_off_topic(monkeypatch):
    from src.evidence.teg import TypedEvidenceGraph
    src = _LadderSource([[_OFF], [_ON]])
    res = _pmer(monkeypatch, src).retrieve(_PREMISE, "the caregiver", TypedEvidenceGraph("c", "d1"))
    assert len(src.queries) == 2, "an all-off-topic rung must not end the ladder"
    assert any("working memory" in c.text for c in res.admitted)


def test_transport_failure_does_not_broaden_the_query(monkeypatch):
    from src.evidence.teg import TypedEvidenceGraph
    src = _LadderSource([None, None, [_OFF], [_OFF]])
    res = _pmer(monkeypatch, src).retrieve(_PREMISE, "the caregiver", TypedEvidenceGraph("c", "d1"))
    assert len(src.queries) == 2, "rung 0 retried once, then the ladder stops"
    assert res.admitted == []


class _Resp:
    def __init__(self, status, retry_after=None, body=None):
        self.status_code = status
        self.headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
        self._body = body or {"results": []}

    def raise_for_status(self):
        import requests
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return self._body


def _fake_get(monkeypatch, responses):
    import src.evidence.sources.base as base
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        return responses.pop(0)
    monkeypatch.setattr(base.requests, "get", get)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return calls


def test_spent_quota_fails_the_case_instead_of_starving_it(monkeypatch):
    from src.evidence.sources.base import SourceBudgetExhausted
    from src.evidence.sources.openalex import OpenAlexClient
    calls = _fake_get(monkeypatch, [_Resp(429, retry_after=10381)])
    with pytest.raises(SourceBudgetExhausted):
        OpenAlexClient().fetch("stress working memory")
    assert len(calls) == 1, "a spent quota is not retried"


def test_short_429_is_still_retried_as_a_burst(monkeypatch):
    from src.evidence.sources.openalex import OpenAlexClient
    calls = _fake_get(monkeypatch, [_Resp(429, retry_after=2), _Resp(200)])
    assert OpenAlexClient().fetch("stress working memory") == []
    assert len(calls) == 2


def test_spent_quota_propagates_out_of_pmer(monkeypatch):
    from src.evidence.sources.base import SourceBudgetExhausted
    from src.evidence.teg import TypedEvidenceGraph

    class _Spent(_LadderSource):
        def fetch(self, q, max_results=5):
            raise SourceBudgetExhausted("spent")

    with pytest.raises(SourceBudgetExhausted):
        _pmer(monkeypatch, _Spent([])).retrieve(_PREMISE, "x", TypedEvidenceGraph("c", "d1"))


def test_openalex_key_goes_in_a_header_not_the_url(monkeypatch):
    import config.settings as cs
    from src.evidence.sources.openalex import OpenAlexClient
    monkeypatch.setattr(cs, "_settings", cs.Settings(openalex_api_key="k-123"))
    calls = _fake_get(monkeypatch, [_Resp(200)])
    OpenAlexClient().fetch("stress")
    assert calls[0]["headers"].get("Authorization") == "Bearer k-123"
    assert "k-123" not in json.dumps(calls[0]["params"])


def test_semantic_scholar_is_off_unless_the_experiment_enables_it():
    from config.settings import get_settings
    from src.evidence.pmer import _sources_for
    from src.evidence.sources.semanticscholar import SemanticScholarClient
    assert get_settings().hp.use_semantic_scholar is False
    for dim in Dimension:
        assert not any(isinstance(c, SemanticScholarClient) for c in _sources_for(dim))
    assert not any(isinstance(c, SemanticScholarClient)
                   for c in _sources_for(Dimension.SOCIO, Discipline.economics))


def test_local_sufficiency_counts_papers_not_windows(monkeypatch):
    import src.evidence.pmer as pm
    from src.evidence.teg import TypedEvidenceGraph

    class _OnePaperVDB(_EmptyVDB):
        def search(self, *a, **k):     # five windows of ONE relevant paper
            return [EvidenceChunk(node_id="", text=f"Chronic stress impairs working memory "
                                  f"in caregivers, window {i}.", doi="10.1/same",
                                  source="openalex", dimension=Dimension.NEURO)
                    for i in range(5)]

    src = _LadderSource([[_ON]])
    monkeypatch.setattr(pm, "_sources_for", lambda dim, discipline=None: [src])
    pm.PMER(vectordb=_OnePaperVDB()).retrieve(_PREMISE, "x", TypedEvidenceGraph("c", "d1"))
    assert src.queries, "one paper is one relevant hit, so the live fetch still runs"


# ------------------------------------------------------------------ chunking
def test_folded_tail_stays_within_the_embedder_limit():
    from src.evidence.chunker import chunk_text
    words = " ".join(f"w{i}" for i in range(140 + 140 - 32 + 30))
    assert max(len(w.split()) for w in chunk_text(words, 140, 32)) <= 140 + 140 // 7


# ------------------------------------------------------------------ vector DB
def _write_corpus(d: Path, n: int):
    import faiss
    d.mkdir(parents=True, exist_ok=True)
    idx = faiss.IndexFlatIP(4)
    idx.add(np.eye(4, dtype="float32")[:n] if n <= 4 else np.ones((n, 4), "float32"))
    faiss.write_index(idx, str(d / "NEURO.index"))
    (d / "NEURO.jsonl").write_text("".join(json.dumps({"text": f"t{i}", "doi": f"d{i}"}) + "\n"
                                           for i in range(n)), encoding="utf-8")


def test_misaligned_index_and_metadata_are_repaired_on_load(monkeypatch, tmp_path):
    import src.storage.vectordb as vdb
    d = tmp_path / "corpus"
    _write_corpus(d, 3)
    with open(d / "NEURO.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"text": "t3", "do')                # killed mid-line, index not updated
    monkeypatch.setattr(vdb, "VECTORDB_DIR", d)
    idx = vdb.DimensionIndex("NEURO")
    assert idx.index.ntotal == len(idx.meta) == 3
    _write_corpus(d, 3)
    import faiss
    big = faiss.IndexFlatIP(4)
    big.add(np.eye(4, dtype="float32"))                # index written, metadata not
    faiss.write_index(big, str(d / "NEURO.index"))
    idx = vdb.DimensionIndex("NEURO")
    assert idx.index.ntotal == len(idx.meta) == 3
    assert idx.search(np.eye(4, dtype="float32")[3], 4)  # no IndexError


def test_failed_case_corpus_additions_are_rolled_back(tmp_path):
    import faiss
    from src.storage.vectordb import corpus_snapshot, rollback_corpus
    d = tmp_path / "corpus"
    _write_corpus(d, 2)
    snap = corpus_snapshot(d)
    _write_corpus(d, 4)                                # the failed case added two
    assert rollback_corpus(snap, d) == {"NEURO": 2}
    assert faiss.read_index(str(d / "NEURO.index")).ntotal == 2
    assert len((d / "NEURO.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_batch_stops_at_a_failed_case_and_rolls_its_evidence_back(monkeypatch, tmp_path):
    import src.pipeline.orchestrate as orch
    import src.storage.vectordb as vdb
    rolled, ran = [], []

    class _Case:
        def __init__(self, sid):
            self.sample_id, self.stratum = sid, "demonstration"

    class _Store:
        def __init__(self, *a, **k):
            pass

        def processed_set(self):
            return set()

        def is_processed(self, sid):
            return False

        def save(self, *a, **k):
            pass

        def finalize_sample(self, *a, **k):
            pass

    def process(case, ctx, store, ablation=None):
        ran.append(case.sample_id)
        if case.sample_id == "002":
            raise RuntimeError("API died")
        return {}

    ctx = type("Ctx", (), {"run_id": "r", "anchor": type("A", (), {"label": "S"})()})()
    monkeypatch.setattr(orch, "_select_cases", lambda *a, **k: [_Case(s) for s in ("001", "002", "003")])
    monkeypatch.setattr(orch.RunContext, "create", classmethod(lambda cls, a: ctx))
    monkeypatch.setattr(orch, "OutputStore", _Store)
    monkeypatch.setattr(orch, "_check_variant_state", lambda *a, **k: None)
    monkeypatch.setattr(orch, "process_sample", process)
    monkeypatch.setattr(vdb, "VECTORDB_DIR", tmp_path)
    monkeypatch.setattr(vdb, "corpus_snapshot", lambda path=None: {"NEURO": 1})
    monkeypatch.setattr(vdb, "rollback_corpus", lambda snap, path=None: rolled.append(snap) or {})
    orch.run_batch("x.csv")
    assert ran == ["001", "002"], "the batch must stop at the failure, not run 003"
    assert rolled == [{"NEURO": 1}]
    assert not (tmp_path / "inflight.json").exists()


def test_killed_case_is_rolled_back_by_the_next_batch(tmp_path):
    # A hard kill (power off) never reaches run_batch's except block; the marker it
    # left must make the next batch undo the dead attempt's evidence.
    import faiss
    import src.storage.vectordb as vdb
    d = tmp_path / "corpus"
    _write_corpus(d, 2)
    vdb.mark_inflight("002", "abl_x", vdb.corpus_snapshot(d), d)
    _write_corpus(d, 4)                                # the killed case added two
    msg = vdb.recover_inflight("abl_x", {"001"}, d)
    assert "002" in msg and "2 corpus chunks" in msg
    assert faiss.read_index(str(d / "NEURO.index")).ntotal == 2
    assert not (d / "inflight.json").exists()
    assert vdb.recover_inflight("abl_x", {"001"}, d) is None


def test_inflight_case_already_ledgered_is_kept(tmp_path):
    import src.storage.vectordb as vdb
    d = tmp_path / "corpus"
    _write_corpus(d, 4)
    vdb.mark_inflight("002", "abl_x", {"NEURO": 2}, d)
    assert vdb.recover_inflight("abl_x", {"001", "002"}, d) is None
    assert vdb.corpus_snapshot(d) == {"NEURO": 4}


def test_inflight_marker_of_another_variant_is_refused(tmp_path):
    import src.storage.vectordb as vdb
    vdb.mark_inflight("002", "abl_x", {}, tmp_path)
    with pytest.raises(SystemExit):
        vdb.recover_inflight("abl_y", set(), tmp_path)


def test_batch_runs_each_case_with_its_snapshot_on_disk(monkeypatch, tmp_path):
    import src.pipeline.orchestrate as orch
    import src.storage.vectordb as vdb
    seen = []

    class _Case:
        sample_id, stratum = "001", "demonstration"

    class _Store:
        def __init__(self, *a, **k):
            pass

        def processed_set(self):
            return set()

        def is_processed(self, sid):
            return False

        def save(self, *a, **k):
            pass

        def finalize_sample(self, *a, **k):
            pass

    def process(case, ctx, store, ablation=None):
        seen.append((tmp_path / "inflight.json").exists())
        return {}

    ctx = type("Ctx", (), {"run_id": "r", "anchor": type("A", (), {"label": "S"})()})()
    monkeypatch.setattr(vdb, "VECTORDB_DIR", tmp_path)
    monkeypatch.setattr(orch, "_select_cases", lambda *a, **k: [_Case()])
    monkeypatch.setattr(orch.RunContext, "create", classmethod(lambda cls, a: ctx))
    monkeypatch.setattr(orch, "OutputStore", _Store)
    monkeypatch.setattr(orch, "_check_variant_state", lambda *a, **k: None)
    monkeypatch.setattr(orch, "process_sample", process)
    orch.run_batch("x.csv")
    assert seen == [True], "the case must run with its snapshot on disk"
    assert not (tmp_path / "inflight.json").exists()


# ------------------------------------------------------------------ premises
def test_premise_stakeholder_ids_are_canonicalised():
    from src.premises.generate import _register_ids
    assert _register_ids(["S2 (Patients)", "S1, S3", "S9", "S1"], {"S1", "S2", "S3"}) == \
        ["S2", "S1", "S3"]


def test_case_proposition_is_one_statement(monkeypatch):
    import src.decisions.generate as gen
    from src.schemas.core import DecisionGeneration

    class _Client:
        def generate_structured(self, **kw):
            assert "CASE:" in kw["user"]
            return DecisionGeneration(decisions=["Turning the trolley is the right course "
                                                 "for the driver.", "extra"])

    monkeypatch.setattr(gen, "get_client", lambda role: _Client())
    ds = gen.generate_case_proposition("c1", "A trolley's brakes have failed ...")
    assert [d.id for d in ds.decisions] == ["d1"]
    assert ds.decisions[0].statement.startswith("Turning the trolley")


# ------------------------------------------------------------------ LLM routing
def _client(monkeypatch, tmp_path, outcomes):
    import src.services.llm_client as lc
    import src.services.telemetry as tel
    from src.services.model_registry import ModelSpec
    monkeypatch.setattr(tel, "OUTPUTS_DIR", tmp_path)
    for reg in ("_REASONING_FLAG_UNUSABLE", "_REQUIRE_PARAMS_UNUSABLE",
                "_REASONING_MANDATORY", "_REASONING_EFFORT_UNUSABLE"):
        monkeypatch.setattr(lc, reg, set())
    sent = []

    class _Completions:
        def create(self, **kw):
            sent.append(kw)
            out = outcomes.pop(0)
            if isinstance(out, Exception):
                raise out
            usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})()
            ch = type("C", (), {"message": type("M", (), {"content": out})(), "finish_reason": "stop"})()
            return type("R", (), {"usage": usage, "choices": [ch]})()

    hp = type("HP", (), {"mandatory_reasoning_effort": "low", "network_outage_wait_s": 0})()
    monkeypatch.setattr(lc, "OpenAI", lambda **kw: type("Cl", (), {
        "chat": type("Ch", (), {"completions": _Completions()})()})())
    monkeypatch.setattr(lc, "get_settings", lambda: type("S", (), {"openrouter_api_key": "k", "hp": hp})())
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    return lc.LLMClient(ModelSpec(role="opponent", provider="openrouter",
                                  model="z-ai/glm-5.3", temperature=0.5)), sent


def test_routing_404_first_keeps_the_reasoning_cap_reachable(monkeypatch, tmp_path):
    routing = Exception("404 No endpoints found that can handle the requested parameters")
    mandatory = Exception("400 Reasoning is mandatory for this endpoint and cannot be disabled")
    c, sent = _client(monkeypatch, tmp_path, [routing, mandatory, "ok"])
    assert c.generate("argue") == "ok"
    assert sent[1]["extra_body"]["reasoning"] == {"enabled": False}, \
        "require_parameters goes first; the reasoning flag stays"
    assert "provider" not in sent[1]["extra_body"]
    assert sent[2]["extra_body"]["reasoning"] == {"effort": "low"}


# ------------------------------------------------------------------ faithfulness
def test_bold_labels_are_not_scored_and_the_anchor_is_its_label_only():
    from src.eval.faithfulness import context_texts, script_sentences
    md = ("## 4. Decision d1\n**Case made that it aligns with Survivability.**\n"
          "- **Evidence.** The studies found that stress impairs control in drivers.\n")
    assert script_sentences(md) == ["The studies found that stress impairs control in drivers."]
    ctx = context_texts("case", "Survivability", "Survivability is the capacity to persist.", [])
    assert "The anchor is Survivability." in ctx
    assert not any("capacity to persist" in t for t in ctx)


# ------------------------------------------------------------------ vacuity
@pytest.mark.parametrize("claim", [
    "Participants in the study did not report increased anxiety after the event.",
    "Data show that whistleblowers who did not report misconduct had higher guilt.",
    "The randomized trials did not demonstrate a mortality benefit.",
    "Caregiver burden was not measured in these trials but mortality was 30% higher.",
    "S2 did not directly report the abuse to anyone.",
    "The field generally agrees that prior offending is a risk factor.",
])
def test_outcome_verbs_are_not_hedges(claim):
    from src.services.vacuity import is_vacuous
    assert not is_vacuous(claim)


def test_lowercase_no_ends_a_sentence():
    from src.services.vacuity import split_sentences
    assert split_sentences("He said no. Then he left.") == ["He said no.", "Then he left."]
    assert len(split_sentences("See item No. 5 in the list.")) == 1


def test_list_leadins_are_scored_only_through_their_items():
    from src.eval.faithfulness import script_sentences
    md = ("The main lines of support were:\n"
          "- The proponent said d1 preserves five lives.\n"
          "- and that it keeps more households above basic needs.\n"
          "The opponent replied that it kills S3 outright: a certain loss.\n")
    assert script_sentences(md) == [
        "The proponent said d1 preserves five lives.",
        "The main lines of support were: and that it keeps more households above basic needs.",
        "The opponent replied that it kills S3 outright: a certain loss.",
    ]
    # the usual arbiter layout puts a blank line between the lead-in and its list
    md2 = "It left open:\n\n- whether warning gives the workmen enough time.\n\nNext paragraph here now."
    assert script_sentences(md2) == ["It left open: whether warning gives the workmen enough time.",
                                     "Next paragraph here now."]
    # ...but a heading ends it
    md3 = "It left open:\n\n## 5. Synthesis\n- whether warning gives the workmen enough time."
    assert script_sentences(md3) == ["whether warning gives the workmen enough time."]


def test_list_items_opening_with_an_id_are_not_fragments():
    """v2.5: "d2: <statement>" was prefixed with its lead-in and NLI rejected it (0.003)."""
    from src.eval.faithfulness import script_sentences
    md = ("The debate materials examine six decisions:\n\n"
          "- d2: Issuing an immediate, binding prohibition is the right course.\n"
          "- p3 holds that rationing stretches the food supply.\n"
          "- and whether rescue arrives in time.\n")
    assert script_sentences(md) == [
        "d2: Issuing an immediate, binding prohibition is the right course.",
        "p3 holds that rationing stretches the food supply.",
        "The debate materials examine six decisions: and whether rescue arrives in time.",
    ]


def test_bracketed_citation_labels_are_not_scored():
    """v2.6: "[E8]" is in no source span and cut entailment 0.61 -> 0.28 on B2 case 001."""
    from src.eval.faithfulness import script_sentences
    md = ("Fatalities raise disputes among the remaining workers [E8]. Trauma impairs "
          "the survivors [E5], [E6], [E10]. Both of these effects persist [E3, TEG-d1-2].\n"
          "TEG-d1-2 is invoked by the proponent for the same point.\n")
    assert script_sentences(md) == [
        "Fatalities raise disputes among the remaining workers.",
        "Trauma impairs the survivors.",
        "Both of these effects persist.",
        "TEG-d1-2 is invoked by the proponent for the same point.",
    ]


def test_ladder_with_nothing_relevant_still_records_its_query(monkeypatch):
    from src.evidence.teg import TypedEvidenceGraph
    from src.services import telemetry
    seen = {}
    monkeypatch.setattr(telemetry, "record_retrieval", lambda **kw: seen.update(kw))
    src = _LadderSource([[_OFF]] * 8)
    _pmer(monkeypatch, src).retrieve(_PREMISE, "the caregiver", TypedEvidenceGraph("c", "d1"))
    assert seen["rung_used"] == "none_relevant"
    assert seen["rung_query"] == src.queries[-1]
