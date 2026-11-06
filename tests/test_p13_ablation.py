"""
Ablation presets + case-range selection (deterministic, no network).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.pipeline.orchestrate import Ablation
from src.schemas.core import Dimension, Discipline, Premise, TypedPremiseSet
from src.testimony.select import select_disciplines


def test_ablation_presets():
    assert Ablation.from_name("none").variant == "main"
    a = Ablation.from_name("no_pmer")
    assert a.no_pmer is True and a.variant == "abl_no_pmer"
    assert Ablation.from_name("fixed_panel").fixed_panel is True
    assert Ablation.from_name("no_testimony").no_testimony is True
    assert Ablation.from_name("no_decision_decomp").no_decision_decomp is True
    with pytest.raises(ValueError):
        Ablation.from_name("bogus")


def test_fixed_panel_selection():
    tps = TypedPremiseSet(case_id="c", decision_id="d1", premises=[
        Premise(id="p1", text="x", dimension=Dimension.SOCIO,
                discipline=Discipline.economics, stakeholder_ids=["S1"], polarity=-1)])
    dyn = select_disciplines(tps, fixed_panel=False)
    fix = select_disciplines(tps, fixed_panel=True)
    # dynamic picks by premises (economics + floor sibling); fixed = the 4 core panel
    assert set(fix.selected) == {Discipline.neuroscience, Discipline.psychology,
                                 Discipline.sociology, Discipline.evolutionary_biology}
    assert Discipline.economics in dyn.selected


def test_case_range_selection():
    from src.pipeline.orchestrate import _select_cases
    csv = Path(__file__).resolve().parent.parent / "data" / "six_cases.csv"
    if not csv.exists():
        pytest.skip("data/six_cases.csv not present")
    rng = _select_cases(str(csv), None, start=1, end=5, sample=None, limit=None)
    assert [c.sample_id for c in rng] == ["001", "002", "003", "004", "005"]
    one = _select_cases(str(csv), None, start=None, end=None, sample=["003"], limit=None)
    assert len(one) == 1 and one[0].sample_id == "003"


# ------------------------------------------------------------ wiring per ablation
def _wire(monkeypatch, tmp_path):
    """Fake every build step; record what process_sample asked each one for."""
    import src.storage.outputs as out
    import src.pipeline.orchestrate as orch
    import src.pipeline.run as run
    from src.arbiter.compile import ArbiterOutput
    from src.evidence.teg import TypedEvidenceGraph
    from src.schemas.core import (
        ActionInventory, DebateResult, Decision, DecisionSet, Premise, ScientistSelection,
        Stakeholder, StakeholderRegister, TestimonySet, TypedPremiseSet,
    )
    monkeypatch.setattr(out, "OUTPUTS_DIR", tmp_path)
    calls = {"decisions": 0, "testimony": [], "debate": [], "statements": [], "debated": []}
    sr = StakeholderRegister(case_id="c", stakeholders=[
        Stakeholder(id="S1", description="driver", type="individual", role="agent")])

    def decisions(case, sr_, ctx):
        calls["decisions"] += 1
        return ActionInventory(), DecisionSet(case_id="c", decisions=[
            Decision(id="d1", statement="divert"), Decision(id="d2", statement="brake")])

    def tps_for(case, d, sr_, ctx, compute_relations=False):
        calls["statements"].append(d.statement)
        return TypedPremiseSet(case_id="c", decision_id=d.id, premises=[
            Premise(id="p1", text="x", dimension=Dimension.NEURO,
                    discipline=Discipline.health_science, stakeholder_ids=["S1"], polarity=1)])

    def testimony(tps, sr_, ctx, teg=None, fixed_panel=False):
        calls["testimony"].append(fixed_panel)
        return (TestimonySet(case_id="c", decision_id=tps.decision_id, testimonies=[]),
                ScientistSelection(case_id="c", decision_id=tps.decision_id, selected=[],
                                   premise_counts={}))

    def debate(tps, sr_, tset, teg, ctx, pmer=None, sit=None, progressive=True,
               decision="", case_text=""):
        calls["debate"].append(progressive)
        calls["debated"].append((decision, case_text))
        return DebateResult(case_id="c", decision_id=tps.decision_id, rounds_run=1,
                            stop_reason="coverage", coverage=1.0, turns=[])

    import src.decisions.generate as gen
    monkeypatch.setattr(gen, "generate_case_proposition", lambda cid, text: DecisionSet(
        case_id=cid, decisions=[Decision(id="d1", statement="Turning the trolley is the "
                                         "right course for the driver.")]))
    monkeypatch.setattr(orch, "ingest_case", lambda case, ctx: sr)
    monkeypatch.setattr(orch, "build_decisions", decisions)
    monkeypatch.setattr(orch, "build_tps_for_decision", tps_for)
    monkeypatch.setattr(orch, "populate_teg",
                        lambda tps, sr_, ctx, pmer=None: TypedEvidenceGraph("c", tps.decision_id))
    monkeypatch.setattr(orch, "build_testimony", testimony)
    monkeypatch.setattr(orch, "build_debate", debate)
    monkeypatch.setattr(run, "build_central_script", lambda case, sr_, m, rows, ctx: (
        ArbiterOutput(script_md="# s", faithfulness=None, ungrounded=[]), {}))
    return orch, calls


@pytest.mark.parametrize("name, n_dec, testimony, progressive", [
    ("none",               2, [False, False], [True, True]),
    ("no_decision_decomp", 1, [False],        [True]),
    ("no_pmer",            2, [False, False], [False, False]),
    ("no_testimony",       2, [],             [True, True]),
    ("fixed_panel",        2, [True, True],   [True, True]),
])
def test_each_ablation_changes_only_its_own_step(monkeypatch, tmp_path, name, n_dec,
                                                 testimony, progressive):
    from src.pipeline.run import RunContext
    from src.storage.cases import MoralCase
    from src.storage.outputs import OutputStore
    orch, calls = _wire(monkeypatch, tmp_path)
    abl = Ablation.from_name(name)
    ctx = RunContext.create("survivability")
    orch.process_sample(MoralCase(case_id="c", text="trolley", stratum="trolley",
                                  sample_id="001"), ctx, OutputStore(ctx.run_id, abl.variant),
                        ablation=abl)
    assert calls["decisions"] == (0 if name == "no_decision_decomp" else 1)
    assert len(calls["statements"]) == n_dec
    assert calls["testimony"] == testimony
    assert calls["debate"] == progressive
    # every debate is told what it debates, and the case it comes from
    assert all(dec and "trolley" in text for dec, text in calls["debated"])
    if name == "no_decision_decomp":
        assert calls["statements"] == ["Turning the trolley is the right course for the driver."]


# ------------------------------------------------------------ variant start guards
def _store(tmp_path, ledger=()):
    from src.storage.outputs import OutputStore
    s = OutputStore("r", "abl_x")         # callers monkeypatch OUTPUTS_DIR to tmp_path
    for sid in ledger:
        s.mark_processed(sid)
    return s


def _corpus(monkeypatch, tmp_path, rows):
    import src.storage.vectordb as vdb
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    (d / "NEURO.jsonl").write_text("\n".join('{"text": "t"}' for _ in range(rows)),
                                   encoding="utf-8")
    monkeypatch.setattr(vdb, "VECTORDB_DIR", d)
    return d


def test_new_variant_refuses_a_warm_corpus(monkeypatch, tmp_path):
    import src.pipeline.orchestrate as orch
    import src.storage.outputs as out
    monkeypatch.setattr(out, "OUTPUTS_DIR", tmp_path)
    _corpus(monkeypatch, tmp_path, rows=24)
    with pytest.raises(SystemExit, match="new id"):
        orch._check_variant_state(_store(tmp_path))
    orch._check_variant_state(_store(tmp_path), allow_warm_corpus=True)


def test_new_variant_on_an_empty_corpus_starts_and_may_resume_its_own(monkeypatch, tmp_path):
    import src.pipeline.orchestrate as orch
    import src.storage.outputs as out
    monkeypatch.setattr(out, "OUTPUTS_DIR", tmp_path)
    _corpus(monkeypatch, tmp_path, rows=0)
    orch._check_variant_state(_store(tmp_path))
    # first case crashed after retrieving: the corpus is its own, so resuming is fine
    _corpus(monkeypatch, tmp_path, rows=7)
    orch._check_variant_state(_store(tmp_path))


def test_variant_finished_by_older_code_is_not_resumed(monkeypatch, tmp_path):
    import src.pipeline.orchestrate as orch
    import src.storage.outputs as out
    monkeypatch.setattr(out, "OUTPUTS_DIR", tmp_path)
    _corpus(monkeypatch, tmp_path, rows=0)
    with pytest.raises(SystemExit, match="archive"):
        orch._check_variant_state(_store(tmp_path, ledger=["001"]))   # no manifest
    s = _store(tmp_path)
    (s.base / "processed_samples.txt").unlink()
    orch._check_variant_state(s)                                     # stamps current
    s.mark_processed("001")
    orch._check_variant_state(s)                                     # same version: ok


def test_resume_on_another_corpus_is_refused(monkeypatch, tmp_path):
    """001 ran on its own worker; 002-006 resumed without --worker hit main's corpus."""
    import src.pipeline.orchestrate as orch
    import src.storage.outputs as out
    import src.storage.vectordb as vdb
    monkeypatch.setattr(out, "OUTPUTS_DIR", tmp_path)
    _corpus(monkeypatch, tmp_path, rows=0)
    s = _store(tmp_path)
    orch._check_variant_state(s)
    s.mark_processed("001")
    other = tmp_path / "main_corpus"
    other.mkdir()
    (other / "NEURO.jsonl").write_text('{"text": "t"}\n' * 24, encoding="utf-8")
    monkeypatch.setattr(vdb, "VECTORDB_DIR", other)
    with pytest.raises(SystemExit, match="was not built by this variant"):
        orch._check_variant_state(s)


def test_resume_with_another_ablation_is_refused(monkeypatch, tmp_path):
    import src.pipeline.orchestrate as orch
    import src.storage.outputs as out
    monkeypatch.setattr(out, "OUTPUTS_DIR", tmp_path)
    _corpus(monkeypatch, tmp_path, rows=0)
    s = _store(tmp_path)
    orch._check_variant_state(s, ablation=Ablation.from_name("no_decision_decomp"))
    with pytest.raises(SystemExit, match="same --ablation"):
        orch._check_variant_state(s, ablation=Ablation.from_name("none"))


def test_own_corpus_built_by_older_code_is_refused(monkeypatch, tmp_path):
    import json
    import src.pipeline.orchestrate as orch
    import src.storage.outputs as out
    monkeypatch.setattr(out, "OUTPUTS_DIR", tmp_path)
    d = _corpus(monkeypatch, tmp_path, rows=0)
    s = _store(tmp_path)
    orch._check_variant_state(s)
    _corpus(monkeypatch, tmp_path, rows=5)          # first case crashed part-way ...
    m = json.loads((s.base / orch.MANIFEST).read_text(encoding="utf-8"))
    m["corpora"][str(d)] = "older-version"           # ... and the code changed since
    (s.base / orch.MANIFEST).write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(SystemExit, match="under pipeline version older-version"):
        orch._check_variant_state(s)


def test_no_testimony_is_stated_to_the_arbiter_and_grounds_saying_so(tmp_path, monkeypatch):
    # With no testimony the arbiter must be told so (not shown an empty heading it is
    # asked to summarise), and its saying so must be a faithfulness source.
    import json
    from src.arbiter.compile import (NO_TESTIMONY, DecisionMaterial, _decision_block,
                                     _source_materials)
    from src.schemas.core import DebateResult, TestimonySet
    tps = TypedPremiseSet(case_id="c", decision_id="d1", premises=[])
    m = DecisionMaterial(decision_id="d1", statement="Do X.", tps=tps,
                         testimony=TestimonySet(case_id="c", decision_id="d1", testimonies=[]),
                         result=DebateResult(case_id="c", decision_id="d1", rounds_run=0,
                                             stop_reason="x", coverage=0.0, turns=[]))
    assert NO_TESTIMONY in _decision_block(m)
    findings, _ = _source_materials([m, m])
    assert findings == [NO_TESTIMONY]

    import scripts.recompute_faithfulness as rf
    d = tmp_path / "v" / "07_testimony" / "001"
    d.mkdir(parents=True)
    (d / "testimony_d1.json").write_text(json.dumps({"testimonies": [], "selection": {}}))
    monkeypatch.setattr(rf, "OUTPUTS_DIR", tmp_path)
    assert rf.load_materials("v", "001")[0] == [NO_TESTIMONY]


def test_arbiter_note_only_when_no_decision_has_testimony():
    from src.prompts.arbiter import NO_TESTIMONY_NOTE, arbiter_user
    with_t = arbiter_user("case", "Survivability", ["block"])
    assert NO_TESTIMONY_NOTE not in with_t
    assert with_t == ("CASE:\ncase\n\nANCHOR: Survivability\n\nDEBATE MATERIALS (per decision):"
                      "\n\nblock\n\nWrite the central script now, following the structure and "
                      "rules exactly. Compile only from the materials above; declare no verdict."), \
        "the prompt every other variant sees must not change"
    assert NO_TESTIMONY_NOTE in arbiter_user("case", "Survivability", ["block"], no_testimony=True)
