"""
Deterministic end-to-end wiring test for the orchestrator: mock every LLM/retrieval
build step so process_sample runs fast, and assert that ALL step outputs are saved,
the full bundle is written, and the resume ledger is updated. Proves the L1->L9
plumbing + storage requirement without any network/LLM.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_process_sample_saves_all_steps(monkeypatch, tmp_path):
    # redirect outputs to temp (variant-namespaced under it)
    import src.storage.outputs as out
    monkeypatch.setattr(out, "OUTPUTS_DIR", tmp_path)

    import src.pipeline.orchestrate as orch
    import src.pipeline.run as run
    from src.arbiter.compile import ArbiterOutput
    from src.evidence.teg import TypedEvidenceGraph
    from src.schemas.core import (
        ActionInventory, Decision, DecisionSet, Dimension, Discipline, EvidenceChunk,
        Premise, ScientistSelection, ScientistTestimony, Stakeholder,
        StakeholderRegister, TestimonySet, DebateResult, DebateTurn, TurnClaim,
        TypedPremiseSet,
    )

    sr = StakeholderRegister(case_id="c", stakeholders=[
        Stakeholder(id="S1", description="wife", type="individual", role="patient")])
    dset = DecisionSet(case_id="c", decisions=[Decision(id="d1", statement="steal")])
    tps = TypedPremiseSet(case_id="c", decision_id="d1", premises=[
        Premise(id="p1", text="radium helps", dimension=Dimension.NEURO,
                discipline=Discipline.health_science, stakeholder_ids=["S1"], polarity=1)])
    teg = TypedEvidenceGraph("c", "d1")
    teg.add(EvidenceChunk(node_id="", text="radium treats cancer", doi="10.1/x",
                          source="pubmed", dimension=Dimension.NEURO,
                          stakeholder_ids=["S1"], citation_count=2))
    tset = TestimonySet(case_id="c", decision_id="d1", testimonies=[
        ScientistTestimony(discipline=Discipline.health_science, claims=[])])
    sel = ScientistSelection(case_id="c", decision_id="d1",
                             selected=[Discipline.health_science],
                             premise_counts={"health_science": 1})
    result = DebateResult(case_id="c", decision_id="d1", rounds_run=1,
                          stop_reason="coverage", coverage=1.0, turns=[
        DebateTurn(round=1, speaker="proponent", premises_addressed=["p1"],
                   evidence_cited=[next(iter(teg.g.nodes))],
                   claims=[TurnClaim(stakeholder_id="S1", dimension=Dimension.NEURO,
                                     polarity=1, evidence_nodes=[next(iter(teg.g.nodes))])],
                   evidential_gap="", argument_text="radium helps survival")])

    monkeypatch.setattr(orch, "ingest_case", lambda case, ctx: sr)
    monkeypatch.setattr(orch, "build_decisions",
                        lambda case, sr_, ctx: (ActionInventory(), dset))
    monkeypatch.setattr(orch, "build_tps_for_decision",
                        lambda case, d, sr_, ctx, compute_relations=False: tps)
    monkeypatch.setattr(orch, "populate_teg",
                        lambda tps_, sr_, ctx, pmer=None: teg)
    monkeypatch.setattr(orch, "build_testimony",
                        lambda tps_, sr_, ctx, teg=None, fixed_panel=False: (tset, sel))
    monkeypatch.setattr(orch, "build_debate",
                        lambda tps_, sr_, tset_, teg_, ctx, pmer=None, sit=None, progressive=True, **kw: result)
    monkeypatch.setattr(run, "build_central_script",
                        lambda case, sr_, materials, sit_rows, ctx: (
                            ArbiterOutput(script_md="# Central\nRadium helps survival.",
                                          faithfulness=0.9, ungrounded=[]),
                            {"faithfulness": {"fraction_entailed": 0.9}, "decisions": []}))

    from src.pipeline.run import RunContext
    from src.storage.cases import MoralCase
    ctx = RunContext.create("survivability")
    store = out.OutputStore(ctx.run_id)
    case = MoralCase(case_id="c", text="Heinz case", stratum="demonstration", sample_id="007")

    summary = process_sample = orch.process_sample(case, ctx, store)
    store.finalize_sample("007", extra={"summary": summary})

    # every step folder has the sample's artifact (under outputs/<variant>/)
    base = tmp_path / "main"
    for step in ["01_ingestion", "02_stakeholders", "03_actions", "04_decisions",
                 "05_premises", "06_evidence_teg", "07_testimony", "08_debate",
                 "09_arbiter"]:
        d = base / step / "007"
        assert d.exists() and any(d.iterdir()), f"missing outputs for {step}"

    # central script md saved, full bundle + ledger
    assert (base / "09_arbiter" / "007" / "central_script.md").exists()
    assert (base / "full_pipeline" / "007.json").exists()
    assert store.is_processed("007")
    assert summary["script_faithfulness"] == 0.9
