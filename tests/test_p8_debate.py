"""
P8 — debate engine logic (round loop, coverage, stopping, SIT writes) with mocked
debaters and mocked NLI, so no live LLM/network is needed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evidence.teg import TypedEvidenceGraph
from src.ingestion.anchor import get_anchor_config
from src.schemas.core import (
    Anchor, Dimension, Discipline, EvidenceChunk, Premise, ScientistTestimony,
    Stakeholder, StakeholderRegister, TurnClaim, DebateTurnDraft,
    TypedPremiseSet,
)
from src.schemas.core import TestimonySet as _TestimonySet  # aliased: avoid pytest collection

_ARG = ("Radium improves survival for S1, per the cited evidence. " * 8).strip()


@pytest.fixture(autouse=True)
def _neutral_stance(monkeypatch):
    # The role check scores a turn's opening with NLI; keep the model out of these tests.
    import src.debate.engine as eng
    monkeypatch.setattr(eng, "_stance", lambda opening, label: (0.0, 0.0))


def _fixture():
    sr = StakeholderRegister(case_id="c1", stakeholders=[
        Stakeholder(id="S1", description="the wife", type="individual", role="patient")])
    tps = TypedPremiseSet(case_id="c1", decision_id="d1", premises=[
        Premise(id="p1", text="Radium extends S1 survival.", dimension=Dimension.NEURO,
                discipline=Discipline.health_science, stakeholder_ids=["S1"], polarity=1),
        Premise(id="p2", text="Theft stress harms S1 family.", dimension=Dimension.PSYCH,
                discipline=Discipline.psychology, stakeholder_ids=["S1"], polarity=-1),
    ])
    teg = TypedEvidenceGraph("c1", "d1")
    teg.add(EvidenceChunk(node_id="", text="Radium treats cancer, improving survival.",
                          doi="10.1/x", source="pubmed", dimension=Dimension.NEURO,
                          stakeholder_ids=["S1"], citation_count=3))
    tset = _TestimonySet(case_id="c1", decision_id="d1", testimonies=[
        ScientistTestimony(discipline=Discipline.health_science, claims=[])])
    return sr, tps, teg, tset


def _add_stress_node(teg):
    return teg.add(EvidenceChunk(
        node_id="", text="Committing theft causes chronic stress that harms the family "
        "of the person who steals.", doi="10.1/y", source="openalex",
        dimension=Dimension.PSYCH, stakeholder_ids=["S1"], citation_count=2))


_CLAIMS = [  # one claim per cited node, about premise p1, p2, ... in order
    dict(premise_id="p1", dimension=Dimension.NEURO, polarity=1,
         statement="Radium treatment improves the survival of S1."),
    dict(premise_id="P2 (theft stress)", dimension=Dimension.PSYCH, polarity=-1,
         statement="Committing theft causes chronic stress that harms S1's family."),
]


class _FakeClient:
    def __init__(self, cited, arg=_ARG, short_first=0):
        self.cited = cited
        self.arg = arg
        self.short_first = short_first
        self.calls = 0
        self.prompts = []

    def generate_structured(self, user, schema, system=None, max_tokens=0):
        self.calls += 1
        self.prompts.append(user)
        text = "See below." if self.calls <= self.short_first else self.arg
        return DebateTurnDraft(
            premises_addressed=["p1", "p2"],          # lists the whole checklist
            evidence_cited=list(self.cited),
            claims=[TurnClaim(stakeholder_id="S1 (the wife)", evidence_nodes=[node], **c)
                    for node, c in zip(self.cited, _CLAIMS)],
            evidential_gap="need trial data on theft-related family stress (p2)",
            argument_text=text,
        )


class _NoPMER:
    def __init__(self):
        self.calls = []

    def retrieve(self, *a, **k):
        from src.evidence.pmer import PMERResult
        self.calls.append(k)
        return PMERResult(admitted=[], rejected=[], mean_novelty=0.0, used_live=False)


def _run(monkeypatch, tmp_path, client, teg=None, progressive=True):
    import src.debate.engine as eng
    sr, tps, teg0, tset = _fixture()
    teg = teg or teg0
    monkeypatch.setattr(eng, "get_client", lambda role: client)
    monkeypatch.setattr(eng.nli, "citation_integrity", lambda a, b: (True, 0.9))
    from src.storage.sit import SITStore
    from src.debate.engine import DebateEngine, DebateDeps
    pmer = _NoPMER()
    deps = DebateDeps(pmer=pmer, sit=SITStore(tmp_path / "sit.db"), run_id="r1")
    result = DebateEngine(deps).run(tps, sr, tset, teg, get_anchor_config(Anchor.survivability),
                                    progressive=progressive)
    return result, deps, pmer


def test_debate_coverage_stop(monkeypatch, tmp_path):
    sr, tps, teg, tset = _fixture()
    radium = next(iter(teg.g.nodes))
    stress = _add_stress_node(teg)
    result, deps, _ = _run(monkeypatch, tmp_path, _FakeClient([radium, stress]), teg=teg)

    assert result.coverage == 1.0
    assert result.stop_reason == "coverage"
    assert result.rounds_run == 1           # covered on the opening round
    assert len(result.turns) == 2           # proponent + opponent
    rows = deps.sit.rows_for("c1", "d1")
    assert len(rows) >= 2 and rows[0]["nli_supported"] == 1
    assert rows[0]["stakeholder_id"] == "S1", "labelled ids are canonicalised to the register"


def test_listing_a_premise_is_not_covering_it(monkeypatch, tmp_path):
    """One radium citation must not cover the theft-stress premise just because the
    turn listed it -- that rule ended 32 of 35 real debates after round 1."""
    _, _, teg, _ = _fixture()
    radium = next(iter(teg.g.nodes))
    result, _, pmer = _run(monkeypatch, tmp_path, _FakeClient([radium]))
    assert result.coverage == 0.5
    # Nothing new is admitted in rounds 2-3 (novelty 0), so the novelty stop ends it.
    assert result.stop_reason == "novelty_exhausted"
    assert result.rounds_run == 3
    # Progressive retrieval targets the uncovered premise, steered by the gap, live.
    assert pmer.calls and all(c.get("force_live") for c in pmer.calls)
    assert "family stress" in pmer.calls[0].get("hint", "")


def test_stuck_debate_stops_when_coverage_stalls(monkeypatch, tmp_path):
    """Retrieval keeps finding 'novel' material the debaters never use: without the
    stall stop this runs all 8 rounds."""
    import src.debate.engine as eng
    from src.evidence.pmer import PMERResult
    _, _, teg, _ = _fixture()
    radium = next(iter(teg.g.nodes))

    class _NovelPMER(_NoPMER):
        def retrieve(self, *a, **k):
            self.calls.append(k)
            from types import SimpleNamespace
            return PMERResult(admitted=[SimpleNamespace(novelty_score=0.6)], rejected=[],
                              mean_novelty=0.6, used_live=True)

    sr, tps, _, tset = _fixture()
    monkeypatch.setattr(eng, "get_client", lambda role: _FakeClient([radium]))
    monkeypatch.setattr(eng.nli, "citation_integrity", lambda a, b: (True, 0.9))
    from src.storage.sit import SITStore
    deps = eng.DebateDeps(pmer=_NovelPMER(), sit=SITStore(tmp_path / "sit.db"), run_id="r1")
    result = eng.DebateEngine(deps).run(tps, sr, tset, teg, get_anchor_config(Anchor.survivability))
    assert result.stop_reason == "coverage_stalled" and result.rounds_run == 3


def test_no_pmer_still_has_a_novelty_stop(monkeypatch, tmp_path):
    _, _, teg, _ = _fixture()
    radium = next(iter(teg.g.nodes))
    result, _, pmer = _run(monkeypatch, tmp_path, _FakeClient([radium]), progressive=False)
    assert pmer.calls == []
    assert result.stop_reason == "novelty_exhausted" and result.rounds_run == 3


def test_empty_argument_is_retried(monkeypatch, tmp_path):
    sr, tps, teg, tset = _fixture()
    radium = next(iter(teg.g.nodes))
    stress = _add_stress_node(teg)
    client = _FakeClient([radium, stress], short_first=1)
    result, _, _ = _run(monkeypatch, tmp_path, client, teg=teg)
    assert result.turns[0].argument_text == _ARG
    assert client.calls == 3          # proponent: short + retry; opponent: one call


def test_debaters_are_told_the_decision_and_the_case(monkeypatch, tmp_path):
    import src.debate.engine as eng
    from src.storage.sit import SITStore
    sr, tps, teg, tset = _fixture()
    client = _FakeClient([next(iter(teg.g.nodes)), _add_stress_node(teg)])
    monkeypatch.setattr(eng, "get_client", lambda role: client)
    monkeypatch.setattr(eng.nli, "citation_integrity", lambda a, b: (True, 0.9))
    deps = eng.DebateDeps(pmer=_NoPMER(), sit=SITStore(tmp_path / "sit.db"), run_id="r1")
    eng.DebateEngine(deps).run(tps, sr, tset, teg, get_anchor_config(Anchor.survivability),
                               decision="Warning the workmen is the right course.",
                               case_text="A trolley's brakes have failed.")
    assert client.prompts and all(
        "DECISION UNDER DEBATE (d1): Warning the workmen is the right course." in p
        and "A trolley's brakes have failed." in p for p in client.prompts)


def test_hedge_statement_is_not_a_supported_citation(monkeypatch, tmp_path):
    import src.debate.engine as eng
    from src.schemas.core import DebateTurn
    from src.storage.sit import SITStore
    sr, tps, teg, _ = _fixture()
    radium = next(iter(teg.g.nodes))
    monkeypatch.setattr(eng, "get_client", lambda role: _FakeClient([radium]))
    monkeypatch.setattr(eng.nli, "citation_integrity", lambda a, b: (True, 0.99))
    deps = eng.DebateDeps(pmer=_NoPMER(), sit=SITStore(tmp_path / "sit.db"), run_id="r1")
    turn = DebateTurn(round=1, speaker="opponent", premises_addressed=["p1"],
                      evidence_cited=[radium], evidential_gap="", argument_text=_ARG,
                      claims=[TurnClaim(stakeholder_id="S1", dimension=Dimension.NEURO,
                                        polarity=-1, evidence_nodes=[radium], premise_id="p1",
                                        statement="The cited nodes do not establish that "
                                                  "radium improves the survival of S1.")])
    eng.DebateEngine(deps)._update_sit(turn, teg, tps, "c1", "d1")
    assert deps.sit.rows_for("c1", "d1")[0]["nli_supported"] == 0


def test_round_novelty_pools_admitted_chunks(monkeypatch, tmp_path):
    """One target admitting fresh chunks among four that admit nothing is not an
    exhausted round."""
    import src.debate.engine as eng
    from types import SimpleNamespace
    from src.evidence.pmer import PMERResult
    from src.storage.sit import SITStore
    _, _, teg, _ = _fixture()
    radium = next(iter(teg.g.nodes))

    class _OneHitPMER(_NoPMER):
        def retrieve(self, *a, **k):
            self.calls.append(k)
            hit = len(self.calls) % 4 == 1
            return PMERResult(admitted=[SimpleNamespace(novelty_score=0.35)] if hit else [],
                              rejected=[], mean_novelty=0.35 if hit else 0.0, used_live=True)

    sr, tps, _, tset = _fixture()
    tps.premises += [Premise(id=f"p{i}", text=f"Unrelated premise {i} about weather.",
                             dimension=Dimension.SOCIO, discipline=Discipline.sociology,
                             stakeholder_ids=["S1"], polarity=1) for i in range(3, 6)]
    monkeypatch.setattr(eng, "get_client", lambda role: _FakeClient([radium]))
    monkeypatch.setattr(eng.nli, "citation_integrity", lambda a, b: (True, 0.9))
    deps = eng.DebateDeps(pmer=_OneHitPMER(), sit=SITStore(tmp_path / "sit.db"), run_id="r1")
    result = eng.DebateEngine(deps).run(tps, sr, tset, teg, get_anchor_config(Anchor.survivability))
    assert result.stop_reason != "novelty_exhausted"


def test_empty_premise_set_is_not_debated(monkeypatch, tmp_path):
    import src.debate.engine as eng
    from src.storage.sit import SITStore
    sr, tps, teg, tset = _fixture()
    tps.premises = []
    client = _FakeClient([next(iter(teg.g.nodes))])
    monkeypatch.setattr(eng, "get_client", lambda role: client)
    deps = eng.DebateDeps(pmer=_NoPMER(), sit=SITStore(tmp_path / "sit.db"), run_id="r1")
    r = eng.DebateEngine(deps).run(tps, sr, tset, teg, get_anchor_config(Anchor.survivability))
    assert r.stop_reason == "no_premises" and r.turns == [] and client.calls == 0


_REVERSED = ("I argue that enacting d1 is NOT sound with respect to Survivability for the "
             "stakeholders, because the harms to S1 outweigh any benefit. " + "x" * 300)


@pytest.mark.parametrize("side, text, flagged", [
    ("proponent", "I argue that enacting d1 is NOT sound with respect to Survivability.", True),
    ("proponent", "I argue that enacting the trial is HARMFUL with respect to Survivability.", True),
    ("proponent", "I argue that enacting d1 is sound with respect to Survivability.", False),
    ("proponent", "I argue that framing the drug is not demonstrably harmful to Survivability.", False),
    ("proponent", "The opponent says d1 is not sound. I disagree.", False),   # not a position
    ("opponent", "I argue that enacting d1 is harmful with respect to Survivability.", False),
    ("opponent", "As the opponent, I argue that enacting d1 is sound for S2.", True),
    ("opponent", "I argue that d1, though some call it sound, is unsound for S3.", False),
])
def test_a_turn_arguing_the_other_side_is_detected(side, text, flagged):
    from src.debate.engine import argues_other_side
    assert argues_other_side(side, text) is flagged


def test_turn_arguing_the_other_side_is_retried_with_its_side(monkeypatch, tmp_path):
    sr, tps, teg, tset = _fixture()
    client = _FakeClient([next(iter(teg.g.nodes)), _add_stress_node(teg)])
    texts = [_REVERSED]

    real = client.generate_structured

    def gen(user, schema, system=None, max_tokens=0):
        d = real(user, schema, system, max_tokens)
        if texts and "PROPONENT" not in user.split("YOUR PREVIOUS ATTEMPT")[-1][:60]:
            d.argument_text = texts.pop(0)
        return d
    client.generate_structured = gen
    result, _, _ = _run(monkeypatch, tmp_path, client, teg=teg)
    assert result.turns[0].argument_text == _ARG
    assert "YOUR PREVIOUS ATTEMPT ARGUED THE OTHER SIDE. You are the PROPONENT" in client.prompts[1]
    assert "is sound with respect to Survivability" in client.prompts[1]


@pytest.mark.parametrize("side, opening, scores, flagged", [
    ("proponent", "The decision to proceed with the rollout undermines survivability for S3.",
     (0.0, 0.99), True),
    ("proponent", "The opponent's critique undermines its own survivability case.", (0.0, 0.99), False),
    ("proponent", "Preserving five lives is the core survivability gain of d1.", (0.9, 0.0), False),
    ("proponent", "The rollout has mixed effects on survivability.", (0.3, 0.6), False),
    ("opponent", "Diverting the trolley preserves the most lives overall.", (0.98, 0.01), True),
    ("opponent", "The proponent's case collapses under its own citations.", (0.98, 0.01), False),
])
def test_unstated_position_is_read_with_nli(monkeypatch, side, opening, scores, flagged):
    import src.debate.engine as eng
    monkeypatch.setattr(eng, "_stance", lambda o, label: scores)
    assert eng.argues_other_side(side, opening, "Survivability") is flagged


def test_persistent_reversal_is_recorded(monkeypatch, tmp_path):
    import src.debate.engine as eng
    sr, tps, teg, tset = _fixture()
    client = _FakeClient([next(iter(teg.g.nodes)), _add_stress_node(teg)], arg=_REVERSED)
    result, _, _ = _run(monkeypatch, tmp_path, client, teg=teg)
    assert {"round": 1, "side": "proponent"} in result.role_reversals
    assert all(r["side"] == "proponent" for r in result.role_reversals)
