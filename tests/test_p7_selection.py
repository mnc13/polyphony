"""
P7 — dynamic scientist selection (pure deterministic logic, no LLM/network).
Covers: multi-discipline pass-through, the floor rule (lone discipline gets a
dimension-sibling), and premise-count reporting.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas.core import (
    Dimension, Discipline, Premise, TypedPremiseSet,
)
from src.testimony.select import select_disciplines


def _p(pid, disc, dim, sid="S1", pol=-1):
    return Premise(id=pid, text=f"claim {pid}", dimension=dim, discipline=disc,
                   stakeholder_ids=[sid], polarity=pol)


def _tps(premises):
    return TypedPremiseSet(case_id="c1", decision_id="d1", premises=premises)


def test_multiple_disciplines_all_selected():
    tps = _tps([
        _p("p1", Discipline.sociology, Dimension.SOCIO),
        _p("p2", Discipline.sociology, Dimension.SOCIO),
        _p("p3", Discipline.health_science, Dimension.NEURO),
        _p("p4", Discipline.psychology, Dimension.PSYCH),
    ])
    sel = select_disciplines(tps)
    assert set(sel.selected) == {Discipline.sociology, Discipline.health_science, Discipline.psychology}
    assert sel.floor_applied is False
    assert sel.premise_counts["sociology"] == 2


def test_floor_lone_discipline_adds_dimension_sibling():
    # Only neuroscience premises -> floor rule must add a second discipline.
    tps = _tps([
        _p("p1", Discipline.neuroscience, Dimension.NEURO),
        _p("p2", Discipline.neuroscience, Dimension.NEURO),
    ])
    sel = select_disciplines(tps)
    assert Discipline.neuroscience in sel.selected
    assert len(sel.selected) == 2
    assert sel.floor_applied is True
    # NEURO's sibling is health_science
    assert Discipline.health_science in sel.selected


def test_ceiling_not_exceeded():
    # All 7 disciplines present -> selected capped at 7, none dropped here.
    dims = {
        Discipline.neuroscience: Dimension.NEURO, Discipline.psychology: Dimension.PSYCH,
        Discipline.sociology: Dimension.SOCIO, Discipline.evolutionary_biology: Dimension.EVOBIO,
        Discipline.economics: Dimension.SOCIO, Discipline.health_science: Dimension.NEURO,
        Discipline.ecology: Dimension.EVOBIO,
    }
    prem = [_p(f"p{i}", d, dims[d]) for i, d in enumerate(dims, start=1)]
    sel = select_disciplines(_tps(prem))
    assert len(sel.selected) == 7 and len(set(sel.selected)) == 7
