"""
P7 — Dynamic scientist selection (Step 7a).

Who testifies, and how many, is decided by the premises — not a constant. Rule:

    a discipline testifies iff it holds >= 1 surviving TPS premise.

So the count is COUNT(DISTINCT discipline) over the TPS. Ceiling = pool size (7).
Floor = 2: if the TPS collapses onto a single discipline, add one more so the debate
has cross-disciplinary grounding (CogniAlign's core value). The added discipline is
a dimension-sibling of the lone discipline where one exists (e.g. NEURO has
neuroscience + health_science), else the first unused discipline in a fixed order. It
testifies on the lone discipline's premises (see assign_premises).

Fully deterministic: a Counter + a SELECT DISTINCT. No LLM decides who speaks.
"""
from __future__ import annotations

from collections import Counter

from config.settings import get_settings
from src.schemas.core import (
    DISCIPLINE_TO_DIMENSION, Discipline, ScientistSelection, TypedPremiseSet,
)

# Reverse map: dimension -> disciplines that can settle it (for the floor rule).
_DIM_TO_DISCIPLINES: dict = {}
for _d, _dim in DISCIPLINE_TO_DIMENSION.items():
    _DIM_TO_DISCIPLINES.setdefault(_dim, []).append(_d)

# Deterministic fallback order when no dimension-sibling exists.
_FALLBACK_ORDER = [
    Discipline.psychology, Discipline.sociology, Discipline.neuroscience,
    Discipline.evolutionary_biology, Discipline.economics,
    Discipline.health_science, Discipline.ecology,
]


# CogniAlign's fixed 4-scientist panel (for the --fixed-panel ablation).
_FIXED_PANEL = [Discipline.neuroscience, Discipline.psychology,
                Discipline.sociology, Discipline.evolutionary_biology]

# Under the fixed panel every premise is heard by the panel member for its dimension.
# CogniAlign's four scientists cover the whole case; giving economics, health-science
# and ecology premises to nobody (37% of premises on the post-fix run) would make the
# ablation measure "drop a third of the checklist" instead of "fixed vs dynamic panel".
FIXED_PANEL_FOR_DIMENSION = {
    DISCIPLINE_TO_DIMENSION[d]: d for d in _FIXED_PANEL
}


def assign_premises(tps: TypedPremiseSet, selection: ScientistSelection,
                    fixed_panel: bool = False) -> dict:
    """
    discipline -> the premises it testifies on.

    Dynamic panel: each discipline takes the premises tagged to it. A discipline added
    by the floor rule has none of its own, and used to be skipped downstream, so the
    floor never produced its second witness; it now testifies on the lone discipline's
    premises as a second, cross-disciplinary reading -- which is what the floor is for.
    """
    by_disc: dict = {}
    if fixed_panel:
        for p in tps.premises:
            by_disc.setdefault(FIXED_PANEL_FOR_DIMENSION[p.dimension], []).append(p)
        return by_disc
    for p in tps.premises:
        by_disc.setdefault(p.discipline, []).append(p)
    if selection.floor_applied and selection.selected:
        lone = selection.selected[0]
        for d in selection.selected[1:]:
            if not by_disc.get(d):
                by_disc[d] = list(by_disc.get(lone, []))
    return by_disc


def select_disciplines(tps: TypedPremiseSet, fixed_panel: bool = False) -> ScientistSelection:
    hp = get_settings().hp
    counts = Counter(p.discipline for p in tps.premises)
    present = [d for d, c in counts.items() if c >= 1]
    floor_applied = False

    if fixed_panel:
        # Ablation: always the 4 core disciplines, ignoring the dynamic count rule.
        return ScientistSelection(
            case_id=tps.case_id, decision_id=tps.decision_id,
            selected=list(_FIXED_PANEL),
            premise_counts={d.value: c for d, c in counts.items()},
            floor_applied=False,
        )

    if len(present) >= hp.scientist_floor:
        # Cap at ceiling by premise count (most-loaded disciplines win ties by order).
        selected = [d for d, _ in counts.most_common()][: hp.scientist_ceiling]
    elif len(present) == 1:
        lone = present[0]
        selected = [lone]
        sibling = next(
            (d for d in _DIM_TO_DISCIPLINES.get(DISCIPLINE_TO_DIMENSION[lone], [])
             if d != lone), None)
        add = sibling or next(d for d in _FALLBACK_ORDER if d != lone)
        selected.append(add)
        floor_applied = True
    else:  # no premises at all (shouldn't happen) — guard with two defaults
        selected = _FALLBACK_ORDER[:hp.scientist_floor]
        floor_applied = True

    return ScientistSelection(
        case_id=tps.case_id,
        decision_id=tps.decision_id,
        selected=selected,
        premise_counts={d.value: c for d, c in counts.items()},
        floor_applied=floor_applied,
    )
