"""
P5 — Decision generation (Step 4).

From the action inventory, generate the space of feasible alternative decisions
d1..dn. Near-duplicate statements are collapsed by embedding cosine, and the set is
capped at `max_decisions`. Decisions are NOT ranked — order carries no meaning; ids
are assigned in generation order purely as labels.
"""
from __future__ import annotations

import numpy as np

from config.settings import get_settings
from src.decisions.actions import decompose_actions
from src.prompts.decisions import (
    CASE_PROPOSITION_SYSTEM, DECISION_SYSTEM, case_proposition_user, decision_user,
)
from src.schemas.core import (
    ActionInventory, Decision, DecisionGeneration, DecisionSet, StakeholderRegister,
)
from src.services.embedder import embed
from src.services.llm_client import get_client


def _dedup(statements: list[str], threshold: float) -> list[str]:
    """Greedy: keep a statement only if it is < threshold cosine to all kept ones."""
    kept: list[str] = []
    kept_vecs: list[np.ndarray] = []
    if not statements:
        return kept
    vecs = embed(statements)
    for stmt, v in zip(statements, vecs):
        if kept_vecs:
            sims = np.array([float(np.dot(v, kv)) for kv in kept_vecs])
            if sims.max() >= threshold:
                continue
        kept.append(stmt.strip())
        kept_vecs.append(v)
    return kept


def generate_decisions(
    case_id: str,
    case_text: str,
    sr: StakeholderRegister,
    inventory: ActionInventory | None = None,
) -> tuple[ActionInventory, DecisionSet]:
    hp = get_settings().hp
    inv = inventory or decompose_actions(case_text, sr)

    client = get_client("decision_generation")
    gen = client.generate_structured(
        user=decision_user(case_text, inv.performed, inv.omitted, inv.alternatives),
        schema=DecisionGeneration,
        system=DECISION_SYSTEM,
        max_tokens=800,
    )

    unique = _dedup([s for s in gen.decisions if s.strip()], hp.decision_dedup_threshold)
    unique = unique[: hp.max_decisions]
    decisions = [Decision(id=f"d{i}", statement=s) for i, s in enumerate(unique, start=1)]
    return inv, DecisionSet(case_id=case_id, decisions=decisions)


def generate_case_proposition(case_id: str, case_text: str) -> DecisionSet:
    """
    The no_decision_decomp ablation's single decision: the case's own contested action,
    stated as one proposition, with no action decomposition and no alternatives.

    The ablation used to debate the fixed string "What is the right course of action in
    this case?". A question has nothing to enact, so the premise model and both debaters
    each picked an option themselves (the old run's premises were all about diverting
    the trolley) and polarity had no defined meaning. Main's decision-generation model
    states the proposition, so the ablation removes the fan-out and nothing else.
    """
    gen = get_client("decision_generation").generate_structured(
        user=case_proposition_user(case_text), schema=DecisionGeneration,
        system=CASE_PROPOSITION_SYSTEM, max_tokens=800)
    statements = [s.strip() for s in gen.decisions if s.strip()]
    if not statements:
        raise ValueError(f"{case_id}: no case proposition returned")
    return DecisionSet(case_id=case_id, decisions=[Decision(id="d1", statement=statements[0])])
