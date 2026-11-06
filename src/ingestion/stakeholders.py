"""
P4 — Stakeholder extraction (Step 2b) + Stakeholder Register assembly (Step 2c).

An LLM (stakeholder_extraction role, low temperature, structured output) lists the
affected parties. IDs are assigned deterministically afterward (S1, S2, ...) so they
are stable and not left to the model. The assembled SR is written to SR.json and is
treated as immutable by every downstream phase.
"""
from __future__ import annotations

from src.prompts.ingestion import STAKEHOLDER_SYSTEM, stakeholder_user
from src.schemas.core import Stakeholder, StakeholderExtraction, StakeholderRegister
from src.services.llm_client import get_client


def extract_stakeholders(case_id: str, case_text: str) -> StakeholderRegister:
    client = get_client("stakeholder_extraction")
    result = client.generate_structured(
        user=stakeholder_user(case_text),
        schema=StakeholderExtraction,
        system=STAKEHOLDER_SYSTEM,
        max_tokens=1200,
    )
    # Deterministic re-id: overwrite whatever the model put in `id`.
    fixed: list[Stakeholder] = []
    for i, s in enumerate(result.stakeholders, start=1):
        fixed.append(Stakeholder(
            id=f"S{i}",
            description=s.description.strip(),
            type=s.type,
            role=s.role,
            vulnerability_flag=s.vulnerability_flag,
        ))
    return StakeholderRegister(case_id=case_id, stakeholders=fixed)
