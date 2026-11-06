"""
P5 — Action decomposition (Step 3). Kept as its own step (not folded into decision
generation), so the derivation of the decision space is auditable.
"""
from __future__ import annotations

from src.prompts.decisions import ACTION_SYSTEM, action_user
from src.schemas.core import ActionInventory, StakeholderRegister
from src.services.llm_client import get_client


def decompose_actions(case_text: str, sr: StakeholderRegister) -> ActionInventory:
    client = get_client("action_decomposition")
    return client.generate_structured(
        user=action_user(case_text, sr),
        schema=ActionInventory,
        system=ACTION_SYSTEM,
        max_tokens=1000,
    )
