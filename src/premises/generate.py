"""
P6 — Premise generation (Step 5a).

Calls the `premise_generation` role (which the registry guarantees is a different
model from the scientist pool, so no scientist later testifies on premises it wrote).
The LLM returns `PremiseDraft`s tagged with a discipline; the dimension is DERIVED
from the discipline (DISCIPLINE_TO_DIMENSION) so the two can never disagree.
Stakeholder ids are validated against the SR — unknown ids are dropped.
"""
from __future__ import annotations

import re

from src.ingestion.anchor import AnchorConfig
from src.prompts.premises import premise_system, premise_user
from src.schemas.core import (
    DISCIPLINE_TO_DIMENSION, Decision, Premise, PremiseDraftSet, StakeholderRegister,
)
from src.services.llm_client import get_client

_SID = re.compile(r"\bS\d+\b")


def _register_ids(raw: list[str], valid_ids: set[str]) -> list[str]:
    """
    Register ids named in the model's stakeholder list, de-duplicated in order.

    Models write "S2 (Patients)" or "S1, S3" as often as "S2"; an exact match dropped
    those premises silently, and a TPS with every premise dropped still went to debate.
    """
    out: list[str] = []
    for item in raw or []:
        for sid in _SID.findall(str(item)):
            if sid in valid_ids and sid not in out:
                out.append(sid)
    return out


def generate_premises(
    decision: Decision,
    case_text: str,
    sr: StakeholderRegister,
    anchor: AnchorConfig,
) -> list[Premise]:
    client = get_client("premise_generation")
    drafts = client.generate_structured(
        user=premise_user(decision, case_text, sr),
        schema=PremiseDraftSet,
        system=premise_system(anchor),
        # The prompt asks for 4-10 premises, each carrying text + discipline +
        # stakeholder_ids + polarity, so a full set is ~1.5-1.8k tokens of JSON before
        # any overhead. At 2000 this truncated constantly: 4 of 8 calls on sample 001
        # returned exactly 2000, the payload failed to parse, and generate_structured's
        # JSON-mode fallback re-ran the whole call — which is why one case logged 8
        # premise calls for 4 decisions, paying twice for half of them. Reasoning models
        # make it worse, since max_completion_tokens also has to cover invisible
        # reasoning tokens before the first character of JSON is emitted. Output is
        # billed on actual usage, not on the ceiling, so raising this costs nothing and
        # removes the duplicate call.
        max_tokens=6000,
    )

    valid_ids = {s.id for s in sr.stakeholders}
    premises: list[Premise] = []
    for i, d in enumerate(drafts.premises, start=1):
        sids = _register_ids(d.stakeholder_ids, valid_ids)
        if not sids:
            # A premise must bind to a real stakeholder; skip orphans.
            continue
        premises.append(Premise(
            id=f"p{i}",
            text=d.text.strip(),
            dimension=DISCIPLINE_TO_DIMENSION[d.discipline],  # derived, not model-set
            discipline=d.discipline,
            stakeholder_ids=sids,
            polarity=d.polarity,
        ))
    return premises
