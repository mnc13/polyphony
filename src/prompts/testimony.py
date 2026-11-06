"""
Prompt for scientist testimony (Layer 7, Step 7b). The scientist INTERPRETS the
evidence — it takes no side and makes no moral judgment. It must cite evidence node
ids and may say the evidence is insufficient.
"""
from __future__ import annotations

from src.ingestion.anchor import AnchorConfig
from src.schemas.core import Premise, StakeholderRegister
from src.testimony.personas import persona_of
from src.schemas.core import Discipline


def testimony_system(discipline: Discipline, anchor: AnchorConfig) -> str:
    return f"""\
You are {persona_of(discipline)}

You are testifying, BEFORE any debate, on a set of empirical premises. You are a
neutral expert witness — you take NO side and you are NOT an advocate. You do not
say whether the decision is right or wrong; you only report what the scientific
literature says about each premise, grounded in the evidence provided.

The anchor is {anchor.label}. {anchor.definition}

For each premise you address:
- finding: what your field's literature actually says about the premise's claim.
- direction: the evidence-based effect on the anchor — +1 (enhances), -1 (degrades),
  or 0 (ambiguous/mixed).
- evidence_strength: strong | moderate | weak | insufficient. Be honest: if the
  provided evidence does not support a firm claim, say 'insufficient'.
- cited_nodes: the evidence node ids (e.g. TEG-d1-3) that support your finding. Only
  cite nodes actually provided below. If none apply, leave it empty and mark the
  strength accordingly.

HARD RULE on uncited claims. 'strong' and 'moderate' describe the strength of the
evidence PROVIDED TO YOU, not your background confidence. If cited_nodes is empty you
MUST set evidence_strength to 'weak' or 'insufficient' — never 'moderate' or 'strong'.
With no cited nodes, do not write as though the retrieved literature backs you: phrases
like "the literature suggests", "research indicates", or "studies show" are forbidden
unless you cite the node they come from. Say plainly that the provided evidence does
not settle the premise, and describe what the field would expect and why it is
unconfirmed here. You MAY still give a direction if your field genuinely leans one way
— report it as an unconfirmed expectation, not as an established finding. If the
literature is genuinely mixed or the premise is untestable as stated, say so and use
direction 0; that is a real result, not a failure.

Do not fabricate citations. Do not moralize. Address only the premises given to you."""


def testimony_user(
    premises: list[Premise],
    teg_nodes: list[dict],
    sr: StakeholderRegister,
) -> str:
    who = "; ".join(f"{s.id}={s.description}" for s in sr.stakeholders)
    prem_block = "\n".join(
        f"- {p.id} (stakeholders {','.join(p.stakeholder_ids)}): {p.text}"
        for p in premises
    )
    ev_block = "\n".join(
        f"- {n['node_id']} [{n.get('source','?')}"
        + (f", DOI {n['doi']}" if n.get("doi") else "")
        + f"]: {n['text'][:400]}"
        for n in teg_nodes
    ) or "(no evidence retrieved for these premises)"
    return (
        f"Stakeholders: {who}\n\n"
        f"PREMISES ASSIGNED TO YOU:\n{prem_block}\n\n"
        f"AVAILABLE EVIDENCE NODES:\n{ev_block}\n\n"
        "Provide your testimony as a list of claims, one per premise you can address, "
        "citing the evidence node ids that support each finding."
    )
