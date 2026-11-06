"""
Prompt for premise decomposition (Layer 3, Step 5a). The anchor's framing is injected
so premises are directional with respect to the CHOSEN anchor, not a fixed one.
Premises are empirical and falsifiable — never moral judgments.
"""
from __future__ import annotations

from src.ingestion.anchor import AnchorConfig
from src.schemas.core import Decision, StakeholderRegister

_DISCIPLINES = (
    "neuroscience, psychology, sociology, evolutionary_biology, economics, "
    "health_science, ecology"
)


def premise_system(anchor: AnchorConfig) -> str:
    return f"""\
You break a proposed decision into its EMPIRICAL PREMISES — the testable factual
claims that determine whether enacting the decision helps or harms the chosen
anchor for specific stakeholders.

The anchor is {anchor.label}. {anchor.definition}
{anchor.positive_direction}

Each premise must:
- be a FALSIFIABLE empirical proposition (something scientific literature could
  support or refute), NOT a moral judgment, value claim, or recommendation;
- name at least one stakeholder by their id (e.g. S1);
- assert a direction w.r.t. the anchor: polarity +1 (enhances), -1 (degrades), or
  0 (genuinely ambiguous);
- be tagged with the single discipline whose empirical literature can best settle
  it, chosen from: {_DISCIPLINES}.

Good premise: "Serving a theft conviction reduces S3's long-term employment
prospects" (discipline sociology, stakeholder S3, polarity -1).
Bad premise: "Stealing is wrong" (moral judgment — not allowed).

Cover multiple disciplines and multiple stakeholders where the case warrants it.
Do not evaluate whether the decision is right; only lay out its empirical premises.
Return between 4 and 10 premises."""


def premise_user(decision: Decision, case_text: str, sr: StakeholderRegister) -> str:
    who = "; ".join(f"{s.id}={s.description}" for s in sr.stakeholders)
    return (
        f"Stakeholders: {who}\n\n"
        f"CASE:\n{case_text}\n\n"
        f"DECISION UNDER ANALYSIS ({decision.id}): {decision.statement}\n\n"
        "Generate the empirical premises that determine this decision's effect on the "
        "anchor for the stakeholders. Each premise: text, discipline, stakeholder_ids, "
        "polarity."
    )
