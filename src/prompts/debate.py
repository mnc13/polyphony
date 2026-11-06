"""
Prompts for the Proponent vs Opponent debate (Layer 8). Debaters argue over the
SCIENTISTS' TESTIMONY and the evidence graph — not raw literature. Every factual
claim must cite a testimony/TEG node; uncited claims are logged as ASSERTED.
No courtroom theatrics — plain adversarial reasoning grounded in evidence.
"""
from __future__ import annotations

from src.ingestion.anchor import AnchorConfig
from src.schemas.core import StakeholderRegister, TypedPremiseSet


def _side_role(side: str, anchor_label: str) -> str:
    # "the decision" is the DECISION UNDER DEBATE named in the user message. Until
    # 2026-10-22 the prompt never stated it, so debaters guessed: on the case-001 pilot
    # both sides of d3 ("warn the workmen") and d4 argued about diverting the trolley,
    # and d5's proponent argued against its own decision.
    if side == "proponent":
        return (f"You are the PROPONENT. You argue that enacting the DECISION UNDER "
                f"DEBATE is sound with respect to {anchor_label} for the stakeholders.")
    return (f"You are the OPPONENT. You argue that enacting the DECISION UNDER DEBATE "
            f"is harmful with respect to {anchor_label} for the stakeholders.")


def debate_system(side: str, anchor: AnchorConfig) -> str:
    return f"""\
{_side_role(side, anchor.label)}

The anchor is {anchor.label}. {anchor.definition}

Rules of engagement:
- Argue from the scientists' testimony and the evidence nodes provided. You MAY use
  your own knowledge to reason and connect points, but every FACTUAL claim must cite
  a supporting evidence node id. Uncited factual claims will be recorded as merely
  'asserted' and are fair targets for rebuttal.
- Rebut the other side by attacking evidential weaknesses: uncited assertions,
  contradicted evidence, or premises they ignored.
- Argue about the DECISION UNDER DEBATE exactly as stated, not about another option
  in the case. Polarity is the effect of enacting THAT decision.
- For each stakeholder you affect, state a structured claim: stakeholder_id (the bare
  register id, e.g. S2), the premise_id it bears on (e.g. p3), dimension, polarity
  (+1/-1/0), the evidence nodes cited, and a one-sentence statement of exactly what
  the claim asserts. The statement is what gets checked against the nodes you cite,
  so cite only nodes that support it, and make one claim per premise it bears on.
- List which premise ids you addressed this turn, and state what evidence you still
  need (evidential_gap), naming the premise id each need belongs to, e.g. "(p3)" —
  this drives the next retrieval round.
- argument_text is your full argument for this turn, written out in prose. Never
  leave it empty or point elsewhere ("see below").
- Do NOT declare a verdict or say the decision is 'right'/'wrong' overall. You make
  the {anchor.label} case for your side; the system does not judge the winner.

Return the structured turn."""


def debate_user(tps: TypedPremiseSet, sr: StakeholderRegister, testimony_block: str,
                teg_block: str, history: str, gap_hint: str = "",
                decision: str = "", case_text: str = "") -> str:
    sign = {1: "+", 0: "0", -1: "-"}
    prem = "\n".join(
        f"- {p.id} [{p.dimension.value}/{p.discipline.value}] "
        f"({sign[p.polarity]} {','.join(p.stakeholder_ids)}): {p.text}"
        for p in tps.premises
    )
    who = "; ".join(f"{s.id}={s.description}" for s in sr.stakeholders)
    parts = []
    if case_text:
        parts.append(f"CASE:\n{case_text}\n")
    if decision:
        parts.append(f"DECISION UNDER DEBATE ({tps.decision_id}): {decision}\n")
    parts += [
        f"Stakeholders: {who}",
        f"\nPREMISES (the checklist to address):\n{prem}",
        f"\nSCIENTIST TESTIMONY:\n{testimony_block}",
        f"\nEVIDENCE NODES:\n{teg_block}",
    ]
    if history:
        parts.append(f"\nDEBATE SO FAR:\n{history}")
    if gap_hint:
        parts.append(f"\nNewly retrieved evidence this round addresses: {gap_hint}")
    parts.append("\nMake your argument for this turn, citing evidence node ids.")
    return "\n".join(parts)
