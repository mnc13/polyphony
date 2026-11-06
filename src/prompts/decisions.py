"""
Prompts for Layer 2 (action decomposition + decision generation). Descriptive and
generative only — never evaluative. No ranking, no "best", no moral commentary.
"""
from __future__ import annotations

from src.schemas.core import StakeholderRegister

ACTION_SYSTEM = """\
You decompose the ACTION STRUCTURE of a moral case. This is descriptive, not \
evaluative — you never say what was right or wrong.

Produce three lists:
- performed: what the central agent actually did (concrete acts).
- omitted: relevant things the agent did NOT do that the situation raised.
- alternatives: other courses of action that were FEASIBLE within the case's stated \
  setting (do not invent options the setting rules out).

Keep each item a short concrete phrase grounded in the case. Refer to parties by \
their situation (e.g. "the husband", "the druggist")."""


def action_user(case_text: str, sr: StakeholderRegister) -> str:
    who = "; ".join(f"{s.id}={s.description}" for s in sr.stakeholders)
    return (
        f"Stakeholders: {who}\n\n"
        f"CASE:\n{case_text}\n\n"
        "Decompose the action structure into performed / omitted / alternatives."
    )


DECISION_SYSTEM = """\
You enumerate the ALTERNATIVE DECISIONS the central agent in a moral case could \
take. Given the action inventory, produce the space of distinct, feasible courses \
of action — including what the agent actually did and genuine alternatives to it.

Each decision is phrased as a debatable proposition about the right course of \
action for the agent, e.g. "Stealing the drug is the right course for Heinz." or \
"Earning the money through work is the right course for the beggar."

Strict rules:
- Each option must be FEASIBLE within the case's stated setting.
- Options must be genuinely DIFFERENT strategies, not rephrasings of each other.
- Produce between 3 and 6 options.
- Do NOT rank them. Do NOT say which is best or most/least moral. Do NOT add any \
  evaluation or commentary. You only lay out the space of choices as equals.

Return the list of decision statements only."""


def decision_user(case_text: str, performed, omitted, alternatives) -> str:
    inv = (
        f"performed: {performed}\n"
        f"omitted: {omitted}\n"
        f"alternatives: {alternatives}"
    )
    return (
        f"ACTION INVENTORY:\n{inv}\n\n"
        f"CASE:\n{case_text}\n\n"
        "Enumerate the distinct feasible decisions the agent could take, each as a "
        "debatable proposition. Do not rank or evaluate them."
    )


# --- The no_decision_decomp ablation: one proposition, the case's own -----------
CASE_PROPOSITION_SYSTEM = """\
You state the ONE course of action a moral case itself puts to its central agent: \
the contested choice the case is about. You do not enumerate alternatives, you do \
not invent options the case does not raise, and you never evaluate.

Phrase it as a single debatable proposition about the right course of action for \
the agent, e.g. for a case ending "Is it morally permissible for you to turn the \
trolley?": "Turning the trolley onto the spur is the right course for the driver."

If the case ends with an open question ("What do you do?") or no question, state the \
action the case presents as the contested or pressured option -- the one the \
dilemma turns on.

Return a list containing exactly one statement."""


def case_proposition_user(case_text: str) -> str:
    return (
        f"CASE:\n{case_text}\n\n"
        "State the one contested course of action this case puts to its central agent, "
        "as a single debatable proposition. Do not rank or evaluate it."
    )
