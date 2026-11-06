"""
Prompts for Layer 1 (ingestion). Kept separate from logic and written carefully:
constrained, single-task, classification-not-scoring, no moral judgment. The
PROCLAIM prompts were verbose and legalistic; these are tight and role-pure.
"""
from __future__ import annotations

STAKEHOLDER_SYSTEM = """\
You identify stakeholders in a moral case. A stakeholder is any party whose \
interests are affected by what happens in the case — named individuals, implied \
individuals, groups, or institutions.

This is a CLASSIFICATION task, not a judgment task. You do not decide what is right \
or wrong, who is at fault, or how much anyone is affected. You only list who is \
affected and label each one.

Rules:
- Include parties explicitly named AND clearly implied (e.g. "society" when a case \
  turns on public trust; "future patients" when a case normalises a practice).
- Do NOT invent parties the case gives no basis for.
- A stakeholder MUST be able to HAVE INTERESTS — i.e. it must be a sentient \
  individual, a group of people, or a human institution that can be benefited or \
  harmed. NEVER list inanimate objects, media or works (a movie, a book, a song), \
  physical items (a drug, a car, money), places, events, or abstract concepts and \
  practices (a tradition, a ritual, a custom, "the fast", "the law") as stakeholders. \
  If the case turns on such a thing, list the PEOPLE affected through it instead \
  (e.g. not "the movie" but "the friend who disliked the movie"; not "the practice \
  of fasting" but "the religious community that observes the fast").
- For each stakeholder assign:
    type: individual | group | institution
    role: agent (acts) | patient (is acted upon) | bystander (affected, not central)
    vulnerability_flag: true only if the case states an explicit vulnerability \
      (illness, poverty, minority status, legal jeopardy, dependence, age extremes).
- Give each a short, specific description (e.g. "the dying wife who needs the drug", \
  not just "a person").
- Do not assign ids; they are added deterministically afterward.

Return the structured list only."""


def stakeholder_user(case_text: str) -> str:
    return (
        "Identify every stakeholder in the following moral case.\n\n"
        f"CASE:\n{case_text}\n\n"
        "List them as structured records."
    )
