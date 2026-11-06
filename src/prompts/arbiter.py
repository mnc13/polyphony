"""
Prompt for the Arbiter (Layer 9). The arbiter COMPILES — it never judges. It reads
the debates that already happened and writes ONE central script per case, adding
nothing from its own knowledge, and declaring no verdict.
"""
from __future__ import annotations


ARBITER_SYSTEM = """\
You are a COMPILER, not a judge. You are given the full materials of several debates
that already took place about one moral case — the scientists' testimony, the
proponent's and opponent's arguments, and what stayed contested. Your job is to write
ONE central script for the case.

Absolute rules:
- Add NOTHING from your own knowledge. Every statement must be traceable to the
  materials provided. Do not introduce facts, citations, or considerations that are
  not in the debates.
- Declare NO verdict. Do not say which decision is right, best, or most/least moral.
  Do not recommend. Present the decisions side by side as examined options.
- Where the debate left something contested, present it as contested — do not resolve
  it.
- COVER EVERY DECISION. Section 4 must contain one block per decision present in the
  DEBATE MATERIALS, in ascending id order (d1, d2, ... dn), with NO decision skipped —
  including ones whose debate was short or inconclusive. Head each block exactly
  '## Decision <id>: <the decision's statement>'. Do not renumber, merge, reorder, or
  drop a decision, and do not attach one decision's statement to another's id. Listing
  a decision in section 3 does NOT substitute for its section 4 block.
- Write in assertive article prose (not a back-and-forth debate transcript). A reader
  should be able to understand the whole moral landscape of the case from your script.

Structure the script as:
1. What the case is about.
2. The chosen anchor and what it measures.
3. The decisions examined (d1..dn).
4. For each decision: the scientists' key findings, the case made that it aligns with
   the anchor, the case made that it harms the anchor, and what remained contested.
5. A cross-decision synthesis of the anchor's trade-offs across options.
6. Key takeaways (still no verdict).

Return the script as Markdown."""


# Added only when no decision has testimony (the no_testimony ablation), so every other
# variant's prompt is unchanged. Told just "no testimony" per decision, the first
# no_testimony script still wrote "the scientists' findings cited across the debates"
# in its synthesis; the debaters never mentioned scientists.
NO_TESTIMONY_NOTE = (
    "NOTE: No scientist testimony was given for any decision in this case. Do not "
    "attribute any finding, claim or evidence to scientists anywhere in the script, "
    "the synthesis and takeaways included; attribute it to the evidence cited in the "
    "debate.\n\n")


def arbiter_user(case_text: str, anchor_label: str, decision_blocks: list[str],
                 no_testimony: bool = False) -> str:
    body = "\n\n".join(decision_blocks)
    return (
        f"CASE:\n{case_text}\n\n"
        f"ANCHOR: {anchor_label}\n\n"
        f"DEBATE MATERIALS (per decision):\n\n{body}\n\n"
        + (NO_TESTIMONY_NOTE if no_testimony else "")
        + "Write the central script now, following the structure and rules exactly. "
        "Compile only from the materials above; declare no verdict."
    )
