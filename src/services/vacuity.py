"""
Non-committal ("vacuous") text: statements ABOUT the evidence base rather than claims.

"The provided evidence does not settle whether X" asserts nothing about X, and an MNLI
model resolves such a premise toward entailment: on the first run a deliberately
absurd hypothesis ("Penguins in Antarctica have formed a labour union") scored 0.996
against several of them, and in the other direction DeBERTa rated a cited abstract as
entailing "The provided evidence does not settle whether ..." at 0.979. Two places
therefore need to recognise this register:

  * testimony support -- a hedge entailed by the paper it mentions is not support for
    the premise (19 of the 22 "NLI-supported" cited testimony claims on the post-fix
    run were hedges of exactly this kind);
  * faithfulness -- a vacuous source sentence cannot ground a substantive script
    sentence.

One pattern, shared, so the two cannot drift apart. It is applied per SENTENCE.

Most hedges name the evidence as their subject ("the provided evidence does not
directly test ..."), so the broader phrasings are anchored to an evidence word. Bare
"not directly", "does not address", "would be expected" or "no data" also occur in
substantive claims ("S4 was not directly affected", "the policy does not address S2's
housing", "no data breaches occurred") and must not be read as hedges.
"""
from __future__ import annotations

import re

# subjects that make a negation a statement about the evidence base
_EV = r"(?:evidence|literature|nodes?|stud(?:y|ies)|data|sources?|papers?|findings?|research|trials?)"
# Verbs of coverage, not of outcome: "the studies did not examine X" says nothing about
# X, while "the trial does not show a benefit" is a negative finding and stays a claim.
_VERBS = (r"(?:test|tested|measure|measured|compare|compared|examine|examined|assess|"
          r"assessed|study|studied|address|addressed|evaluate|evaluated|investigate|"
          r"investigated|speak|settle|settled)")
# "Does not establish" and its kin, added after the case-001 pilot, where "The provided
# evidence does not (directly|empirically) establish that ..." opened three of the four
# testimony claims stamped as supported. These verbs also describe OUTCOMES ("the trials
# did not demonstrate a mortality benefit", "participants did not report anxiety"), so
# they count as a hedge only with the evidence base itself as the subject, directly
# before the negation ("the provided evidence does not establish", "the cited nodes do
# not confirm"), or in the passive with the provided/retrieved/cited evidence as agent.
_EST = r"(?:establish|demonstrate|confirm|verify)"
_EST_PASSIVE = (r"(?:established|demonstrated|confirmed|verified|reported|quantified|"
                r"tested|examined|studied|measured|assessed|addressed|evaluated|"
                r"investigated|compared)")
_BASE = r"(?:evidence|literature|nodes?|material|record|sources)"
_GIVEN = r"(?:provided|retrieved|available|cited|supplied)"

VACUOUS = re.compile(
    # the original register
    r"does not settle|do not settle|is not settled|not (?:been )?settled by|"
    r"no retrieved|insufficient (?:evidence|literature|data)|"
    r"no (?:direct |relevant )?literature|no direct evidence|"
    r"provides? no (?:direct )?evidence|cannot be determined|evidence is silent|"
    r"neither .{0,40}address|not confirmed by|unconfirmed|cannot confirm|"
    r"is not supported by the (?:provided|retrieved|available) (?:evidence|literature)|"
    # evidence-subject negations: "the provided evidence does not (directly) test ..."
    # (the adverb slot takes any -ly word: "directly", "empirically", "specifically" ...)
    rf"\b{_EV}\b[^.;]{{0,40}}\b(?:does|do|did|has|have)(?: not|n't)(?: been)? (?:\w+ly )?{_VERBS}\b|"
    # "the provided evidence does not establish ...", "the cited nodes do not confirm ..."
    rf"\b{_BASE}\s+(?:does|do|did|has|have)(?: not|n't)(?: been)? (?:\w+ly )?{_EST}|"
    # ... and the passive: "has not been established by the retrieved studies"
    rf"\bnot (?:been )?(?:\w+ly )?{_EST_PASSIVE} (?:by|in) (?:the |this |these |any )?"
    rf"{_GIVEN} (?:{_BASE}|stud(?:y|ies)|papers?|trials?)\b|"
    rf"\bnot directly (?:{_VERBS}|relevant|bear)|"
    rf"\bonly indirect(?:ly)? (?:support|evidence|relevan|bear|relat)|"
    rf"\bindirect(?:ly)? (?:support|evidence|relevan|bear)|"
    r"\bpartial(?:ly)? support(?:s|ed)? (?:for )?(?:the|this) (?:premise|claim|finding)|"
    r"\babsence of (?:direct |any |relevant )?evidence|\blimited (?:direct )?evidence|"
    rf"\bno {_EV}\b[^.;]{{0,20}}\b(?:directly|that|on|about|address|examin|test|measur)|"
    r"\b(?:remains?|is) (?:unclear|uncertain|untested|unresolved)\b|"
    # statements about the premise itself, and the field's unconfirmed expectations
    # (the testimony prompt asks scientists to label those as unconfirmed)
    r"^\s*the premise\b|\bthe premise (?:asserts|claims|states|as stated)\b|"
    r"\bone would (?:generally )?expect\b|\bwe would (?:generally )?expect\b|"
    r"\bwould generally be expected\b|"
    r"\bthe field (?:would|might) (?:\w+ly )?expect\b|"
    r"\bthe field has (?:competing|mixed) expectations\b",
    re.I,
)

_SENT = re.compile(r"(?<=[.!?])\s+")
# A period after one of these does not end a sentence. "Mrs. Foot (S4)" used to split
# into "...for Mrs." and "Foot (S4) of the workman (S3) surviving versus being killed.",
# and that second fragment scored 0.98 entailment against a cited node.
# ("No." only capitalised, as in "No. 5": "He said no." ends a sentence.)
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Sr|Jr|vs|cf|al|Fig|(?-i:No)|approx|e\.g|"
                     r"i\.e)\.$", re.I)


def split_sentences(text: str | None) -> list[str]:
    """Split on terminal punctuation, but not after common abbreviations."""
    out: list[str] = []
    for part in _SENT.split(text or ""):
        if out and _ABBREV.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def is_vacuous(text: str | None) -> bool:
    return bool(VACUOUS.search(text or ""))


def sentences(text: str | None, min_chars: int = 16) -> list[str]:
    return [s.strip() for s in split_sentences(text) if len(s.strip()) >= min_chars]


def committed_sentences(text: str | None, min_chars: int = 16) -> list[str]:
    """The sentences of `text` that assert something (are not hedges)."""
    return [s for s in sentences(text, min_chars) if not is_vacuous(s)]
