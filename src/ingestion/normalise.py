"""
P4 — Case normalisation (Step 2a). Deterministic, rule-based, no LLM.

Cleans raw case text and separates its parts so downstream LLM steps get tidy,
segmented input:
  - collapse whitespace, strip artifacts (mojibake already handled at load);
  - split a "Setting:" preamble from the actual dilemma/question body (the
    demonstration and trolley cases follow "Setting: ... Your Choices: ..." shape);
  - tag quoted dialogue spans so they are not mistaken for narration.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalisedCase:
    setting: str          # scene-setting preamble (may be empty)
    body: str             # the actual dilemma / question / action
    choices: str          # explicit "Your Choices:" block if present (may be empty)
    full_text: str        # cleaned, recombined
    quoted_spans: tuple[str, ...]


_WS = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")
_QUOTE = re.compile(r"[\"“”]([^\"“”]{3,}?)[\"“”]")


def _clean(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WS.sub(" ", s)
    s = _MULTINL.sub("\n\n", s)
    return s.strip()


def normalise(text: str) -> NormalisedCase:
    clean = _clean(text)

    setting = ""
    body = clean
    choices = ""

    # Split off an explicit choices block ("Your Choices:", "Your choices:", etc.)
    m_choice = re.search(r"\n\s*Your Choices:\s*\n", clean, flags=re.IGNORECASE)
    if not m_choice:
        m_choice = re.search(r"\n\s*Your choices:\s*\n", clean)
    if m_choice:
        choices = clean[m_choice.end():].strip()
        body = clean[:m_choice.start()].strip()

    # Split off a "Setting:" preamble from the body.
    m_set = re.match(r"\s*Setting:\s*\n?(.*?)\n\s*(?:The dilemma:|The Dilemma:|Your|Is it|Should|What|Suppose)",
                     body, flags=re.IGNORECASE | re.DOTALL)
    if body.lower().startswith("setting:"):
        # Take everything up to the first clear question/dilemma marker as setting.
        marker = re.search(r"(The dilemma:|The Dilemma:|Your Choices:|Is it |Should |What do you)",
                           body, flags=re.IGNORECASE)
        if marker:
            setting = body[len("setting:"):marker.start()].strip()
            body = body[marker.start():].strip()
        else:
            setting = body[len("setting:"):].strip()
            body = ""

    quoted = tuple(m.group(1).strip() for m in _QUOTE.finditer(clean))

    recombined = "\n\n".join(p for p in (
        (f"Setting: {setting}" if setting else ""),
        body,
        (f"Your Choices:\n{choices}" if choices else ""),
    ) if p)

    return NormalisedCase(
        setting=setting, body=body, choices=choices,
        full_text=recombined or clean, quoted_spans=quoted,
    )
