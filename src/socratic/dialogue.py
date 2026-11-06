"""
P10 — Socratic dialogue (Step 10b).

A session grounded on one case's central script + testimony. It answers opinion-seeking
questions with questions and factual questions with facts — never a verdict. Runs on
the `socratic` role (distinct model from the arbiter). Turns are logged so the dialogue
is saved with the case (real data on what humans interrogate).

"Never a verdict" used to rest on the prompt alone: whatever the model wrote was
returned. A reply is now checked by rule (verdict_sentences): a declarative sentence
that recommends, ranks or pronounces an option right or wrong flags it, and a flagged
reply is regenerated once with a reminder. A reply still flagged after that is returned
and recorded in `flags`, so the saved dialogue shows where the guard failed. Questions
are exempt: "What should matter most here?" is the method, not a verdict.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.prompts.socratic import SOCRATIC_SYSTEM, socratic_context, socratic_user
from src.services.llm_client import get_client

_VERDICT = re.compile(
    r"\b(you should|you ought to|(?:i|we) (?:would )?(?:recommend|suggest|advise)|"
    r"my (?:verdict|recommendation|advice)|i would (?:choose|pick|go with)|"
    r"the (?:right|best|correct|better|moral|ethical) (?:choice|option|course|decision|"
    r"answer|thing to do) (?:is|would be)|"
    r"(?:is|was|would be) (?:morally |ethically )?(?:right|wrong|justified|unjustified|"
    r"permissible|impermissible|the (?:better|best|right) (?:option|choice))\b|"
    r"rank(?:ed|ing)? (?:first|highest|above))", re.I)

_REMINDER = ("\n\nREMINDER: your previous reply stated a verdict or recommendation. Do "
             "not tell the user what is right, best or what they should do, and do not "
             "rank the options. Answer factual questions factually and turn evaluative "
             "ones back to the user as questions.")


def verdict_sentences(reply: str) -> list[str]:
    """Declarative sentences of `reply` that state a verdict, recommendation or ranking."""
    sents = re.split(r"(?<=[.!?])\s+", reply.strip())
    return [s for s in sents if s and not s.rstrip().endswith("?") and _VERDICT.search(s)]


@dataclass
class SocraticSession:
    central_script: str
    testimony_summary: str
    history: list[tuple[str, str]] = field(default_factory=list)  # (user, system)
    flags: list[bool] = field(default_factory=list)  # per turn: verdict survived the retry

    def _context(self) -> str:
        return socratic_context(self.central_script, self.testimony_summary)

    def _history_text(self, last: int = 6) -> str:
        return "\n".join(f"USER: {u}\nSYSTEM: {s}" for u, s in self.history[-last:])

    def respond(self, query: str) -> str:
        client = get_client("socratic")
        user = socratic_user(self._context(), self._history_text(), query)
        reply = client.generate(user=user, system=SOCRATIC_SYSTEM, max_tokens=800).strip()
        if verdict_sentences(reply):
            reply = client.generate(user=user + _REMINDER, system=SOCRATIC_SYSTEM,
                                    max_tokens=800).strip()
        self.flags.append(bool(verdict_sentences(reply)))
        self.history.append((query, reply))
        return reply


def testimony_summary_from_bundle(bundle: dict) -> str:
    lines = []
    for d in bundle.get("decisions", []):
        for t in d.get("testimony", []):
            disc = t.get("discipline", "?")
            for c in t.get("claims", []):
                lines.append(f"[{disc}] {c.get('finding','')[:180]}")
    return "\n".join(lines[:40]) or "(no testimony)"
