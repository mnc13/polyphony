"""
Arbiter faithfulness (SummaC-style): is each sentence of the central script entailed by
some span of the materials the arbiter was given -- the case text, the anchor, the
decision statements, the testimony findings and the debate turns?

Definition v2. Four defects in v1 are fixed here, all of which moved the number
(the fourth, missing case/anchor/decision context, is described at context_texts):

1. HYPOTHESES WERE NOT SENTENCES. v1 split the Markdown script on [.!?] only, so a
   numbered heading became its own "sentence" ("## 2.") and an unpunctuated heading was
   glued onto the next sentence. DeBERTa scores a bare "## 2." as entailed by almost any
   span (0.70-0.93 measured), which inflated case 002 by ~4.5 points. Headings, rules and
   list markers are now stripped; each remaining line is split into sentences and only
   sentences of at least four words are scored.

2. SOURCE SPANS DID NOT FIT THE MODELS. v1 scored against whole debate turns, median
   ~690 DeBERTa tokens (60 of 80 over the 512-token limit), so NLI saw only the first
   ~480 tokens of each turn and the embedding prefilter (MiniLM, 256 tokens) ranked a
   turn by its opening alone. A script sentence summarising the end of a turn was
   ungrounded by construction. Spans are now windows of up to three consecutive
   sentences (<= 120 words, stride two sentences), which both models see whole.

3. VACUOUS SPANS. A non-committal statement ("the provided evidence does not settle
   whether X") entails nearly any hypothesis under MNLI. v1's offline recompute dropped
   such testimony claims -- but kept vacuous debate text, and also dropped every UNCITED
   testimony finding, although those are materials the arbiter was given and may
   faithfully report. v2 drops every vacuous SENTENCE from the sources before
   windowing (so a hedge never takes its neighbours with it); what remains grounds
   substantive sentences, and the dropped sentences ground nothing. A script sentence that is
   itself about the state of the evidence ("the evidence for d1 was thin") is a meta
   sentence: it is counted, reported, and not scored, since the only spans that could
   ground it are the vacuous ones that entail anything.

faithfulness = entailed substantive sentences / substantive sentences.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from config.settings import get_settings
from src.services.vacuity import is_vacuous, split_sentences

# v2.1: the vacuity pattern gained the "does not establish" family of hedges, and
# sentences no longer split after abbreviations ("Mrs. Foot"). Both change which
# sentences are scored, meta or dropped, so v2 figures are not comparable.
# v2.2: bold section labels ("**Case made that it harms Survivability.**", two per
# decision) are headings, not sentences -- 10 of 93 scored pilot sentences, all
# "grounded" by the anchor definition; and the anchor enters the sources as its label
# only, since that is all the arbiter is shown.
# v2.3: a list lead-in ("The main lines of support were:") asserts nothing until its
# items are read, and was scored alone -- 3 of 13 ungrounded sentences on the first
# no_decision_decomp case. It is no longer scored; a list item that is a fragment of it
# ("and the conditions under which agents continue to function ...") is scored with
# the lead-in in front, so it is checked as the statement it completes.
# v2.4: v2.3 dropped the lead-in at a blank line, and arbiter scripts nearly always put
# one between a lead-in and its list (1-7 times per script on the 2026-10-23 runs), so
# those lists' fragments were still scored alone.
# v2.5: an item was taken for a fragment whenever it began lower-case, and an item that
# opens with a decision id ("d2: Issuing an immediate, binding prohibition ...") is a
# complete statement. Prefixed with its lead-in ("The debate materials examine six
# decisions: d2: ...") NLI rejected it at 0.003 against the decision statement it
# copies; alone it scores 0.995. 6 of 39 ungrounded sentences on main case 006, 3 of 29
# on 001.
# v2.6: bracketed citation labels ("... remaining workers [E8].", "[E5], [E6]") are not
# claims, and no source span contains them, so they counted against the sentence: B2's
# "workplace fatalities ... increased disputes [E8]" scored 0.28 against E8 itself and
# 0.61 without the label. They are removed before scoring. The pipeline's scripts name
# TEG ids in prose ("TEG-d1-2 is invoked ...") and contain no bracketed labels, so their
# sentences are unchanged.
FAITHFULNESS_DEF = "v2.6-sentences-windows-substantive-context-leadins-ids-cites"

_HEADING = re.compile(r"^\s{0,3}#{1,6}(\s|$)")
_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_LIST_MARK = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_EMPHASIS = re.compile(r"\*{1,3}|`")
# A line that is nothing but emphasis is a label ("**Case made that it harms ...**").
_LABEL_LINE = re.compile(r"^\s*(\*{1,3}|_{1,3})[^*_]+\1\s*:?\s*$")
# ... and so is a bold lead-in ending in a stop or colon ("**Evidence.** The studies ...").
_LEAD_LABEL = re.compile(r"^\s*\*{2,3}[^*]{1,80}?[.:]\s*\*{2,3}\s*")
_LETTER = re.compile(r"[A-Za-z]")
# An item opening with an id ("d2: ...", "p3 ...") is lower-case but not a fragment.
_ID_START = re.compile(r"^[a-z]\d+\b")
# A run of bracketed citation labels: " [E8]", " [E5], [E6], [E10]", " [E3, TEG-d1-2]".
_CITE_ID = r"(?:E\d+|TEG-d\d+-\d+)"
_CITES = re.compile(rf"(?:\s*,?\s*\[\s*{_CITE_ID}(?:\s*[,;]\s*{_CITE_ID})*\s*\])+")

WINDOW_SENTENCES = 3
WINDOW_STRIDE = 2
WINDOW_MAX_WORDS = 120
MIN_SENTENCE_WORDS = 4


def script_sentences(md: str) -> list[str]:
    """Scorable sentences of a Markdown script: headings, rules and list markers removed."""
    out: list[str] = []
    lead = ""          # the open list's lead-in sentence, if the last prose line had one
    for line in (md or "").splitlines():
        if not line.strip():
            continue     # a blank line between a lead-in and its list keeps the lead-in
        if _HEADING.match(line) or _RULE.match(line):
            lead = ""
            continue
        is_item = bool(_LIST_MARK.match(line))
        line = _LIST_MARK.sub("", line)
        if _LABEL_LINE.match(line):
            lead = ""
            continue
        text = _CITES.sub("", _EMPHASIS.sub("", _LEAD_LABEL.sub("", line))).strip()
        if is_item and lead and text[:1].islower() and not _ID_START.match(text):
            text = f"{lead} {text}"
        elif not is_item:
            lead = ""
        sents = [s.strip() for s in split_sentences(text)]
        if sents and sents[-1].endswith(":"):
            # A lead-in is scored only through the items it introduces.
            if not is_item:
                lead = sents[-1]
            sents = sents[:-1]
        for s in sents:
            if len(s.split()) >= MIN_SENTENCE_WORDS and _LETTER.search(s):
                out.append(s)
    return out


def _source_sentences(text: str) -> list[str]:
    sents: list[str] = []
    for s in split_sentences(" ".join((text or "").split())):
        words = s.split()
        # A run-on "sentence" longer than a window is split by words, so no window
        # outgrows the embedder.
        for i in range(0, len(words), WINDOW_MAX_WORDS):
            sents.append(" ".join(words[i:i + WINDOW_MAX_WORDS]))
    return sents


def windows(text: str) -> list[str]:
    """Overlapping windows of consecutive sentences, each short enough for NLI/MiniLM."""
    return _windows_of(_source_sentences(text))


def _windows_of(sents: list[str]) -> list[str]:
    if not sents:
        return []
    out: list[str] = []
    i = 0
    while i < len(sents):
        win: list[str] = []
        words = 0
        for s in sents[i:i + WINDOW_SENTENCES]:
            n = len(s.split())
            if win and words + n > WINDOW_MAX_WORDS:
                break
            win.append(s)
            words += n
        out.append(" ".join(win))
        if i + len(win) >= len(sents):
            break
        i += max(1, min(WINDOW_STRIDE, len(win)))
    return out


@dataclass
class FaithfulnessResult:
    fraction: float                    # nan when there is nothing substantive to score
    n_sentences: int                   # all scorable sentences in the script
    n_scored: int                      # substantive sentences (the denominator)
    n_meta: int                        # sentences about the evidence itself, not scored
    n_ungrounded: int
    ungrounded_detail: list[dict] = field(default_factory=list)
    n_windows: int = 0                 # substantive source windows scored against
    n_vacuous_sentences: int = 0       # source sentences dropped as non-committal
    definition: str = FAITHFULNESS_DEF


def source_pools(testimony_findings: list[str], debate_turns: list[str]) -> tuple[list[str], int]:
    """
    (substantive windows, number of vacuous source sentences dropped).

    Vacuous SENTENCES are removed before windowing. Classifying whole windows instead
    let one hedge discard its neighbours: most testimony findings open with "the
    provided evidence does not directly test ...", which took 398 of 484 substantive
    testimony sentences out of the grounding pool with it.
    """
    substantive: list[str] = []
    vacuous = 0
    for text in list(testimony_findings) + list(debate_turns):
        kept = []
        for s in _source_sentences(text):
            if is_vacuous(s):
                vacuous += 1
            else:
                kept.append(s)
        substantive.extend(w for w in _windows_of(kept) if len(w) > 15)
    return substantive, vacuous


def context_texts(case_text: str, anchor_label: str, anchor_definition: str,
                  decisions: list[tuple[str, str]]) -> list[str]:
    """
    The rest of what the arbiter is given besides testimony and debate: the case, the
    anchor and the decision statements. Sections 1-3 of every script ("what the case is
    about", "the chosen anchor", "the decisions examined") are compiled from exactly
    these, and v1 left them out of the sources -- so on case 001 "This case concerns a
    trolley whose brakes have failed ..." scored 0.00 and every decision statement was
    ungrounded by construction.

    The anchor enters as its LABEL only: the arbiter is shown the label, not the
    definition, so a script sentence the definition would ground came from somewhere
    else. (`anchor_definition` is accepted and ignored, for callers' signatures.)
    """
    out = [case_text or "", f"The anchor is {anchor_label}."]
    out += [f"Decision {did}: {stmt}" for did, stmt in decisions]
    return [t for t in out if t.strip()]


def score_script(script_md: str, testimony_findings: list[str], debate_turns: list[str],
                 top_k: int | None = None, context: list[str] | None = None) -> FaithfulnessResult:
    from src.services import nli
    from src.services.embedder import embed

    hp = get_settings().hp
    k = top_k if top_k is not None else hp.nli_faithfulness_top_k
    sents = script_sentences(script_md)
    meta = [s for s in sents if is_vacuous(s)]
    scored = [s for s in sents if not is_vacuous(s)]
    spans, n_vac = source_pools(list(context or []) + list(testimony_findings), debate_turns)
    base = dict(n_sentences=len(sents), n_scored=len(scored), n_meta=len(meta),
                n_windows=len(spans), n_vacuous_sentences=n_vac)
    if not scored or not spans:
        return FaithfulnessResult(fraction=float("nan"), n_ungrounded=0, **base)

    span_vecs = embed(spans)
    sims = embed(scored) @ span_vecs.T          # both L2-normalised
    threshold = hp.nli_entailment_threshold
    entailed = 0
    detail: list[dict] = []
    for i, sent in enumerate(scored):
        order = sims[i].argsort()[::-1][:k]
        scores = nli.entails_batch([(spans[j], sent) for j in order])
        if not scores:
            detail.append({"sentence": sent, "best_score": 0.0, "best_span": ""})
            continue
        b = max(range(len(scores)), key=scores.__getitem__)
        if scores[b] >= threshold:
            entailed += 1
        else:
            detail.append({"sentence": sent, "best_score": round(scores[b], 3),
                           "best_span": spans[order[b]][:300]})
    return FaithfulnessResult(fraction=entailed / len(scored), n_ungrounded=len(detail),
                              ungrounded_detail=detail, **base)
