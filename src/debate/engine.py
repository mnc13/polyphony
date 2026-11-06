"""
P8 — Debate orchestration (Layer 8): Proponent vs Opponent over the testimony.

Per decision, a dynamic-round debate:
  Round 1  : proponent opening, opponent opening (seed TEG + testimony only).
  Round k≥2: progressive P-MER on still-uncovered premises, steered by the debaters'
             evidential gaps -> new evidence into the TEG; proponent turn; opponent turn.
  After each turn: a deterministic SIT updater writes one row per survivability claim
             (NLI-gated; written by code, never the LLM). Rebuttal (an opposite-
             polarity claim by the other side on the same stakeholder+dimension) is
             derived from row order when rows are read -- see src/storage/sit.py.
  Stop at the earliest of:
    - coverage ≥ 0.90 (premises addressed WITH topically relevant cited evidence /
      total; see src/debate/coverage.py),
    - P-MER novelty < 0.10 for two consecutive rounds (novelty of what was ADMITTED;
      a round that admits nothing scores 0),
    - coverage not improved for `coverage_stall_rounds` (2) consecutive rounds -- a
      cost guard added with the stricter coverage rule, switchable in settings,
    - max_rounds (8).

All quality signals here are rule-based / NLI — no LLM scoring (Design Law 1).
Transcript and SIT rows are the outputs consumed by the arbiter (P9).
"""
from __future__ import annotations

import re

from dataclasses import dataclass

from config.settings import get_settings
from src.debate.coverage import claim_premise_id, covered_premises
from src.evidence.pmer import PMER
from src.evidence.teg import TypedEvidenceGraph
from src.ingestion.anchor import AnchorConfig
from src.prompts.debate import debate_system, debate_user
from src.schemas.core import (
    DebateResult, DebateTurn, DebateTurnDraft, StakeholderRegister, TestimonySet,
    TurnClaim, TypedPremiseSet,
)
from src.services import nli
from src.services.embedder import embed
from src.services.llm_client import get_client
from src.services.vacuity import is_vacuous, split_sentences
from src.storage.sit import SITRow, SITStore, canonical_stakeholder_id

# A debate turn shorter than this is not an argument. glm-5.3 returned turns of one
# character and "See below." (its content went nowhere), and the arbiter then compiled
# those decisions with no opposing case at all.
MIN_ARGUMENT_CHARS = 300
MAX_TURN_ATTEMPTS = 3

# A turn that argues the other side. Seen twice in 144 turns of the 2026-10-23/24 runs,
# both from the proponent: on "roll out a crime-prediction model despite known racial
# bias" its opening read "I argue that enacting d1 ... is NOT sound", and on another
# decision it wrote the opponent's turn ("is HARMFUL ... the proponent's claims rest on
# ..."). Both sides then argued against the decision, so nothing was debated. Turns
# open with a stated position ("I argue that enacting d1 ... is sound/harmful with
# respect to <anchor>"), so the opening sentence is checked, deterministically.
_POSITION = re.compile(r"^\W*(?:as the \w+,\s*)?i (?:will |would )?(?:argue|contend|maintain) that\b",
                       re.IGNORECASE)
_SOUND = re.compile(r"\bis\s+sound\b", re.IGNORECASE)
_NOT_SOUND = re.compile(r"\bnot\s+sound\b|\bunsound\b|\bis\s+harmful\b", re.IGNORECASE)
# An opening about the other side ("The opponent's critique ... undermines its own ...")
# is a rebuttal, whatever NLI reads into it: 3 of the 4 near-misses on the stored turns.
_ABOUT_OTHER_SIDE = re.compile(r"^\W*(?:(?:the|my)\s+)?(?:opponent|proponent)\b|^\W*as the (?:opponent|proponent)\b",
                               re.IGNORECASE)
# The rerun of that case opened every proponent turn without a stated position ("The
# decision to proceed ... undermines survivability across multiple stakeholder
# dimensions"), so wording alone missed it. Such openings are scored with NLI against
# "the decision supports / harms <anchor>". On 190 stored turns, entailment >= 0.95 for
# the wrong side with <= 0.05 for the right one flagged exactly the 6 reversals found by
# reading; the nearest legitimate turns scored 0.75-0.91 and opened about the opponent.
STANCE_ENTAILMENT = 0.95
STANCE_OTHER_MAX = 0.05


def _stance(opening: str, anchor_label: str) -> tuple[float, float]:
    """(entailment of 'supports', entailment of 'harms') for the opening sentence."""
    p = nli.entails_batch([(opening, f"Enacting the decision supports {anchor_label}."),
                           (opening, f"Enacting the decision harms {anchor_label}.")])
    return p[0], p[1]


def argues_other_side(side: str, argument_text: str, anchor_label: str = "") -> bool:
    first = (split_sentences(" ".join((argument_text or "").split())) or [""])[0]
    if _POSITION.search(first):
        pro, con = bool(_SOUND.search(first)), bool(_NOT_SOUND.search(first))
        if pro or con:
            return (con and not pro) if side == "proponent" else (pro and not con)
    if not anchor_label or _ABOUT_OTHER_SIDE.search(first):
        return False
    p_for, p_against = _stance(first[:600], anchor_label)
    wrong, right = (p_against, p_for) if side == "proponent" else (p_for, p_against)
    return wrong >= STANCE_ENTAILMENT and right <= STANCE_OTHER_MAX

# Context limits for what a debater sees. The old limits cut every testimony finding to
# 220 characters (median 736) -- which in practice left only the hedge that opens most
# findings -- and every earlier turn to 500 characters (median 3,419).
FINDING_CHARS = 1200
RECENT_TURN_CHARS = 6000
OLDER_TURN_CHARS = 800
TEG_BLOCK_LIMIT = 60

# P-MER calls per progressive round.
RETRIEVALS_PER_ROUND = 4


@dataclass
class DebateDeps:
    pmer: PMER
    sit: SITStore
    run_id: str


def _testimony_block(tset: TestimonySet) -> str:
    sign = {1: "+1", 0: "0", -1: "-1"}
    lines = []
    for t in tset.testimonies:
        for c in t.claims:
            tag = "cited" if c.cited_nodes else "asserted"
            cites = f" cites {','.join(c.cited_nodes)}" if c.cited_nodes else ""
            lines.append(f"[{t.discipline.value}] {c.premise_id} ({c.evidence_strength},{tag},"
                         f" direction {sign.get(c.direction, c.direction)}{cites}): "
                         f"{c.finding[:FINDING_CHARS]}")
    return "\n".join(lines) or "(no testimony)"


def _teg_block(teg: TypedEvidenceGraph, limit: int = TEG_BLOCK_LIMIT) -> str:
    lines = []
    for _, d in list(teg.g.nodes(data=True))[:limit]:
        lines.append(f"{d['node_id']} [{d.get('source','?')}"
                     + (f", DOI {d['doi']}" if d.get("doi") else "") + f"]: {d['text'][:200]}")
    return "\n".join(lines) or "(no evidence yet)"


def _history(turns: list[DebateTurn], last: int = 4) -> str:
    """The last two turns in full (the exchange being answered), earlier ones abridged."""
    recent = turns[-last:]
    out = []
    for i, t in enumerate(recent):
        limit = RECENT_TURN_CHARS if i >= len(recent) - 2 else OLDER_TURN_CHARS
        out.append(f"[R{t.round} {t.speaker}] {t.argument_text[:limit]}")
    return "\n\n".join(out)


def _closest_sentence(evidence: str, argument: str, fallback_chars: int = 400) -> str:
    """
    The sentence of `argument` most similar to `evidence` by embedding cosine.

    Cosine is used to pick the sentence because it is cheap (milliseconds for a whole
    turn) and only has to shortlist; the entailment judgment itself stays with NLI.
    Falls back to the leading characters if the argument has no sentence worth scoring.
    Used only for claims that arrive without a statement of their own.
    """
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", argument or "") if len(x.strip()) > 15]
    if not sents:
        return (argument or "")[:fallback_chars]
    if len(sents) == 1:
        return sents[0]
    vecs = embed(sents)
    ev_vec = embed([evidence or ""])[0]
    sims = vecs @ ev_vec
    return sents[int(sims.argmax())]


_PID = re.compile(r"\bp\d+\b")
_CLAUSE_SPLIT = re.compile(r";|,(?=\s*(?:and\s+)?that\b)|\.\s+")


def gap_hints(turns: list[DebateTurn], premise_ids: set[str]) -> dict[str, str]:
    """
    Per premise id, the parts of the debaters' evidential gaps that name it.

    The debate prompt has always said the gap "drives the next retrieval round", but
    nothing read it: round k re-ran round 1's query for the same premise and got the
    same local hits back. Debaters are now asked to tag each need with its premise id;
    clauses carrying an id become that premise's retrieval hint.
    """
    hints: dict[str, list[str]] = {}
    for t in turns:
        gap = (t.evidential_gap or "").strip()
        if not gap:
            continue
        for clause in _CLAUSE_SPLIT.split(gap):
            ids = set(_PID.findall(clause)) & premise_ids
            text = _PID.sub(" ", clause).strip(" ()[]")
            for pid in ids:
                if text:
                    hints.setdefault(pid, []).append(text)
    return {pid: " ".join(v) for pid, v in hints.items()}


def pick_retrieval_targets(uncovered: list, hints: dict[str, str], rnd: int,
                           k: int = RETRIEVALS_PER_ROUND) -> list:
    """
    Premises to retrieve for this round: ones the debaters asked about first, then the
    rest in an order that rotates by round. Always taking uncovered[:4] meant premises
    later in the list never got a progressive retrieval at all.
    """
    asked = [p for p in uncovered if p.id in hints]
    rest = [p for p in uncovered if p.id not in hints]
    if rest:
        off = ((rnd - 2) * k) % len(rest)
        rest = rest[off:] + rest[:off]
    return (asked + rest)[:k]


class DebateEngine:
    def __init__(self, deps: DebateDeps):
        self.deps = deps
        self.decision, self.case_text = "", ""
        self.role_reversals: list[dict] = []
        self.hp = get_settings().hp
        self.prop = get_client("proponent")
        self.opp = get_client("opponent")

    def _run_turn(self, side, client, tps, sr, tset, teg, turns, anchor, rnd, gap_hint):
        system = debate_system(side, anchor)
        user = debate_user(tps, sr, _testimony_block(tset), _teg_block(teg),
                           _history(turns), gap_hint, decision=self.decision,
                           case_text=self.case_text)
        draft: DebateTurnDraft | None = None
        retry_note = ""
        for attempt in range(MAX_TURN_ATTEMPTS):
            prompt = user + retry_note
            draft = client.generate_structured(
                user=prompt, schema=DebateTurnDraft, system=system,
                # glm-5.3 writes long turns and truncated mid-string at 1800. The
                # client escalates on truncation, but each escalation costs a wasted
                # call, and with ~200 debate turns over six cases it is cheaper to
                # start high — output bills on usage, not on the ceiling.
                max_tokens=12000)
            short = len((draft.argument_text or "").strip()) < MIN_ARGUMENT_CHARS
            other_side = not short and argues_other_side(side, draft.argument_text, anchor.label)
            if not short and not other_side:
                break
            if short:
                retry_note = (f"\n\nYOUR PREVIOUS ATTEMPT RETURNED AN EMPTY OR NEAR-EMPTY "
                              f"argument_text. Write the full argument for this turn "
                              f"(at least {MIN_ARGUMENT_CHARS} characters of prose) in "
                              f"argument_text itself; do not refer to any other field.")
            else:
                stance = "sound" if side == "proponent" else "harmful"
                retry_note = (f"\n\nYOUR PREVIOUS ATTEMPT ARGUED THE OTHER SIDE. You are the "
                              f"{side.upper()}: argue that enacting the DECISION UNDER DEBATE "
                              f"is {stance} with respect to {anchor.label}. You may concede "
                              f"what the evidence does not support, but this turn must make "
                              f"that case; the other side is argued by your opponent.")
            try:
                from src.services import telemetry
                telemetry.record_event("debate_role_reversal" if other_side else "short_debate_turn",
                                       side=side, round=rnd, decision_id=tps.decision_id,
                                       attempt=attempt + 1,
                                       chars=len((draft.argument_text or "").strip()))
            except Exception:
                pass
        if len((draft.argument_text or "").strip()) < MIN_ARGUMENT_CHARS:
            print(f"[debate] WARNING {tps.decision_id} R{rnd} {side}: argument_text still "
                  f"{len(draft.argument_text.strip())} chars after {MAX_TURN_ATTEMPTS} attempts")
        elif other_side:
            # Kept (there is no better turn to use) but recorded in the debate summary, so
            # a debate in which one side never argued its case is visible as such.
            self.role_reversals.append({"round": rnd, "side": side})
            print(f"[debate] WARNING {tps.decision_id} R{rnd} {side}: still argues the other "
                  f"side after {MAX_TURN_ATTEMPTS} attempts")

        valid_nodes = {n for n, _ in teg.g.nodes(data=True)}
        valid_pids = {p.id for p in tps.premises}
        register = {s.id for s in sr.stakeholders}
        claims = []
        for c in draft.claims:
            # "S2 (Patients)" -> "S2" when the bare id is in the register; anything
            # else is kept as written so it stays visible in the audit trail.
            sid = canonical_stakeholder_id(c.stakeholder_id)
            pid = claim_premise_id(c)
            claims.append(TurnClaim(
                stakeholder_id=sid if sid in register else c.stakeholder_id,
                dimension=c.dimension, polarity=c.polarity,
                evidence_nodes=[n for n in c.evidence_nodes if n in valid_nodes],
                statement=(c.statement or "").strip(),
                premise_id=pid if pid in valid_pids else ""))
        turn = DebateTurn(
            round=rnd, speaker=side,
            premises_addressed=[p for p in draft.premises_addressed if p in valid_pids],
            evidence_cited=[n for n in draft.evidence_cited if n in valid_nodes],
            claims=claims, evidential_gap=draft.evidential_gap.strip(),
            argument_text=draft.argument_text.strip(),
        )
        return turn

    def _update_sit(self, turn: DebateTurn, teg, tps, case_id, decision_id):
        node_text = {n: d["text"] for n, d in teg.g.nodes(data=True)}
        for c in turn.claims:
            node = c.evidence_nodes[0] if c.evidence_nodes else None
            nli_ok = False
            if c.evidence_nodes:
                # Does any cited node entail what THIS claim asserts?
                #
                # History: this first tested turn.argument_text[:400], the speech's
                # opening framing, identical for every claim in the turn, and
                # citation_integrity came out at 0.024. Its replacement scored the turn
                # sentence nearest the evidence, which still ignored the claim: every
                # claim citing a node got that node's verdict. Claims now carry their
                # own one-sentence statement; the nearest sentence is only a fallback
                # for claims without one. The supporting node, not just the first
                # cited, is the one recorded.
                #
                # A hedge ("the cited nodes do not establish that ...") is entailed by
                # the paper it describes and asserts nothing, so it is never support --
                # the same rule testimony applies (3 of 25 supported rows on the pilot).
                hyp_own = c.statement if len(c.statement or "") > 15 else None
                for nid in c.evidence_nodes:
                    ev = node_text.get(nid, "")
                    hyp = hyp_own or _closest_sentence(ev, turn.argument_text)
                    if is_vacuous(hyp):
                        continue
                    ok, _ = nli.citation_integrity(ev, hyp)
                    if ok:
                        nli_ok = True
                        node = nid
                        break
            self.deps.sit.append(SITRow(
                run_id=self.deps.run_id, case_id=case_id, decision_id=decision_id,
                round=turn.round, speaker=turn.speaker, stakeholder_id=c.stakeholder_id,
                dimension=c.dimension.value, polarity=c.polarity,
                evidence_node=node, rebutted=False, nli_supported=nli_ok,
            ))

    def _covered(self, turns: list[DebateTurn], tps: TypedPremiseSet,
                 teg: TypedEvidenceGraph) -> set[str]:
        """Premise ids addressed WITH at least one topically relevant cited node."""
        node_text = {n: d["text"] for n, d in teg.g.nodes(data=True)}
        return covered_premises(tps.premises, turns, node_text)

    def run(self, tps: TypedPremiseSet, sr: StakeholderRegister, tset: TestimonySet,
            teg: TypedEvidenceGraph, anchor: AnchorConfig, progressive: bool = True,
            decision: str = "", case_text: str = "") -> DebateResult:
        # What is being debated, and the case it comes from, go into every turn's prompt.
        self.decision, self.case_text = decision, case_text
        self.role_reversals = []
        if not tps.premises:
            # Every premise was dropped upstream: there is no checklist to debate, and
            # running would bill turns against a coverage of 0/1.
            return DebateResult(case_id=tps.case_id, decision_id=tps.decision_id,
                                rounds_run=0, stop_reason="no_premises", coverage=0.0,
                                turns=[])
        turns: list[DebateTurn] = []
        total = max(1, len(tps.premises))
        low_novelty_streak = 0
        best_coverage, stalled = -1.0, 0
        stop_reason = "max_rounds"
        sr_by_id = {s.id: s.description for s in sr.stakeholders}
        pids = {p.id for p in tps.premises}
        rnd = 0

        for rnd in range(1, self.hp.max_rounds + 1):
            gap_hint = ""
            # Progressive retrieval from round 2 onward, on still-uncovered premises.
            # Disabled by the --no-pmer ablation (progressive=False).
            if rnd >= 2 and progressive:
                covered = self._covered(turns, tps, teg)
                uncovered = [p for p in tps.premises if p.id not in covered]
                last_round = [t for t in turns if t.round == rnd - 1]
                hints = gap_hints(last_round, pids)
                # Novelty of what the round ADMITTED, pooled over its chunks. Averaging
                # per retrieval scored a target that admitted nothing as 0, so one call
                # admitting chunks at novelty < 0.40 among four read as an exhausted
                # round and could stop a debate that was still finding evidence.
                admitted_novelty: list[float] = []
                for p in pick_retrieval_targets(uncovered, hints, rnd):
                    desc = "; ".join(sr_by_id.get(sid, "") for sid in p.stakeholder_ids)
                    res = self.deps.pmer.retrieve(p, desc, teg, hint=hints.get(p.id, ""),
                                                  force_live=True)
                    admitted_novelty += [float(ch.novelty_score) for ch in res.admitted]
                    if res.admitted:
                        gap_hint += f"{p.id} "
                round_novelty = (sum(admitted_novelty) / len(admitted_novelty)
                                 if admitted_novelty else 0.0)
                low_novelty_streak = low_novelty_streak + 1 if round_novelty < self.hp.novelty_stop_threshold else 0
            elif rnd >= 2:
                # --no-pmer: no retrieval means nothing new, i.e. novelty 0. Without this
                # the ablation had no novelty stop at all and ran to max_rounds whenever
                # coverage stalled, so it would differ from main in debate length as
                # well as in retrieval.
                low_novelty_streak += 1

            # Proponent then Opponent.
            pt = self._run_turn("proponent", self.prop, tps, sr, tset, teg, turns, anchor, rnd, gap_hint)
            turns.append(pt)
            self._update_sit(pt, teg, tps, tps.case_id, tps.decision_id)
            ot = self._run_turn("opponent", self.opp, tps, sr, tset, teg, turns, anchor, rnd, gap_hint)
            turns.append(ot)
            self._update_sit(ot, teg, tps, tps.case_id, tps.decision_id)

            coverage = len(self._covered(turns, tps, teg)) / total
            if coverage >= self.hp.coverage_stop_threshold:
                stop_reason = "coverage"
                break
            if coverage > best_coverage + 1e-9:
                best_coverage, stalled = coverage, 0
            else:
                stalled += 1
            if rnd >= 2 and low_novelty_streak >= self.hp.novelty_stop_consecutive_rounds:
                stop_reason = "novelty_exhausted"
                break
            if self.hp.coverage_stall_rounds and stalled >= self.hp.coverage_stall_rounds:
                stop_reason = "coverage_stalled"
                break

        coverage = len(self._covered(turns, tps, teg)) / total
        return DebateResult(
            case_id=tps.case_id, decision_id=tps.decision_id, rounds_run=rnd,
            stop_reason=stop_reason, coverage=round(coverage, 3), turns=turns,
            role_reversals=list(self.role_reversals),
        )
