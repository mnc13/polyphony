"""
P9 — Arbiter central-script compilation + NLI faithfulness gate + audit bundle.

The arbiter reads the debate materials for every decision and writes one central
script (no verdict). A separate NLI check then verifies every script sentence is
entailed by some span of the source materials — the empirical alternative to trusting
the summarizer. The audit bundle collects everything a reviewer needs to verify any
claim without trusting the system.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.ingestion.anchor import AnchorConfig
from src.prompts.arbiter import ARBITER_SYSTEM, arbiter_user
from src.schemas.core import (
    DebateResult, ScientistTestimony, StakeholderRegister, TestimonySet,
    TypedPremiseSet,
)
from src.services.llm_client import get_client


@dataclass
class DecisionMaterial:
    decision_id: str
    statement: str
    tps: TypedPremiseSet
    testimony: TestimonySet
    result: DebateResult
    teg: dict = field(default_factory=dict)   # TEG.to_dict() for provenance/audit


@dataclass
class ArbiterOutput:
    script_md: str
    # None when faithfulness was not computed in-run (the default; it is recomputed
    # offline). It used to be 1.0 in that case, a placeholder indistinguishable from a
    # perfect score that reports picked up as if it were measured.
    faithfulness: float | None
    ungrounded: list[str] = field(default_factory=list)
    source_spans: list[str] = field(default_factory=list)
    # Decisions that never got their own section-4 block. Empty is the healthy case;
    # a non-empty list means the script silently dropped a debate we actually ran.
    missing_decisions: list[str] = field(default_factory=list)
    measured: bool = False
    faithfulness_detail: dict = field(default_factory=dict)


# A section-4 block heading, e.g. "## Decision d3: Inviting the friend to ...".
_DECISION_HEADING = re.compile(
    r"^\s{0,3}#{1,6}[^\n]*?\bDecision\s+(d\d+)\b", re.IGNORECASE | re.MULTILINE
)


def _missing_decisions(script: str, materials: list[DecisionMaterial]) -> list[str]:
    """Decision ids with no heading of their own in the compiled script."""
    covered = {m.group(1).lower() for m in _DECISION_HEADING.finditer(script)}
    return [m.decision_id for m in materials if m.decision_id.lower() not in covered]


# Said in place of testimony when a decision has none (the no_testimony ablation). The
# arbiter is told to report "the scientists' key findings" per decision; under an empty
# heading it could pass debate claims off as findings. The same sentence is a faithfulness
# source, so the script saying so is grounded rather than scored as unsupported.
NO_TESTIMONY = "No scientist testimony was given for this decision."


def _has_testimony(tset: TestimonySet | None) -> bool:
    return bool(tset and any(t.claims for t in tset.testimonies))


def _decision_block(m: DecisionMaterial) -> str:
    lines = [f"### Decision {m.decision_id}: {m.statement}", "", "Scientist testimony:"]
    for t in m.testimony.testimonies:
        for c in t.claims:
            tag = "evidence-backed" if c.nli_supported else ("cited" if c.cited_nodes else "asserted")
            lines.append(f"- [{t.discipline.value}] ({c.evidence_strength}, {tag}) {c.finding}")
    if not _has_testimony(m.testimony):
        lines.append(f"- {NO_TESTIMONY}")
    lines.append("")
    lines.append("Debate turns:")
    for turn in m.result.turns:
        lines.append(f"- [R{turn.round} {turn.speaker}] {turn.argument_text}")
    lines.append(f"(debate stopped: {m.result.stop_reason}, coverage {m.result.coverage})")
    return "\n".join(lines)


def _source_spans(materials: list[DecisionMaterial]) -> list[str]:
    """Every span the arbiter was given (testimony findings + debate turns)."""
    t, d = _source_materials(materials)
    return [s for s in t + d if s and len(s) > 15]


def _source_materials(materials: list[DecisionMaterial]) -> tuple[list[str], list[str]]:
    """(testimony findings, debate turn texts) -- the inputs to the faithfulness check."""
    findings: list[str] = []
    turns: list[str] = []
    for m in materials:
        if m.testimony is not None:
            for t in m.testimony.testimonies:
                findings.extend(c.finding for c in t.claims)
        if not _has_testimony(m.testimony) and NO_TESTIMONY not in findings:
            findings.append(NO_TESTIMONY)
        if m.result is not None:
            turns.extend(turn.argument_text for turn in m.result.turns)
    return findings, turns


# The arbiter writes ~3.1-3.5k tokens for a six-decision case. The old 3000 cap cut four
# of six scripts mid-sentence on the first main run; the client now also retries a
# truncated reply with a larger budget, so this is a starting point, not a ceiling.
ARBITER_MAX_TOKENS = 6000


def compile_central_script(
    case_text: str,
    anchor: AnchorConfig,
    materials: list[DecisionMaterial],
    check_faithfulness: bool = True,
) -> ArbiterOutput:
    blocks = [_decision_block(m) for m in materials]
    client = get_client("arbiter")
    base_user = arbiter_user(case_text, anchor.label, blocks,
                             no_testimony=not any(_has_testimony(m.testimony) for m in materials))
    script = client.generate(user=base_user, system=ARBITER_SYSTEM,
                             max_tokens=ARBITER_MAX_TOKENS, retry_truncated=True)

    # Post-hoc completeness gate: the prompt asks for every decision, but the model
    # can still drop one (mentioning it in the roster is not the same as analysing
    # it). Detect that and give exactly one corrective retry naming what was lost;
    # keep whichever attempt covers more, and surface any residue instead of letting
    # a whole decision's debate vanish silently.
    missing = _missing_decisions(script, materials)
    if missing:
        by_id = {m.decision_id: m.statement for m in materials}
        corrective = (
            "\n\nYOUR PREVIOUS DRAFT OMITTED these decisions from section 4: "
            + ", ".join(f"{d} ({by_id[d]})" for d in missing)
            + ".\nRewrite the FULL script. Section 4 must contain one "
            "'## Decision <id>: <statement>' block for EVERY decision in ascending id "
            "order, including the omitted ones. Change nothing else."
        )
        retry = client.generate(user=base_user + corrective, system=ARBITER_SYSTEM,
                                max_tokens=ARBITER_MAX_TOKENS, retry_truncated=True)
        retry_missing = _missing_decisions(retry, materials)
        if len(retry_missing) < len(missing):
            script, missing = retry, retry_missing

    spans = _source_spans(materials)
    faith, ungrounded, measured, detail = None, [], False, {}
    if check_faithfulness:
        from src.eval.faithfulness import context_texts, score_script
        findings, turns = _source_materials(materials)
        ctx = context_texts(case_text, anchor.label, getattr(anchor, "definition", ""),
                            [(m.decision_id, m.statement) for m in materials])
        res = score_script(script, findings, turns, context=ctx)
        measured = True
        faith = None if res.fraction != res.fraction else round(res.fraction, 3)  # nan -> None
        ungrounded = [d["sentence"] for d in res.ungrounded_detail]
        detail = {"definition": res.definition, "n_sentences": res.n_sentences,
                  "n_scored": res.n_scored, "n_meta": res.n_meta,
                  "n_ungrounded": res.n_ungrounded, "n_windows": res.n_windows,
                  "n_vacuous_sentences": res.n_vacuous_sentences,
                  "ungrounded_detail": res.ungrounded_detail}
    return ArbiterOutput(script_md=script, faithfulness=faith,
                         ungrounded=ungrounded, source_spans=spans,
                         missing_decisions=missing, measured=measured,
                         faithfulness_detail=detail)


def build_audit_bundle(sr: StakeholderRegister, materials: list[DecisionMaterial],
                       arbiter: ArbiterOutput, sit_rows: list[dict]) -> dict:
    return {
        "stakeholder_register": sr.model_dump(mode="json"),
        "decisions": [{
            "decision_id": m.decision_id, "statement": m.statement,
            "tps": m.tps.model_dump(mode="json"),
            "testimony": [t.model_dump(mode="json") for t in m.testimony.testimonies],
            "debate": {"rounds_run": m.result.rounds_run, "stop_reason": m.result.stop_reason,
                       "coverage": m.result.coverage,
                       "turns": [t.model_dump(mode="json") for t in m.result.turns]},
            "teg": m.teg,
        } for m in materials],
        "sit_rows": sit_rows,
        "faithfulness": {"fraction_entailed": arbiter.faithfulness,
                         "measured": arbiter.measured,
                         "ungrounded_sentences": arbiter.ungrounded,
                         "missing_decisions": arbiter.missing_decisions,
                         **({"definition": arbiter.faithfulness_detail.get("definition")}
                            if arbiter.faithfulness_detail else {})},
    }
