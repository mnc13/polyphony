"""
Run context + Layer 1 orchestration.

RunContext binds a run to its anchor and writes a manifest snapshotting the anchor,
the hyperparameter surface, and the resolved model table — everything needed to
reproduce and audit the run. `ingest_case` executes Layer 1: normalise -> extract
stakeholders -> assemble & persist the (immutable) Stakeholder Register.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from config.settings import get_settings
from src.ingestion.anchor import AnchorConfig, select_anchor
from src.ingestion.normalise import normalise
from src.ingestion.stakeholders import extract_stakeholders
from src.schemas.core import Anchor, DecisionSet, StakeholderRegister, TypedPremiseSet
from src.services.model_registry import get_registry
from src.storage.cases import MoralCase
from src.storage.paths import CasePaths, manifest_path


@dataclass
class RunContext:
    run_id: str
    anchor: AnchorConfig

    @classmethod
    def create(cls, anchor_choice: str | int | Anchor, run_id: str | None = None) -> "RunContext":
        if not run_id:
            # Worker id in the run id: two workers started in the same second would
            # otherwise share runs/<run_id>/ and its manifest.
            import os
            worker = os.getenv("COGNI_WORKER", "").strip()
            rid = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
            if worker:
                rid += f"_{worker}"
        else:
            rid = run_id
        anchor = select_anchor(anchor_choice)
        ctx = cls(run_id=rid, anchor=anchor)
        ctx._write_manifest()
        return ctx

    def _write_manifest(self) -> None:
        from src.services.llm_client import forced_model, resolved_spec
        reg = get_registry()
        # The specs the run actually uses: under B4's forced model every role is that
        # model, and listing the registry roster instead misrecorded the B4 run.
        specs = {r: resolved_spec(r) for r in reg.roles()}
        manifest = {
            "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "anchor": {
                "name": self.anchor.anchor.value,
                "label": self.anchor.label,
                "dimensions": [d.value for d in self.anchor.dimensions],
            },
            "models": {r: {"model": sp.model, "family": sp.family,
                           "temperature": sp.temperature} for r, sp in specs.items()},
            "forced_model": forced_model(),
            "mode": reg.mode,
            **get_settings().manifest(),
        }
        with open(manifest_path(self.run_id), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    def paths(self, case_id: str) -> CasePaths:
        return CasePaths(self.run_id, case_id)


def ingest_case(case: MoralCase, ctx: RunContext) -> StakeholderRegister:
    """Layer 1: clean the case and produce the immutable Stakeholder Register."""
    norm = normalise(case.text)
    sr = extract_stakeholders(case.case_id, norm.full_text)

    paths = ctx.paths(case.case_id)
    # Persist normalised text (for inspection) and the SR (immutable artifact).
    (paths.root / "normalised.json").write_text(
        json.dumps({
            "setting": norm.setting, "body": norm.body,
            "choices": norm.choices, "full_text": norm.full_text,
            "quoted_spans": list(norm.quoted_spans),
        }, indent=2),
        encoding="utf-8",
    )
    paths.sr.write_text(sr.model_dump_json(indent=2), encoding="utf-8")
    return sr


def load_sr(ctx: RunContext, case_id: str) -> StakeholderRegister:
    return StakeholderRegister.model_validate_json(ctx.paths(case_id).sr.read_text(encoding="utf-8"))


def build_decisions(case: MoralCase, sr: StakeholderRegister, ctx: RunContext):
    """Layer 2: action decomposition + decision generation. Returns (inventory, decisions)."""
    from src.decisions.generate import generate_decisions

    norm = normalise(case.text)
    inv, dset = generate_decisions(case.case_id, norm.full_text, sr)

    paths = ctx.paths(case.case_id)
    paths.actions.write_text(inv.model_dump_json(indent=2), encoding="utf-8")
    paths.decisions.write_text(dset.model_dump_json(indent=2), encoding="utf-8")
    return inv, dset


def load_decisions(ctx: RunContext, case_id: str) -> DecisionSet:
    return DecisionSet.model_validate_json(ctx.paths(case_id).decisions.read_text(encoding="utf-8"))


def build_tps_for_decision(
    case: MoralCase, decision, sr: StakeholderRegister, ctx: RunContext,
    compute_relations: bool = True,
) -> TypedPremiseSet:
    """Layer 3 for ONE decision: generate premises -> typed premise set. Persists TPS_<d>.json."""
    from src.premises.generate import generate_premises
    from src.premises.tps import build_tps

    norm = normalise(case.text)
    premises = generate_premises(decision, norm.full_text, sr, ctx.anchor)
    tps, rels = build_tps(case.case_id, decision.id, premises, compute_relations=compute_relations)

    # Save premises + relations (relations ignored on reload by the schema).
    out = {**tps.model_dump(mode="json"), "relations": [r.model_dump() for r in rels]}
    ctx.paths(case.case_id).tps(decision.id).write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return tps


def build_all_tps(case: MoralCase, dset: DecisionSet, sr: StakeholderRegister, ctx: RunContext,
                  compute_relations: bool = True) -> dict[str, TypedPremiseSet]:
    """Fan-out: one TPS per decision."""
    return {d.id: build_tps_for_decision(case, d, sr, ctx, compute_relations)
            for d in dset.decisions}


def load_tps(ctx: RunContext, case_id: str, decision_id: str) -> TypedPremiseSet:
    return TypedPremiseSet.model_validate_json(
        ctx.paths(case_id).tps(decision_id).read_text(encoding="utf-8"))


def build_debate(tps, sr, tset, teg, ctx: RunContext, pmer=None, sit=None, progressive=True,
                 decision: str = "", case_text: str = ""):
    """Layer 8: Proponent vs Opponent debate over the testimony. Persists the transcript.
    progressive=False disables round>=2 P-MER retrieval (the --no-pmer ablation).
    decision / case_text: the statement being debated and the case it belongs to; both
    go into every debate prompt."""
    from src.debate.engine import DebateEngine, DebateDeps
    from src.evidence.pmer import PMER
    from src.storage.sit import SITStore

    deps = DebateDeps(pmer=pmer or PMER(), sit=sit or SITStore(), run_id=ctx.run_id)
    result = DebateEngine(deps).run(tps, sr, tset, teg, ctx.anchor, progressive=progressive,
                                    decision=decision, case_text=case_text)
    ctx.paths(tps.case_id).transcript(tps.decision_id).write_text(
        "\n".join(t.model_dump_json() for t in result.turns), encoding="utf-8")
    return result


def build_central_script(case, sr, materials, sit_rows, ctx: RunContext):
    """Layer 9: arbiter compiles all decisions into one central script + audit bundle."""
    import json
    from src.arbiter.compile import build_audit_bundle, compile_central_script

    from config.settings import get_settings
    out = compile_central_script(
        case.text, ctx.anchor, materials,
        check_faithfulness=get_settings().hp.inrun_faithfulness)
    paths = ctx.paths(case.case_id)
    paths.central_script.write_text(out.script_md, encoding="utf-8")
    bundle = build_audit_bundle(sr, materials, out, sit_rows)
    paths.audit_bundle.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return out, bundle


def populate_teg(tps: TypedPremiseSet, sr: StakeholderRegister, ctx: RunContext,
                 pmer=None, teg=None):
    """
    Initial P-MER pass (Layer 6): retrieve evidence for every premise into the
    decision's TEG. Returns the TEG. Persists TEG_<d>.json.
    """
    from src.evidence.pmer import PMER
    from src.evidence.teg import TypedEvidenceGraph

    pmer = pmer or PMER()
    teg = teg or TypedEvidenceGraph(tps.case_id, tps.decision_id)
    sr_by_id = {s.id: s.description for s in sr.stakeholders}
    for p in tps.premises:
        desc = "; ".join(sr_by_id.get(sid, "") for sid in p.stakeholder_ids)
        pmer.retrieve(p, desc, teg)
    teg.export(ctx.paths(tps.case_id).teg(tps.decision_id))
    return teg


def build_testimony(tps: TypedPremiseSet, sr: StakeholderRegister, ctx: RunContext,
                    teg=None, fixed_panel=False):
    """
    Layer 7 for one decision: dynamic scientist selection -> per-scientist testimony
    over the TEG, NLI-gated. Persists testimony_<d>.json (+ the selection inside it).
    fixed_panel=True forces the CogniAlign 4-scientist panel (the --fixed-panel ablation).
    """
    from src.testimony.select import assign_premises, select_disciplines
    from src.testimony.testify import testify
    from src.schemas.core import ScientistTestimony, TestimonySet

    if teg is None:
        teg = populate_teg(tps, sr, ctx)

    selection = select_disciplines(tps, fixed_panel=fixed_panel)
    by_disc = assign_premises(tps, selection, fixed_panel=fixed_panel)

    testimonies: list[ScientistTestimony] = []
    for disc in selection.selected:
        prem = by_disc.get(disc, [])
        if not prem:
            continue
        testimonies.append(testify(disc, prem, sr, teg, ctx.anchor))

    tset = TestimonySet(case_id=tps.case_id, decision_id=tps.decision_id,
                        testimonies=testimonies)
    out = {**tset.model_dump(mode="json"), "selection": selection.model_dump(mode="json")}
    ctx.paths(tps.case_id).testimony(tps.decision_id).write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return tset, selection
