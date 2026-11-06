"""
Full per-sample orchestrator + resumable batch runner.

`process_sample` runs the whole pipeline for ONE case and — following the storage
requirement — saves EVERY step's output at runtime into the step-wise `outputs/`
tree AND folds it into the sample's full bundle. On success it finalizes the bundle
and records the sample id in the resume ledger.

`run_batch` iterates the dataset, SKIPS samples already in the ledger, and resumes
from the first unprocessed one — so a crash at sample N is resumed at N, never
re-running finished samples.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass

from src.ingestion.normalise import normalise
from src.pipeline.run import (
    RunContext, build_debate, build_decisions, build_testimony,
    build_tps_for_decision, ingest_case, populate_teg,
)
from src.storage.cases import MoralCase
from src.storage.outputs import OutputStore

# Ablation presets: name -> (variant tag, flags). "none" is the full pipeline.
ABLATIONS = {
    "none":                {"variant": "main",                    "no_decision_decomp": False, "no_pmer": False, "no_testimony": False, "fixed_panel": False},
    "no_decision_decomp":  {"variant": "abl_no_decision_decomp",  "no_decision_decomp": True,  "no_pmer": False, "no_testimony": False, "fixed_panel": False},
    "no_pmer":             {"variant": "abl_no_pmer",             "no_decision_decomp": False, "no_pmer": True,  "no_testimony": False, "fixed_panel": False},
    "no_testimony":        {"variant": "abl_no_testimony",        "no_decision_decomp": False, "no_pmer": False, "no_testimony": True,  "fixed_panel": False},
    "fixed_panel":         {"variant": "abl_fixed_panel",         "no_decision_decomp": False, "no_pmer": False, "no_testimony": False, "fixed_panel": True},
}


@dataclass
class Ablation:
    variant: str = "main"
    no_decision_decomp: bool = False
    no_pmer: bool = False
    no_testimony: bool = False
    fixed_panel: bool = False

    @classmethod
    def from_name(cls, name: str) -> "Ablation":
        if name not in ABLATIONS:
            raise ValueError(f"unknown ablation '{name}'. choices: {list(ABLATIONS)}")
        return cls(**ABLATIONS[name])


def process_sample(case: MoralCase, ctx: RunContext, store: OutputStore,
                   compute_relations: bool = False, ablation: Ablation | None = None) -> dict:
    """Run L1..L9 for one sample, saving each step. Returns a compact summary."""
    abl = ablation or Ablation()
    sid = case.sample_id or case.case_id
    # Start from a clean slate: per-decision artifacts from an earlier run of this
    # sample would otherwise survive alongside this run's, since decision counts vary
    # between runs and the filenames are decision-keyed.
    store.reset_sample(sid)

    # L1 — ingestion + anchor + stakeholders
    norm = normalise(case.text)
    store.save("ingestion", sid, "normalised", {
        "setting": norm.setting, "body": norm.body, "choices": norm.choices,
        "full_text": norm.full_text, "quoted_spans": list(norm.quoted_spans),
        "stratum": case.stratum, "case_id": case.case_id}, echo=False)
    store.save("ingestion", sid, "anchor", {
        "anchor": ctx.anchor.anchor.value, "label": ctx.anchor.label,
        "dimensions": [d.value for d in ctx.anchor.dimensions]}, echo=False)
    sr = ingest_case(case, ctx)
    store.save("stakeholders", sid, "stakeholder_register", sr, echo=False)

    # L2 — actions + decisions  (ablation: --no-decision-decomp => one whole-case debate)
    if abl.no_decision_decomp:
        # One whole-case debate over the case's own contested action: no action
        # decomposition, no alternatives (see generate_case_proposition).
        from src.decisions.generate import generate_case_proposition
        from src.schemas.core import ActionInventory
        inv = ActionInventory()
        dset = generate_case_proposition(case.case_id, norm.full_text)
    else:
        inv, dset = build_decisions(case, sr, ctx)
    store.save("actions", sid, "action_inventory", inv, echo=False)
    store.save("decisions", sid, "decisions", dset, echo=False)

    # L3..L8 — per decision (the fan-out)
    from src.arbiter.compile import DecisionMaterial
    from src.evidence.pmer import PMER
    from src.pipeline.run import build_central_script
    from src.storage.sit import SITStore
    pmer, sit = PMER(), SITStore()
    per_decision = []
    materials: list[DecisionMaterial] = []
    for d in dset.decisions:
        tps = build_tps_for_decision(case, d, sr, ctx, compute_relations=compute_relations)
        store.save("premises", sid, f"TPS_{d.id}", tps, echo=False)

        teg = populate_teg(tps, sr, ctx, pmer=pmer)
        store.save("evidence_teg", sid, f"TEG_{d.id}", teg.to_dict(), echo=False)

        # L7 — testimony (ablation: --no-testimony => empty; --fixed-panel => 4 core)
        if abl.no_testimony:
            from src.schemas.core import ScientistSelection, TestimonySet
            tset = TestimonySet(case_id=tps.case_id, decision_id=tps.decision_id, testimonies=[])
            selection = ScientistSelection(case_id=tps.case_id, decision_id=tps.decision_id,
                                           selected=[], premise_counts={})
        else:
            tset, selection = build_testimony(tps, sr, ctx, teg=teg, fixed_panel=abl.fixed_panel)
        store.save("testimony", sid, f"testimony_{d.id}",
                   {"testimonies": [t.model_dump(mode="json") for t in tset.testimonies],
                    "selection": selection.model_dump(mode="json")}, echo=False)

        result = build_debate(tps, sr, tset, teg, ctx, pmer=pmer, sit=sit,
                              progressive=not abl.no_pmer, decision=d.statement,
                              case_text=norm.full_text)
        # Re-save the TEG: progressive rounds add evidence during the debate, and the
        # pre-debate save above would otherwise be the only copy in 06_evidence_teg.
        store.save("evidence_teg", sid, f"TEG_{d.id}", teg.to_dict(), echo=False)
        store.save("debate", sid, f"transcript_{d.id}",
                   [t.model_dump(mode="json") for t in result.turns], echo=False)
        store.save("debate", sid, f"debate_summary_{d.id}", {
            "rounds_run": result.rounds_run, "stop_reason": result.stop_reason,
            "coverage": result.coverage, "role_reversals": result.role_reversals}, echo=False)

        materials.append(DecisionMaterial(decision_id=d.id, statement=d.statement,
                                          tps=tps, testimony=tset, result=result,
                                          teg=teg.to_dict()))
        per_decision.append({"decision_id": d.id, "premises": len(tps.premises),
                             "scientists": [x.value for x in selection.selected],
                             "teg_nodes": len(teg), "rounds": result.rounds_run,
                             "coverage": result.coverage, "stop": result.stop_reason})

    # L9 — arbiter central script + faithfulness + audit bundle
    # This run's rows only: a resumed case would otherwise carry a crashed attempt's
    # rows too. Rebuttal is derived from row order (the store is append-only).
    from src.storage.sit import mark_rebuttals
    sit_rows = mark_rebuttals([dict(r) for r in sit.rows_for(case.case_id, run_id=ctx.run_id)])
    out, bundle = build_central_script(case, sr, materials, sit_rows, ctx)
    store.save("arbiter", sid, "central_script", out.script_md, fmt="md", echo=False)
    store.save("arbiter", sid, "faithfulness", {
        "fraction_entailed": out.faithfulness,
        "measured": out.measured,
        "ungrounded_sentences": out.ungrounded,
        "missing_decisions": out.missing_decisions,
        **out.faithfulness_detail}, echo=False)
    store.save("arbiter", sid, "audit_bundle", bundle, echo=False)

    summary = {"sample_id": sid, "case_id": case.case_id, "stratum": case.stratum,
               "anchor": ctx.anchor.label, "variant": abl.variant,
               "n_decisions": len(dset.decisions),
               "script_faithfulness": out.faithfulness,
               "missing_decisions": out.missing_decisions, "decisions": per_decision}
    return summary


def _interactive_socratic(case, ctx: RunContext, store: OutputStore | None = None) -> None:
    """Open a Socratic dialogue on a just-finished case (gate said Y)."""
    import json
    from src.socratic.dialogue import SocraticSession, testimony_summary_from_bundle
    from src.socratic.gate import read_line

    paths = ctx.paths(case.case_id)
    script = paths.central_script.read_text(encoding="utf-8")
    bundle = json.loads(paths.audit_bundle.read_text(encoding="utf-8"))
    session = SocraticSession(central_script=script,
                              testimony_summary=testimony_summary_from_bundle(bundle))
    print("\n[Socratic mode] Ask about the case. Blank line to exit.\n")
    log = []
    while True:
        # Through the gate's reader: a second reader on stdin would race it for lines.
        q = (read_line("you> ") or "").strip()
        if not q:
            break
        reply = session.respond(q)
        print(f"\n{reply}\n")
        log.append({"user": q, "system": reply, "verdict_flagged": session.flags[-1]})
    if log:
        paths.dialogue.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in log),
                                  encoding="utf-8")
        if store is not None:
            # The case's bundle is already finalized; the dialogue joins its step folder.
            store.save("socratic", case.sample_id or case.case_id, "dialogue", log,
                       fmt="jsonl", echo=False)


def _select_cases(csv_path, only_strata, start, end, sample, limit=None):
    from src.storage.cases import load_cases
    cases = load_cases(csv_path)
    if sample:
        want = {s if len(s) == 3 else f"{int(s):03d}" for s in sample}
        cases = [c for c in cases if c.sample_id in want]
    if only_strata:
        cases = [c for c in cases if c.stratum in only_strata]
    if start is not None:
        cases = [c for c in cases if int(c.sample_id) >= start]
    if end is not None:
        cases = [c for c in cases if int(c.sample_id) <= end]
    if limit:
        # Callers that resume from a ledger must NOT pass limit here: cutting before
        # the skip re-selects the same already-processed samples on every run and makes
        # no progress. run_batch applies its limit after the ledger check.
        cases = cases[:limit]
    return cases


MANIFEST = "variant_manifest.json"


def _check_variant_state(store: OutputStore, allow_warm_corpus: bool = False,
                         ablation: "Ablation | None" = None, uses_corpus: bool = True,
                         config: dict | None = None) -> None:
    """
    Refuse to start or resume a variant from state that would quietly corrupt it.

    1. Stale results. The ledger skips every case it lists. A variant folder whose cases
       were finished by older pipeline code would be "resumed" by skipping all of them,
       leaving pre-fix results in place as if the new code had produced them.
    2. A different ablation. --variant names the folder and --ablation decides what
       runs, so a resume that drops --ablation would fill an ablation's folder with
       full-pipeline cases. The flags are recorded on first start and must match.
    3. A warm corpus. Each variant starts from an empty evidence corpus, as main does,
       or whichever runs second retrieves from what the first gathered. A corpus that
       already holds chunks is accepted only if this variant built it under the current
       pipeline version (a resume, or a first case that crashed part-way); this holds
       whether or not cases have finished, so resuming under another --worker, or with
       none, is refused too.
    4. A different configuration. `config` (a baseline's model, k, rounds ...; B4's
       forced model) is recorded on first start like the ablation flags and must match.

    uses_corpus=False (baselines that retrieve nothing) skips the corpus check: their
    results do not depend on what any corpus holds.
    """
    import json
    from dataclasses import asdict
    from config.settings import PIPELINE_VERSION
    from src.storage import vectordb

    path = store.base / MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    corpus = str(vectordb.VECTORDB_DIR)
    corpora: dict = manifest.get("corpora", {})
    flags = {k: v for k, v in asdict(ablation).items() if k != "variant"} if ablation else None
    finished = bool(store.processed_set())

    if finished and manifest.get("pipeline_version") != PIPELINE_VERSION:
        raise SystemExit(
            f"outputs/{store.variant}/ already holds finished cases from pipeline "
            f"version {manifest.get('pipeline_version') or 'unknown (before versioning)'}"
            f", not {PIPELINE_VERSION}. Resuming would skip them and keep the old "
            f"results. Move that folder to archive/ (or pass --variant <new name>) "
            f"and start the variant again.")
    if flags is not None and manifest.get("ablation") not in (None, flags):
        raise SystemExit(
            f"outputs/{store.variant}/ was started with ablation flags "
            f"{manifest['ablation']}, not {flags}. Resume it with the same --ablation.")
    if config is not None and manifest.get("config") not in (None, config):
        raise SystemExit(
            f"outputs/{store.variant}/ was started with configuration "
            f"{manifest['config']}, not {config}. Resume it with the same options.")
    n = vectordb.corpus_chunk_count() if uses_corpus else 0
    if n and corpora.get(corpus) != PIPELINE_VERSION and not allow_warm_corpus:
        why = ("was built by this variant under pipeline version " + corpora[corpus]
               if corpus in corpora else "was not built by this variant")
        raise SystemExit(
            f"the evidence corpus {corpus} holds {n} chunks and {why}. Every variant "
            f"must run on its own corpus, started empty: pass the --worker this variant "
            f"was started with, or a new id for a new variant (or --allow-warm-corpus "
            f"to proceed anyway).")

    if not finished:
        # Nothing finished yet means nothing to be stale: (re)stamp the current version.
        manifest["pipeline_version"] = PIPELINE_VERSION
    if flags is not None:
        manifest["ablation"] = flags
    if config is not None:
        manifest["config"] = config
    if uses_corpus:
        corpora[corpus] = PIPELINE_VERSION
    manifest["corpora"] = corpora
    manifest.pop("corpus_dirs", None)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_batch(csv_path: str, anchor_choice="survivability", limit: int | None = None,
              only_strata: list[str] | None = None, interactive_gate: bool = False,
              start: int | None = None, end: int | None = None,
              sample: list[str] | None = None, ablation: str = "none",
              variant: str | None = None, allow_warm_corpus: bool = False,
              force_model: str | None = None) -> None:
    """
    Resumable batch over the dataset. Skips samples already in THIS variant's ledger.

    - start/end/sample : select a case range (for parallel windows) or single cases.
    - ablation         : "none" (full) or an ablation preset -> its own variant/ledger.
    - interactive_gate : after each sample, the Y/10s gate offers Socratic dialogue.
    - force_model      : run every LLM role on this one model (baseline B4). Set before
                         the run context exists, so its manifest records that model.

    Parallel safety: run each window with a distinct --start/--end range and a distinct
    COGNI_WORKER (namespaces the vector DB + SIT). outputs/<variant>/ and the ledger are
    shared, but samples are disjoint by range, so nothing is run twice or corrupted.
    """
    abl = Ablation.from_name(ablation)
    if variant:
        # Output namespace only (outputs/<variant>/ + its own ledger); the ablation
        # flags still decide what the pipeline does.
        import re
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", variant):
            raise ValueError(f"variant must be a plain folder name, got {variant!r}")
        abl.variant = variant
    cases = _select_cases(csv_path, only_strata, start, end, sample)

    from src.services.llm_client import set_force_model
    set_force_model(force_model)
    try:
        _run_cases(cases, anchor_choice, abl, limit, start, end, interactive_gate,
                   allow_warm_corpus, force_model)
    finally:
        set_force_model(None)


def _run_cases(cases, anchor_choice, abl: Ablation, limit, start, end, interactive_gate,
               allow_warm_corpus, force_model) -> None:
    from src.socratic.gate import ask_gate

    ctx = RunContext.create(anchor_choice)
    store = OutputStore(ctx.run_id, variant=abl.variant)
    _check_variant_state(store, allow_warm_corpus, ablation=abl,
                         config={"force_model": force_model} if force_model else None)
    from src.storage import vectordb
    recovered = vectordb.recover_inflight(abl.variant, store.processed_set())
    if recovered:
        print(recovered)
    if limit:
        # --limit N = the next N UNPROCESSED samples.
        processed = store.processed_set()
        pending = [c for c in cases if c.sample_id not in processed]
        keep = {c.sample_id for c in pending[:limit]}
        cases = [c for c in cases if c.sample_id in processed or c.sample_id in keep]
    rng = f"[{start or '-'}..{end or '-'}]"
    print(f"Batch {ctx.run_id} variant={abl.variant} range={rng}: "
          f"{len(cases)} candidate samples, anchor={ctx.anchor.label}")

    done = skipped = failed = 0
    for case in cases:
        sid = case.sample_id
        if store.is_processed(sid):
            skipped += 1
            print(f"[skip] {abl.variant} sample {sid} already processed")
            continue
        print(f"\n=== {abl.variant} sample {sid} ({case.stratum}) ===")
        from src.services import telemetry
        telemetry.set_context(abl.variant, sid, ctx.run_id)
        snapshot = vectordb.corpus_snapshot()
        vectordb.mark_inflight(sid, abl.variant, snapshot)
        try:
            summary = process_sample(case, ctx, store, ablation=abl)
            store.save("eval", sid, "run_summary", summary, echo=False)
            store.finalize_sample(sid, extra={"summary": summary})
            vectordb.clear_inflight()
            done += 1
        except KeyboardInterrupt:
            vectordb.rollback_corpus(snapshot)
            vectordb.clear_inflight()
            print(f"\n[stopped] {sid} interrupted; its corpus additions were rolled back")
            raise
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] sample {sid}: {e}")
            traceback.print_exc()
            # do NOT mark processed -> it will be retried on the next run. Its evidence
            # is rolled back so the retry starts from the corpus it first started from,
            # and the batch stops here: later cases must not grow the corpus without
            # this one's evidence, or the corpus would no longer grow in case order.
            removed = vectordb.rollback_corpus(snapshot)
            vectordb.clear_inflight()
            if removed:
                print(f"[FAIL] rolled back {sum(removed.values())} corpus chunks added by {sid}")
            print(f"[FAIL] stopping the batch at {sid}; re-run the same command to retry it")
            break
        # The case is finished and ledgered; a problem in the optional dialogue must not
        # be reported as a failed case.
        if interactive_gate:
            try:
                if ask_gate():
                    _interactive_socratic(case, ctx, store)
            except Exception as e:  # noqa: BLE001
                print(f"[socratic] dialogue for {sid} ended with an error: {e}")
    print(f"\nBatch complete ({abl.variant}): {done} done, {skipped} skipped, {failed} failed.")
