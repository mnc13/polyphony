"""
Baseline runner (resumable, own variant per baseline).

  python scripts/run_baseline.py --baseline b1 --sample 001
  python scripts/run_baseline.py --baseline b2 --teg-from main_audit
  python scripts/run_baseline.py --baseline b4 --worker b4_audit --variant baseline_b4_audit
  python scripts/run_baseline.py --baseline b1 --strata demonstration trolley

Every baseline writes the layout the eval reads (src/eval/baselines.py:save_outputs),
so each one is scored like the main pipeline:

  python scripts/recompute_faithfulness.py --variant baseline_b3 --write
  python scripts/build_eval.py --variant baseline_b3

B2 reads the TEG of a FINISHED pipeline variant (--teg-from) and retrieves nothing.
B4 is the full pipeline with every role forced onto one model (--model); it runs
through run_batch with all of main's guards, on its own corpus: --worker defaults to
the variant name, so it never reads or grows another run's corpus.

Outputs land under outputs/<variant>/ with a resumable ledger per variant. A variant
records its pipeline version and options on first start and refuses a resume under
different ones (src/pipeline/orchestrate.py:_check_variant_state).
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Baselines run on the same six-case dataset as the main pipeline, so the comparison
# is like-for-like (data/six_cases.csv).
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "six_cases.csv"


def main():
    ap = argparse.ArgumentParser(description="Polyphony baseline runner")
    ap.add_argument("--baseline", required=True, choices=["b1", "b2", "b3", "b4", "b5"])
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--anchor", default="survivability",
                    choices=["survivability", "well_being", "flourishing"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--sample", nargs="*", default=None)
    ap.add_argument("--strata", nargs="*", default=None,
                    choices=["ordinary", "demonstration", "trolley"])
    ap.add_argument("--variant", default=None,
                    help="output folder under outputs/ (default baseline_<b>)")
    ap.add_argument("--worker", default=None,
                    help="worker id (namespaces vector DB + SIT); B4 defaults to the variant")
    ap.add_argument("--k", type=int, default=5, help="B1 self-consistency samples")
    ap.add_argument("--agents", type=int, default=3, help="B3 agents")
    ap.add_argument("--rounds", type=int, default=2, help="B3 rounds (Du et al.: 2)")
    ap.add_argument("--model", default="gpt-4o", help="B4/B5 single model")
    ap.add_argument("--teg-from", default=None,
                    help="B2: the finished pipeline variant whose TEG evidence B2 reads")
    ap.add_argument("--allow-warm-corpus", action="store_true",
                    help="B4: start on a corpus that already holds chunks")
    args = ap.parse_args()

    variant = args.variant or f"baseline_{args.baseline}"
    import re
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", variant):
        raise SystemExit(f"variant must be a plain folder name, got {variant!r}")
    if args.baseline == "b2" and not args.teg_from:
        raise SystemExit("B2 needs --teg-from <variant>: the finished run whose TEG it reads")
    # Must be set BEFORE importing settings-dependent modules. B4 retrieves, so it gets
    # its own corpus by default; it used to share data/vectordb with main.
    worker = args.worker or (variant if args.baseline == "b4" else None)
    if worker:
        os.environ["COGNI_WORKER"] = str(worker)

    if args.baseline == "b4":
        from src.pipeline.orchestrate import run_batch
        run_batch(args.csv, anchor_choice=args.anchor, limit=args.limit,
                  only_strata=args.strata, start=args.start, end=args.end,
                  sample=args.sample, variant=variant, force_model=args.model,
                  allow_warm_corpus=args.allow_warm_corpus)
        return

    from src.eval import baselines
    from src.pipeline.orchestrate import _check_variant_state, _select_cases
    from src.pipeline.run import RunContext
    from src.services.model_registry import get_model
    from src.storage.outputs import OutputStore

    cases = _select_cases(args.csv, args.strata, args.start, args.end, args.sample)
    ctx = RunContext.create(args.anchor)
    store = OutputStore(ctx.run_id, variant=variant)
    config = {
        "b1": {"baseline": "b1", "model": get_model("action_decomposition").model,
               "k": args.k, "temperature": 0.7, "vote": baselines.vote_label()},
        "b2": {"baseline": "b2", "model": get_model("scientist_pool").model,
               "teg_from": args.teg_from},
        "b3": {"baseline": "b3", "model": get_model("scientist_pool").model,
               "agents": args.agents, "rounds": args.rounds, "temperature": 0.7,
               "vote": baselines.vote_label()},
        "b5": {"baseline": "b5", "model": args.model, "temperature": 0.2},
    }[args.baseline]
    config["anchor"] = ctx.anchor.anchor.value
    _check_variant_state(store, uses_corpus=False, config=config)
    if args.limit:
        # --limit N = the next N UNPROCESSED samples (cutting before the ledger skip
        # re-selected finished samples on every run and made no progress).
        processed = store.processed_set()
        pending = [c for c in cases if c.sample_id not in processed]
        keep = {c.sample_id for c in pending[:args.limit]}
        cases = [c for c in cases if c.sample_id in processed or c.sample_id in keep]
    print(f"Baseline {args.baseline} -> outputs/{variant}: {len(cases)} candidate samples")

    from src.ingestion.normalise import normalise

    def _text(c):
        return normalise(c.text).full_text

    runner = {
        "b1": lambda c: baselines.run_b1_monolithic(_text(c), ctx.anchor, k=args.k),
        "b2": lambda c: baselines.run_b2_rag_only(_text(c), c.sample_id, args.teg_from,
                                                  ctx.anchor),
        "b3": lambda c: baselines.run_b3_standard_mad(_text(c), ctx.anchor,
                                                      n_agents=args.agents,
                                                      rounds=args.rounds),
        "b5": lambda c: baselines.run_b5_cognialign(_text(c), model=args.model),
    }[args.baseline]

    done = skipped = failed = 0
    for case in cases:
        sid = case.sample_id
        if store.is_processed(sid):
            skipped += 1
            continue
        print(f"=== {variant} sample {sid} ===")
        from src.services import telemetry
        telemetry.set_context(variant, sid, ctx.run_id)
        try:
            store.reset_sample(sid)   # never mix this run's output with an older one
            result = runner(case)
            summary = baselines.save_outputs(store, case, ctx.anchor, args.baseline, result)
            store.finalize_sample(sid, extra={"baseline": args.baseline, "summary": summary})
            done += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {sid}: {e}")
            traceback.print_exc()
    print(f"\nBaseline {args.baseline} complete: {done} done, {skipped} skipped, {failed} failed.")
    print(f"next:\n  python scripts/recompute_faithfulness.py --variant {variant} --write"
          f"\n  python scripts/build_eval.py --variant {variant}")


if __name__ == "__main__":
    main()
