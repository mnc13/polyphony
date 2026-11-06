"""
Full / subset / parallel / ablation run driver (resumable).

Common runs:
  # single case (test one end to end):
  python scripts/run_full.py --sample 001

  # primary-analysis subset (deep dilemmas + trolley), unattended:
  python scripts/run_full.py --strata demonstration trolley

  # full 62-case run under Survivability (overnight, resumable):
  python scripts/run_full.py

  # PARALLEL windows — give each a disjoint range + its own worker id:
  #   window A:  python scripts/run_full.py --start 1  --end 15 --worker A
  #   window B:  python scripts/run_full.py --start 16 --end 30 --worker B
  # outputs/<variant>/ and the ledger are shared; ranges are disjoint so no case runs
  # twice and no store is corrupted (worker namespaces the vector DB + SIT).

  # an ABLATION (own variant + own ledger, won't collide with the main run):
  python scripts/run_full.py --ablation no_pmer --strata demonstration trolley

Resume is automatic per variant: samples already in
outputs/<variant>/processed_samples.txt are skipped.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The experiment runs on the six dilemmas in data/six_cases.csv (options already stripped,
# text pre-repaired, strata resolved; see tests/test_p17_six_cases.py). Pass --csv to
# point at any other dataset with a "Moral Case" column.
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "six_cases.csv"


def main():
    ap = argparse.ArgumentParser(description="Polyphony batch runner (resumable)")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="dataset CSV path")
    ap.add_argument("--anchor", default="survivability",
                    choices=["survivability", "well_being", "flourishing"])
    ap.add_argument("--limit", type=int, default=None, help="max samples this run")
    ap.add_argument("--start", type=int, default=None, help="first sample id (inclusive), e.g. 1")
    ap.add_argument("--end", type=int, default=None, help="last sample id (inclusive), e.g. 15")
    ap.add_argument("--sample", nargs="*", default=None, help="specific sample id(s), e.g. 001 005")
    ap.add_argument("--strata", nargs="*", default=None,
                    choices=["ordinary", "demonstration", "trolley"],
                    help="restrict to these strata")
    ap.add_argument("--worker", default=None,
                    help="worker id for parallel safety (namespaces vector DB + SIT)")
    ap.add_argument("--ablation", default="none",
                    choices=["none", "no_decision_decomp", "no_pmer", "no_testimony", "fixed_panel"],
                    help="run an ablation variant (own outputs + ledger)")
    ap.add_argument("--variant", default=None,
                    help="write to outputs/<variant>/ with its own ledger instead of the "
                         "ablation's default folder (e.g. a pilot run that must not touch "
                         "outputs/main). Pipeline behaviour is still set by --ablation.")
    ap.add_argument("--allow-warm-corpus", action="store_true",
                    help="start a new variant on a corpus that already holds chunks from "
                         "another run (refused by default: it makes variants incomparable)")
    ap.add_argument("--interactive", action="store_true",
                    help="offer the Y/10s Socratic gate after each sample")
    args = ap.parse_args()

    # Must set the worker env BEFORE importing settings-dependent modules.
    if args.worker:
        os.environ["COGNI_WORKER"] = str(args.worker)

    if not Path(args.csv).exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    from src.pipeline.orchestrate import run_batch
    run_batch(args.csv, anchor_choice=args.anchor, limit=args.limit,
              only_strata=args.strata, interactive_gate=args.interactive,
              start=args.start, end=args.end, sample=args.sample, ablation=args.ablation,
              variant=args.variant, allow_warm_corpus=args.allow_warm_corpus)


if __name__ == "__main__":
    main()
