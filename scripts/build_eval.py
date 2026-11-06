"""
Build the evaluation report from finalized sample bundles for a given variant.

  python scripts/build_eval.py                       # main pipeline
  python scripts/build_eval.py --variant abl_no_pmer # an ablation
  python scripts/build_eval.py --variant baseline_b1 # a baseline

Reads outputs/<variant>/09_arbiter/<sid>/audit_bundle.json for every completed sample,
computes the Tier-1 metric suite, aggregates by stratum, and writes
outputs/<variant>/11_eval/eval_report.{json,md} plus per-sample central scripts for the
human raters under outputs/<variant>/11_eval/human_eval/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.report import build_report

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    args = ap.parse_args()
    agg = build_report(variant=args.variant)
    print(f"\nOverall Tier-1 metrics ({args.variant}):")
    for k, v in agg["overall"].items():
        print(f"  {k}: {v}")
