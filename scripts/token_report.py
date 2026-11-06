"""
Token + time report for a run variant (main / an ablation / a baseline).

  python scripts/token_report.py --variant main
  python scripts/token_report.py --variant baseline_b1
  python scripts/token_report.py --variant main --project 62   # OPTIONAL projection

By default it reports averages over ONLY the samples actually run so far (1 run -> that
one; N runs -> average of N). Pass --project N to ALSO get a projection to N samples
(for budgeting the full 62). Writes outputs/<variant>/telemetry/token_report.{json,md}.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR
from src.eval.tokens import aggregate_tokens, report_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    ap.add_argument("--run-id", default="all",
                    help="'all' (default: every record, which sums every run and attempt "
                         "in the file), 'latest' (the most recent run id), or a specific "
                         "run id. Parallel --worker windows and resumed invocations each "
                         "get their own run id, so one variant's run can span several.")
    ap.add_argument("--project", type=int, default=None,
                    help="OPTIONAL: also project totals to N samples (e.g. 62)")
    args = ap.parse_args()

    rid = None if args.run_id == "all" else args.run_id
    agg = aggregate_tokens(args.variant, project_to=args.project, run_id=rid)
    out_dir = OUTPUTS_DIR / args.variant / "telemetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "token_report.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    md = report_md(agg)
    (out_dir / "token_report.md").write_text(md, encoding="utf-8")

    print(md)
    print(f"\nsaved -> {out_dir/'token_report.md'}")


if __name__ == "__main__":
    main()
