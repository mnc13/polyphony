"""
Re-run B1's or B3's vote on the answers already sampled, under the current judge.

  python scripts/revote_baseline.py --variant baseline_b1_audit2            # dry run
  python scripts/revote_baseline.py --variant baseline_b3_audit --write

No new samples or debate turns: only the judge's one-word calls. The previous vote is
kept under "superseded_votes" in 12_baseline/<sid>/b<N>.json. When the returned answer
changes, the central script and source materials are rewritten and the sample is listed
for a faithfulness recompute (--force, since the definition did not change). The variant
config's "vote" is updated, so the remaining samples run under the same judge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR  # noqa: E402


def _load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def _dump(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--sample", nargs="*", default=None)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    from src.eval import baselines
    from src.services import telemetry

    base = OUTPUTS_DIR / args.variant
    manifest_p = base / "variant_manifest.json"
    manifest = _load(manifest_p)
    old_label = (manifest.get("config") or {}).get("vote")
    judge = baselines.vote_judge()
    rescore = []
    sids = args.sample or sorted(d.name for d in (base / "12_baseline").iterdir() if d.is_dir())
    for sid in sids:
        files = sorted((base / "12_baseline" / sid).glob("b[13].json"))
        if not files:
            print(f"  {sid}: no B1/B3 result - skipped")
            continue
        rp = files[0]
        r = _load(rp)
        b3 = "transcript" in r
        agents = [f"Agent{i + 1}" for i in range(r.get("n_agents") or 0)] if b3 else []
        texts = [r["final"][a] for a in agents] if b3 else r["analyses"]
        telemetry.set_context(args.variant, sid)
        v = baselines.majority_vote(r["final_positions"], texts, judge)
        old = {k: r.get(k) for k in ("clusters", "majority", "selected", "vote_share")}
        same_pick = v["selected"] == old["selected"]
        print(f"  {sid}: {old['clusters']} share {old['vote_share']} -> {v['clusters']} "
              f"share {v['vote_share']}" + ("" if same_pick else
                                             f"   RETURNED ANSWER {old['selected']} -> {v['selected']}"))
        if not args.write:
            continue
        r.setdefault("superseded_votes", []).append({**old, "vote": old_label})
        r.update(v)
        r["vote"] = baselines.vote_label()
        if b3:
            chosen = agents[v["selected"]]
            r["selected_agent"] = chosen
            r["script"] = r["final"][chosen]
            r["sources"] = [t["text"] for t in r["transcript"]
                            if not (t["round"] == r["rounds"] and t["agent"] == chosen)]
        else:
            r["script"] = r["analyses"][v["selected"]]
        _dump(rp, r)
        if not same_pick:
            arb = base / "09_arbiter" / sid
            (arb / "central_script.md").write_text(r["script"], encoding="utf-8")
            sm = _load(arb / "source_materials.json")
            sm["texts"] = r["sources"]
            _dump(arb / "source_materials.json", sm)
            if r["sources"]:
                rescore.append(sid)
    if args.write:
        manifest.setdefault("config", {})["vote"] = baselines.vote_label()
        _dump(manifest_p, manifest)
        print(f"config vote: {old_label} -> {manifest['config']['vote']}")
        if rescore:
            print("faithfulness must be recomputed for: " + " ".join(rescore) +
                  f"\n  python scripts/recompute_faithfulness.py --variant {args.variant} "
                  f"--sample {' '.join(rescore)} --write --force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
