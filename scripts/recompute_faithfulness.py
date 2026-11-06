"""
Recompute the arbiter faithfulness metric offline, from stored artifacts.

  python scripts/recompute_faithfulness.py                    # report, write nothing
  python scripts/recompute_faithfulness.py --write            # update the stored figures
  python scripts/recompute_faithfulness.py --variant abl_no_pmer --write
  python scripts/recompute_faithfulness.py --write --force    # redo cases already done

The metric's definition lives in src/eval/faithfulness.py (FAITHFULNESS_DEF); read its
docstring for what changed between v1 and v2 and why. In short, v2 scores real
sentences (Markdown headings no longer count), against short windows of the source
materials that DeBERTa and MiniLM see whole, and never lets a non-committal span
("the evidence does not settle ...") ground a substantive sentence.

Why offline: the pipeline does not need this metric to produce its outputs, and NLI on
this CPU costs seconds per pair. v2 windows are a fraction of the old 512-token spans,
so a pass is far faster than the ~15 s/pair measured under v1, but it is still a
multi-hour CPU job for six cases. It is resumable: each case is written as it finishes,
and a case already computed under the CURRENT definition is skipped with --write.

--write updates three places, because three readers exist:
  09_arbiter/<sid>/faithfulness.json   full detail, incl. every ungrounded sentence
  09_arbiter/<sid>/audit_bundle.json   what src/eval/metrics.py reads
  11_eval/<sid>/run_summary.json       script_faithfulness (it kept the placeholder)
A value computed under the previous definition is kept under "v1" so earlier numbers
stay traceable.

--skip-truncated is accepted for compatibility with older instructions and ignored:
v2 has no "truncated" column (its windows are never truncated).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import OUTPUTS_DIR  # noqa: E402
from src.eval.faithfulness import FAITHFULNESS_DEF, context_texts, score_script  # noqa: E402

_V1_KEYS = ("fraction_entailed", "fraction_entailed_as_run", "fraction_entailed_truncated",
            "n_sentences", "n_ungrounded", "n_source_spans", "n_grounded_spans",
            "n_vacuous_spans_dropped", "ungrounded_detail", "nli_max_length")


def load_materials(variant: str, sample: str) -> tuple[list[str], list[str]]:
    """
    (testimony findings, debate turn texts) exactly as the arbiter received them.

    A baseline has no testimony or debate files; it lists the texts its final answer
    was compiled from in 09_arbiter/<sid>/source_materials.json (B2's evidence, B3's
    other turns, B5's arguments and rebuttals), returned here as the turn pool.
    """
    base = OUTPUTS_DIR / variant
    sm = base / "09_arbiter" / sample / "source_materials.json"
    if sm.exists():
        return [], [t for t in json.load(open(sm, encoding="utf-8")).get("texts", []) if t]
    findings: list[str] = []
    turns: list[str] = []
    from src.arbiter.compile import NO_TESTIMONY
    for p in sorted(glob.glob(str(base / "07_testimony" / sample / "testimony_*.json"))):
        claims = [c for t in json.load(open(p, encoding="utf-8"))["testimonies"] for c in t["claims"]]
        findings.extend(c.get("finding") or "" for c in claims)
        if not claims and NO_TESTIMONY not in findings:
            findings.append(NO_TESTIMONY)
    for p in sorted(glob.glob(str(base / "08_debate" / sample / "transcript_*.json"))):
        turns.extend(turn.get("argument_text") or "" for turn in json.load(open(p, encoding="utf-8")))
    return findings, turns


def load_context(variant: str, sample: str) -> list[str]:
    """Case text, anchor and decision statements, as given to the arbiter."""
    base = OUTPUTS_DIR / variant
    case_text, label, definition, decisions = "", "", "", []
    norm = base / "01_ingestion" / sample / "normalised.json"
    if norm.exists():
        case_text = json.load(open(norm, encoding="utf-8")).get("full_text", "")
    anc = base / "01_ingestion" / sample / "anchor.json"
    if anc.exists():
        a = json.load(open(anc, encoding="utf-8"))
        label = a.get("label", "")
        try:
            from src.ingestion.anchor import get_anchor_config
            from src.schemas.core import Anchor
            definition = get_anchor_config(Anchor(a.get("anchor"))).definition
        except Exception:
            definition = ""
    dec = base / "04_decisions" / sample / "decisions.json"
    if dec.exists():
        decisions = [(d["id"], d["statement"])
                     for d in json.load(open(dec, encoding="utf-8")).get("decisions", [])]
    return context_texts(case_text, label, definition, decisions)


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, 3)


def persist_case(variant: str, s: str, res) -> None:
    arb = OUTPUTS_DIR / variant / "09_arbiter" / s
    frac = _num(res.fraction)

    fj = arb / "faithfulness.json"
    data = json.load(open(fj, encoding="utf-8")) if fj.exists() else {}
    # In-run figures carry "definition"; offline ones "faithfulness_def". Either kind of
    # earlier measurement is archived, not overwritten without a trace.
    prev_def = data.get("faithfulness_def") or data.get("definition")
    data.pop("definition", None)
    if prev_def != FAITHFULNESS_DEF and (data.get("recomputed_offline_done")
                                         or data.get("measured") is True):
        if prev_def:
            # an earlier v2.x figure: kept under its own definition name, not as "v1"
            data.setdefault("superseded", {})[prev_def] = {
                k: data.get(k) for k in ("fraction_entailed", "n_scored", "n_meta", "n_ungrounded")}
        else:
            data["v1"] = {k: data[k] for k in _V1_KEYS if k in data}
    for k in _V1_KEYS:
        data.pop(k, None)
    data.update({
        "fraction_entailed": frac,
        "measured": True,
        "faithfulness_def": FAITHFULNESS_DEF,
        "n_sentences": res.n_sentences, "n_scored": res.n_scored, "n_meta": res.n_meta,
        "n_ungrounded": res.n_ungrounded,
        "n_windows": res.n_windows, "n_vacuous_sentences": res.n_vacuous_sentences,
        "ungrounded_sentences": [d["sentence"] for d in res.ungrounded_detail],
        "ungrounded_detail": res.ungrounded_detail,
        "recomputed_offline_done": True,
    })
    json.dump(data, open(fj, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    ab = arb / "audit_bundle.json"
    if ab.exists():
        bundle = json.load(open(ab, encoding="utf-8"))
        fb = bundle.setdefault("faithfulness", {})
        if fb.get("definition") != FAITHFULNESS_DEF and (fb.get("recomputed_offline")
                                                         or fb.get("measured") is True):
            if fb.get("definition"):
                fb.setdefault("superseded", {})[fb["definition"]] = {
                    k: fb.get(k) for k in ("fraction_entailed", "n_scored", "n_meta", "n_ungrounded")}
            else:
                fb["v1"] = {k: fb[k] for k in ("fraction_entailed", "fraction_entailed_truncated",
                                              "n_ungrounded", "n_grounded_spans",
                                              "n_vacuous_spans_dropped") if k in fb}
        for k in ("fraction_entailed_as_run", "fraction_entailed_truncated",
                  "n_grounded_spans", "n_vacuous_spans_dropped"):
            fb.pop(k, None)
        fb.update({"fraction_entailed": frac, "measured": True, "recomputed_offline": True,
                   "definition": FAITHFULNESS_DEF, "n_scored": res.n_scored,
                   "n_meta": res.n_meta, "n_ungrounded": res.n_ungrounded,
                   "ungrounded_sentences": [d["sentence"] for d in res.ungrounded_detail]})
        json.dump(bundle, open(ab, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    rs = OUTPUTS_DIR / variant / "11_eval" / s / "run_summary.json"
    if rs.exists():
        summ = json.load(open(rs, encoding="utf-8"))
        summ["script_faithfulness"] = frac
        summ["script_faithfulness_def"] = FAITHFULNESS_DEF
        json.dump(summ, open(rs, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    ap.add_argument("--write", action="store_true", help="persist the recomputed figures")
    ap.add_argument("--force", action="store_true",
                    help="recompute cases already done under the current definition")
    ap.add_argument("--skip-truncated", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--sample", nargs="*", default=None)
    args = ap.parse_args()

    base = OUTPUTS_DIR / args.variant / "09_arbiter"
    if not base.is_dir():
        raise SystemExit(f"no arbiter outputs for variant {args.variant}")
    samples = args.sample or sorted(
        d for d in os.listdir(base) if (base / d / "central_script.md").exists())
    if not samples:
        raise SystemExit("no central_script.md found — nothing to recompute")

    print(f"variant={args.variant}  definition={FAITHFULNESS_DEF}\n")
    print(f"  {'case':6} {'scored':>7} {'meta':>5} {'windows/vac':>13} {'faithful':>9} {'ungrounded':>11}")
    rows = []
    for s in samples:
        fj = base / s / "faithfulness.json"
        if args.write and not args.force and fj.exists():
            d0 = json.load(open(fj, encoding="utf-8"))
            done_def = d0.get("faithfulness_def") or d0.get("definition")
            if done_def == FAITHFULNESS_DEF and (d0.get("recomputed_offline_done")
                                                 or d0.get("measured") is True):
                print(f"  {s:6} already recomputed under {FAITHFULNESS_DEF} - skipped")
                rows.append(d0.get("fraction_entailed"))
                continue
        script = (base / s / "central_script.md").read_text(encoding="utf-8")
        findings, turns = load_materials(args.variant, s)
        if not findings and not turns:
            # Faithfulness to sources is undefined without sources (baseline B1).
            print(f"  {s:6} no source materials - not applicable, skipped")
            continue
        res = score_script(script, findings, turns, context=load_context(args.variant, s))
        print(f"  {s:6} {res.n_scored:7} {res.n_meta:5} {res.n_windows:6}/{res.n_vacuous_sentences:<6} "
              f"{res.fraction:9.3f} {res.n_ungrounded:5}/{res.n_scored:<5}")
        rows.append(_num(res.fraction))
        if args.write:
            persist_case(args.variant, s, res)

    vals = [v for v in rows if isinstance(v, (int, float))]
    if vals:
        print(f"\n  mean faithfulness = {sum(vals)/len(vals):.3f}  (n={len(vals)})")
    print("\nwrote figures per case as each finished" if args.write
          else "\nreport only — pass --write to persist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
