"""
Export each case to one readable .txt — everything the pipeline produced, in order.

  python scripts/export_case_text.py                  # all cases, variant main
  python scripts/export_case_text.py --sample 002     # one case
  python scripts/export_case_text.py --variant abl_no_pmer

Writes outputs/<variant>/11_eval/readable/case_<id>.txt, one file per case, plus
all_cases.txt with the lot concatenated.

The per-stage JSON under outputs/<variant>/ is the machine-readable record and stays the
source of truth. This is for reading: the case, the stakeholders, the decision space,
then for each decision its premises, the evidence actually retrieved, what each scientist
testified, the full debate transcript turn by turn, and finally the arbiter's central
script. Nothing is summarised or truncated — it is the same content, laid out to be read
top to bottom rather than opened file by file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import OUTPUTS_DIR  # noqa: E402
from src.storage.sit import mark_rebuttals  # noqa: E402

W = 96
NAMES = {
    "001": "Trolley driver", "002": "Memory implantation", "003": "Pharma drug release",
    "004": "Border-town refugee", "005": "Predictive-policing bias",
    "006": "Stranded crew rations",
}


def rule(ch="=", label=""):
    if not label:
        return ch * W
    pad = W - len(label) - 4
    return f"{ch*2} {label} {ch*max(pad,0)}"


def wrap(text, indent=""):
    out = []
    for para in (text or "").split("\n"):
        if not para.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(para, width=W - len(indent),
                                 initial_indent=indent, subsequent_indent=indent))
    return "\n".join(out)


def load(p, default=None):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return default


def export(variant: str, sid: str) -> str | None:
    base = OUTPUTS_DIR / variant
    arb_md = base / "09_arbiter" / sid / "central_script.md"
    if not arb_md.exists():
        return None

    L: list[str] = []
    norm = load(base / "01_ingestion" / sid / "normalised.json", {}) or {}
    sr = load(base / "02_stakeholders" / sid / "stakeholder_register.json", {}) or {}
    inv = load(base / "03_actions" / sid / "action_inventory.json", {}) or {}
    dec = load(base / "04_decisions" / sid / "decisions.json", {}) or {}
    bundle = load(base / "09_arbiter" / sid / "audit_bundle.json", {}) or {}

    L.append(rule("#"))
    L.append(f"#  CASE {sid} — {NAMES.get(sid, '')}")
    L.append(f"#  stratum: {norm.get('stratum','?')}   variant: {variant}")
    L.append(rule("#"))
    L.append("")

    L.append(rule("=", "THE CASE"))
    L.append(wrap(norm.get("full_text") or ""))
    L.append("")

    # A baseline has no stakeholders, decisions or debate of the pipeline's kind: its own
    # process (samples and vote, evidence, agent turns, arguments) is written instead.
    bl = sorted((base / "12_baseline" / sid).glob("b*.json"))
    if bl:
        L.extend(_baseline_section(load(bl[0], {}) or {}))
        L.append(rule("#"))
        L.append("#  FINAL OUTPUT (scored as this baseline's central script)")
        L.append(rule("#"))
        L.append(arb_md.read_text(encoding="utf-8"))
        L.extend(_faithfulness_section(base / "09_arbiter" / sid, bundle))
        return "\n".join(L)

    stks = sr.get("stakeholders", [])
    L.append(rule("=", f"STAKEHOLDERS ({len(stks)})"))
    for s in stks:
        vul = "  [vulnerable]" if s.get("vulnerability_flag") else ""
        L.append(f"  {s.get('id'):4} {s.get('type','?'):11} {s.get('role','?'):9} "
                 f"{s.get('description','')}{vul}")
    L.append("")

    L.append(rule("=", "ACTION INVENTORY"))
    for k in ("performed", "omitted", "alternatives"):
        L.append(f"  {k}:")
        for a in inv.get(k, []):
            L.append(wrap(f"- {a}", indent="    "))
    L.append("")

    decs = dec.get("decisions", [])
    L.append(rule("=", f"DECISION SPACE ({len(decs)})"))
    for d in decs:
        L.append(wrap(f"{d.get('id')}: {d.get('statement','')}", indent="  "))
    L.append("")

    # rebutted is derived from row order (the SIT store never sets it on disk)
    sit = mark_rebuttals(bundle.get("sit_rows", []))
    for d in decs:
        did = d.get("id")
        L.append("")
        L.append(rule("#"))
        L.append(f"#  DECISION {did}")
        L.append(wrap(d.get("statement", ""), indent="#  "))
        L.append(rule("#"))

        tps = load(base / "05_premises" / sid / f"TPS_{did}.json", {}) or {}
        prem = tps.get("premises", [])
        L.append("")
        L.append(rule("-", f"PREMISES ({len(prem)})"))
        for p in prem:
            pol = {1: "+1 enhances", -1: "-1 degrades", 0: " 0 ambiguous"}.get(
                p.get("polarity"), str(p.get("polarity")))
            L.append(f"  [{p.get('id')}] {p.get('discipline','?')} / "
                     f"{p.get('dimension','?')} / {pol} / "
                     f"stakeholders={','.join(p.get('stakeholder_ids') or [])}")
            L.append(wrap(p.get("text", ""), indent="      "))
        L.append("")

        teg = load(base / "06_evidence_teg" / sid / f"TEG_{did}.json", {}) or {}
        nodes = teg.get("nodes", [])
        L.append(rule("-", f"EVIDENCE RETRIEVED ({len(nodes)} nodes)"))
        if not nodes:
            L.append("  (no literature admitted for this decision)")
        for n in nodes:
            L.append(f"  [{n.get('node_id')}] {n.get('venue') or 'n/a'} "
                     f"({n.get('year') or 'n/a'}) cites={n.get('citation_count')} "
                     f"doi={n.get('doi') or 'n/a'} src={n.get('source')}")
            L.append(wrap(n.get("text", ""), indent="      "))
        L.append("")

        tst = load(base / "07_testimony" / sid / f"testimony_{did}.json", {}) or {}
        tms = tst.get("testimonies", [])
        L.append(rule("-", f"SCIENTIST TESTIMONY ({len(tms)} scientists)"))
        for t in tms:
            L.append(f"  -- {t.get('discipline','?')} --")
            for c in t.get("claims", []):
                tag = ("evidence-backed" if c.get("nli_supported")
                       else ("cited" if c.get("cited_nodes") else "asserted"))
                L.append(f"     on {c.get('premise_id')} | direction={c.get('direction')} "
                         f"| strength={c.get('evidence_strength')} | {tag} "
                         f"| nodes={','.join(c.get('cited_nodes') or []) or 'none'}")
                L.append(wrap(c.get("finding", ""), indent="        "))
        L.append("")

        turns = load(base / "08_debate" / sid / f"transcript_{did}.json", []) or []
        summ = load(base / "08_debate" / sid / f"debate_summary_{did}.json", {}) or {}
        L.append(rule("-", f"DEBATE TRANSCRIPT ({len(turns)} turns, "
                           f"{summ.get('rounds_run','?')} rounds, "
                           f"stop={summ.get('stop_reason','?')})"))
        for t in turns:
            L.append("")
            L.append(f"  ---- round {t.get('round')} | {str(t.get('speaker','')).upper()} ----")
            L.append(f"       premises addressed: {', '.join(t.get('premises_addressed') or []) or 'none'}")
            L.append(f"       evidence cited:     {', '.join(t.get('evidence_cited') or []) or 'none'}")
            L.append(wrap(t.get("argument_text", ""), indent="       "))
            if t.get("evidential_gap"):
                L.append(wrap(f"[evidential gap flagged] {t['evidential_gap']}", indent="       "))
        L.append("")

        rows = [r for r in sit if r.get("decision_id") == did]
        if rows:
            cited = sum(1 for r in rows if r.get("evidence_node"))
            nli_ok = sum(1 for r in rows if r.get("nli_supported"))
            L.append(rule("-", f"SIT ROWS FOR {did} ({len(rows)})"))
            L.append(f"  cited={cited}  asserted={len(rows)-cited}  "
                     f"nli_supported={nli_ok}  rebutted={sum(1 for r in rows if r.get('rebutted'))}")
            L.append("")

    L.append("")
    L.append(rule("#"))
    L.append("#  ARBITER CENTRAL SCRIPT")
    L.append(rule("#"))
    L.append(arb_md.read_text(encoding="utf-8"))
    L.extend(_faithfulness_section(base / "09_arbiter" / sid, bundle))
    return "\n".join(L)


def _vote_lines(r: dict, names: list[str]) -> list[str]:
    groups = "  ".join("[" + ", ".join(names[i] for i in c) + "]" for c in r.get("clusters", []))
    return [f"  answer groups (same course of action): {groups}",
            f"  majority: {', '.join(names[i] for i in r.get('majority', []))}   "
            f"vote share: {r.get('vote_share')}   returned: {names[r['selected']]}"
            if r.get("selected") is not None else "", ""]


def _baseline_section(r: dict) -> list[str]:
    L = [rule("=", f"BASELINE {r.get('baseline', '?')}   model: {r.get('model', '?')}"), ""]
    if "analyses" in r:                                    # B1: k samples and a vote
        names = [f"sample {i + 1}" for i in range(len(r["analyses"]))]
        for i, (a, fp) in enumerate(zip(r["analyses"], r.get("final_positions", []))):
            L.append(rule("-", f"{names[i].upper()} (T={r.get('temperature')})"))
            L.append(wrap(a, indent="  "))
            L.append(wrap(f"[final position parsed] {fp or '(none)'}", indent="  "))
            L.append("")
        L.append(rule("-", "SELF-CONSISTENCY VOTE"))
        L.extend(_vote_lines(r, names))
    elif "transcript" in r:                                # B3: agents over rounds, a vote
        for t in r["transcript"]:
            L.append(rule("-", f"ROUND {t.get('round')} | {t.get('agent')}"))
            L.append(wrap(t.get("text", ""), indent="  "))
            L.append("")
        agents = [f"Agent{i + 1}" for i in range(r.get("n_agents") or len(r.get("final", {})))]
        L.append(rule("-", "VOTE OVER THE LAST ROUND"))
        L.extend(_vote_lines(r, agents))
    elif "evidence_labels" in r:                           # B2: the TEG evidence it was given
        labels = list(r["evidence_labels"].items())
        L.append(rule("-", f"EVIDENCE GIVEN ({len(labels)} chunks from {r.get('teg_variant')}'s TEG)"))
        for (lab, node), text in zip(labels, r.get("sources", [])):
            L.append(f"  [{lab}] {node}")
            L.append(wrap(text, indent="      "))
        L.append("")
    elif "arguments" in r:                                 # B5: arguments, rebuttals, verdict
        for k, title in (("arguments", "ARGUMENT"), ("rebuttals", "REBUTTAL")):
            for disc, text in (r.get(k) or {}).items():
                L.append(rule("-", f"{title} | {disc}"))
                L.append(wrap(text, indent="  "))
                L.append("")
    return L


def _faithfulness_section(arb, bundle: dict) -> list[str]:
    """The recompute's record (faithfulness.json) first: it carries every ungrounded
    sentence with its best score; the bundle's copy may be the in-run figure."""
    f = load(arb / "faithfulness.json", None) or bundle.get("faithfulness", {}) or {}
    L = ["", rule("=", "FAITHFULNESS")]
    if f.get("applicable") is False:
        L += ["  not applicable: this output was compiled from no source materials", ""]
        return L
    L.append(f"  faithfulness: {f.get('fraction_entailed')}   definition: "
             f"{f.get('faithfulness_def') or f.get('definition')}")
    L.append(f"  scored sentences: {f.get('n_scored')}   meta (not scored): {f.get('n_meta')}   "
             f"ungrounded: {f.get('n_ungrounded')}   source windows: {f.get('n_windows')}")
    detail = f.get("ungrounded_detail") or [{"sentence": s, "best_score": None}
                                            for s in f.get("ungrounded_sentences") or []]
    if detail:
        L.append("  ungrounded sentences (best entailment score against any source window):")
        for d in detail:
            L.append(wrap(f"- ({d.get('best_score')}) {d.get('sentence')}", indent="      "))
    L.append("")
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    ap.add_argument("--sample", nargs="*", default=None)
    args = ap.parse_args()

    base = OUTPUTS_DIR / args.variant / "09_arbiter"
    if not base.is_dir():
        raise SystemExit(f"no arbiter outputs for variant {args.variant}")
    sids = args.sample or sorted(
        d for d in os.listdir(base) if (base / d / "central_script.md").exists())

    out_dir = OUTPUTS_DIR / args.variant / "11_eval" / "readable"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    for sid in sids:
        txt = export(args.variant, sid)
        if txt is None:
            print(f"  {sid}: skipped (no central script)")
            continue
        p = out_dir / f"case_{sid}.txt"
        p.write_text(txt, encoding="utf-8")
        combined.append(txt)
        print(f"  wrote {p}  ({len(txt):,} chars, {txt.count(chr(10)):,} lines)")
    if len(combined) > 1:
        allp = out_dir / "all_cases.txt"
        allp.write_text("\n\n".join(combined), encoding="utf-8")
        print(f"  wrote {allp}  ({sum(len(c) for c in combined):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
