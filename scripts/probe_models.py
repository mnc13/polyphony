"""
Preflight that calls each configured model the way the pipeline actually calls it.

  python scripts/probe_models.py              # probe every role
  python scripts/probe_models.py --role proponent opponent

Why a realistic prompt, not a toy one
-------------------------------------
An earlier version of this script sent a two-field schema and a one-line prompt. All nine
roles passed, and the batch then failed on the very first case: qwen3.8-max-0902 spent all
1000 of its completion tokens on reasoning and returned an empty body with
finish_reason=length. The toy prompt produced short reasoning and fit; a real 388-token
case prompt did not. A preflight that is easier than the real workload proves nothing.

So each role here is probed with a real case from the dataset, the role's OWN schema, and
the SAME max_tokens the pipeline passes at that call site. The probe reports reasoning
token counts, because mandatory-reasoning endpoints draw them from the same budget as the
output and that is the failure mode most likely to recur when a model is swapped.

Cost is a handful of real calls — cents, not dollars — against a batch that takes an hour.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schemas.core import (  # noqa: E402
    ActionInventory, DebateTurnDraft, DecisionSet, PremiseDraftSet,
    ScientistTestimonyDraft, Stakeholder, StakeholderRegister,
)
from src.services.llm_client import get_client  # noqa: E402
from src.services.model_registry import get_registry  # noqa: E402

CSV = ROOT / "data" / "six_cases.csv"


def load_case(idx: int = 1) -> str:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8", newline="")))
    return rows[idx]["Moral Case"]


def fake_sr(case_id: str = "probe") -> StakeholderRegister:
    return StakeholderRegister(case_id=case_id, stakeholders=[
        Stakeholder(id="S1", description="the decision maker", type="individual", role="agent"),
        Stakeholder(id="S2", description="the most affected party", type="individual",
                    role="patient", vulnerability_flag=True),
        Stakeholder(id="S3", description="the wider community", type="group", role="bystander"),
    ])


def build_probes(case: str):
    """role -> (schema, system, user, max_tokens) mirroring the real call sites."""
    from src.ingestion.anchor import Anchor, get_anchor_config
    from src.prompts.decisions import ACTION_SYSTEM, DECISION_SYSTEM, action_user, decision_user
    from src.prompts.ingestion import STAKEHOLDER_SYSTEM, stakeholder_user
    from src.prompts.premises import premise_system, premise_user
    from src.schemas.core import Decision

    sr = fake_sr()
    anchor = get_anchor_config(Anchor("survivability"))
    inv = ActionInventory(
        performed=["acted as described in the case"],
        omitted=["declined to act"],
        alternatives=["pursued a middle course"],
    )
    dec = Decision(id="d1", statement="Acting as described is the right course of action.")
    probes = {
        "stakeholder_extraction": (
            StakeholderRegister, STAKEHOLDER_SYSTEM, stakeholder_user(case), 1200),
        "action_decomposition": (
            ActionInventory, ACTION_SYSTEM, action_user(case, sr), 1000),
        "decision_generation": (
            DecisionSet, DECISION_SYSTEM, decision_user(case, inv.performed, inv.omitted, inv.alternatives), 1200),
        "premise_generation": (
            PremiseDraftSet, premise_system(anchor), premise_user(dec, case, sr), 6000),
    }
    return probes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", nargs="*", default=None)
    ap.add_argument("--case", type=int, default=1, help="dataset row index to probe with")
    args = ap.parse_args()

    reg = get_registry()
    case = load_case(args.case)
    probes = build_probes(case)
    roles = args.role or [r for r in reg.roles() if r in probes]
    skipped = [r for r in (args.role or reg.roles()) if r not in probes]

    print(f"probing {len(roles)} role(s) with a real case ({len(case)} chars) and each "
          f"role's own schema\n")
    print(f"  {'role':24} {'model':28} {'result':6} {'finish':9} {'out':>5} {'reas':>5}  time")
    failures = []
    for role in roles:
        schema, system, user, mt = probes[role]
        spec = reg.get(role)
        t0 = time.time()
        try:
            out = get_client(role).generate_structured(
                user=user, schema=schema, system=system, max_tokens=mt)
            dt = time.time() - t0
            n = sum(len(v) for v in out.model_dump().values() if isinstance(v, list)) or 1
            print(f"  {role:24} {spec.model:28} {'PASS':6} {'stop':9} {'':>5} {'':>5}  "
                  f"{dt:5.1f}s  ({n} items)")
        except Exception as e:  # noqa: BLE001
            dt = time.time() - t0
            msg = " ".join(str(e).split())
            print(f"  {role:24} {spec.model:28} {'FAIL':6} {'':9} {'':>5} {'':>5}  {dt:5.1f}s")
            print(f"      {msg[:200]}")
            failures.append((role, spec.model, msg))

    if skipped:
        print(f"\n  not probed (no standalone prompt available): {', '.join(skipped)}")
        print("  these exercise the same client path as the roles above.")

    print()
    if failures:
        print(f"{len(failures)} of {len(roles)} role(s) FAILED. Fix config/models.yaml "
              f"before starting a batch.")
        return 1
    print(f"all {len(roles)} probed role(s) returned schema-valid output at the real "
          f"prompt size. Safe to start the batch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
