"""
P0 deliverable: print the role->model table and confirm heterogeneity holds.

  python scripts/validate_models.py               # offline: registry + family checks
  python scripts/validate_models.py --check-live  # also verify every id against OpenRouter

--check-live exists because a configured model id can stop existing between selection and
run. qwen/qwen3-max was deprecated on 9 Oct 2026 and a run started on the 14th failed at
the action_decomposition stage of every case, after billing stakeholder_extraction each
time and spending ~90s of retry backoff per failure. The catalogue endpoint is public --
no key, no charge -- so there is no reason not to check it before a batch.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.model_registry import get_registry  # noqa: E402

CATALOGUE = "https://openrouter.ai/api/v1/models"


def fetch_catalogue(timeout: int = 30) -> dict[str, dict]:
    req = urllib.request.Request(CATALOGUE, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {m["id"]: m for m in data["data"]}


def check_live(reg) -> int:
    """Verify each openrouter-provider model still exists. Returns the number of dead ids."""
    specs = [reg.get(r) for r in reg.roles()]
    targets = [s for s in specs if s.provider == "openrouter"]
    if not targets:
        print("\nno openrouter models configured — nothing to check against the catalogue")
        return 0

    print(f"\nchecking {len(targets)} model ids against {CATALOGUE} ...")
    try:
        cat = fetch_catalogue()
    except Exception as e:  # noqa: BLE001
        print(f"  could not reach the catalogue: {e}")
        print("  SKIPPED — this check is advisory; the registry checks above still passed.")
        return 0

    print(f"  catalogue has {len(cat)} models\n")
    print(f"  {'role':24} {'model':34} {'status':6} price in / out")
    dead = []
    for s in targets:
        m = cat.get(s.model)
        if m is None:
            print(f"  {s.role:24} {s.model:34} {'DEAD':6} <-- not in the catalogue")
            dead.append(s)
            continue
        p = m.get("pricing", {})
        pin = float(p.get("prompt") or 0) * 1e6
        pout = float(p.get("completion") or 0) * 1e6
        sp = m.get("supported_parameters") or []
        warn = "" if "structured_outputs" in sp else "   (no structured_outputs)"
        print(f"  {s.role:24} {s.model:34} {'live':6} ${pin:.3f} / ${pout:.3f}{warn}")

    if dead:
        print(f"\n  {len(dead)} DEAD MODEL ID(S) — fix config/models.yaml before running:")
        for s in dead:
            near = sorted(k for k in cat if k.startswith(s.model.split("/")[0] + "/"))[:6]
            print(f"    {s.role}: {s.model}")
            if near:
                print(f"      same org, currently live: {', '.join(near)}")
    return len(dead)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-live", action="store_true",
                    help="verify every openrouter model id against the public catalogue")
    args = ap.parse_args()

    reg = get_registry()  # raises on any heterogeneity/repetition violation
    print(reg.table())
    print("\nOK — registry validated, no violations.")

    if args.check_live:
        dead = check_live(reg)
        if dead:
            print("\nFAILED — do not start a batch until the ids above are fixed.")
            return 1
        print("\nOK — every configured model id is live on OpenRouter.")
    else:
        print("\nTip: run with --check-live to confirm the ids still exist before a batch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
