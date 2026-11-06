"""
Central output store — saves EVERYTHING at every step, at runtime, in a relevant
format, so no intermediate result is ever lost.

Everything is namespaced by a VARIANT (the experimental condition) so the main
pipeline, each ablation, and each baseline keep separate outputs + separate resume
ledgers and never overwrite or skip one another:

  outputs/<variant>/
    <NN_step>/<sample_id>/<artifact>.<ext>     # STEP-WISE view (per step, per sample)
    full_pipeline/<sample_id>/                  # FULL bundle for one sample
    full_pipeline/<sample_id>.json              # single combined bundle file per sample
    processed_samples.txt                       # resume ledger for THIS variant

Variants: "main" (full pipeline), "abl_no_pmer", "abl_no_testimony",
"abl_fixed_panel", "abl_no_decision_decomp", "baseline_b1", ...

The full-bundle file is named by the DATASET SAMPLE ID (e.g. 001), so a batch run can
consult this variant's ledger and resume from the first sample not yet completed —
never re-running a finished one. A sample id is written to the ledger ONLY after its
whole pipeline finishes and everything is saved.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import OUTPUTS_DIR

# Canonical, ordered step folders. Every phase writes into one of these.
STEPS = {
    "ingestion": "01_ingestion",
    "stakeholders": "02_stakeholders",
    "actions": "03_actions",
    "decisions": "04_decisions",
    "premises": "05_premises",
    "evidence_teg": "06_evidence_teg",
    "testimony": "07_testimony",
    "debate": "08_debate",
    "arbiter": "09_arbiter",
    "socratic": "10_socratic",
    "eval": "11_eval",
    "baseline": "12_baseline",
}


def _to_serialisable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):        # pydantic
        return obj.model_dump(mode="json")
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    return obj


class OutputStore:
    """One instance per (run, variant). All saves also accumulate the full bundle."""

    def __init__(self, run_id: str, variant: str = "main"):
        self.run_id = run_id
        self.variant = variant
        self.base = OUTPUTS_DIR / variant
        self.ledger = self.base / "processed_samples.txt"
        self.full_dir = self.base / "full_pipeline"
        self.base.mkdir(parents=True, exist_ok=True)
        self.full_dir.mkdir(parents=True, exist_ok=True)
        self._bundles: dict[str, dict] = {}   # sample_id -> accumulating bundle

    # ---- step-wise saving -------------------------------------------------
    def _step_dir(self, step: str, sample_id: str) -> Path:
        folder = STEPS.get(step, step)
        d = self.base / folder / sample_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def reset_sample(self, sample_id: str) -> list[str]:
        """
        Drop every artifact this sample already has, before re-running it.

        Step folders are keyed by decision id (TPS_d5.json, testimony_d5.json, ...).
        A re-run that yields FEWER decisions than the previous one therefore leaves
        the surplus files sitting beside the current ones, where nothing marks them
        as dead — they read as live output to anyone inspecting the step folders.
        Clearing first makes the folder always describe exactly one run.

        The ledger is deliberately NOT touched: whether a sample should be re-run at
        all is the caller's decision, and wiping resume state here would silently
        undo it.
        """
        removed: list[str] = []
        for folder in sorted(set(STEPS.values())):
            d = self.base / folder / sample_id
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                removed.append(f"{folder}/{sample_id}")
        stale_dir = self.full_dir / sample_id
        if stale_dir.exists():
            shutil.rmtree(stale_dir, ignore_errors=True)
            removed.append(f"full_pipeline/{sample_id}")
        combined = self.full_dir / f"{sample_id}.json"
        if combined.exists():
            combined.unlink()
            removed.append(f"full_pipeline/{sample_id}.json")
        self._bundles.pop(sample_id, None)
        return removed

    def save(self, step: str, sample_id: str, name: str, obj: Any,
             fmt: str = "json", echo: bool = True) -> Path:
        d = self._step_dir(step, sample_id)
        data = _to_serialisable(obj)
        if fmt == "json":
            path = d / f"{name}.json"
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        elif fmt in ("md", "txt"):
            path = d / f"{name}.{fmt}"
            path.write_text(str(obj), encoding="utf-8")
        elif fmt == "jsonl":
            path = d / f"{name}.jsonl"
            lines = data if isinstance(data, list) else [data]
            path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines),
                            encoding="utf-8")
        else:
            raise ValueError(f"unknown fmt {fmt}")

        b = self._bundles.setdefault(sample_id, {"sample_id": sample_id, "variant": self.variant,
                                                 "run_id": self.run_id, "steps": {}})
        b["steps"].setdefault(STEPS.get(step, step), {})[name] = data
        if echo:
            print(f"[save] {self.variant}/{step}/{sample_id}/{name}.{fmt}")
        return path

    # ---- full bundle + ledger --------------------------------------------
    def finalize_sample(self, sample_id: str, extra: dict | None = None) -> Path:
        b = self._bundles.get(sample_id, {"sample_id": sample_id, "variant": self.variant,
                                          "run_id": self.run_id, "steps": {}})
        b["completed_utc"] = datetime.now(timezone.utc).isoformat()
        if extra:
            b.update(extra)
        (self.full_dir / sample_id).mkdir(parents=True, exist_ok=True)
        (self.full_dir / sample_id / "bundle.json").write_text(
            json.dumps(b, indent=2, ensure_ascii=False), encoding="utf-8")
        combined = self.full_dir / f"{sample_id}.json"
        combined.write_text(json.dumps(b, indent=2, ensure_ascii=False), encoding="utf-8")
        self.mark_processed(sample_id)
        print(f"[done] {self.variant} sample {sample_id} finalized (ledger updated)")
        return combined

    # ---- resume ledger (per variant) -------------------------------------
    def processed_set(self) -> set[str]:
        if not self.ledger.exists():
            return set()
        return {ln.strip() for ln in self.ledger.read_text(encoding="utf-8").splitlines() if ln.strip()}

    def is_processed(self, sample_id: str) -> bool:
        return sample_id in self.processed_set()

    def mark_processed(self, sample_id: str) -> None:
        if self.is_processed(sample_id):
            return
        with open(self.ledger, "a", encoding="utf-8") as fh:
            fh.write(sample_id + "\n")
