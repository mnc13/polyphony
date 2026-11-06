"""
P1 — Case-folder conventions.

runs/<run_id>/<case_id>/ holds every per-case artifact. One helper so paths are
never string-built ad hoc across the codebase.
"""
from __future__ import annotations

from pathlib import Path

from config.settings import RUNS_DIR


class CasePaths:
    def __init__(self, run_id: str, case_id: str):
        self.root = RUNS_DIR / run_id / case_id
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def sr(self) -> Path: return self.root / "SR.json"

    @property
    def decisions(self) -> Path: return self.root / "decisions.json"

    @property
    def actions(self) -> Path: return self.root / "actions.json"

    def tps(self, decision_id: str) -> Path: return self.root / f"TPS_{decision_id}.json"

    def testimony(self, decision_id: str) -> Path: return self.root / f"testimony_{decision_id}.json"

    def teg(self, decision_id: str) -> Path: return self.root / f"TEG_{decision_id}.json"

    def transcript(self, decision_id: str) -> Path: return self.root / f"transcript_{decision_id}.jsonl"

    @property
    def central_script(self) -> Path: return self.root / "central_script.md"

    @property
    def audit_bundle(self) -> Path: return self.root / "audit_bundle.json"

    @property
    def dialogue(self) -> Path: return self.root / "dialogue.jsonl"


def manifest_path(run_id: str) -> Path:
    p = RUNS_DIR / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p / "manifest.json"
