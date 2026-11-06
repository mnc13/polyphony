"""
P1 — Case loader.

Reads the moral-case CSV, repairs the mojibake the source files carry
(â€™ -> ', etc.), and tags each case with its stratum (ordinary / demonstration /
trolley) so every downstream metric can break results down by case type.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Common UTF-8-decoded-as-cp1252 artifacts seen in the source CSVs.
_MOJIBAKE = {
    "â€™": "'", "â€˜": "'", "â€œ": '"', "â€\x9d": '"', "â€”": "—",
    "â€“": "–", "â€¦": "…", "Â": "", "﻿": "",
}

_SOURCE_TO_STRATUM = {
    "ordinary_cases.csv": "ordinary",
    "demonstration_cases.csv": "demonstration",
    "batteryiii_trolley.csv": "trolley",
}


def repair_text(s: str) -> str:
    for bad, good in _MOJIBAKE.items():
        s = s.replace(bad, good)
    # A second pass: try the canonical re-decode for anything left.
    try:
        s = s.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore") if "Ã" in s else s
    except Exception:
        pass
    return s.strip()


@dataclass(frozen=True)
class MoralCase:
    case_id: str
    text: str
    stratum: str  # ordinary | demonstration | trolley | unknown
    sample_id: str = ""   # stable sequential dataset id (001..NNN); the ledger/key


def _case_id(text: str, idx: int) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"case{idx:03d}_{h}"


def load_cases(csv_path: str | Path) -> list[MoralCase]:
    """Load the combined CSV. Expects a 'Moral Case' column and optional 'Source_File'."""
    path = Path(csv_path)
    cases: list[MoralCase] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        # Normalize header keys (strip BOM/space).
        for i, row in enumerate(reader, start=1):
            row = { (k or "").strip().lstrip("﻿"): v for k, v in row.items() }
            raw = row.get("Moral Case") or row.get("Moral_Case") or ""
            text = repair_text(raw)
            if not text:
                continue
            src = (row.get("Source_File") or "").strip()
            stratum = _SOURCE_TO_STRATUM.get(src, "unknown")
            sid = f"{len(cases) + 1:03d}"   # stable sequential dataset sample id
            cases.append(MoralCase(case_id=_case_id(text, i), text=text,
                                   stratum=stratum, sample_id=sid))
    return cases


def iter_cases(csv_path: str | Path) -> Iterator[MoralCase]:
    yield from load_cases(csv_path)
