"""
P1 — Stakeholder Impact Trajectory store (append-only SQLite).

The framework's primary non-LLM quality signal. Every survivability claim made in
any debate turn becomes one row, written by deterministic parsing code (Phase 8) —
never by an LLM. This module exposes INSERT and SELECT only. There is no update or
delete API by design; a guard also rejects such SQL.

Rebuttal is therefore DERIVED, not stored. A claim is rebutted when a later claim by
the other debater, in the same decision, on the same stakeholder and dimension, carries
the opposite non-zero polarity. Marking that at write time would mean updating the
earlier row, which an append-only store cannot do -- and nothing ever did: the debate
engine wrote rebutted=False on every row, so 0 of 433 rows on the post-fix run were
rebutted although 45 stakeholder/dimension groups held opposing claims from the two
sides. `mark_rebuttals` computes the flag from row order whenever rows are read for an
audit bundle or a metric; the column stays 0 on disk.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from config.settings import SIT_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sit (
    row_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    case_id         TEXT NOT NULL,
    decision_id     TEXT NOT NULL,
    round           INTEGER NOT NULL,
    speaker         TEXT NOT NULL,          -- proponent | opponent | <discipline>
    stakeholder_id  TEXT NOT NULL,
    dimension       TEXT NOT NULL,          -- NEURO | PSYCH | SOCIO | EVOBIO
    polarity        INTEGER NOT NULL,       -- -1 | 0 | 1
    evidence_node   TEXT,                   -- TEG node id, or NULL if merely asserted
    rebutted        INTEGER NOT NULL DEFAULT 0,
    nli_supported   INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_sit_case ON sit(case_id, decision_id);
"""


@dataclass
class SITRow:
    run_id: str
    case_id: str
    decision_id: str
    round: int
    speaker: str
    stakeholder_id: str
    dimension: str
    polarity: int
    evidence_node: str | None = None
    rebutted: bool = False
    nli_supported: bool = False


class SITStore:
    def __init__(self, db_path: str | Path | None = None):
        self.path = Path(db_path) if db_path else SIT_DB_PATH
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def append(self, row: SITRow) -> int:
        cur = self._conn.execute(
            """INSERT INTO sit
               (run_id, case_id, decision_id, round, speaker, stakeholder_id,
                dimension, polarity, evidence_node, rebutted, nli_supported)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (row.run_id, row.case_id, row.decision_id, row.round, row.speaker,
             row.stakeholder_id, row.dimension, row.polarity, row.evidence_node,
             int(row.rebutted), int(row.nli_supported)),
        )
        self._conn.commit()
        return cur.lastrowid

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Read-only query helper. Rejects any mutating statement except INSERT."""
        head = sql.strip().split(None, 1)[0].upper()
        if head in {"UPDATE", "DELETE", "DROP", "ALTER", "REPLACE"}:
            raise PermissionError(f"SIT is append-only; '{head}' is not permitted.")
        return self._conn.execute(sql, params).fetchall()

    def rows_for(self, case_id: str, decision_id: str | None = None,
                 run_id: str | None = None) -> list[sqlite3.Row]:
        """
        Rows for a case, optionally one decision and/or one run.

        Pass run_id when building an audit bundle. The store accumulates across runs and
        attempts, so a case resumed after a crash otherwise picks up the dead attempt's
        rows too: cases 005 and 006 of the first main run carried 146 and 156 rows for
        77 and 64 actual debate claims.
        """
        sql = "SELECT * FROM sit WHERE case_id=?"
        params: list = [case_id]
        if decision_id:
            sql += " AND decision_id=?"
            params.append(decision_id)
        if run_id:
            sql += " AND run_id=?"
            params.append(run_id)
        return self.query(sql + " ORDER BY row_id", tuple(params))

    def close(self) -> None:
        self._conn.close()


# Debaters are asked for a bare stakeholder id but sometimes answer with the label
# attached -- "S2 (Patients)", "S5 (pharmaceutical company as an institution)". The
# register holds the bare form, so the two spellings must be reconciled before rows are
# compared with each other or with the register.
_SID = re.compile(r"\bS\d+\b")


def canonical_stakeholder_id(raw: str) -> str:
    """'S2 (Patients)' -> 'S2'. Anything without an S<n> prefix is returned as given."""
    m = _SID.search(raw or "")
    return m.group(0) if m else (raw or "")


_SIDES = {"proponent", "opponent"}


def _rebuttal_key(row: dict) -> tuple:
    return (row.get("decision_id"), canonical_stakeholder_id(row.get("stakeholder_id", "")),
            row.get("dimension"))


def mark_rebuttals(rows: list[dict]) -> list[dict]:
    """
    Copies of `rows` with `rebutted` derived from row order (see the module docstring).

    Rows must be in write order (row_id ascending, as rows_for returns them, or the
    order stored in a bundle). Rows from speakers other than the two debaters are
    never marked.
    """
    out = [dict(r) for r in rows]
    for r in out:
        r["rebutted"] = False
    for i, r in enumerate(out):
        pol = int(r.get("polarity") or 0)
        if r.get("speaker") not in _SIDES or pol == 0:
            continue
        key = _rebuttal_key(r)
        for later in out[i + 1:]:
            if (later.get("speaker") in _SIDES and later.get("speaker") != r.get("speaker")
                    and int(later.get("polarity") or 0) == -pol
                    and _rebuttal_key(later) == key):
                r["rebutted"] = True
                break
    return out


