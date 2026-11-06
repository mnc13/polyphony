"""
P3 tests. Network-dependent source tests are marked and skippable offline.
The gate/chunker/TEG logic is tested without network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.evidence.chunker import chunk_text
from src.schemas.core import Dimension, Discipline, EvidenceChunk, Premise
from src.evidence.teg import TypedEvidenceGraph


def test_chunker_overlap():
    words = " ".join(str(i) for i in range(600))
    chunks = chunk_text(words, size=256, overlap=32)
    assert len(chunks) >= 2
    # short text passes through whole
    assert chunk_text("just a short one", 256, 32) == ["just a short one"]


def test_teg_add_and_export(tmp_path):
    teg = TypedEvidenceGraph("c1", "d1")
    ch = EvidenceChunk(
        node_id="", text="Chronic stress elevates cortisol and impairs cognition.",
        doi="10.1/x", source="pubmed", dimension=Dimension.NEURO,
        stakeholder_ids=["S1"], citation_count=5,
    )
    nid = teg.add(ch)
    assert nid.startswith("TEG-d1-")
    assert len(teg) == 1
    out = tmp_path / "teg.json"
    teg.export(out)
    assert out.exists()


def test_premise_schema_roundtrip():
    p = Premise(
        id="p1", text="Theft conviction reduces long-term employment prospects.",
        dimension=Dimension.SOCIO, discipline=Discipline.sociology,
        stakeholder_ids=["S1"], polarity=-1,
    )
    assert p.polarity == -1
    j = p.model_dump_json()
    assert "SOCIO" in j
