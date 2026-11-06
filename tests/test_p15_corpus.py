"""
P15 — evidence-corpus hygiene and retrieval telemetry.

  1. the vector store refuses duplicate rows (same DOI + same text)
  2. re-adding after a reload still refuses them (the seen-set survives persistence)
  3. P-MER records whether a premise was served locally or needed a live fetch
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas.core import Dimension, EvidenceChunk


def _chunk(text, doi):
    return EvidenceChunk(node_id="", text=text, doi=doi, source="openalex",
                         dimension=Dimension.SOCIO, stakeholder_ids=["S1"],
                         citation_count=3)


A = ("Shared communal rituals measurably increase group cohesion and cooperative "
     "behaviour among participants over sustained periods.")
B = ("Economic scarcity reduces measured prosocial behaviour in controlled laboratory "
     "settings across several independent replications.")


def test_vectordb_refuses_duplicates(monkeypatch, tmp_path):
    import src.storage.vectordb as vdb_mod
    monkeypatch.setattr(vdb_mod, "VECTORDB_DIR", tmp_path)

    vdb = vdb_mod.VectorDB()
    vdb.add_chunks([_chunk(A, "10.1/a")])
    vdb.add_chunks([_chunk(A, "10.1/a")])          # exact repeat
    vdb.add_chunks([_chunk("  " + A + " ", "10.1/A")])  # whitespace + case variant
    assert vdb.size(Dimension.SOCIO) == 1, "same paper + same text must store once"

    vdb.add_chunks([_chunk(B, "10.1/b")])
    assert vdb.size(Dimension.SOCIO) == 2, "genuinely different evidence must still store"

    rows = [json.loads(l) for l in (tmp_path / "SOCIO.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 2, "metadata rows must stay aligned with the index"


def test_dedupe_survives_reload(monkeypatch, tmp_path):
    """The seen-set is rebuilt on load, so a later process cannot re-append a duplicate."""
    import src.storage.vectordb as vdb_mod
    monkeypatch.setattr(vdb_mod, "VECTORDB_DIR", tmp_path)

    vdb_mod.VectorDB().add_chunks([_chunk(A, "10.1/a")])
    fresh = vdb_mod.VectorDB()                      # simulates the next run
    fresh.add_chunks([_chunk(A, "10.1/a")])
    assert fresh.size(Dimension.SOCIO) == 1


def test_content_key_normalises():
    from src.storage.vectordb import content_key
    k1 = content_key({"doi": "10.1/X", "text": "a  b\nc"})
    k2 = content_key({"doi": "10.1/x", "text": " a b c "})
    assert k1 == k2
    assert content_key({"doi": "10.1/x", "text": "a b c"}) != \
           content_key({"doi": "10.1/y", "text": "a b c"})


def test_retrieval_telemetry_records_local_vs_live(monkeypatch, tmp_path):
    """used_live was computed and thrown away; it must now reach disk."""
    import src.services.telemetry as tel
    monkeypatch.setattr(tel, "OUTPUTS_DIR", tmp_path)

    tel.set_context("main", "001")
    tel.record_retrieval(premise_id="p1", dimension="SOCIO", local_hits=5,
                         local_relevant=0, used_live=False, candidates=5,
                         admitted=0, rejected={"off_topic": 5}, mean_novelty=0.44)
    tel.record_retrieval(premise_id="p2", dimension="PSYCH", local_hits=1,
                         local_relevant=1, used_live=True, candidates=9,
                         admitted=3, rejected={"low_novelty": 6}, mean_novelty=0.71)

    path = tmp_path / "main" / "telemetry" / "retrieval.jsonl"
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(recs) == 2
    r0 = recs[0]
    assert r0["sample_id"] == "001" and r0["variant"] == "main"
    assert r0["used_live"] is False and r0["local_hits"] == 5
    # the starvation signature: local satisfied top_k, yet nothing survived the gate
    assert r0["local_relevant"] == 0 and r0["admitted"] == 0
    assert r0["rejected"] == {"off_topic": 5}
    assert recs[1]["used_live"] is True
