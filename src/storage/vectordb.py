"""
P1 — Persistent vector database (the growing evidence corpus).

One FAISS index per survivability dimension (NEURO/PSYCH/SOCIO/EVOBIO). Cosine via
IndexFlatIP over normalized embeddings. Every P-MER fetch is written here and kept,
so the corpus scales across cases and later cases retrieve locally before hitting
any live API (the "local-first" property).

Persistence: each dimension has <dim>.index (FAISS) + <dim>.jsonl (metadata,
line-aligned with vector rows). Append-only in spirit; rebuild-on-load is avoided
by persisting after each add.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from config.settings import VECTORDB_DIR
from src.schemas.core import Dimension, EvidenceChunk
from src.services.embedder import embed


def corpus_chunk_count(path: Path | None = None) -> int:
    """Chunks stored in a corpus directory (metadata rows), without loading FAISS."""
    d = Path(path) if path is not None else VECTORDB_DIR
    n = 0
    for f in d.glob("*.jsonl"):
        with open(f, encoding="utf-8") as fh:
            n += sum(1 for line in fh if line.strip())
    return n


def content_key(meta: dict) -> str:
    """Identity of a stored chunk: same paper + same text is the same evidence."""
    return f"{(meta.get('doi') or '').strip().lower()}||{' '.join((meta.get('text') or '').split())}"


def _read_meta(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break           # a line cut off mid-write: it and anything after it go
    return rows


def _write_meta(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for m in rows:
            fh.write(json.dumps(m) + "\n")
    os.replace(tmp, path)


def _write_index(index, path: Path) -> None:
    import faiss
    tmp = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)


def _truncated(index, n: int):
    """The first n vectors of a flat index, as a new index."""
    import faiss
    out = faiss.IndexFlatIP(index.d)
    if n:
        out.add(index.reconstruct_n(0, n))
    return out


def corpus_snapshot(path: Path | None = None) -> dict[str, int]:
    """Rows per dimension file, to roll a failed case's additions back to."""
    d = Path(path) if path is not None else VECTORDB_DIR
    return {f.stem: len(_read_meta(f)) for f in d.glob("*.jsonl")}


# run_batch rolls a failed case's evidence back in an except block, which a killed
# process never reaches. On 2026-10-23 the machine was powered off mid-case on two
# workers at once and both corpora kept the dead attempts' chunks (43 and 21), which
# their retries would have found locally. The pre-case snapshot is therefore also
# written to disk, and a batch that finds one left behind rolls back to it first.
_INFLIGHT = "inflight.json"


def mark_inflight(sample_id: str, variant: str, snapshot: dict[str, int],
                  path: Path | None = None) -> None:
    d = Path(path) if path is not None else VECTORDB_DIR
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (_INFLIGHT + ".tmp")
    tmp.write_text(json.dumps({"sample": sample_id, "variant": variant,
                               "snapshot": snapshot}), encoding="utf-8")
    os.replace(tmp, d / _INFLIGHT)


def clear_inflight(path: Path | None = None) -> None:
    d = Path(path) if path is not None else VECTORDB_DIR
    (d / _INFLIGHT).unlink(missing_ok=True)


def recover_inflight(variant: str, processed: set[str], path: Path | None = None) -> str | None:
    """
    Undo a case that was killed after its pre-case snapshot was marked. Returns a line
    to print, or None when nothing was in flight. A case that reached the ledger before
    the kill is kept (only its marker is cleared).
    """
    d = Path(path) if path is not None else VECTORDB_DIR
    f = d / _INFLIGHT
    if not f.exists():
        return None
    data = json.loads(f.read_text(encoding="utf-8"))
    if data.get("variant") != variant:
        raise SystemExit(
            f"{d} holds an interrupted case of variant {data.get('variant')!r}, not "
            f"{variant!r}. Resume that variant first (each variant needs its own --worker).")
    sid = data.get("sample")
    if sid in processed:
        clear_inflight(d)
        return None
    removed = rollback_corpus(data["snapshot"], d)
    clear_inflight(d)
    return (f"[recover] sample {sid} was interrupted last time (process killed); rolled "
            f"back {sum(removed.values())} corpus chunks it had added")


def rollback_corpus(snapshot: dict[str, int], path: Path | None = None) -> dict[str, int]:
    """
    Drop the chunks added since `snapshot` (a case that failed part-way). Without this
    its retry finds its own dead attempt's evidence locally -- including chunks fetched
    in that attempt's debate rounds -- and skips live fetches the first attempt made.
    Returns the rows removed per dimension.
    """
    import faiss
    d = Path(path) if path is not None else VECTORDB_DIR
    removed: dict[str, int] = {}
    for meta_path in d.glob("*.jsonl"):
        keep = snapshot.get(meta_path.stem, 0)
        rows = _read_meta(meta_path)
        if len(rows) <= keep:
            continue
        index_path = meta_path.with_suffix(".index")
        if index_path.exists():
            index = faiss.read_index(str(index_path))
            _write_index(_truncated(index, min(keep, index.ntotal)), index_path)
        _write_meta(meta_path, rows[:keep])
        removed[meta_path.stem] = len(rows) - keep
    return removed


class DimensionIndex:
    def __init__(self, dimension: str):
        import faiss
        self.dimension = dimension
        self.dir = VECTORDB_DIR
        self.index_path = self.dir / f"{dimension}.index"
        self.meta_path = self.dir / f"{dimension}.jsonl"
        self._faiss = faiss
        self.meta: list[dict] = []
        self._seen: set[str] = set()
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.meta = _read_meta(self.meta_path)
            if self.index.ntotal != len(self.meta):
                # A run killed between the two writes. Keep the rows both files hold;
                # a misaligned pair would hand back the wrong chunk for a vector.
                n = min(self.index.ntotal, len(self.meta))
                print(f"[vectordb] {dimension}: index has {self.index.ntotal} vectors, "
                      f"metadata {len(self.meta)} rows; truncating both to {n}")
                self.index = _truncated(self.index, n)
                self.meta = self.meta[:n]
                _write_meta(self.meta_path, self.meta)
                _write_index(self.index, self.index_path)
            self._seen = {content_key(m) for m in self.meta}
        else:
            from config.settings import get_settings
            self.index = faiss.IndexFlatIP(get_settings().hp.embedding_dim)

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, vectors: np.ndarray, metas: list[dict]) -> None:
        if vectors.shape[0] == 0:
            return
        self.index.add(vectors)
        self.meta.extend(metas)
        self._persist(vectors_only=metas)

    def _persist(self, vectors_only: list[dict]) -> None:
        # Metadata first, then the index swapped in whole: an interruption can leave
        # extra metadata rows (repaired on load) but never a half-written index.
        with open(self.meta_path, "a", encoding="utf-8") as fh:
            for m in vectors_only:
                fh.write(json.dumps(m) + "\n")
        _write_index(self.index, self.index_path)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[dict, float]]:
        if self.index.ntotal == 0:
            return []
        scores, idxs = self.index.search(query_vec.reshape(1, -1), min(top_k, self.index.ntotal))
        out: list[tuple[dict, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append((self.meta[idx], float(score)))
        return out


class VectorDB:
    """Facade over the four dimension indices."""

    def __init__(self):
        self._indices: dict[str, DimensionIndex] = {}

    def _index(self, dimension: str) -> DimensionIndex:
        if dimension not in self._indices:
            self._indices[dimension] = DimensionIndex(dimension)
        return self._indices[dimension]

    def add_chunks(self, chunks: list[EvidenceChunk]) -> None:
        """Group by dimension, embed, persist. Chunks without a dimension are skipped."""
        by_dim: dict[str, list[EvidenceChunk]] = {}
        for c in chunks:
            if c.dimension is None:
                continue
            by_dim.setdefault(c.dimension.value, []).append(c)
        for dim, cs in by_dim.items():
            idx = self._index(dim)
            # Novelty is judged against the DECISION's evidence pool, not this corpus,
            # so re-running a case re-admits the same papers. Without this guard the
            # store accumulates duplicate rows of identical evidence indefinitely.
            fresh, metas = [], []
            for c in cs:
                m = c.model_dump(mode="json")
                k = content_key(m)
                if k in idx._seen:
                    continue
                idx._seen.add(k)
                fresh.append(c)
                metas.append(m)
            if not fresh:
                continue
            vecs = embed([c.text for c in fresh])
            idx.add(vecs, metas)

    def search(self, query: str, dimension: Dimension, top_k: int = 5) -> list[EvidenceChunk]:
        qv = embed([query])[0]
        hits = self._index(dimension.value).search(qv, top_k)
        results: list[EvidenceChunk] = []
        for meta, score in hits:
            chunk = EvidenceChunk.model_validate(meta)
            results.append(chunk)
        return results

    def size(self, dimension: Dimension) -> int:
        return len(self._index(dimension.value))

    def total_size(self) -> int:
        return sum(len(self._index(d.value)) for d in Dimension)
