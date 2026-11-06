"""
P3 — Typed Evidence Graph (TEG).

The shared evidence pool for one decision's debate, as a graph. Nodes are admitted
EvidenceChunks (typed with dimension / stakeholder / DOI / novelty). Edges are
contradiction links between nodes, detected by the NLI model — not an LLM. The TEG
is append-only during debate: P-MER adds nodes; nothing is removed, so every
citation stays checkable.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from config.settings import get_settings
from src.schemas.core import EvidenceChunk
from src.services import nli

_CONTRADICTION_THRESHOLD = 0.55

# NLI judges entailment between PROPOSITIONS. A bare title ("Cooperation, Coordination,
# and Collective Action") is a noun phrase with no proposition in it, so scoring one
# against another produces confident nonsense — observed 0.945/0.987 "contradictions"
# between two titles that plainly do not contradict. Only score text substantial enough
# to assert something, and never score a source against itself.
_MIN_CONTRADICTION_TOKENS = 12


def _scorable(text: str | None) -> bool:
    """True if `text` is a claim we can meaningfully run NLI on (not a bare title)."""
    if not text:
        return False
    return len(text.split()) >= _MIN_CONTRADICTION_TOKENS


class TypedEvidenceGraph:
    def __init__(self, case_id: str, decision_id: str):
        self.case_id = case_id
        self.decision_id = decision_id
        self.g = nx.DiGraph()
        self._counter = 0
        self._vecs: dict = {}      # node_id -> embedding, for the contradiction topic floor

    def _vec(self, node_id: str, text: str):
        if node_id not in self._vecs:
            from src.services.embedder import embed
            self._vecs[node_id] = embed([text])[0]
        return self._vecs[node_id]

    def _new_node_id(self) -> str:
        self._counter += 1
        return f"TEG-{self.decision_id}-{self._counter}"

    def add(self, chunk: EvidenceChunk) -> str:
        """Add a chunk, wiring contradiction edges vs existing nodes. Returns node id."""
        if not chunk.node_id:
            chunk.node_id = self._new_node_id()
        self.g.add_node(chunk.node_id, **chunk.model_dump(mode="json"))
        if not _scorable(chunk.text):
            return chunk.node_id
        dim = chunk.dimension.value if chunk.dimension else None
        topic_floor = get_settings().hp.contradiction_topic_cosine
        new_vec = self._vec(chunk.node_id, chunk.text)
        for other_id, data in list(self.g.nodes(data=True)):
            if other_id == chunk.node_id:
                continue
            if data.get("dimension") != dim:
                continue
            if not _scorable(data.get("text")):
                continue
            # Same paper retrieved twice cannot contradict itself; without this the
            # same DOI pair scores identically under every decision that retrieves it.
            # (DOIs compared case-insensitively: PubMed keeps the publisher's case,
            # OpenAlex lowercases.)
            if chunk.doi and (data.get("doi") or "").lower() == chunk.doi.lower():
                continue
            # DOI-less records (admitted on a PMID / OpenAlex id) are the same paper
            # when their source id matches; two title-prefixed windows of one abstract
            # pass the topic floor against each other easily.
            if chunk.ext_id and data.get("ext_id") == chunk.ext_id:
                continue
            if data["text"].strip() == chunk.text.strip():
                continue
            # Two abstracts can only contradict each other if they are about the same
            # thing. Sharing a dimension is not that: on the post-fix run all 25 edges
            # linked the same four unrelated NEURO abstracts (panic-disorder steroids,
            # a brain-ecology editorial, moral-dilemma framing, mild TBI), because NLI
            # resolves two unrelated propositions toward "contradiction" readily.
            if float(new_vec @ self._vec(other_id, data["text"])) < topic_floor:
                continue
            # Contradiction must hold in BOTH directions — a one-way NLI hit is
            # usually asymmetric specificity, not genuine disagreement.
            fwd = nli.contradicts(data["text"], chunk.text)
            if fwd < _CONTRADICTION_THRESHOLD:
                continue
            rev = nli.contradicts(chunk.text, data["text"])
            c = min(fwd, rev)
            if c >= _CONTRADICTION_THRESHOLD:
                self.g.add_edge(other_id, chunk.node_id, type="contradicts", score=round(c, 3))
        return chunk.node_id

    def node(self, node_id: str) -> dict | None:
        return self.g.nodes.get(node_id)

    def texts(self) -> list[str]:
        return [d["text"] for _, d in self.g.nodes(data=True)]

    def __len__(self) -> int:
        return self.g.number_of_nodes()

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "decision_id": self.decision_id,
            "nodes": [self.g.nodes[n] for n in self.g.nodes],
            "edges": [
                {"from": u, "to": v, **self.g.edges[u, v]}
                for u, v in self.g.edges
            ],
        }

    def export(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
