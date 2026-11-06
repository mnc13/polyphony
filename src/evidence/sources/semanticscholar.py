"""Semantic Scholar Graph API (free; citation graph + influence signals)."""
from __future__ import annotations

from config.settings import get_settings
from src.evidence.sources.base import RawDoc, SourceClient, clean_markup

SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarClient(SourceClient):
    name = "semanticscholar"
    _warned = False

    def fetch(self, query: str, max_results: int = 5) -> list[RawDoc]:
        key = get_settings().semantic_scholar_api_key
        if not key:
            # Unauthenticated requests to this endpoint are rate-limited hard enough to
            # be useless: on the first six-case run S2 contributed 0 of 195 evidence
            # nodes and logged a 429 for every premise, which buried real retrieval
            # errors in noise. Skipping costs nothing and keeps the log readable. A free
            # key from semanticscholar.org re-enables the source.
            if not SemanticScholarClient._warned:
                SemanticScholarClient._warned = True
                print("[semanticscholar] no SEMANTIC_SCHOLAR_API_KEY set - source skipped "
                      "(unauthenticated requests are rate-limited to uselessness)")
            return []
        params = {
            "query": query, "limit": max_results,
            "fields": "title,abstract,year,venue,citationCount,externalIds",
        }
        r = self._get(SEARCH, params, headers={"x-api-key": key})
        if r is None:
            return []
        try:
            data = r.json().get("data", []) or []
        except Exception:
            return []
        docs: list[RawDoc] = []
        for it in data:
            ext = it.get("externalIds") or {}
            doc = RawDoc(
                title=clean_markup(it.get("title")),
                abstract=clean_markup(it.get("abstract")),
                source=self.name,
                doi=ext.get("DOI"),
                venue=(it.get("venue") or None),
                year=it.get("year"),
                citation_count=int(it.get("citationCount") or 0),
                ext_id=it.get("paperId"),
            )
            if doc.has_content():
                docs.append(doc)
        return docs
