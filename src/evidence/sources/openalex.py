"""OpenAlex (~250M works, abstracts + citation counts + concepts).

Free, but keyless requests now draw on a small daily budget shared by the whole IP;
a free key (OPENALEX_API_KEY) has its own. The key goes in a header, not the URL, so
it never reaches the failure messages, which print the URL.
"""
from __future__ import annotations

from src.evidence.sources.base import RawDoc, SourceClient, clean_markup

WORKS = "https://api.openalex.org/works"


def _deinvert_abstract(inv: dict | None) -> str:
    """OpenAlex stores abstracts as an inverted index; reconstruct the text."""
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


class OpenAlexClient(SourceClient):
    name = "openalex"

    def __init__(self, concept_filter: str | None = None):
        # e.g. "psychology" or "sociology" to bias the corpus toward a discipline.
        self.concept_filter = concept_filter

    def fetch(self, query: str, max_results: int = 5) -> list[RawDoc]:
        from config.settings import get_settings
        # Bias toward the discipline by folding it into the search terms (the
        # `concepts.display_name.search` FILTER is not valid on this endpoint).
        # Keep the string lean — OpenAlex rejects overly long searches.
        base = f"{self.concept_filter} {query}" if self.concept_filter else query
        q = " ".join(base.split())[:180]
        params = {
            "search": q, "per-page": max_results,
            "mailto": get_settings().contact_email,
        }
        key = get_settings().openalex_api_key
        r = self._get(WORKS, params, headers={"Authorization": f"Bearer {key}"} if key else None)
        if r is None:
            return []
        try:
            results = r.json().get("results", [])
        except Exception:
            return []
        docs: list[RawDoc] = []
        for it in results:
            venue = None
            src = (it.get("primary_location") or {}).get("source") or {}
            if src:
                venue = src.get("display_name")
            doc = RawDoc(
                title=clean_markup(it.get("title")),
                abstract=clean_markup(_deinvert_abstract(it.get("abstract_inverted_index"))),
                source=self.name,
                doi=(it.get("doi") or "").replace("https://doi.org/", "") or None,
                venue=venue,
                year=it.get("publication_year"),
                citation_count=int(it.get("cited_by_count") or 0),
                ext_id=it.get("id"),
            )
            if doc.has_content():
                docs.append(doc)
        return docs
