"""CrossRef (free; DOI resolution + venue/year metadata for provenance)."""
from __future__ import annotations

from config.settings import get_settings
from src.evidence.sources.base import RawDoc, SourceClient, clean_markup

WORKS = "https://api.crossref.org/works"


class CrossRefClient(SourceClient):
    name = "crossref"

    def fetch(self, query: str, max_results: int = 5) -> list[RawDoc]:
        params = {
            "query": query, "rows": max_results,
            "mailto": get_settings().contact_email,
            "select": "title,abstract,container-title,issued,DOI,is-referenced-by-count",
        }
        r = self._get(WORKS, params)
        if r is None:
            return []
        try:
            items = r.json().get("message", {}).get("items", [])
        except Exception:
            return []
        docs: list[RawDoc] = []
        for it in items:
            title = clean_markup(" ".join(it.get("title") or []))
            venue = " ".join(it.get("container-title") or []).strip() or None
            year = None
            issued = (it.get("issued") or {}).get("date-parts") or [[None]]
            if issued and issued[0] and issued[0][0]:
                year = issued[0][0]
            # CrossRef abstracts are JATS XML (<jats:p>...</jats:p>), not text.
            abstract = clean_markup(it.get("abstract"))
            doc = RawDoc(
                title=title, abstract=abstract, source=self.name,
                doi=it.get("DOI"), venue=venue, year=year,
                citation_count=int(it.get("is-referenced-by-count") or 0),
            )
            # CrossRef often lacks abstracts; it is a provenance/fallback source.
            if title:
                docs.append(doc)
        return docs
