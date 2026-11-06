"""Europe PMC REST (free; open-access full text + abstracts)."""
from __future__ import annotations

from src.evidence.sources.base import RawDoc, SourceClient, clean_markup

SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class EuropePMCClient(SourceClient):
    name = "europepmc"

    def fetch(self, query: str, max_results: int = 5) -> list[RawDoc]:
        params = {
            "query": query, "format": "json", "pageSize": max_results,
            "resultType": "core",
        }
        r = self._get(SEARCH, params)
        if r is None:
            return []
        try:
            results = r.json().get("resultList", {}).get("result", [])
        except Exception:
            return []
        docs: list[RawDoc] = []
        for it in results:
            year = it.get("pubYear")
            doc = RawDoc(
                title=clean_markup(it.get("title")),
                abstract=clean_markup(it.get("abstractText")),
                source=self.name,
                doi=it.get("doi"),
                venue=(it.get("journalTitle") or None),
                year=int(year) if year and str(year).isdigit() else None,
                citation_count=int(it.get("citedByCount") or 0),
                ext_id=it.get("id"),
            )
            if doc.has_content():
                docs.append(doc)
        return docs
