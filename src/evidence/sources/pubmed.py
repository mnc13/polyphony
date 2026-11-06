"""PubMed via NCBI E-utilities (free; NCBI_API_KEY optional, raises rate limit)."""
from __future__ import annotations

import xml.etree.ElementTree as ET

from config.settings import get_settings
from src.evidence.sources.base import RawDoc, SourceClient

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _all_text(el) -> str:
    """
    Every piece of text under `el`, inline markup included.

    `el.text` stops at the first child element, and PubMed puts <i>, <sup>, <sub> and
    <b> inside titles and abstracts routinely, so an abstract with a single italic
    species name or a superscript statistic was cut at that point -- one stored
    abstract jumped from "(F[2,51] = 5.5," straight to its conclusions.
    """
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def _article_doi(art) -> str | None:
    """
    The article's own DOI.

    `.//ArticleId` also matches the ArticleIds inside ReferenceList, i.e. the DOIs of
    the papers this one cites, and the old loop kept the LAST match -- so a record
    could be stored under a cited reference's DOI. Only the article-level id list and
    the ELocationID are the article's own.
    """
    for eid in art.findall("./PubmedData/ArticleIdList/ArticleId"):
        if eid.get("IdType") == "doi" and (eid.text or "").strip():
            return eid.text.strip()
    for loc in art.findall("./MedlineCitation/Article/ELocationID"):
        if loc.get("EIdType") == "doi" and (loc.text or "").strip():
            return loc.text.strip()
    return None


class PubMedClient(SourceClient):
    name = "pubmed"

    def fetch(self, query: str, max_results: int = 5) -> list[RawDoc]:
        key = get_settings().ncbi_api_key
        params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
        if key:
            params["api_key"] = key
        r = self._get(ESEARCH, params)
        if r is None:
            return []
        try:
            ids = r.json().get("esearchresult", {}).get("idlist", [])
        except Exception:
            return []
        if not ids:
            return []
        fparams = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
        if key:
            fparams["api_key"] = key
        fr = self._get(EFETCH, fparams)
        if fr is None:
            return []
        return self._parse(fr.text)

    def _parse(self, xml_text: str) -> list[RawDoc]:
        docs: list[RawDoc] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return docs
        for art in root.findall(".//PubmedArticle"):
            title = _all_text(art.find(".//ArticleTitle"))
            # Structured abstracts carry one AbstractText per section (Background,
            # Methods, ...); keep the label so the sections stay readable once joined.
            parts = []
            for t in art.findall(".//Abstract/AbstractText"):
                body = _all_text(t)
                label = (t.get("Label") or "").strip()
                if body:
                    parts.append(f"{label}: {body}" if label else body)
            abstract = " ".join(parts)
            year = art.findtext(".//PubDate/Year")
            venue = (art.findtext(".//Journal/Title") or "").strip() or None
            # Direct child only: CommentsCorrections nests other articles' PMIDs.
            pmid = (art.findtext("./MedlineCitation/PMID") or "").strip() or None
            doi = _article_doi(art)
            doc = RawDoc(
                title=title, abstract=abstract, source=self.name,
                doi=doi, venue=venue,
                year=int(year) if year and year.isdigit() else None,
                ext_id=pmid,
            )
            if doc.has_content():
                docs.append(doc)
        return docs
