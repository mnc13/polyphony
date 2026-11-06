"""
P3 — Progressive Moral Evidence Retrieval (P-MER) orchestrator.

Ties the evidence layer together for one retrieval call:

  premise (+ stakeholder desc + dimension)  ->  query
     -> LOCAL-FIRST: search the growing vector DB for this dimension
     -> if fewer than `local_sufficient_hits` local hits are RELEVANT to the premise,
        fetch LIVE from the premise's sources
     -> chunk -> embed -> ADMISSION GATE (novelty + relevance + credibility)
     -> persist admitted chunks to the vector DB (corpus grows)
     -> add admitted chunks to the decision's TEG (with contradiction edges)
  returns admitted chunks + novelty stats (for the debate stopping logic).

Source routing (by dimension, with discipline overrides):
  NEURO, EVOBIO -> PubMed + Europe PMC
  PSYCH         -> OpenAlex(psychology) + Semantic Scholar
  SOCIO         -> OpenAlex(sociology)  + Semantic Scholar
  economics     -> OpenAlex(economics)  + Semantic Scholar
  ecology       -> OpenAlex(ecology)    + Europe PMC
  (CrossRef is a metadata/provenance fallback, queried when others come back empty.)
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field

import numpy as np

from config.settings import get_settings
from src.evidence.chunker import chunk_doc
from src.evidence.gate import AdmissionGate, Candidate
from src.evidence.sources.base import RawDoc, failure_snapshot
from src.evidence.sources.crossref import CrossRefClient
from src.evidence.sources.europepmc import EuropePMCClient
from src.evidence.sources.openalex import OpenAlexClient
from src.evidence.sources.pubmed import PubMedClient
from src.evidence.sources.semanticscholar import SemanticScholarClient
from src.evidence.teg import TypedEvidenceGraph
from src.schemas.core import Dimension, Discipline, EvidenceChunk, Premise
from src.services.embedder import embed
from src.storage.vectordb import VectorDB

_DIM_KEYWORDS = {
    Dimension.NEURO: "neuroscience stress cognition",
    Dimension.PSYCH: "psychology behavior wellbeing",
    Dimension.SOCIO: "sociology social cohesion institutions",
    Dimension.EVOBIO: "evolutionary biology cooperation fitness",
}

# Three of the seven disciplines share a dimension with another discipline (economics
# -> SOCIO, health_science -> NEURO, ecology -> EVOBIO) and were queried with that other
# field's vocabulary and sources: economics premises went to OpenAlex prefixed with
# "sociology". None of the 49 economics or 8 ecology testimony claims on the post-fix
# run cited anything. The dimension still decides where a chunk is stored and scored;
# the discipline decides how the literature is searched.
_DISC_KEYWORDS = {
    Discipline.economics: "economics incentives welfare",
    Discipline.health_science: "health outcomes mortality",
    Discipline.ecology: "ecology resources sustainability",
}


def _field_keywords(premise: Premise) -> str:
    return _DISC_KEYWORDS.get(premise.discipline) or _DIM_KEYWORDS.get(premise.dimension, "")


# Literature APIs differ in how they match a query. PubMed, EuropePMC and OpenAlex are
# term matchers: a full premise sentence ("Undisclosed use of an imperfect memory-implant
# protocol raises the probability that S1 suffers licensure loss ...") matches zero
# records, because no indexed work contains that construction. CrossRef does fuzzy
# bibliographic matching and so always returns something -- titles, usually without an
# abstract. Sending sentences therefore silently routed almost the whole corpus through
# the CrossRef fallback: on the first six-case run, 184 of 195 evidence nodes came from
# CrossRef and 163 of 195 were title-only, which is why claims so often reported that the
# evidence did not settle their premise. Querying with content terms instead returns real
# abstracts from the primary sources.
_STOP = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with
without within into onto about over under between among during against as is are was
were be been being am do does did doing have has had having will would shall should can
could may might must not no nor only own same so too very just also however therefore
thus hence raises raise increases increase decreases decrease probability likelihood
risk chance causes cause caused leads lead led results result resulting suffers suffer
suffered experiences experience its their his her our your they them it he she we you i
who whom whose which what when where why how more most less least many much few several
such other others any all both each either neither per via due given across through
because become becomes became known compared versus relative rather whereas while
""".split())

# Stakeholder placeholders (S1, S2...), decision/premise ids, bare numbers: identifiers
# from this pipeline's own vocabulary, meaningless to a literature index.
_ID_TOKEN = re.compile(r"^(?:[sSdDpP]\d+|\d+|TEG[\w-]*)$")
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]+")

MAX_QUERY_TERMS = 6

# Words debaters use to ASK for evidence, which say nothing about what to look for.
_GAP_BOILERPLATE = frozenset("""
need needs needed still direct directly evidence evidential whether data study studies
show shows showing showed support supporting supports supported find finding findings
would could close case cases specific empirical relevant further additional more
measure measuring measured quantify quantifying quantified rate rates effect effects
claim claims premise premises node nodes literature research particular actual actually
""".split())


def keywords(text: str, limit: int = MAX_QUERY_TERMS) -> list[str]:
    """
    Content terms from a premise, in order of first appearance, deduplicated.

    Order is preserved rather than ranked by frequency: a premise states its subject
    early, and these strings are one sentence long, so frequency carries no signal.
    Hyphens are kept ("memory-implant"); indexes tokenise them themselves.
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in _WORD.finditer(text or ""):
        w = m.group(0)
        lw = w.lower().strip("-")
        if len(lw) < 4 or lw in _STOP or lw in seen or _ID_TOKEN.match(w):
            continue
        seen.add(lw)
        out.append(lw)
        if len(out) >= limit:
            break
    return out


def _sources_for(dim: Dimension, discipline: Discipline | None = None) -> list:
    # Semantic Scholar answers only with an API key, so whether a key happened to be set
    # used to decide, unrecorded, whether PSYCH/SOCIO/economics premises had a second
    # source. It is now an explicit experiment setting (off: the pilot and main ran
    # without it), so two machines cannot silently search different literatures.
    pm, epmc = PubMedClient(), EuropePMCClient()
    s2 = [SemanticScholarClient()] if get_settings().hp.use_semantic_scholar else []
    if discipline == Discipline.economics:
        return [OpenAlexClient(concept_filter="economics"), *s2]
    if discipline == Discipline.ecology:
        return [OpenAlexClient(concept_filter="ecology"), epmc]
    if dim in (Dimension.NEURO, Dimension.EVOBIO):
        return [pm, epmc]
    if dim == Dimension.PSYCH:
        return [OpenAlexClient(concept_filter="psychology"), *s2]
    if dim == Dimension.SOCIO:
        return [OpenAlexClient(concept_filter="sociology"), *s2]
    return [pm, *s2]


def _paper_key(ch) -> str:
    """One paper, whichever chunk of it: DOI, else the source's own id, else the text."""
    return ((ch.doi or "").strip().lower() or (ch.ext_id or "").strip()
            or " ".join((ch.text or "").split())[:200])


@dataclass
class PMERResult:
    admitted: list[EvidenceChunk] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (text_head, reason)
    # Mean novelty of the chunks ADMITTED by this call; 0.0 when nothing was admitted.
    # It used to average over every candidate, rejected ones included, and default to
    # 1.0 with no candidates at all, so a call that added nothing still read as ~0.8
    # "novel" and the debate's novelty stop (< 0.10 for two rounds) could never fire.
    mean_novelty: float = 0.0
    used_live: bool = False


class PMER:
    def __init__(self, vectordb: VectorDB | None = None):
        self.vdb = vectordb or VectorDB()
        self.gate = AdmissionGate()
        self.hp = get_settings().hp

    # Belt and braces: even a term query should not grow unbounded.
    MAX_QUERY_CHARS = 200

    def build_query_ladder(self, premise: Premise, stakeholder_desc: str,
                           hint: str = "") -> list[str]:
        """
        Queries from most to least specific, to be tried in order until one returns hits.

        Content terms alone are not enough. PubMed and EuropePMC conjoin terms, so a
        premise's own vocabulary ("destabilized autobiographical memories impair accurate
        recall") still matches nothing: every term is individually common, but no work
        contains all six. Measured on real premises from the first run, the six-term query
        returned 0 documents while the same premise cut to three terms returned 8, every
        one of them with an abstract.

        That measurement was taken WITHOUT field keywords, and the ladder used to append
        the dimension's three keywords to every rung, so rung 0 sent up to ten terms that
        PubMed and Europe PMC AND together. Field vocabulary now enters only once the
        premise's own terms have failed, and then as a single term before the field
        keywords alone. The field is the discipline's where one is defined (economics,
        health science, ecology), else the dimension's.

        `hint` is the debaters' evidential gap for this premise during a debate round;
        when given, its terms lead, so a progressive round asks for what was missing
        instead of replaying the query that already ran.

        So specificity is relaxed in steps rather than abandoned: take the best hits that
        exist, and only reach the title-only CrossRef fallback when the field itself
        returns nothing. Precision is not lost by broadening -- the admission gate scores
        every candidate's cosine relevance against the premise and drops what does not
        pertain, so a broad rung contributes only its genuinely relevant documents.
        """
        field_kw = _field_keywords(premise)
        terms = keywords(premise.text)
        stake = keywords(stakeholder_desc.split(";")[0], limit=1)
        rungs = []
        if hint:
            # Gap sentences are mostly request boilerplate ("I still need direct
            # evidence whether ..."); on the stored transcripts a bare gap rung came out
            # as "need direct evidence" or "still need direct", which full-text indexes
            # happily answer -- and the first rung that answers wins. So boilerplate is
            # dropped, and a gap rung always keeps the premise's own leading terms.
            gap = [w for w in keywords(hint, limit=12)
                   if w not in _GAP_BOILERPLATE and w not in terms][:3]
            if gap:
                rungs.append(" ".join(gap + terms[:2]))
        rungs += [
            " ".join(terms + stake),                     # the premise, as stated
            " ".join(terms[:3]),                         # its three leading terms
            " ".join(terms[:2] + field_kw.split()[:1]),  # two terms, anchored to the field
            field_kw,                                    # the field alone
        ]
        out: list[str] = []
        for r in rungs:
            r = " ".join(r.split())
            if len(r) > self.MAX_QUERY_CHARS:
                r = r[: self.MAX_QUERY_CHARS].rsplit(" ", 1)[0]
            if r and r not in out:
                out.append(r)
        return out

    def build_query(self, premise: Premise, stakeholder_desc: str) -> str:
        """The most specific rung. Kept for callers that want a single query string."""
        return self.build_query_ladder(premise, stakeholder_desc)[0]

    def _fetch_rung(self, clients: list, rung: str, top_k: int) -> list[RawDoc] | None:
        """
        One rung against every source. None when it came back empty only because every
        source failed in transport, on a second try too; [] when the sources answered
        and simply had nothing.
        """
        import time
        for attempt in range(2):
            before = failure_snapshot()
            docs: list[RawDoc] = []
            for client in clients:
                docs.extend(client.fetch(rung, max_results=top_k))
            after = failure_snapshot()
            failed = sum(1 for c in clients
                         if after.get(getattr(c, "name", ""), 0) > before.get(getattr(c, "name", ""), 0))
            if docs or failed < len(clients):
                return docs
            if attempt == 0:
                time.sleep(5)
        return None

    def _any_relevant(self, docs: list[RawDoc], premise_vec: np.ndarray) -> bool:
        chunks = [c for d in docs for c in chunk_doc(d)]
        if not chunks:
            return False
        return bool(np.max(embed(chunks) @ premise_vec) >= self.gate.relevance_threshold)

    def retrieve(
        self,
        premise: Premise,
        stakeholder_desc: str,
        teg: TypedEvidenceGraph,
        top_k: int | None = None,
        hint: str = "",
        force_live: bool = False,
    ) -> PMERResult:
        """
        One retrieval call for one premise into one decision's TEG.

        hint       : the debaters' evidential gap for this premise (debate rounds).
        force_live : skip the local-sufficiency shortcut. Debate rounds retrieve only
                     for premises the debate left uncovered, which is itself evidence the
                     local corpus did not cover them.
        """
        top_k = top_k or self.hp.per_corpus_top_k
        dim = premise.dimension
        failures_before = failure_snapshot()

        # Pool for novelty = current TEG nodes (per-decision pool).
        pool_texts = teg.texts()
        pool_vecs = embed(pool_texts) if pool_texts else np.zeros((0, self.hp.embedding_dim), "float32")
        premise_vec = embed([premise.text])[0]  # for topical relevance in the gate

        # 1. LOCAL-FIRST -- semantic search of the persistent corpus with the premise
        # itself (plus the gap, in a debate round). The keyword ladder is built for
        # term-matching APIs; an embedding index is better served by the sentence.
        local_query = f"{hint} {premise.text}".strip() if hint else premise.text
        candidates: list[Candidate] = []
        local_hits = self.vdb.search(local_query, dim, top_k=top_k)
        for ch in local_hits:
            candidates.append(Candidate(
                text=ch.text, doi=ch.doi, citation_count=ch.citation_count,
                source=ch.source, venue=ch.venue, year=ch.year, ext_id=ch.ext_id,
            ))

        # How many local hits actually pertain to this premise. The search always
        # returns min(top_k, corpus size) results, relevant or not, so the raw count
        # says nothing about coverage once a dimension holds top_k chunks -- which is
        # how cases 003-006 of the first post-fix run made zero live fetches.
        # Counted per PAPER: one long abstract split into several windows used to count
        # as several relevant hits (one OpenAlex record was 6 of a TEG's 24 nodes).
        if local_hits:
            lv = embed([ch.text for ch in local_hits])
            rel = lv @ premise_vec >= self.gate.relevance_threshold
            local_relevant = len({_paper_key(ch) for ch, ok in zip(local_hits, rel) if ok})
        else:
            local_relevant = 0

        used_live = False
        rung_used = None
        rung_query = None
        transport_failed = False
        # 2. LIVE fetch unless the corpus already covers the premise.
        if force_live or local_relevant < self.hp.local_sufficient_hits:
            used_live = True
            docs: list[RawDoc] = []
            clients = _sources_for(dim, premise.discipline)
            # Walk the ladder only as far as needed: stop at the first rung that yields
            # a candidate the gate could admit (topical cosine >= the relevance bar), so
            # a premise whose own wording is indexable never gets broadened. Stopping at
            # the first rung with ANY hit left 28 of 49 pilot fetches at a rung whose
            # every candidate was then rejected as off topic, broader rungs untried.
            for i, rung in enumerate(self.build_query_ladder(premise, stakeholder_desc, hint)):
                # Telemetry keeps the last rung tried, so a ladder that ends with nothing
                # relevant (or in a transport failure) still says what it searched.
                rung_query = rung
                got = self._fetch_rung(clients, rung, top_k)
                if got is None:
                    # Every source failed in transport on this rung, twice. An empty
                    # answer here is not "no literature", so the ladder is not broadened
                    # on it (that silently swaps in vaguer evidence); the premise keeps
                    # what it has and the failure is recorded.
                    transport_failed = True
                    break
                docs.extend(got)
                if got and self._any_relevant(got, premise_vec):
                    rung_used, rung_query = i, rung
                    break
            if not docs and not transport_failed:
                # Last resort. CrossRef matches bibliographically and so nearly always
                # answers, but usually with a title and no abstract -- evidence that can
                # ground almost nothing. Reaching this point is worth knowing about.
                rung_query = self.build_query(premise, stakeholder_desc)
                docs.extend(CrossRefClient().fetch(rung_query, max_results=top_k))
                rung_used = "crossref" if docs else "none"
            elif rung_used is None and not transport_failed:
                rung_used = "none_relevant"
            for d in docs:
                for chunk in chunk_doc(d):
                    candidates.append(Candidate(
                        text=chunk, doi=d.doi, citation_count=d.citation_count,
                        source=d.source, venue=d.venue, year=d.year, ext_id=d.ext_id,
                    ))

        # 3. GATE + persist + add to TEG.
        result = PMERResult(used_live=used_live)
        novelties: list[float] = []
        new_chunks: list[EvidenceChunk] = []
        n_local = len(local_hits)
        local_admitted = 0
        for i, cand in enumerate(candidates):
            gr = self.gate.evaluate(cand, premise, pool_vecs, premise_vec=premise_vec)
            if not gr.admitted:
                result.rejected.append((cand.text[:80], gr.reason))
                continue
            novelties.append(gr.novelty)
            if i < n_local:
                local_admitted += 1
            chunk = EvidenceChunk(
                node_id="", text=cand.text, doi=cand.doi, source=cand.source,
                venue=cand.venue, year=cand.year, citation_count=cand.citation_count,
                dimension=dim, stakeholder_ids=list(premise.stakeholder_ids),
                novelty_score=round(gr.novelty, 3), ext_id=cand.ext_id,
            )
            node_id = teg.add(chunk)
            chunk.node_id = node_id
            new_chunks.append(chunk)
            result.admitted.append(chunk)
            # grow the novelty pool as we admit within this call
            pool_vecs = np.vstack([pool_vecs, embed([cand.text])]) if pool_vecs.shape[0] else embed([cand.text])

        # 4. Persist newly admitted chunks to the global corpus.
        if new_chunks:
            self.vdb.add_chunks(new_chunks)

        result.mean_novelty = float(np.mean(novelties)) if novelties else 0.0

        # Telemetry: was this premise served locally, or did it need a live fetch?
        try:
            from collections import Counter
            from src.services import telemetry
            failures_after = failure_snapshot()
            source_errors = {k: v - failures_before.get(k, 0) for k, v in failures_after.items()
                             if v - failures_before.get(k, 0) > 0}
            reasons = Counter(r.split("(")[0] for _, r in result.rejected)
            telemetry.record_retrieval(
                premise_id=premise.id, dimension=getattr(dim, "value", str(dim)),
                local_hits=len(local_hits), local_relevant=local_relevant,
                used_live=used_live, candidates=len(candidates),
                admitted=len(result.admitted), rejected=dict(reasons),
                mean_novelty=result.mean_novelty, decision_id=teg.decision_id,
                local_admitted=local_admitted, source_errors=source_errors,
                rung_used=rung_used, rung_query=rung_query, transport_failed=transport_failed,
                forced=bool(force_live), hinted=bool(hint))
        except Exception:
            pass  # telemetry must never break a run

        return result
