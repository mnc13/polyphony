"""
Central configuration: environment, filesystem paths, and every hyperparameter.

All tunable numbers live here (not scattered through the code) so a reviewer can
see the full parameter surface in one place and the run manifest can snapshot it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# --- Filesystem layout -------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
VECTORDB_DIR = DATA_DIR / "vectordb"
SIT_DB_PATH = DATA_DIR / "sit.db"
RUNS_DIR = ROOT / "runs"
OUTPUTS_DIR = ROOT / "outputs"          # step-wise + full per-sample deliverables
MODELS_YAML = CONFIG_DIR / "models.yaml"

# Parallel-safety: when COGNI_WORKER is set, namespace the writable stores (vector DB
# + SIT) per worker so several windows can run at once without corrupting shared FAISS
# / SQLite files. outputs/ and the ledger stay SHARED (samples are disjoint by range).
_WORKER = os.getenv("COGNI_WORKER", "").strip()
if _WORKER:
    VECTORDB_DIR = DATA_DIR / f"worker_{_WORKER}" / "vectordb"
    SIT_DB_PATH = DATA_DIR / f"worker_{_WORKER}" / "sit.db"

for _d in (DATA_DIR, VECTORDB_DIR, RUNS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Bump whenever a code change alters what a run produces. A variant folder
# (outputs/<variant>/) records the version it was started under, and run_batch refuses
# to resume one started under another: the resume ledger would otherwise skip every
# case finished by older code and leave those results standing beside new ones.
PIPELINE_VERSION = "2026-10-22.openalex-budget"


@dataclass(frozen=True)
class Hyperparams:
    """Every threshold in the pipeline. Snapshotted into each run manifest."""

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    # Window size in WORDS (the name predates that). MiniLM truncates at 256 word-pieces
    # and abstracts run ~1.3 pieces per word; 140 words leaves room for the title prefix
    # and a folded tail. See src/evidence/chunker.py.
    chunk_tokens: int = 140
    chunk_overlap: int = 32

    # P-MER admission gate (inherited from PROCLAIM, empirically validated)
    novelty_threshold: float = 0.20          # admit only if novelty >= this
    novelty_stop_threshold: float = 0.10     # debate-stop signal when below this
    redundancy_ratio_threshold: float = 0.70
    per_corpus_top_k: int = 5
    # Local-first retrieval goes live unless the corpus already holds at least this many
    # chunks that are RELEVANT to the premise (cosine >= relevance_cosine_threshold).
    # Gating on the raw hit count instead meant that once a dimension held five chunks
    # of anything, no premise in that dimension ever fetched again: cases 003-006 of the
    # first post-fix run made zero live fetches, and all six cases shared 24 chunks.
    local_sufficient_hits: int = 3
    # TEG contradiction edges are only scored between chunks this topically similar.
    # Without the floor, NLI "found" contradictions between unrelated abstracts that
    # merely shared a dimension (panic-disorder steroids vs a brain-ecology editorial).
    contradiction_topic_cosine: float = 0.50
    # Semantic Scholar only answers with an API key, so whether one was set used to
    # decide, unrecorded, which literature PSYCH/SOCIO/economics premises searched. Off
    # for the experiment (the pilot and main ran without it); every variant must match.
    use_semantic_scholar: bool = False

    # NLI (the no-LLM-scoring backbone)
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    nli_entailment_threshold: float = 0.60   # citation integrity / testimony fidelity cutoff
    # DeBERTa-v3-base is trained to 512 positions (max_position_embeddings=512), but its
    # tokenizer advertises model_max_length=1e19, so passing truncation=True WITHOUT a
    # length silently truncates nothing. Over-length premises then scored spuriously
    # high: spans over 1200 chars averaged 0.902 entailment against a deliberately
    # absurd hypothesis, against 0.293 for shorter ones.
    nli_max_length: int = 512
    # Candidate spans sent to NLI per summary sentence. All-pairs scoring is
    # len(sentences) x len(spans) calls — 12,240 on a six-decision case, over fourteen
    # hours at this machine's measured per-call cost. Spans are pre-ranked by embedding
    # cosine and only this many are scored.
    nli_faithfulness_top_k: int = 5
    # Compute arbiter faithfulness DURING the run? Default off, and deliberately.
    # Faithfulness is a post-hoc evaluation metric: the pipeline does not need it to
    # produce decisions, premises, testimony, debate or the central script. Measured on
    # this machine, DeBERTa costs ~3.5 s per pair whether batched or not, so even with
    # the top-k prefilter one case is ~900 pairs, around 52 minutes — against ~35 minutes
    # for everything else in the case combined. Case 005 sat for 7.5 hours in this one
    # call after its arbiter output had already succeeded. Run it offline instead, for
    # every case at once, with scripts/recompute_faithfulness.py.
    inrun_faithfulness: bool = False
    # Admission-gate relevance is TOPICAL (embedding cosine), not entailment: an
    # abstract rarely *entails* a specific premise, so entailment wrongly rejects
    # on-topic evidence. NLI entailment is reserved for citation-time checks.
    # 0.28 -> 0.35 after the case-001 pilot. Of its 83 admitted chunks, 29 scored below
    # 0.35 against their best premise. Inspected by title, about 16 of those 29 were off
    # topic (syphilis epidemiology, antenatal care, war poetry, a death-tax essay), against
    # about 3 of the 54 at or above 0.35. That is one case and a judgment by title, not a
    # labelled benchmark. The same bar decides coverage (src/debate/coverage.py) and
    # testimony relevance, so it also lowers measured coverage: round-1 coverage on the
    # pilot transcripts falls from 0.67-1.0 to 0.60-0.75, and debates run more rounds.
    relevance_cosine_threshold: float = 0.35

    # Premise dedup
    dedup_cosine_threshold: float = 0.85
    # Decision dedup — higher than premises: decision statements share a template
    # ("X is the right course for the agent"), which inflates cosine similarity, so
    # only genuine restatements (measured ~0.94) should collapse, not distinct
    # options (measured ~0.80-0.85). See P5 doc.
    decision_dedup_threshold: float = 0.90

    # Scientist selection (dynamic count)
    scientist_pool_size: int = 7
    scientist_floor: int = 2                 # min disciplines that must testify
    scientist_ceiling: int = 7

    # Decision fan-out
    max_decisions: int = 6

    # Debate
    max_rounds: int = 8
    coverage_stop_threshold: float = 0.90
    novelty_stop_consecutive_rounds: int = 2
    # Stop when coverage has not improved for this many consecutive rounds (0 = off).
    # Added with the stricter coverage rule: re-scored, only 2 of 35 stored debates
    # reach 0.90 after round 1, and while retrieval keeps admitting anything the novelty
    # stop does not fire, so a stuck debate would otherwise run all 8 rounds (16 turns
    # instead of 2) without addressing anything new. Not part of the original design.
    coverage_stall_rounds: int = 2

    # LLM roles run with reasoning disabled unless models.yaml marks them reasoning: true.
    # A model whose endpoint refuses to disable it is sent this reasoning effort instead
    # of being left to reason without limit ("" = send nothing, the old behaviour). See
    # _REASONING_MANDATORY in src/services/llm_client.py.
    mandatory_reasoning_effort: str = "low"

    # A local network outage (DNS stops resolving) is waited out for up to this long, in
    # both evidence fetches and LLM calls, instead of failing the work in flight. On the
    # case-001 pilot a ~4 minute outage failed 31 evidence requests for decision d5.
    network_outage_wait_s: int = 900

    # Human gate
    query_gate_timeout_s: int = 10


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    use_openrouter: bool = field(default_factory=lambda: os.getenv("USE_OPENROUTER", "false").lower() == "true")
    ncbi_api_key: str = field(default_factory=lambda: os.getenv("NCBI_API_KEY", ""))
    semantic_scholar_api_key: str = field(default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""))
    openalex_api_key: str = field(default_factory=lambda: os.getenv("OPENALEX_API_KEY", ""))
    contact_email: str = field(default_factory=lambda: os.getenv("CONTACT_EMAIL", "anonymous@example.com"))
    hp: Hyperparams = field(default_factory=Hyperparams)

    def manifest(self) -> dict[str, Any]:
        """Serializable snapshot of the parameter surface (no secrets)."""
        return {"hyperparams": asdict(self.hp), "use_openrouter": self.use_openrouter}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
