"""
P3 — Chunker. Splits a document's text into overlapping windows.

Word-based (no tiktoken dependency). Two properties matter downstream:

  * A window must fit the embedder. all-MiniLM-L6-v2 truncates at 256 word-pieces, and
    abstracts run ~1.3 word-pieces per word, so the old 256-WORD windows were cut
    silently: 15 of the 24 distinct chunks in the post-fix corpus exceeded 256 tokens,
    and relevance, novelty and local search all ignored their endings. Windows are now
    140 words (Hyperparams.chunk_tokens), so even a folded tail plus the title stays
    under the limit.

  * The last window must carry new text. Stepping by size-overlap left final windows
    that were mostly overlap -- one tail was 37 words with 5 new -- and each was admitted
    as a separate piece of "evidence". A tail with fewer than a quarter of a window of
    new words is folded into the window before it instead.

Every window is prefixed with the document title, so a tail is never evidence with no
indication of which paper it came from.
"""
from __future__ import annotations

from config.settings import get_settings
from src.evidence.sources.base import RawDoc


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text.strip()] if text.strip() else []
    step = max(1, size - overlap)
    starts = list(range(0, max(1, len(words) - overlap), step))
    windows: list[tuple[int, int]] = []
    for s in starts:
        e = min(s + size, len(words))
        windows.append((s, e))
        if e >= len(words):
            break
    # Fold a tail that adds too little beyond the previous window's end -- unless the
    # folded window would outgrow the embedder: folding up to size + size//4 words made
    # windows of 174 words plus the title, past MiniLM's 256 word-pieces (1 of 49 pilot
    # chunks). Such a tail stays separate; the novelty gate drops it if it adds nothing.
    if len(windows) >= 2:
        prev_s, prev_e = windows[-2]
        new_words = windows[-1][1] - prev_e
        if new_words < max(1, size // 4) and windows[-1][1] - prev_s <= size + size // 7:
            windows[-2:] = [(prev_s, windows[-1][1])]
    return [" ".join(words[s:e]) for s, e in windows]


def chunk_doc(doc: RawDoc) -> list[str]:
    hp = get_settings().hp
    title = (doc.title or "").strip()
    body = (doc.abstract or "").strip()
    if not body:
        return chunk_text(doc.text, hp.chunk_tokens, hp.chunk_overlap)
    pieces = chunk_text(body, hp.chunk_tokens, hp.chunk_overlap)
    out = []
    for i, p in enumerate(pieces):
        out.append(f"{title}\n\n{p}".strip() if title else p)
    return out
