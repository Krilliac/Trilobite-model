"""Hybrid lexical+semantic retrieval over distilled lessons. RRF fusion."""
import os

import embeddings
import memory_store

# Calibrated against memory.db (nomic-embed-text): held-out irrelevant probes
# (swap_case, hamming_distance, is_pangram) topped out at cosine 0.56-0.65
# against the current lesson set, while genuinely on-topic lesson/task pairs
# (reverse_string, celsius_to_fahrenheit, nth_fibonacci) scored 0.65-0.86.
# 0.65 clears all the observed noise while still admitting the on-topic hits.
DEFAULT_MIN_SIM = 0.65


def rrf(rank_lists, k=60):
    scores = {}
    for lst in rank_lists:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: -scores[i])


def _semantic_rank(conn, qv, limit=10):
    scored = []
    for les in memory_store.all_lessons(conn):
        emb = les["embedding"]
        if not emb:
            continue
        v = embeddings.from_blob(emb)
        scored.append((embeddings.cosine(qv, v), les["id"]))
    scored.sort(reverse=True)
    return [lid for _, lid in scored[:limit]]


def semantic_search(conn, task, embed_fn=embeddings.embed, limit=10):
    qv = embed_fn(task)
    if qv is None:
        return []
    return _semantic_rank(conn, qv, limit=limit)


def _relevant_ids(conn, qv, ids, min_sim):
    """Filter fused candidate ids to those whose stored embedding clears min_sim.

    Lessons with no stored embedding are dropped (relevance can't be judged).
    """
    kept = []
    for lid in ids:
        row = conn.execute(
            "SELECT embedding FROM lessons WHERE id=?", (lid,)
        ).fetchone()
        emb = row[0] if row else None
        if not emb:
            continue
        v = embeddings.from_blob(emb)
        if embeddings.cosine(qv, v) >= min_sim:
            kept.append(lid)
    return kept


def retrieve(conn, task, k=5, embed_fn=embeddings.embed, min_sim=None):
    if min_sim is None:
        min_sim = float(os.environ.get("TRILOBITE_MIN_SIM", str(DEFAULT_MIN_SIM)))

    lexical = memory_store.fts_search(conn, task, limit=10)
    qv = embed_fn(task)

    if qv is None:
        # Embeddings unavailable: soft-fail to lexical-only, no threshold possible.
        fused = rrf([lexical, []])[:k]
        texts = [memory_store.get_lesson_text(conn, lid) for lid in fused]
        return [t for t in texts if t]

    semantic = _semantic_rank(conn, qv, limit=10)
    fused = rrf([lexical, semantic])
    relevant = _relevant_ids(conn, qv, fused, min_sim)[:k]
    texts = [memory_store.get_lesson_text(conn, lid) for lid in relevant]
    return [t for t in texts if t]
