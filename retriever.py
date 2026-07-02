"""Hybrid lexical+semantic retrieval over distilled lessons. RRF fusion."""
import embeddings
import memory_store


def rrf(rank_lists, k=60):
    scores = {}
    for list_idx, lst in enumerate(rank_lists):
        weight = 2 ** list_idx  # exponential boost for later lists
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank + 1)
    return sorted(scores, key=lambda i: -scores[i])


def semantic_search(conn, task, embed_fn=embeddings.embed, limit=10):
    qv = embed_fn(task)
    if qv is None:
        return []
    scored = []
    for les in memory_store.all_lessons(conn):
        emb = les["embedding"]
        if not emb:
            continue
        v = embeddings.from_blob(emb)
        scored.append((embeddings.cosine(qv, v), les["id"]))
    scored.sort(reverse=True)
    return [lid for _, lid in scored[:limit]]


def retrieve(conn, task, k=5, embed_fn=embeddings.embed):
    lexical = memory_store.fts_search(conn, task, limit=10)
    semantic = semantic_search(conn, task, embed_fn=embed_fn, limit=10)
    fused = rrf([lexical, semantic])[:k]
    texts = [memory_store.get_lesson_text(conn, lid) for lid in fused]
    return [t for t in texts if t]
