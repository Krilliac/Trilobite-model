import embeddings as e
import memory_store as ms
import retriever as r


def test_rrf_rewards_agreement():
    # B appears in BOTH lists; A appears in only one (even at rank 0).
    # Standard RRF rewards cross-list agreement, so B wins.
    fused = r.rrf([["A", "B"], ["B", "C"]])
    assert fused[0] == "B"


def test_semantic_search_uses_embeddings():
    c = ms.connect(":memory:")
    ms.add_lesson(c, "near", "x", e.to_blob([1.0, 0.0]), "i")
    ms.add_lesson(c, "far", "y", e.to_blob([0.0, 1.0]), "i")
    hits = r.semantic_search(c, "query", embed_fn=lambda t: [0.9, 0.1])
    assert hits[0] == "near"


def test_semantic_search_empty_when_no_embeddings():
    c = ms.connect(":memory:")
    ms.add_lesson(c, "near", "x", e.to_blob([1.0, 0.0]), "i")
    assert r.semantic_search(c, "q", embed_fn=lambda t: None) == []


def test_retrieve_returns_texts_and_degrades_to_lexical():
    c = ms.connect(":memory:")
    ms.add_lesson(c, "L1", "always release the threading lock", None, "i")
    ms.add_lesson(c, "L2", "prefer RRF for hybrid ranking", None, "i")
    # embeddings unavailable -> lexical only, still finds the lock lesson.
    texts = r.retrieve(c, "threading lock release", embed_fn=lambda t: None)
    assert any("threading lock" in t for t in texts)
