# MMR diversity reranker for retrieval

Implements Maximal Marginal Relevance: given a query vector and a candidate
pool of `(id, vector)` pairs, greedily builds a top-k list that trades off
relevance-to-query against redundancy-with-already-picked, using
`score = lambda * relevance - (1 - lambda) * max_similarity_to_selected`.

It's valuable because `retriever.py`'s RRF fusion currently has no
de-duplication pressure: five near-identical restatements of the same lesson
(a common shape after repeated `feedback.py`/`reflection.py` distillation
passes) can all clear `min_sim` and crowd out a genuinely different lesson
that's slightly less similar but covers new ground — the k-slot budget gets
spent on redundant copies instead of diverse, useful context.

To integrate: after `retriever._relevant_ids(conn, qv, fused, min_sim)`
produces its filtered id list, decode each id's stored embedding (already
loaded once in `_semantic_rank`) and pass `(id, vector)` pairs plus `qv`
through `mmr_from_blobs` (or `mmr_rerank` directly, if vectors are already in
hand) before truncating to `k`, instead of just slicing the RRF order —
everything else in `retrieve()` stays untouched. `lambda_mult` can start at
0.5 and be tuned the same way `DEFAULT_MIN_SIM` was, via a small held-out set
of queries with known duplicate-lesson clusters.
