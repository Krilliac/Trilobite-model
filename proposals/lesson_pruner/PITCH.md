# lesson_pruner

Clusters lessons in the memory_store by embedding cosine similarity (single-linkage,
threshold 0.93 by default) and reports which are near-duplicate restatements of each
other, keeping the longest/earliest as the survivor of each cluster. Dry-run by default;
deletion only happens via `apply_plan`/`prune(..., dry_run=False)`, and only through
`memory_store.delete_lesson`.

It's valuable because the lessons table only grows (every solver self-repair /
reflection pass can mint a new lesson) and near-duplicates dilute both FTS and semantic
retrieval — `retriever.py`'s RRF fusion and top-k semantic search waste slots on
restatements of the same fact, crowding out genuinely distinct lessons and degrading
recall for the solver loop.

Integration: run `python proposals/lesson_pruner/lesson_pruner.py --db memory.db` (or
import `lesson_pruner.prune(conn)`/`build_plan(conn)` directly) periodically — e.g. as a
step in `curriculum_run.py` or a standalone maintenance cron — review the printed
report, then re-run with `--apply` (or `dry_run=False`) once satisfied. No changes to
`memory_store.py`, `embeddings.py`, or any other existing module were required or made.
