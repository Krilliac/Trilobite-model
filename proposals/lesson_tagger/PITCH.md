# lesson_tagger

Infers zero or more coarse domain/language tags (python, sql, cpp, javascript,
concurrency, security, algorithm, io, testing, git, networking, regex,
performance, windows) from a lesson's raw text using deterministic
keyword/regex signals — `tag(text) -> sorted list[str]`, plus `tag_scores`
for per-tag hit-count weighting and `filter_by_tag` as a ready-made
pre-filter over a list of lesson rows/dicts.

It's valuable because `retriever.py`'s FTS5 + embedding fusion has no notion
of domain: a task about a SQL injection bug can currently pull back a
Python-list-comprehension lesson purely because the embeddings landed
nearby, wasting a retrieval slot and diluting the solver's context. Tags let
callers narrow the candidate pool to same-domain lessons before/alongside
the existing RRF fusion, and let `curriculum_run.py`/reflection-style
reporting bucket "how many concurrency lessons have we learned" without any
model call.

Integration: `import proposals.lesson_tagger.lesson_tagger as lesson_tagger`
(or copy into the repo root) and call `lesson_tagger.tag(lesson_text)` when
storing a lesson (`memory_store.add_lesson`) to attach tags out-of-band, or
call it at query time in `retriever.retrieve`/`semantic_search` to bias/filter
candidates whose tags overlap the current task's own `tag(task)` output. No
existing file was modified — this is purely additive and read-only against
the rest of the codebase.
