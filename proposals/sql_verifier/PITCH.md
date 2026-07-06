# sql_verifier

Grounds SQL artifacts against a real, throwaway in-memory sqlite3 database instead of
eyeballing syntax: `sql_valid(artifact, spec)` optionally applies a `schema` (DDL/seed
DML) via `executescript()`, then prepares/executes the artifact — single statement
(with param binding and a row/column preview) or a multi-statement script — and returns
`Verdict(passed, reason, detail)`, with `detail` carrying either a result preview or the
full traceback for a self-repair loop to consume. It also supports `dry_run` (an
`EXPLAIN` prefix) to validate destructive INSERT/UPDATE/DELETE statements without any
side effect. This is valuable because SQL is one of the most common artifacts a coding
model produces and gets subtly wrong (bad column/table refs, syntax slips, type
mistakes) — until now `verifiers.py` had no way to ground SQL the way it grounds Python,
C++, and pytest, so any solver/reward loop had to fall back to the weak `llm_judge`
oracle for it. Stdlib-only (sqlite3), no GPU/network/external binary, deterministic
tests.

**Integration:** add one entry to `verifiers.REGISTRY` — `"sql_valid": sql_verifier.sql_valid`
(after `import sql_verifier` in `verifiers.py`, or by copying the function body in) — and
it becomes callable from `solver.solve_verified()` / the reward loop exactly like
`python_exec` or `cpp_compile`, with `spec={"schema": ..., "params": ..., "mode": ...,
"dry_run": ...}` as the task-specific contract.
