# difficulty_estimator

Scores a coding task's text (no execution, no model call) into a difficulty in
`[0, 1]` from three deterministic signals: saturating length, keyword hits
across recursion/DP/concurrency/parsing/regex (plus graph/numerics), and a
count of explicit constraints ("must", "at most N", numbered requirements,
complexity bounds). It's valuable because `solver.solve()` / `best_of_n()`
currently spend a flat `max_attempts` on every task regardless of how hard it
actually is — this lets the orchestrator front-load repair budget onto tasks
that need it and cut it short on trivial ones, saving both wall-clock and
token spend on the easy majority. Integration is a one-line call: before
invoking `solver.solve(prompt, check, generate_fn, ...)`, compute
`max_attempts=difficulty_estimator.suggest_max_attempts(prompt)` (or log
`difficulty_estimator.classify(score)` alongside `reward.score()` outcomes to
see whether repair success correlates with estimated difficulty). No changes
to solver.py, verifiers.py, or reward.py are required — it's a pure
pre-filter that produces a number those modules already accept as a parameter.
