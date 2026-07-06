# ruff_verifier

Adds a `ruff_check` backend to the verifier registry: it pipes a code string into `ruff check -` over stdin and turns the exit code into the same `Verdict(passed, reason, detail)` shape every other backend produces (`rc==0` -> pass, `rc==1` -> fail with ruff's own diagnostics as `detail`, anything else or a missing `ruff` executable -> `VerifierUnavailable`, matching how `typecheck`/`cpp_compile` treat "tool absent" as distinct from "artifact failed").

It's valuable as a near-zero-cost pre-filter ahead of `python_exec`/`pytest_run` in the self-repair ladder: style/correctness smells (unused imports, undefined names, bare excepts) get caught and fed back to the solver in milliseconds, before spending a real interpreter/subprocess round-trip on an artifact that a linter would have flagged instantly.

To integrate: copy `ruff_verifier.py` into the repo root (or merge its one entry into `verifiers.REGISTRY`) and call `verifiers.verify("ruff_check", code, spec)` the same way any other backend is called — no other code changes needed. `_run` is monkeypatch-swappable exactly like `verifiers._run`, so the solver's repair loop and tests never need a real `ruff` binary to exercise the logic; only real usage requires `pip install ruff`.
