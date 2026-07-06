# import_autofix

Detects a `NameError` for an un-imported stdlib module or symbol in a real execution traceback (`detect_missing_import`) and mechanically prepends the missing `import`/`from ... import ...` line to the offending code (`fix_missing_imports`). Covers `random`, `math`, `os`, `sys`, `json`, `re`, `collections`, `itertools`, `time` (direct `modname.attr` use) plus common `typing`/`collections`/`itertools` symbols pulled in via `from X import Y`.

This targets the single most common "breakout-class" solver failure — the model writes correct logic but forgets `import random` (or `typing.List`, etc.) — which currently burns a whole repair-loop generation on a one-line mechanical fix the model usually doesn't even notice on retry.

**Integration:** call `fix_missing_imports(code, error)` inside `solver.solve()`'s repair branch, right after a failed `run_code_fn` and before building the next `_repair_prompt` — if it returns changed code, re-run the verifier immediately (free, no model call) and only fall back to a repair prompt if it still fails. Same seam applies to `verifiers.python_exec`/`pytest_run` callers and `game_ladder`'s repair loop, since all of them already pass `(code, output)` through this exact shape.

Zero dependencies beyond the stdlib `re` module; pure functions of `(code, traceback_text)` with no model/GPU/network involved, so it's deterministic and safe to run on every repair-loop iteration at negligible cost.
