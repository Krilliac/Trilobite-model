# Few-shot prompt builder (fewshot_injector)

`build_fewshot_prompt(task, recalls, k=3) -> str` turns the top-k most similar past
(task, solution) recalls into a structured few-shot prompt (`Task: ... / Solution: ...`
per example, in similarity order) instead of the terse `"- task -> response"` bullet
list `orchestrator.build_prompt` currently emits for its recall block. Few-shot
exemplars with explicit task/solution pairing are a well-established lift for a
frozen model's code-generation accuracy over a flat context dump, at zero extra
inference cost since the recalls are already retrieved.

It accepts `recall.recall()`'s native `"task -> response"` strings as-is, plus dicts
(`task`/`query`/`prompt` + `solution`/`response`/`answer`/`code`, optional
`score`/`similarity`/`sim`) and `(task, solution[, score])` tuples, so any current or
future recall source can feed it without reshaping. It is pure stdlib with no model,
GPU, or I/O dependency — trivially unit-testable and safe to drop in anywhere a
prompt string is assembled.

**Integration**: swap the `RECALL_HEADER`/bullet-join block in `orchestrator.build_prompt`
(orchestrator.py:15-25) for `fewshot_injector.build_fewshot_prompt(task, recalls, k)`, or
call it directly ahead of `solver.solve`/`solve_verified` when composing a prompt that
should carry few-shot exemplars from `recall.recall()`'s output.
