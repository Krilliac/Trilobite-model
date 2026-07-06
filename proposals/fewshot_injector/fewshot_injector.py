"""Few-shot prompt builder: turns top-k similar past (task, solution) recalls
into a structured few-shot prompt, instead of the single-line bullet list
orchestrator.build_prompt currently uses for its RECALL_HEADER block.

Recalls arrive from recall.recall() as "task -> response" strings (its
_format()), but this module also accepts dicts ({"task"/"query"/"prompt":...,
"solution"/"response"/"answer"/"code": ..., optional "score"/"similarity"/"sim"})
and (task, solution[, score]) tuples/lists, so it can sit in front of any
recall source without the caller reshaping its data first.

Pure and dependency-free: no model, no GPU, no I/O. If every recall carries a
numeric score it re-sorts by score descending (defensive — callers may pass
an unranked pool); otherwise the given order is trusted as already-ranked
(recall.recall()'s contract).
"""

MAX_SOLUTION_CHARS = 600

FEWSHOT_HEADER = (
    "# Similar task(s) solved before ({n} example(s)). Use them as style/pattern "
    "references -- adapt, don't copy verbatim unless the new task is identical."
)
EXAMPLE_TEMPLATE = "\n--- Example {i} ---\nTask: {task}\nSolution:\n{solution}\n"
FOOTER_TEMPLATE = "\n# Now solve this new task:\n{task}"


def _extract(entry):
    """Normalize one raw recall entry into (task_text, solution_text, score).

    score is None when the entry carries no similarity/score field -- callers
    should treat None-scored entries as pre-ranked rather than sortable.
    """
    if isinstance(entry, str):
        if " -> " in entry:
            task, solution = entry.split(" -> ", 1)
        else:
            task, solution = entry, ""
        return task.strip(), solution.strip(), None

    if isinstance(entry, (tuple, list)):
        if len(entry) >= 3:
            task, solution, score = entry[0], entry[1], entry[2]
        elif len(entry) == 2:
            task, solution, score = entry[0], entry[1], None
        elif len(entry) == 1:
            task, solution, score = entry[0], "", None
        else:
            task, solution, score = "", "", None
        return (task or "").strip(), (solution or "").strip(), score

    if isinstance(entry, dict):
        task = entry.get("task") or entry.get("query") or entry.get("prompt") or ""
        solution = (
            entry.get("solution") or entry.get("response")
            or entry.get("answer") or entry.get("code") or ""
        )
        score = entry.get("score")
        if score is None:
            score = entry.get("similarity")
        if score is None:
            score = entry.get("sim")
        return task.strip(), solution.strip(), score

    # Unknown shape: best-effort stringify rather than raising.
    return str(entry).strip(), "", None


def _truncate(text, max_chars):
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + " ..."
    return text


def build_fewshot_prompt(task, recalls, k=3, max_solution_chars=MAX_SOLUTION_CHARS):
    """Build a few-shot-augmented prompt from up to k past (task, solution) recalls.

    task: the new task text.
    recalls: iterable of past recalls, in any of the shapes _extract() handles.
    k: max number of examples to include (<=0 or falsy recalls -> task returned
       unchanged, no header/footer scaffolding added).
    max_solution_chars: per-example solution truncation (0/None disables it).

    Never raises on malformed entries; a fully-empty entry is dropped. Returns
    a single str ready to hand to a generate_fn.
    """
    task = task or ""
    if not recalls or k <= 0:
        return task

    normalized = [_extract(r) for r in recalls]
    normalized = [(t, s, sc) for t, s, sc in normalized if t or s]
    if not normalized:
        return task

    if all(sc is not None for _, _, sc in normalized):
        normalized.sort(key=lambda item: -item[2])

    top = normalized[:k]
    parts = [FEWSHOT_HEADER.format(n=len(top))]
    for i, (t, s, _sc) in enumerate(top, 1):
        parts.append(EXAMPLE_TEMPLATE.format(
            i=i,
            task=t or "(no task text)",
            solution=_truncate(s, max_solution_chars) or "(no solution recorded)",
        ))
    parts.append(FOOTER_TEMPLATE.format(task=task))
    return "".join(parts)
