"""eval_duel — compare single-model vs CROSS-MODEL reasoning strategies on hard
tasks, grounded by real execution. Tests the hypothesis that a second, different
model (rotated generator, or a dedicated critic) converts failures that a single
model's self-repair cannot.

Strategies compared (all served locally, execution-verified):
  * coder pass@1            baseline single-shot (qwen2.5-coder:7b)
  * coder self-repair x3    same model repairs itself (the loop that showed no lift)
  * r1 self-repair x3       is a reasoning model alone enough?
  * rotate coder<->r1 x4    rotate the generator across two models each attempt
  * gen coder / critic r1   coder writes, r1 diagnoses failures
  * gen r1 / critic coder   r1 writes, coder diagnoses failures

NOTE: on 6GB VRAM only one 7B is resident at a time, so cross-model strategies
pay a model-swap reload between alternating calls — slow but correct. Run in the
background. Usage: python eval_duel.py [n_tasks]
"""
import sys

import grounding
import server
import solver
import training_tasks

HARD = [
    "eval_expr", "base64_encode_manual", "topological_sort",
    "levenshtein_distance", "int_to_roman", "merge_intervals",
]

CODER = "qwen2.5-coder:7b"
R1 = "deepseek-r1:7b"


def mk(model, temp=0.3):
    # reasoning models emit long <think> traces -> give them room so the fenced
    # code block is never truncated before extraction.
    return server._make_generate(model, "", temp, 3072, 8192)


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else len(HARD)
    tasks = [t for t in training_tasks.TASKS if t["name"] in HARD][:n]
    gc, gr = mk(CODER), mk(R1)

    strategies = [
        ("coder pass@1",         lambda t: solver.solve(t["prompt"], t["check"], gc, max_attempts=1)),
        ("coder self-repair x3", lambda t: solver.solve(t["prompt"], t["check"], gc, max_attempts=3)),
        ("r1 self-repair x3",    lambda t: solver.solve(t["prompt"], t["check"], gr, max_attempts=3)),
        ("rotate coder<->r1 x4", lambda t: solver.rotate_solve(t["prompt"], t["check"], [gc, gr], max_attempts=4)),
        ("gen coder/critic r1",  lambda t: solver.solve_with_critic(t["prompt"], t["check"], gc, gr, max_attempts=3)),
        ("gen r1/critic coder",  lambda t: solver.solve_with_critic(t["prompt"], t["check"], gr, gc, max_attempts=3)),
    ]

    tally = {name: 0 for name, _ in strategies}
    for t in tasks:
        marks = []
        for name, fn in strategies:
            try:
                ok = bool(fn(t)["passed"])
            except Exception as e:
                ok = False
                print("  ! %s on %s: %r" % (name, t["name"], e))
            tally[name] += 1 if ok else 0
            marks.append("Y" if ok else ".")
        print("%-22s %s" % (t["name"], " ".join("%s=%s" % (s[0][:14], m) for s, m in zip(strategies, marks))))

    m = len(tasks)
    print("\n=== pass-rate on %d hard tasks ===" % m)
    for name, _ in strategies:
        print("  %-24s %d/%d" % (name, tally[name], m))


if __name__ == "__main__":
    main(sys.argv)
