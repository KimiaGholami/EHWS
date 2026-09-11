"""Zero-shot downstream accuracy via lm-evaluation-harness.

Task set matches ELSA's own zero-shot protocol (ELSA paper, Section 5.1 /
Appendix B.2): ARC-Easy, ARC-Challenge, BoolQ, HellaSwag, OpenBookQA, RTE,
Winogrande, all 0-shot -- since the Proposed Work is a direct extension of
ELSA, reusing ELSA's exact task list keeps results comparable to the
paper this method is built on.
"""

from __future__ import annotations

import lm_eval
from lm_eval.models.huggingface import HFLM

ELSA_TASKS = ["arc_challenge", "arc_easy", "boolq", "hellaswag", "openbookqa", "rte", "winogrande"]


def run_zeroshot(model, tokenizer, device, tasks: list[str] | None = None, batch_size: int = 8) -> dict:
    tasks = tasks or ELSA_TASKS
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=str(device))
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, log_samples=False)
    summary = {}
    for task in tasks:
        task_res = results["results"].get(task, {})
        acc_key = next((k for k in task_res if k.startswith("acc") and "stderr" not in k), None)
        summary[task] = task_res.get(acc_key) if acc_key else None
    summary["average"] = sum(v for v in summary.values() if v is not None) / len(
        [v for v in summary.values() if v is not None]
    )
    return summary
