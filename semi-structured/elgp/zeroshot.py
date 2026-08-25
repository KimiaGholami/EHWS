"""Zero-shot downstream accuracy via lm-evaluation-harness.

Task set: ARC-Easy, ARC-Challenge, BoolQ, HellaSwag, OpenBookQA, RTE,
Winogrande, all 0-shot -- the standard 7-task set used when evaluating
against ELSA's own published numbers.
"""

from __future__ import annotations

import lm_eval
from lm_eval.models.huggingface import HFLM

TASKS = ["arc_challenge", "arc_easy", "boolq", "hellaswag", "openbookqa", "rte", "winogrande"]


def run_zeroshot(model, tokenizer, device, tasks: list[str] | None = None, batch_size: int = 8) -> dict:
    tasks = tasks or TASKS
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=str(device))
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, log_samples=False)
    summary = {}
    for task in tasks:
        task_res = results["results"].get(task, {})
        acc_key = next((k for k in task_res if k.startswith("acc") and "stderr" not in k), None)
        summary[task] = task_res.get(acc_key) if acc_key else None
    scored = [v for v in summary.values() if v is not None]
    summary["average"] = sum(scored) / len(scored)
    return summary
