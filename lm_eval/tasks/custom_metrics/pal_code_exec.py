import evaluate
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from lm_eval.tasks.executors.python_executor import run_program
import numpy as np
import warnings

# adapted from https://github.com/huggingface/evaluate/blob/main/metrics/code_eval/code_eval.py

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval" metric executes untrusted model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions,
set the environment variable HF_ALLOW_CODE_EVAL="1". Within Python you can to this
with:
>>> import os
>>> os.environ["HF_ALLOW_CODE_EVAL"] = "1"
################################################################################\
"""


def compute(predictions, references, num_workers=4, timeout=3.0, majority_voting=False):
    """Returns the scores"""

    if os.getenv("HF_ALLOW_CODE_EVAL", 0) != "1":
        raise ValueError(_WARNING)

    if os.name == "nt":
        raise NotImplementedError("This metric is currently not supported on Windows.")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        for task_id, candidates in enumerate(predictions):
            for candidate in candidates:
                args = (candidate, timeout, task_id, completion_id[task_id])
                future = executor.submit(run_program, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

        for future in as_completed(futures):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    answers = [None]*len(results)
    for result in results.values():
        result.sort()
        task_id = result[0][1]['task_id']
        eval_answers = [r[1]["result"] for r in result]
        if majority_voting:
            counter = Counter(eval_answers)
            eval_answers = [counter.most_common()[0][0]]

        if not majority_voting and len(eval_answers) > 1:
            warnings.warn(
                f"Multiple generations found for a task without setting `majority_voting` to True, defaulting answers from first generation"
            )
        answers[task_id] = eval_answers[0]

    scores = []
    # Number of code generated that failed execution.
    errored = 0
    for task_id,(ans, ref) in enumerate(zip(answers, references)):
        try:
            score = 1 if abs(float(ans) - float(ref)) < 1e-3 else 0
        except ValueError as e:
            warnings.warn(
                f"""ValueError - {e} during scoring task_id- {task_id},answer - {ans},reference -{ref}\ndefaulting evaluation score for this answer to 0"""
            )
            errored += 1
            score = 0

        scores.append(score)

    return {"accuracy": sum(scores) / len(scores),"num_failed_execution":errored}
