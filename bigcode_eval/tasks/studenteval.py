"""
StudentEval is a dataset of 1,749 prompts for 48 problems, authored by 80
students who have only completed a one-semester Python programming class.
Unlike many other benchmarks, it has multiple prompts per problem and multiple
attempts by the same participant.

Web page: https://huggingface.co/datasets/wellesley-easel/StudentEval
"""

from bigcode_eval.base import Task
from datasets import load_dataset
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import tempfile
import pandas as pd
import numpy as np
import subprocess

_CITATION = """\
@misc{babe2023studenteval,
      title={StudentEval: A Benchmark of Student-Written Prompts for Large Language Models of Code}, 
      author={Hannah McLean Babe and Sydney Nguyen and Yangtian Zi and Arjun Guha and Molly Q Feldman and Carolyn Jane Anderson},
      year={2023},
      eprint={2306.04556},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}"""

EXECUTION_TIMEOUT = 15


# Source: Chen at al. Evaluating Large Language Models of Code. 2021
def _estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    assert c <= n, "c must be less than n"
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def _run_assembled_program(item):
    """
    Runs the program with a timeout. The result dictionary has a "success" key
    that is 1 on success and 0 on failure. It also includes keys necessary to
    group results (problem, prompt, and group) and report results for each
    subset of StudentEval.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(item["program"])
        f.flush()
        try:
            result = subprocess.run(
                ["python3", f.name],
                timeout=EXECUTION_TIMEOUT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = 1
    return {
        "problem": item["problem"],
        "prompt": item["prompt"],
        "group": item["group"],
        "success": 1 if exit_code == 0 else 0,
    }


def _get_group(item):
    """
    These boolean flags are mutually exclusive in the dataset. We turn them into a
    a string for easy grouping with Pandas.
    """
    if item["is_first_success"]:
        return "First Success"
    if item["is_last_success"]:
        return "Last Success"
    if item["is_first_failure"]:
        return "First Failure"
    if item["is_last_failure"]:
        return "Last Failure"
    return None


class StudentEval(Task):
    DATASET_PATH = "wellesley-easel/StudentEval"

    def __init__(self):
        self.stop_words = ["\ndef", "\nclass", "\nif", "\nprint"]
        self.requires_execution = True
        self.dataset = load_dataset(path=self.DATASET_PATH)
        # NOTE(Arjun Guha): Avoiding .filter so that we don't get a datasets
        # cache item on disk.
        self.dataset = [
            item for item in self.dataset["test"] if _get_group(item) is not None
        ]

    def get_dataset(self):
        return self.dataset

    def get_prompt(self, doc):
        return doc["prompt"].rstrip()

    # For a task with tests, the reference solution is the suite of tests.
    def get_reference(self, doc):
        return {
            "prompt": doc["prompt"],
            "assertions": doc["assertions"],
            "problem": doc["problem"],
            "group": _get_group(doc),
        }

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        prompt = self.get_prompt(self.dataset[idx])
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list({ "assertions": list(str), "problem": str })
            list of reference solutions
        """

        worklist = []
        for generations, reference in zip(generations, references):
            # NOTE(Arjun Guha): This can be more efficient. At low temperature, we get lots of
            # repeated completions. So, this will end up running the same program repeatedly.
            # The original StudentEval code runs each generation once.
            for generation in generations:
                item = {
                    "program": generation + "\n\n" + reference["assertions"],
                    "prompt": reference["prompt"],
                    "problem": reference["problem"],
                    "group": reference["group"],
                }
                worklist.append(item)

        with ThreadPoolExecutor(max_workers=cpu_count() - 1) as executor:
            results_df = pd.DataFrame(
                list(
                    tqdm(
                        executor.map(_run_assembled_program, worklist),
                        total=len(worklist),
                    )
                )
            )

        # Calculate pass@1 for each prompt
        results_df = results_df.groupby(["problem", "prompt", "group"]).agg(
            c=("success", np.sum), n=("success", "count")
        )
        results_df.reset_index(inplace=True)
        results_df["pass1"] = results_df.apply(
            lambda row: _estimator(row["n"], row["c"], 1), axis=1
        )

        # Calculate mean pass@1 for each group
        results_df = results_df.groupby(["group"]).agg(pass1=("pass1", np.mean))

        # Turn into JSON
        results_df.reset_index(inplace=True)
        results_df = results_df.to_dict(orient="records")
        return results_df
