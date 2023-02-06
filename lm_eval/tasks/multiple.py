"""MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation
https://arxiv.org/abs/2107.03374

MultiPL-E is a dataset for evaluating large language models for code generation that supports 18 programming languages.
It takes the OpenAI "HumanEval" and the MBPP Python benchmarks and uses little compilers to translate them to other languages.

Homepage: https://nuprl.github.io/MultiPL-E/
"""

import json
import os
import re
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from time import time

import numpy as np
from custom_metrics.multiple_metrics.evaluation import evaluate_problem
from custom_metrics.multiple_metrics.single_experiment_pass_k import for_file
from tqdm import tqdm

from lm_eval.base import Task

_CITATION = """
@article{cassano2022scalable,
  title={A Scalable and Extensible Approach to Benchmarking NL2Code for 18 Programming Languages},
  author={Cassano, Federico and Gouwar, John and Nguyen, Daniel and Nguyen, Sydney and Phipps-Costin, Luna and Pinckney, Donald and Yee, Ming Ho and Zi, Yangtian and Anderson, Carolyn Jane and Feldman, Molly Q and others},
  journal={arXiv preprint arXiv:2208.08227},
  year={2022}
}
"""

LANGUAGES = [
    "py",
    "bs",
    "cpp",
    "cs",
    "d",
    "go",
    "java",
    "js",
    "jl",
    "lua",
    "pl",
    "php",
    "r",
    "rkt",
    "rb",
    "rs",
    "scala",
    "swift",
    "ts",
]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-python: Task, multiple-java: Task}
    """
    return {f"multiple-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class MultiPLE(GeneralMultiPLE):
        def __init__(self):
            super().__init__(language)

    return MultiPLE


class GeneralMultiPLE(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "nuprl/MultiPL-E"
    DATASET_NAME = None

    def __init__(self, language):
        self.DATASET_NAME = f"humaneval-{language}"
        stop_words = self.get_dataset()[0]["stop_tokens"]
        super().__init__(
            stop_words=stop_words,
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return doc["prompt"].strip()

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["tests"]

    @staticmethod
    def remove_last_block(string, stop_words):
        # Remove the last block of the code containing stop_words for HumanEval
        string_list = re.split("(%s)" % "|".join(stop_words), string)
        # last string should be ""
        return "".join(string_list[:-2])

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        return self.remove_last_block(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """

        # create a temp json file for each problem
        prompts = [doc["prompt"] for doc in self.get_dataset()][: len(generations)]
        list_files = []
        for i, (prompt, generation, reference) in enumerate(
            zip(prompts, generations, references)
        ):
            problem = {
                "prompt": prompt,
                "completions": generation,
                "tests": reference,
                "language": self.language,
            }
            temp_dir = tempfile.gettempdir()
            temp_file_name = os.path.join(temp_dir, f"problem-{i}.json")
            list_files.append(temp_file_name)
            with open(temp_file_name, "wt") as f:
                json.dump(problem, f)

        # execute the problems to evaluate them
        output_dir = os.path.join(temp_dir, f"results/")
        max_workers = cpu_count() - 1 if cpu_count() > 1 else 1
        start_t = time.time()
        for file in tqdm(list_files):
            evaluate_problem(output_dir, file, max_workers, temp_dir)
        end_t = time.time()
        print(f"Execution took {end_t - start_t} seconds")

        # compute pass@k scores
        result_array = np.array(
            [for_file(p) for p in Path(temp_dir).glob("*.results.json")]
        )
        result = result_array.mean(axis=0)
        name = (
            temp_dir.split("/")[-1]
            if temp_dir.split("/")[-1] != ""
            else temp_dir.split("/")[-2]
        )
        print(f"{name},1,{result[0]:.2f}")
        print(f"{name},10,{result[1]:.2f}")
        print(f"{name},100,{result[2]:.2f}")
        results = {f"pass@{k}": v for k, v in zip([1, 10, 100], result)}
        # print but add .2f to the values in the dict
        print({k: f"{v:.2f}" for k, v in results.items()})
        return results
