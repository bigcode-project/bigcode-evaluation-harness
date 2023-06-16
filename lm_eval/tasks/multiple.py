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
from datasets import load_dataset
from tqdm import tqdm

from lm_eval.base import Task
from lm_eval.tasks.custom_metrics.multiple_metrics.evaluation import \
    get_test_results_json_path, evaluate_programs
from lm_eval.tasks.custom_metrics.multiple_metrics.single_experiment_pass_k import \
    for_result

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
        e.g. {multiple-py: Task, multiple-java: Task}
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
    DATASET_REVISION = "5d2abbb8ced9a0e37db985c47d24c24f45a16655"

    def __init__(self, language):
        self.language = language
        self.DATASET_NAME = f"humaneval-{language}"
        # we need the dataset to get stop words for each language
        self.dataset = load_dataset(
            GeneralMultiPLE.DATASET_PATH,
            self.DATASET_NAME,
            revision=self.DATASET_REVISION)
        stop_words = self.dataset["test"][0]["stop_tokens"]
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

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        completion = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(completion, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        # get prompts and problem names
        prompts_names = [
            {"prompt": doc["prompt"], "name": doc["name"]}
            for i, doc in enumerate(self.get_dataset())
            if i < len(generations)
        ]
        # a common temp dir for all the problems
        with tempfile.TemporaryDirectory() as temp_dir:
            list_files = []
            for (prompt_name, generation, reference) in zip(
                prompts_names, generations, references
            ):
                problem = {
                    "name": prompt_name["name"],
                    "language": self.language,
                    "prompt": prompt_name["prompt"],
                    "completions": generation,
                    "tests": reference,
                }
                # each problem is save in a json file
                temp_file_name = os.path.join(temp_dir, f"{prompt_name['name']}.json")
                list_files.append(temp_file_name)
                with open(temp_file_name, "wt") as f:
                    json.dump(problem, f)
            print(
                f"Saved {len(list_files)} problems in {temp_dir} for evaluation, each problem has {len(generations[0])} completions"
            )

            # execute the problems to evaluate them
            max_workers = cpu_count() - 1 if cpu_count() > 1 else 1

            programs, test_results_list, languages =  self.unroll_problems(list_files)

            evaluate_programs(programs, test_results_list, languages, max_workers)

            # Purge duplicates
            result_array = []
            [result_array.append(result) for result in test_results_list if result not in result_array]

            #print([[r["stderr"] for r in result["results"]] for result in result_array])

            # compute pass@k scores
            result = np.mean([for_result(result) for result in result_array], axis=0)
        results = {
            f"pass@{k}": v
            for k, v in zip([1, 10, 100], result)
            if k <= len(generations[0])
        }
        return results
    

    def unroll_problems(self, problem_json_paths):
        programs = list()
        test_results_list = list()
        languages = list()
        for problem_json_path in problem_json_paths:
            with open(problem_json_path, "r") as f:
                problem = json.load(f)
            print(problem)
            test_results = problem.copy()
            del test_results["completions"]
            test_results["results"] = []

            num_problems = len(problem["completions"])
            min_problem = len(test_results["results"])

            for index in range(min_problem, num_problems): 
                programs.append(problem["completions"][index] + "\n" + problem["tests"])
                test_results_list.append(test_results)
                languages.append(problem["language"])
            
        return programs, test_results_list, languages
