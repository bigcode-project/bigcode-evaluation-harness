import json
import re

from evaluate import load
from lm_eval.base import Task


LANGUAGES = ["python", "cpp", "js", "java", "go", "rust"]

# Taken from https://huggingface.co/datasets/nuprl/MultiPL-E/ & https://github.com/THUDM/CodeGeeX
LANGUAGE_TO_STOP_WORDS = {
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L164
    "python": ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\nassert"],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L185
    "cpp": [],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L188
    "js": [],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L177
    "go": ["\n//", "\nfunc main(", "struct", "\nfunc"],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L169
    "java": [],
    "rust": [],
}

LANGUAGE_TO_TIMEOUT = {
    "python": 10,
    "cpp": 10,
    "js": 10,
    "java": 10,
    "go": 20,
    "rust": 300, # Necessary for first-time compilation of cargo
}

# Java sometimes fails with more workers; For JS it's twice as fast with 4 workers
LANGUAGE_TO_NUM_WORKERS = {
    "python": 4,
    "cpp": 4,
    "js": 4,
    "java": 1,
    "go": 4,
    "rust": 1,
}

# https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L6
IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "go": [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp": [
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<math.h>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
    ],
}

def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {apps-interview: Task, apps-competitoon: Task}
    """
    return {f"humaneval-x-explain-generate-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class HumanEvalXExplainGenerate(GeneralHumanEvalXExplainGenerate):
        def __init__(self, mutate_method="prompt", language=language):
            super().__init__(mutate_method=mutate_method, language=language)

    return HumanEvalXExplainGenerate


class GeneralHumanEvalXExplainGenerate(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, mutate_method="prompt", language="python", load_data_path=None):

        self.DATASET_NAME = language
        self.descriptions = None
        assert load_data_path is not None, "load_data_path must be specified"
        with open(load_data_path) as fp:
            self.descriptions = json.load(fp)
            print(f"{len(self.descriptions)} descriptions with {len(self.descriptions[0])} description candidates loaded.")            

        self.mutate_method = mutate_method        
        stop_words = LANGUAGE_TO_STOP_WORDS[language]
        if self.mutate_method.startswith("edit"):
            stop_words.extend([
                "<commit_before>",
                "<commit_msg>",
                "<commit_after>",
            ])

        stop_words.append("<|endoftext|>")
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        # Use declaration instead of prompt to hide the docstring
        if self.mutate_method == "edit":
            prompt = "<commit_before><commit_after>" + doc["declaration"] + doc["canonical_solution"]
            prompt += "<commit_msg>"
        elif self.mutate_method == "instruct":
            prompt = doc["declaration"] + doc["canonical_solution"]
            prompt += f"\nProvide a detailed natural language description of the above function such that you would be able to reconstruct the function given the description. You are given a budget of {self.token_budget} tokens, everything afterwards will be cut off. Do not include any code."
        
        return prompt.strip()

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)
        gen = generation[len(prompt):].strip()[:self.token_budget].rstrip()
        return gen

    def get_reference(self, doc, get_solution=False):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        if get_solution:
            return doc["prompt"] + doc["canonical_solution"]
        else:
            test_func = doc["test"]
            # check(func_name) is already included
            return "\n" + test_func
            
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("Muennighoff/code_eval")
        timeout = LANGUAGE_TO_TIMEOUT[self.DATASET_NAME]
        num_workers = LANGUAGE_TO_NUM_WORKERS[self.DATASET_NAME]
        language = self.DATASET_NAME if self.DATASET_NAME != "js" else "javascript"

        # See https://github.com/THUDM/CodeGeeX/blob/ebeb850f227a90c79de39f7e26b1302f374f3240/codegeex/benchmark/evaluate_humaneval_x.py
        if language == "python":
            python_imports = "\n".join(IMPORT_HELPER["python"])
            generations = [
                [(python_imports + "\n" + g).strip() for g in gen] for gen in generations
            ]
        elif language == "cpp":
            for gen in generations:
                for i, g in enumerate(gen):
                    for s in IMPORT_HELPER["cpp"]:
                        if s not in g:
                            gen[i] = s + "\n" + g
        elif language == "go":
            ds = self.get_dataset().select(range(len(generations)))
            for gen, doc in zip(generations, ds):
                import_string = doc["import"]
                test_setup = doc["test_setup"]
                for i, g in enumerate(gen):
                    other_pkgs = []
                    for pkg in IMPORT_HELPER["go"]:
                        if pkg not in test_setup:
                            p = pkg.split("/")[-1]
                            if p + "." in g:
                                # The problem is that it could appear in a comment
                                # For example in problem 158, the docstring is:
                                # // ... a list of strings.
                                # but the "strings" package is never used
                                # Golang throws an error if the package is not used
                                # Hence search for the package & make sure it's not in a commented line
                                #other_pkgs.append(f"\"{pkg}\"")
                                lines = g.split("\n")
                                for line in lines:
                                    if p + "." in line and not line.startswith("//"):
                                        other_pkgs.append(f"\"{pkg}\"")
                                        break

                    gen[i] = g.replace(import_string, "")
                    if other_pkgs:
                        import_other_pkgs = "import (\n" + "    ".join([p + "\n" for p in other_pkgs]) + ")"
                        gen[i] = test_setup + "\n" + import_other_pkgs + "\n" + gen[i]
                    else:
                        gen[i] = test_setup + "\n" + gen[i]
        elif language == "rust":
            ds = self.get_dataset().select(range(len(generations)))
            main = "\nfn main(){ \n } \n"
            for gen, doc in zip(generations, ds):
                declaration = doc["declaration"]
                for i, g in enumerate(gen):
                    gen[i] = main + declaration + g

        results, logs = code_metric.compute(
            references=references,
            predictions=generations,
            language=language,
            timeout=timeout,
            num_workers=num_workers,
        )
        """Debugging help
        for i, (gen, ref) in enumerate(zip(generations, references)):
            import time
            starttime = time.time()            
            results, log = code_metric.compute(
                references=[ref],
                predictions=[gen],
                language=language,
                timeout=timeout,
            )
            print("TOOK: ", time.time() - starttime)
            with open("errors.txt", "a") as f:
                f.write(log[0][0][1]["result"] + "\n")
            if ("compilation error" in log[0][0][1]["result"]) or (results["pass@1"] != 0):
                print("XXXXX")
                print(results)
                print(log)
                print(i)
                print(gen[0])
                print(ref)        
        """
        return results