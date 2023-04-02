"""WIP

Homepage: https://github.com/bigcode-project/commits
"""

import re
from evaluate import load
from lm_eval.base import Task


_CITATION = """
"""

# TODO: Possibly check for gen finished via brackets
# https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L115

LANGUAGES = ["python", "cpp", "js", "java", "go", "rust"]


# Taken from https://huggingface.co/datasets/nuprl/MultiPL-E/ & https://github.com/THUDM/CodeGeeX
LANGUAGE_TO_STOP_WORDS = {
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L164
    "python": ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\nassert"],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L185
    "cpp": ["\n}"],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L188
    "js": ["\n}"],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L177
    "go": ["\n//", "\nfunc main(", "struct", "\nfunc"],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L169
    "java": ["\n }\n"],
    "rust": ["\n}"],
}

LANGUAGE_TO_TIMEOUT = {
    "python": 10,
    "cpp": 10,
    "js": 10,
    "java": 10,
    "go": 10,
    "rust": 300, # Necessary for first-time compilation of cargo
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
    return {f"humaneval-x-bugs-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class HumanEvalXBugs(GeneralHumanEvalXBugs):
        def __init__(self, mutate_method="prompt", language=language):
            super().__init__(mutate_method=mutate_method, language=language)

    return HumanEvalXBugs


class GeneralHumanEvalXBugs(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "bigcode/humaneval-x-bugs"
    DATASET_NAME = None

    def __init__(self, mutate_method="prompt", language="python"):
        
        self.DATASET_NAME = language
        stop_words = LANGUAGE_TO_STOP_WORDS[language]
        self.mutate_method = mutate_method
        if self.mutate_method.startswith("edit"):
            stop_words += [
                "<commit_before>", 
                "<commit_msg>", 
                "<commit_after>", 
                "<|endoftext|>",
            ]
        elif self.mutate_method == "instruct":
            stop_words += ["<|endoftext|>"]

        super().__init__(
            stop_words=stop_words,
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        if self.mutate_method == "edit":
            prompt = "<commit_before>" + doc["prompt"] + doc["buggy_solution"]
            prompt += "<commit_msg>" + "Fix bug in " + doc["entry_point"]
            prompt += "<commit_after>" + doc["prompt"]
        elif self.mutate_method == "edit-type":
            prompt = "<commit_before>" + doc["prompt"] + doc["buggy_solution"]
            prompt += "<commit_msg>" + "Fix " + doc["bug_type"] + " in " + doc["entry_point"]
            prompt += "<commit_after>" + doc["prompt"]
        elif self.mutate_method == "prompt":
            prompt = "# Buggy function"
            prompt += "\n" + doc["prompt"] + doc["buggy_solution"] + "\n"
            prompt += "# Fixed function\n" + doc["prompt"]
        elif self.mutate_method == "instruct":
            # input_template = "Instructions: {instruction}\nInput: {input} Output: "
            prompt = f"Instructions: Fix bug in {doc['entry_point']}\nInput: {doc['buggy_solution']} Output:"
        else:
            raise ValueError(f"Unknown mutate_method: {mutate_method}")
        # Strip off the final \n as it seems like its easier for small models to generate
        # \n\t than \t based on experiments from @lvwerra
        return prompt.strip()

    def get_reference(self, doc, get_solution=False):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        if get_solution:
            return doc["prompt"] + doc["canonical_solution"]
            # To check that all buggy solutions result in a 0 score:
            # return doc["prompt"] + doc["buggy_solution"]
        else:
            test_func = doc["test"]
            # check(func_name) is already included
            return "\n" + test_func

    @staticmethod
    def remove_last_block(string, stop_words):
        stop_words = [re.escape(word) for word in stop_words] # Escape e.g. | in <|endoftext|>
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
            (not used for Humaneval-Task)
        """
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)
        # Keep the defining part of the function; Strip on the right to maintain same
        # behavior as with get_prompt
        generation = doc["prompt"].rstrip() + generation[len(prompt):]
        return self.remove_last_block(generation, self.stop_words).strip()


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
        language = self.DATASET_NAME if self.DATASET_NAME != "js" else "javascript"

        # Apply the diff to the input
        if self.mutate_method == "diff":
            ds = self.get_dataset().select(range(len(generations)))
            end_of_diff = re.compile("\n[^ +-@]+")
            for gen in generations:
                for i, g in enumerate(gen):
                    # truncate diff hunk at the first line not starting with " ", "+", "-", or "@"
                    diff_hunk: str = end_of_diff.split(g)[0]
                    # apply the diff hunk to the input
                    # apply_diff(function_str, diff_hunk)
                    # WIP


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

        results, _ = code_metric.compute(
            references=references,
            predictions=generations,
            language=language,
            timeout=timeout,
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
