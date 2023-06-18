import json

from evaluate import load
from datasets import load_dataset
from lm_eval.base import Task


LANGUAGES = ["python", "cpp", "js", "java", "go", "rust"]

LANGUAGE_TO_NAME = {
    "python": "Python",
    "cpp": "C++",
    "js": "JavaScript",
    "java": "Java",
    "go": "Go",
    "rust": "Rust",
}

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
        def __init__(self, mutate_method="prompt", language=language, load_data_path=None):
            super().__init__(mutate_method=mutate_method, language=language, load_data_path=load_data_path)

    return HumanEvalXExplainGenerate


class GeneralHumanEvalXExplainGenerate(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
    DATASET_PATH = "bigcode/humaneval-x-bugs"
    DATASET_NAME = None

    def __init__(self, mutate_method="prompt", language="python", load_data_path=None):

        self.DATASET_NAME = language
        self.descriptions = None
        assert load_data_path is not None, "load_data_path must be specified"
        with open(load_data_path) as fp:
            self.descriptions = json.load(fp)
            print(f"{len(self.descriptions)} descriptions with {len(self.descriptions[0])} description candidates loaded.")            
        self.dataset = load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)

        self.mutate_method = mutate_method        
        self.stop_words = LANGUAGE_TO_STOP_WORDS[language] + ["<|endoftext|>"]
        if self.mutate_method.startswith("edit"):
            self.stop_words.extend([
                "<commit_before>",
                "<commit_msg>",
                "<commit_after>",
            ])
        self.requires_execution = True

    def check_fn(self, code):
        """
        Adapted from https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L115

        Checks whether the generated code is finished
        """
        if any([w in code for w in self.stop_words]):
            return True

        # The heuristics below do not hold for diff generation
        if self.mutate_method.startswith("diff"):
            return False

        if self.DATASET_NAME == "python":
            for line in code.split("\n"):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    return True
        elif self.DATASET_NAME == "java":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif self.DATASET_NAME == "go":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif self.DATASET_NAME == "js":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif self.DATASET_NAME == "cpp":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif self.DATASET_NAME == "rust":
            if code.count("{") + 1 == code.count("}"):
                return True
        return False
    
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = []
        for description, sample in zip(self.descriptions, self.dataset["test"]):
            for description_candidate in description:
                dataset.append({"description": description_candidate} | sample)
        return dataset

    def get_prompt_encoder(self, doc):
        """Encoder input for models with Enc-Dec architecture like CodeT5"""
        assert self.mutate_method == "instruct", "Only instruct mutation is supported for Enc-Dec models"
        prompt = doc["description"]
        prompt += f"\nWrite functional code in {LANGUAGE_TO_NAME[self.DATASET_NAME]} that follows the description above."

        return prompt
    
    def get_prompt_base(self, doc):
        # See 
        # https://github.com/roG0d/CodeGeeX/blob/f66205b5f615a4eead9c26d7ec297e14738ea18d/codegeex/benchmark/evaluate_humaneval_x.py#L78
        # https://github.com/THUDM/CodeGeeX/pull/76#issuecomment-1500653190
        if self.DATASET_NAME == "rust":
            main = "\nfn main(){ \n } \n"
            prompt_base = main + doc["declaration"]
        else:
            prompt_base = doc["declaration"]
        return prompt_base
    
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt_base = self.get_prompt_base(doc)
        if self.mutate_method == "instruct":
            prompt = doc["description"]
            prompt += f"\nWrite functional code in {LANGUAGE_TO_NAME[self.DATASET_NAME]} that follows the description above."
            prompt += f"\n{prompt_base}"

        return prompt.strip()

    def remove_last_block(self, code):
        """
        Adapted from https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L151
        """
        for w in self.stop_words:
            if w in code:
                code = code[:code.rfind(w)]

        if self.mutate_method.startswith("diff"):
            return code

        if self.DATASET_NAME == "python":
            for i, line in enumerate(code.split("\n")):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    return "\n".join(code.split("\n")[:i])
        elif self.DATASET_NAME == "java":
            main_pos = code.find("public static void main")
            if main_pos != -1:
                code = code[:main_pos] + '}'
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
            if code.count('{') + 1 == code.count('}'):
                code += "\n}"
        elif self.DATASET_NAME == "go":
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
        elif self.DATASET_NAME == "cpp":
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
        elif self.DATASET_NAME == "js":
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
        elif self.DATASET_NAME == "rust":
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
        return code
    
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
        gen = self.remove_last_block(generation[len(prompt):].rstrip())
        prompt_base = self.get_prompt_base(doc)
        # Strip to maintain same behavior as with get_prompt
        return prompt_base.rstrip() + gen

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