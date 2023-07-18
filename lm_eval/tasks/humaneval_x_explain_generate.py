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
    "cpp": 60,
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
        "using namespace std;",      
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<cmath>",
        "#include<math.h>",
        "#include<numeric>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<set>",
        "#include<map>",
        "#include<queue>",
        "#include<stack>",
        "#include<list>",
        "#include<deque>",
        "#include<boost/any.hpp>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
        "#include<sstream>",
        "#include<fstream>",
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
        elif self.mutate_method == "starchat":
            self.stop_words.extend(["<|end|>"])            
        self.requires_execution = True
        # TODO: Switch back to super

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
        instruction = f"Write functional code in {LANGUAGE_TO_NAME[self.DATASET_NAME]} according to the description."
        if self.mutate_method == "instructcodet5p":
            # https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/humaneval/generate_codet5p.py#L89
            prompt = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n{doc["description"]}\n\n### Response:'       
        else:
            raise NotImplementedError
        return prompt
    
    def get_prompt_base(self, doc):
        # See 
        # https://github.com/roG0d/CodeGeeX/blob/f66205b5f615a4eead9c26d7ec297e14738ea18d/codegeex/benchmark/evaluate_humaneval_x.py#L78
        # https://github.com/THUDM/CodeGeeX/pull/76#issuecomment-1500653190
        if self.DATASET_NAME == "rust":
            main = "fn main(){}\n"
            prompt_base = main + doc["declaration"]
        else:
            prompt_base = doc["declaration"]
        return prompt_base
    
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt_base = self.get_prompt_base(doc)
        instruction = f"Write functional code in {LANGUAGE_TO_NAME[self.DATASET_NAME]} according to the description."

        if self.mutate_method == "instruct":
            prompt = doc["description"] + "\n" + instruction + "\n" + prompt_base
        elif self.mutate_method == "instruct-qa":
            prompt = f'Question: {instruction}\n{doc["description"]}\n\nAnswer:\n{prompt_base}'
        elif self.mutate_method == "instructcodet5p":
            # https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/humaneval/generate_codet5p.py#L89
            prompt = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n{doc["description"]}\n\n### Response:{prompt_base}'       
        elif self.mutate_method == "starchat":
            prompt = f'<|system|>\n<|end|>\n<|user|>\n{instruction}\n{doc["description"]}<|end|>\n<|assistant|>\n{prompt_base}'
        elif self.mutate_method == "starcodercommit":
            prompt = f'<commit_before><commit_msg>{instruction}\n{doc["description"]}<commit_after>{prompt_base}'
        elif self.mutate_method == "wizardcoder":
            prompt = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n{doc["description"]}\n\n### Response:\n{prompt_base}'
        else:
            raise NotImplementedError

        return prompt.strip()

    def remove_last_block(self, code):
        """
        Adapted from https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L151
        """
        for w in self.stop_words:
            if w in code:
                code = code[:code.find(w)]

        ### Find the first occassion where a chain of { } is closed
        if self.DATASET_NAME == "python":
            for i, line in enumerate(code.split("\n")):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    return "\n".join(code.split("\n")[:i])
        elif self.DATASET_NAME in ["java", "js", "go", "cpp", "rust"]:
            open_brackets = 1
            cut = False
            for i, c in enumerate(code):
                if c == '{':
                    open_brackets += 1
                elif c == '}':
                    open_brackets -= 1
                if open_brackets == 0:
                    code = code[:i+1]
                    cut = True
                    break
            if not cut:
                if self.DATASET_NAME == "java":
                    main_pos = code.find("public static void main")
                    if main_pos != -1:
                        code = code[:main_pos] + '}'
                    if '}' in code:
                        code = code[:code.rfind('}')] + '}'
                    if code.count('{') - 1 == code.count('}'):
                        code += "\n}"
                elif '}' in code:
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
        # Strip to maintain same behavior as with get_prompt
        return self.get_prompt_base(doc).rstrip() + gen

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
        """Not necessary as scores are the same if grouped or not grouped
        # Reformat from [[sample1], [sample2] ...] -> [[samples until n_samples], [samples until 2*n_samples] ...]
        n_samples = 1
        while n_samples < len(references) and references[n_samples] == references[0]:
            n_samples += 1 
        generations = [
            [g[0] for g in generations[i:i+n_samples]] for i in range(0, len(generations), n_samples)
        ]
        if self.DATASET_NAME in ["go", "rust"]:
            ds = self.get_dataset()
            ds = [
                ds[i] for i in range(0, len(references), n_samples)
            ]
        references = [
            references[i] for i in range(0, len(references), n_samples)
        ]
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
            cpp_imports = "\n".join(IMPORT_HELPER["cpp"])
            generations = [
                [(cpp_imports + "\n" + g.split("int main")[0]).strip() for g in gen] for gen in generations
            ]
            # Legacy bug
            if len(generations) > 77:
                generations[77] = [g.replace("iscuber", "iscube") for g in generations[77]]    
            # Remove int main if present
            generations = [[g.split("int main")[0] for g in gens] for gens in generations]
        elif language == "go":
            for gen, ref, doc in zip(generations, references, ds):
                for line in doc["import"].split("\n"):
                    line = line.replace("import", "").replace("(", "").replace(")", "").replace('"', "").strip()
                    if line: assert line in IMPORT_HELPER["go"], doc["import"] # Will be added later
                test_setup_str = doc["test_setup"] + "\n"
                for i, g in enumerate(gen):
                    for line in test_setup_str.split("\n"):
                        line = line.replace("import", "").replace("(", "").replace(")", "").strip()
                        if line.startswith('"') and line in g:
                            test_setup_str = test_setup_str.replace(line, "")
                    g = test_setup_str + g + "\n" + ref
                    other_pkgs = set()
                    for pkg in IMPORT_HELPER["go"]:
                        if ('"' + pkg + '"' not in g):
                            p = pkg.split("/")[-1]
                            # Check if the package is used
                            if (p + "." in g):
                                # The problem is that it could appear in a comment
                                # E.g. in problem 158, the docstring is:
                                # // ... a list of strings.
                                # but the "strings" pkg is never used
                                # Golang throws an error if the pkg is not used
                                # Thus search for the package & make sure it's not in a commented line
                                lines = g.split("\n")
                                for line in lines:
                                    if (p + "." in line) and not(line.strip().startswith("//")):
                                        other_pkgs.add('"' + p + '"')
                                        break
                    other_pkgs_str = ""
                    if other_pkgs:
                        other_pkgs_str = "import (\n" + "\n".join(["    " + p for p in other_pkgs]) + "\n)\n"
                    if ("package main" in gen[i]) and ("package main" in test_setup_str):
                        gen[i] = gen[i].replace("package main", "")
                    gen[i] = test_setup_str + other_pkgs_str + gen[i]
        elif language == "rust":
            main = "fn main(){}"
            for gen, doc in zip(generations, ds):
                declaration = doc["declaration"]
                for i, g in enumerate(gen):
                    new_gen = ""
                    if "fn main()" not in g:
                        new_gen += main
                    for line in declaration.split("\n"):
                        if line.strip() not in g:
                            new_gen += line.strip() + "\n"
                    # If fn main() is present twice, cut off before the second one
                    g = "fn main()".join(g.split("fn main()")[0:2])
                    new_gen += g
                    gen[i] = new_gen
            # Legacy bug
            if len(generations) > 77:
                generations[77] = [g.replace("iscuber", "iscube") for g in generations[77]]                        

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
            if ("compilation error" in log[0][0][1]["result"]) or (results["pass@1"] != 1):
                print("XXXXX")
                print(results)
                print(log)
                print(i)
                print(gen[0])
                print(ref)        
        """
        return results