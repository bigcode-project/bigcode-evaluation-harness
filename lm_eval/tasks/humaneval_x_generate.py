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
    "cpp": 60,
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
    return {f"humaneval-x-generate-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class HumanEvalXGenerate(GeneralHumanEvalXGenerate):
        def __init__(self, mutate_method="prompt", language=language):
            super().__init__(mutate_method=mutate_method, language=language)

    return HumanEvalXGenerate


class GeneralHumanEvalXGenerate(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
    DATASET_PATH = "bigcode/humaneval-x-bugs"
    DATASET_NAME = None

    def __init__(self, mutate_method="prompt", language="python"):
        
        self.DATASET_NAME = language
        self.mutate_method = mutate_method        
        stop_words = LANGUAGE_TO_STOP_WORDS[language]
        if self.mutate_method.startswith("edit"):
            stop_words.extend([
                "<commit_before>",
                "<commit_msg>",
                "<commit_after>",
            ])
        elif self.mutate_method == "instruct-wizard-or":
            stop_words = []
        elif self.mutate_method == "starchat":
            stop_words.extend(["<|end|>"])

        stop_words.append("<|endoftext|>")
        
        super().__init__(
            stop_words=stop_words,
            requires_execution=True,
        )

    def check_fn(self, code):
        """
        Adapted from https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L115

        Checks whether the generated code is finished.

        This is suboptimal as models may (rarely) split their answer in multiple functions.
        This will force the generation to end after the first function.
        """
        if any([w in code for w in self.stop_words]):
            return True

        # The heuristics below do not hold for diff generation
        if (self.mutate_method.startswith("diff")) or (self.mutate_method == ("instruct-wizard-or")):
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
        return self.dataset["test"]

    def get_prompt_encoder(self, doc):
        """Encoder input for models with Enc-Dec architecture like CodeT5"""
        if self.mutate_method == "instruct":
            return doc["instruction"].strip()
        elif self.mutate_method == "instructcodet5p":
            return f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{doc["instruction"].strip()}\n\n### Response:'
        else:
            raise NotImplementedError
    
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt_base = doc["prompt"]

        if self.mutate_method == "continue":
            prompt = prompt_base
        elif self.mutate_method == "instruct":
            prompt = doc["instruction"].strip() + "\n\n" + prompt_base
        elif self.mutate_method == "instruct-qa":
            prompt = f'Question: {doc["instruction"].strip()}\n\nAnswer:\n{prompt_base}'
        elif self.mutate_method == "starchat":
            # https://huggingface.co/HuggingFaceH4/starchat-beta
            prompt = f'<|system|>\n<|end|>\n<|user|>\n{doc["instruction"].strip()}<|end|>\n<|assistant|>\n{prompt_base}'
        elif self.mutate_method == "starcodercommit":
            prompt = f'<commit_before><commit_msg>{doc["instruction"].strip()}<commit_after>{prompt_base}'
        elif self.mutate_method == "starcodercommit2":
            prompt = f'<commit_before>{prompt_base}<commit_msg>{doc["instruction"].strip().replace("Write a", "Complete")}<commit_after>{prompt_base}'
        elif self.mutate_method == "instructcodet5p":
            # https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/humaneval/generate_codet5p.py#L89
            prompt = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{doc["instruction"].strip()}\n\n### Response:{prompt_base}'       
        elif self.mutate_method == "wizardcoder":
            # https://github.com/nlpxucan/WizardLM/blob/main/WizardCoder/src/humaneval_gen.py#L37
            prompt = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{doc["instruction"].strip()}\n\n### Response:\n{prompt_base}'
        else:
            raise NotImplementedError            
        
        return prompt.rstrip()

    def get_reference(self, doc, get_solution=False):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        if get_solution:
            return doc["prompt"] + doc["canonical_solution"]
        else:
            # check(func_name) is already included
            return "\n" + doc["test"]
        
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
                    if code.count('{') + 1 == code.count('}'):
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
        return doc["prompt"].rstrip() + gen
    
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
            cpp_imports = "\n".join(IMPORT_HELPER["cpp"])
            # Remove main in case present
            generations = [
                [(cpp_imports + "\n" + g.split("int main")[0]).strip() for g in gen] for gen in generations
            ]
            # Legacy bug
            if len(generations) > 77:
                generations[77] = [g.replace("iscuber", "iscube") for g in generations[77]]
        elif language == "java":
            generations = [
                [g.replace("public class Main {\n    }", "").strip() for g in gen] for gen in generations
            ]
        elif language == "go":
            ds = self.get_dataset().select(range(len(generations)))
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
            ds = self.get_dataset().select(range(len(generations)))
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
