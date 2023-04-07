"""WIP

Homepage: https://github.com/bigcode-project/commits

Recommended evaluation:
--max_length_generation 2048
    - The longest solution by split using the santacoder tokenizer:
        - CPP: 562
        - Java: 399
        - JS: 465
        - Go: 349
        - Rust: 753
    - In addition comes:
        - the function docstring (~100 tokens max)
        - the instruction (~10 tokens)
        - the generation/duplication with fixed bug (i.e. docstring again & solution) 
    - So for e.g. Rust the entire thing may be ~1800 tokens (worst case)
--do_sample False
    - Greedy evaluation seems to work best; More experiments needed
"""

import re
from evaluate import load
from lm_eval.base import Task


_CITATION = """
"""

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
        self.mutate_method = mutate_method
        
        stop_words = LANGUAGE_TO_STOP_WORDS[language]
        if self.mutate_method.startswith("edit"):
            stop_words.extend([
                "<commit_before>",
                "<commit_msg>",
                "<commit_after>",
            ])
        elif self.mutate_method.startswith("diff"):
            stop_words = ["<commit_before>", "<commit_msg>", "<commit_after>"]
        stop_words.append("<|endoftext|>")

        super().__init__(
            stop_words=stop_words,
            requires_execution=True,
        )

    def check_fn(self, code):
        """
        Adapted from https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L115

        Checks whether the generated code is finished
        """
        if any([w in code for w in self.stop_words]):
            return True

        # The heuristics below do not hold for diff generation
        if self.mutate_method == "diff":
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
        elif self.mutate_method == "diff":
            prompt = "<commit_before>" + doc["prompt"] + doc["buggy_solution"]
            prompt += "<commit_msg>" + "Fix bug in " + doc["entry_point"]
            prompt += "<commit_after>"
        elif self.mutate_method == "diff-carper":
            prompt = "<BEF>" + doc["prompt"] + doc["buggy_solution"]
            prompt += "<MSG>" + "Fix bug in " + doc["entry_point"]
            prompt += "<DFF>"       
        elif self.mutate_method == "prompt":
            prompt = "# Buggy function"
            prompt += "\n" + doc["prompt"] + doc["buggy_solution"] + "\n"
            prompt += "# Fixed function\n" + doc["prompt"]
        elif self.mutate_method == "prompt-plain":
            prompt = doc["prompt"] + doc["buggy_solution"]
            prompt += "\n" + "Fix bug in " + doc["entry_point"] # This will be cut-off, so it will compile
            prompt += "\n" + doc["prompt"]     
        elif self.mutate_method == "instruct":
            # input_template = "Instructions: {instruction}\nInput: {input} Output: "
            # https://github.com/SivilTaram/santacoder-finetuning-commit/blob/82a5598d632d299b7350c8b2ffb4af39527befa3/train.py#L115
            prompt = f"Instructions: Fix bug in {doc['entry_point']}\n"
            prompt += f"Input: {doc['prompt'] + doc['buggy_solution']} "
            prompt += f"Output: " + doc["prompt"]
        else:
            raise ValueError(f"Unknown mutate_method: {mutate_method}")
        # Strip off the final \n to make the tokens more natural
        # Essentially, we want to make sure that if there was no distrinction between
        # input & output, the tokens would be the same
        # E.g. for SantaCoder:
        # tokenize("""def hi()\n   return""")
        # ['def', 'Ġhi', '()', 'ĊĠĠ', 'Ġreturn']
        # So we need to split before the \n so that the input is
        # ['def', 'Ġhi', '()'] and the model can generate ['ĊĠĠ', 'Ġreturn']
        # If instead we provide def hi()\n the tokens will be
        # ['def', 'Ġhi', '()', 'Ċ'] and the model would need to generate ['ĠĠ', 'Ġreturn']
        # Which would be harder, as it's not the usual way these tokens are tokenized
        # i.e. the model has never seen the token sequence of ['()', 'Ċ', 'ĠĠ'], but only ['()', 'ĊĠĠ']
        # The same holds for Java, JS, Go, Rust, C++ tho the start sequences are slightly different
        return prompt.strip()

    def get_reference(self, doc, get_solution=False):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        if get_solution:
            if self.mutate_method == "diff":
                from diff_match_patch import diff_match_patch
                text1 = doc["prompt"] + doc["buggy_solution"]
                text2 = doc["prompt"] + doc["canonical_solution"]
                dmp = diff_match_patch()
                patches = dmp.patch_make(text1, text2)
                diff = dmp.patch_toText(patches)
                return diff
            else:
                return doc["prompt"] + doc["canonical_solution"]
                # To check that all buggy solutions result in a 0 score:
                # return doc["prompt"] + doc["buggy_solution"]
        else:
            test_func = doc["test"]
            # check(func_name) is already included
            return "\n" + test_func

    def remove_last_block(self, code):
        """
        Adapted from https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L151
        """
        for w in self.stop_words:
            if w in code:
                code = code[:code.rfind(w)]

        if self.mutate_method == "diff":
            return code

        if self.DATASET_NAME == "java":
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
        if self.mutate_method == "diff":
            return gen
        else:
            # Strip on the right to maintain same behavior as with get_prompt
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

        # Apply the diff to the input
        if self.mutate_method == "diff":
            # !wget https://raw.githubusercontent.com/google/diff-match-patch/master/python3/diff_match_patch.py
            from diff_match_patch import diff_match_patch
            dmp = diff_match_patch()
            ds = self.get_dataset().select(range(len(generations)))
            for gen, doc in zip(generations, ds):
                old_code = doc["prompt"] + doc["buggy_solution"]
                for i, diff in enumerate(gen): 
                    try:
                        # Strip away anything to the left such as \n
                        patches = dmp.patch_fromText(diff.lstrip())
                        fixed_code, _ = dmp.patch_apply(patches, old_code)
                    except Exception as e:
                        print(f"Failed with {e} when applying patch to buggy code: {diff}")
                        fixed_code = ""
                    gen[i] = fixed_code
        # 
        elif self.mutate_method == "diff-carper":
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
