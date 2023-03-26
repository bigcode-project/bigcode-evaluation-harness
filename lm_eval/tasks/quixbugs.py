"""QuixBugs
"""

import re
from evaluate import load
from lm_eval.base import Task

_CITATION = """

"""


class QuixBugs(Task):

    DATASET_PATH = "Muennighoff/quixbugs"

    def __init__(self, mutate_method="prompt"):
        self.mutate_method = mutate_method
        if self.mutate_method == "edit":
            self.stop_words = [
                "<commit_before>",
                "<commit_msg>", 
                "<commit_after>", 
                "<|endoftext|>",
                "\nprint",
                "\nif",
                "\nclass",
            ]
        elif self.mutate_method == "prompt":
            self.stop_words = [
                "\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "<|endoftext|>"
            ]
        else:
            raise ValueError(f"Unknown mutate_method: {self.mutate_method}")

        super().__init__(
            stop_words=self.stop_words,
            requires_execution=True,
        )
        self.max_length_multiplier = 2.25 # Allow 2.25 times the length of the prompt

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["train"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        if self.mutate_method == "edit":
            prompt = "<commit_before>" + doc["buggy_program"]
            prompt += "<commit_msg>" + "Fix bug in " + doc["name"]
            prompt += "<commit_after>"
        elif self.mutate_method == "prompt":
            # https://arxiv.org/pdf/2111.03922.pdf, Prenner et al.
            prompt = "### fix the bug in the following function"
            prompt += doc["buggy_program"] + "\n"
            prompt += "### fixed function"
        else:
            raise ValueError(f"Unknown mutate_method: {mutate_method}")

        return prompt.strip()

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return (doc["name"], doc["tests"].strip())
        #test_func = doc["test"]
        #entry_point = f"check({doc['entry_point']})"
        #return "\n" + test_func + "\n" + entry_point

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
        generation = generation[len(prompt):]
        if self.mutate_method.startswith("prompt"):
            generation = "def" + generation # Add def which is in the prompt back to the output        
        return self.remove_last_block(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("code_eval")
        results = {}
        for i, (gen, (name, ref)) in enumerate(zip(generations, references)):
            sub_results, _ = code_metric.compute(
                references=[ref],
                predictions=[gen],
                timeout=10, # Levenshtein distance is slow
            )
            results[name] = sub_results
        # Provide average of all metrics computed
        if results:
            results["all"] = {
                k: sum(v[k] for v in results.values()) / len(results) for k in results[list(results.keys())[0]]
            }
        return results
