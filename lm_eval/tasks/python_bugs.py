"""Python Bugs
https://proceedings.mlr.press/v162/he22a.html

This dataset is taken from the preprossing done by CarperAI (https://carper.ai/diff-models-a-new-way-to-edit-code).
It is uploaded here: https://huggingface.co/datasets/Muennighoff/python-bugs
"""

import re
from evaluate import load
from lm_eval.base import Task


_CITATION = """
@inproceedings{he2022distribution,
  title={On distribution shift in learning-based bug detectors},
  author={He, Jingxuan and Beurer-Kellner, Luca and Vechev, Martin},
  booktitle={International Conference on Machine Learning},
  pages={8559--8580},
  year={2022},
  organization={PMLR}
}
"""

BIN_OP_PROMPT = "Fix binary operator"
VAR_MISUSE_PROMPT = "Fix incorrect variable name"

class PythonBugs(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "Muennighoff/python-bugs"

    def __init__(self):
        super().__init__(
            stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/"],
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["train"]
        return dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        # TODO: Special Tokens for commit models
        description = doc["prompt_code"]
        if doc["task"] == "bin-op":
            prompt = f'{description}\n{BIN_OP_PROMPT}'
        elif doc["task"] == "var-misuse":
            prompt = f'{description}\n{VAR_MISUSE_PROMPT}'
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["correct_code"]

    @staticmethod
    def first_block(string, stop_words):
        """Split off first block of code by scanning for class, def etc. on newlines."""
        return re.split("|".join(stop_words), string)[0].rstrip()

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        output = generation[len(prompt) :]
        return self.first_block(output, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        exact_match = load("exact_match")
        results, _ = exact_match.compute(
            references=references,
            predictions=generations,
        )
        return results
