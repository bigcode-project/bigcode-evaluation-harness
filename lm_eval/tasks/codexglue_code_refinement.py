"""
An Empirical Study on Learning Bug-Fixing Patches in the Wild via Neural Machine Translation
https://arxiv.org/pdf/1812.08693.pdf
Code Refinement task from CodeXGlue (code refactoring):
* For both subsets (i.e. small and medium) based on the function length, the source side i.e the Java function with bugs is given as a prompt (all the function and variable names are normalized). 
"""
import json

from evaluate import load
from lm_eval.base import Task

_CITATION = """
@article{10.1145/3340544,
author = {Tufano, Michele and Watson, Cody and Bavota, Gabriele and Penta, Massimiliano Di and White, Martin and Poshyvanyk, Denys},
title = {An Empirical Study on Learning Bug-Fixing Patches in the Wild via Neural Machine Translation},
year = {2019},
issue_date = {October 2019},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {28},
number = {4},
issn = {1049-331X},
url = {https://doi.org/10.1145/3340544},
doi = {10.1145/3340544},
journal = {ACM Trans. Softw. Eng. Methodol.},
month = {sep},
articleno = {19},
numpages = {29},
keywords = {Neural machine translation, bug-fixes}
}
"""

FUNCTION_LENGTH = {
    "small": "small",
    "medium": "medium",
}


def create_all_tasks():
    """Creates a dictionary of tasks from a list of sizes based on the function length
    :return: {task_name: task}
        e.g. {codexglue_code_refinement-small: Task, codexglue_code_refinement-medium: Task}
    """
    return {
        f"codexglue_code_refinement-{func_length}": create_task(func_length)
        for func_length in FUNCTION_LENGTH
    }


def create_task(func_length):
    class CodexGLUECodeRefinementTask(CodexGLUECodeRefinement):
        def __init__(self):
            super().__init__(func_length)

    return CodexGLUECodeRefinementTask


class CodexGLUECodeRefinement(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "code_x_glue_cc_code_refinement"
    DATASET_NAME = None

    def __init__(self, func_length):
        self.DATASET_NAME = func_length
        super().__init__(
            stop_words=["\n"],
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        with open(
            "lm_eval/tasks/few_shot_examples/codexglue_code_refinement_few_shot_prompts.json",
            "r",
        ) as file:
            examples = json.load(file)
        return examples

    @staticmethod
    def two_shot_prompt(entry, text, examples):
        """Two shot prompt format as Buggy Version of Java Code & Fixed Version of Java Code"""
        prompt = f"\nBuggy function:\n{examples['buggy version of java code1']}\
                   \nFixed function:\n{examples['fixed version of java code1']}\
                   \nBuggy function:\n{examples['buggy version of java code2']}\
                   \nFixed function:\n{examples['fixed version of java code2']}\
                   \nBuggy function:\n{text}\
                   \nFixed function:\n"
        return entry + prompt

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        func_length = FUNCTION_LENGTH[self.DATASET_NAME]
        text = doc["buggy"]
        entry = "Fix this  buggy Java function:\n"
        examples = self.fewshot_examples()
        examples = examples[func_length]
        prompt = self.two_shot_prompt(entry, text, examples)
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["fixed"].strip()

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        output = generation.split("\nFixed function:\n", 3)[-1].strip()
        return output

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing references
        """
        bleu = load("bleu")
        gens = [gen[0] for gen in generations]
        results = bleu.compute(
            references=references, predictions=gens, max_order=4, smooth=True
        )
        return results
