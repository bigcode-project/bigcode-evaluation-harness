"""WIP

Homepage: https://github.com/bigcode-project/commits
"""

import re
from evaluate import load
from lm_eval.base import Task


_CITATION = """
"""

LANGUAGES = ["python", "cpp", "js", "java", "go", "rust"]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {apps-interview: Task, apps-competitoon: Task}
    """
    return {f"humaneval-x-bugs-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class HumanEvalXBugs(GeneralHumanEvalXBugs):
        def __init__(self, mutate_method="prompt", language=language):
            super().__init__(language=language)

    return HumanEvalXBugs


class GeneralHumanEvalXBugs(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "Muennighoff/humaneval-x-bugs"
    DATASET_NAME = None

    def __init__(self, mutate_method="prompt", language="python"):
        
        self.DATASET_NAME = language
        stop_words = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
        self.mutate_method = mutate_method
        if self.mutate_method == "edit":
            stop_words = [
                "<commit_before>", 
                "<commit_msg>", 
                "<commit_after>", 
                "<|endoftext|>",
            ]

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
        else:
            raise ValueError(f"Unknown mutate_method: {mutate_method}")

        return prompt.strip()

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
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
        # Keep the defining part of the function
        cutoff = len(prompt) - len(doc["prompt"])
        generation = generation[cutoff:]
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
        results, _ = code_metric.compute(
            references=references,
            predictions=generations,
        )
        return results
