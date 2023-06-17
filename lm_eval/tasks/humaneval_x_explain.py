import re
from evaluate import load
from lm_eval.base import Task


LANGUAGES = ["python", "cpp", "js", "java", "go", "rust"]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {apps-interview: Task, apps-competitoon: Task}
    """
    return {f"humaneval-x-bugs-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class HumanEvalXExplain(GeneralHumanEvalXExplain):
        def __init__(self, mutate_method="prompt", language=language):
            super().__init__(mutate_method=mutate_method, language=language)

    return HumanEvalXExplain


class GeneralHumanEvalXExplain(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
    DATASET_PATH = "bigcode/humaneval-x-bugs"
    DATASET_NAME = None

    def __init__(self, mutate_method="prompt", language="python", token_budget=500):

        self.DATASET_NAME = language
        self.mutate_method = mutate_method
        self.token_budget = token_budget
        
        stop_words = ["<|endoftext|>"]

        super().__init__(
            stop_words=stop_words,
            requires_execution=True,
        )

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
    
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        raise ValueError("""
        HumanEval-X-Explain should be run with the flag `--generation_only`.
        Once generations are done run HumanEval-X-Generate with `--mutate_method path/to/generations.json`
        It will load the explanations, generate from them and evaluate.
        """)