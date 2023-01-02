# This template file is adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/templates/new_task.py

# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.
TODO: Write a Short Description of the task.
Homepage: TODO: Add the URL to the task's Homepage here.
"""
from lm_eval.base import Task

# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


# TODO: Replace `NewTask` with the name of your Task.
class NewTask(Task):
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = ""
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def __init__(self):
        super().__init__(
            # TODO: Specify the list of stop words in `stop_words` for the code generation task \
            # and if the evaluation requires executing the generated code in `requires_execution`.
            stop_words=[],
            requires_execution=False,
        )

    def get_dataset(self):
        # TODO: retrieve the evaluation subset from the loaded dataset (e.g. `self.dataset["test"]`)
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return []

    def fewshot_examples(self):
        # TODO: load few-shot examples (from lm_eval/tasks/fewshot_examples) if they exist
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    def get_prompt(self, doc):
        # TODO: build the prompt for the language model from a sample `doc` from the dataset
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return ""

    def get_reference(self, doc):
        # TODO: get the reference solution from a sample `doc` from the dataset
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return ""

    def postprocess_generation(self, generation, idx):
        # TODO: define the postprocessing for the LM generation
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        return ""

    def process_results(self, generations, references):
        # TODO: define how the evaluation score is computed from list of \
        # generations and reference solutions
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        return {}
