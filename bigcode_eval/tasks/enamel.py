"""How efficient is LLM-generated code? A rigorous & high-standard benchmark
https://arxiv.org/pdf/2406.06647

ENAMEL is a rigorous & high-standard benchmark for evaluating the efficiency of generated code
compared with **expert-written** reference solutions under 142 HumanEval problems

Homepage: https://github.com/q-rz/enamel
"""

from warnings import warn
from bigcode_eval.humaneval import GeneralHumanEval

_CITATION = """
@article{qiu2024enamel,
  title={How efficient is {LLM}-generated code? A rigorous \& high-standard benchmark},
  author={Qiu, Ruizhong and Zeng, Weiliang Will and Tong, Hanghang and Ezick, James and Lott, Christopher},
  journal={arXiv preprint arXiv:2406.06647},
  year={2024}
}
"""


class ENAMEL(GeneralHumanEval):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "q-rz/enamel"
    DATASET_NAME = None

    def __init__(self, strip_prompt, k=[1, 10, 100], num_workers=16, timeout_factor=): # TODO
        super().__init__(strip_prompt=strip_prompt, k=k, num_workers=num_workers, timeout=None)
        # TODO

    def get_dataset(self):
        # TODO: retrieve the evaluation subset from the loaded dataset (e.g. `self.dataset["test"]`)
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return []

    def fewshot_examples(self):
        # TODO: load few-shot examples (from bigcode_eval/tasks/fewshot_examples) if they exist
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
