"""How efficient is LLM-generated code? A rigorous & high-standard benchmark
https://arxiv.org/pdf/2406.06647

ENAMEL is a rigorous & high-standard benchmark for evaluating the efficiency of generated
Python code compared with expert-written reference solutions under 142 HumanEval problems

Homepage: https://github.com/q-rz/enamel
"""

from warnings import warn
from bigcode_eval.tasks.humaneval import GeneralHumanEval

_CITATION = """
@article{qiu2024enamel,
  title={How efficient is {LLM}-generated code? A rigorous \& high-standard benchmark},
  author={Qiu, Ruizhong and Zeng, Weiliang Will and Tong, Hanghang and Ezick, James and Lott, Christopher},
  journal={arXiv preprint arXiv:2406.06647},
  year={2024}
}
"""


class GeneralENAMEL(GeneralHumanEval):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "q-rz/enamel"
    DATASET_NAME = "ENAMEL_HumanEval"
    DATASET_SUBSETS = {
        "ENAMEL": sorted(set(range(164)) - {2, 23, 41, 45, 53, 60, 71, 92, 97, 99, 102, 123, 124, 135, 137, 138, 144, 148, 156, 157, 159, 160}),
        "ENAMEL_Algo": [10, 18, 36, 39, 40, 43, 46, 49, 55, 59, 63, 76, 83, 96, 107, 109, 114, 129, 147, 154],
        "ENAMEL_Impl": [1, 5, 8, 9, 11, 12, 15, 16, 17, 19, 21, 22, 24, 25, 26, 27, 31, 33, 37, 38, 44, 48, 49, 50, 51, 52, 56, 57, 58, 59, 61, 64, 66, 69, 70, 72, 73, 74, 75, 78, 80, 82, 85, 87, 89, 91, 93, 94, 95, 96, 98, 100, 104, 105, 108, 110, 111, 112, 113, 116, 117, 118, 121, 122, 125, 127, 128, 131, 140, 142, 143, 150, 152, 155, 161],
    }

    def __init__(self,
        subset, # list of problem IDs, or one of the predefined subsets
        hardness=[0., 3., 3., 4.], memory_giga=4., timeout_factor=2., tolerence_sec=0.01, tests_path="cache/eval~tests.pkl",
        strip_prompt=True, k=[1, 10, 100], num_workers=16,
    ):
        super().__init__(strip_prompt=strip_prompt, k=k, num_workers=num_workers, timeout=None) # each problem has a different time limit
        if isinstance(subset, list):
            self.subset = subset
        else:
            assert subset in self.DATASET_SUBSETS, f"unknown subset {repr(subset)}"
            self.subset = self.DATASET_SUBSETS[subset]
        self.hardness = hardness
        self.memory_giga = memory_giga
        self.timeout_factor = timeout_factor
        self.tolerence_sec = tolerence_sec
        self.tests_path = tests_path
        # TODO: load dataset and tests

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["ENAMEL_HumanEval"].iloc[np.array(self.subset), :]

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


def create_task(subset):
    class ENAMEL(GeneralENAMEL):
        __name__ = subset
        __qualname__ = subset
        def __init__(self, *args, **kwargs):
            super().__init__(subset = subset, *args, **kwargs)
    return ENAMEL

def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
    """
    return {
        "enamel": create_task(subset = "ENAMEL"),
        "enamel-algo": create_task(subset = "ENAMEL_Algo"),
        "enamel-impl": create_task(subset = "ENAMEL_Impl"),
    }
