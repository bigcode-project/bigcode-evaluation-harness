"""How efficient is LLM-generated code? A rigorous & high-standard benchmark
https://arxiv.org/pdf/2406.06647

ENAMEL is a rigorous & high-standard benchmark for evaluating the efficiency of generated
Python code compared with expert-written reference solutions under 142 HumanEval problems

Homepage: https://github.com/q-rz/enamel
"""

_CITATION = """
@article{qiu2024enamel,
  title={How efficient is {LLM}-generated code? A rigorous \& high-standard benchmark},
  author={Qiu, Ruizhong and Zeng, Weiliang Will and Tong, Hanghang and Ezick, James and Lott, Christopher},
  journal={arXiv preprint arXiv:2406.06647},
  year={2024}
}
"""


import pickle
from warnings import warn
import numpy as np
from bigcode_eval.tasks.humaneval import GeneralHumanEval
from bigcode_eval.custom_metrics.enamel_eval import Unpickler, evaluate_all, might_catch_timeout_signal


class GeneralENAMEL(GeneralHumanEval):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "q-rz/enamel"
    DATASET_NAME = "ENAMEL_HumanEval"

    def __init__(self, subset, # list of problem IDs
        hardness=[0., 3., 3., 4.], n_reps = 6, memory_giga=4., timeout_factor=2., tolerence_sec=0.01, tests_path="cache/eval~tests.pkl",
        strip_prompt=True, k=[1, 10, 100],
    ):
        super().__init__(strip_prompt=strip_prompt, k=k, num_workers=1, timeout=None) # each problem has a different time limit
        if isinstance(subset, list):
            self.subset = subset
        else:
            assert subset in self.DATASET_SUBSETS, f"unknown subset {repr(subset)}"
            self.subset = self.DATASET_SUBSETS[subset]
        self.dataset[self.__name__] = self.dataset["ENAMEL_HumanEval"].iloc[np.array(self.subset), :] # TODO
        self.hardness = hardness
        self.n_levels = len(self.hardness)
        self.n_reps = [n_reps if self.hardness[j] else 1 for j in range(self.n_levels)] # no need to repeat if it does not count into the efficiency score
        self.memory_giga = memory_giga
        self.timeout_factor = timeout_factor
        self.tolerence_sec = tolerence_sec
        self.tests_path = tests_path
        # TODO: load tests from tests_path

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset[self.__name__]

    def get_reference(self, doc):
        # TODO: get the reference solution from a sample `doc` from the dataset
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict{str: str}
            sample from the test dataset
        :return: str
        """
        return "" # TODO: include tests

    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        generation = self._stop_at_stop_token(generation, self.stop_words)
        if (not self.warned_dead_loop) and might_catch_timeout_signal(generation):
            warn(might_catch_timeout_signal.WARNING)
            self.warned_dead_loop = True
        return generation

    def process_results(self, generations, references):
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        return evaluate_all(
            generations, references, k=self.k, hardness=self.hardness, n_reps=self.n_reps,
            memory_giga=self.memory_giga, timeout_factor=self.timeout_factor, tolerence_sec=self.tolerence_sec,
        )


def create_task(name, subset):
    class ENAMEL(GeneralENAMEL):
        __name__ = name
        __qualname__ = name
        SUBSET = subset
        def __init__(self, *args, **kwargs):
            super().__init__(subset=self.SUBSET, *args, **kwargs)
    return ENAMEL

def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
    """
    return {
        "enamel": create_task(name="ENAMEL", subset=sorted(set(range(164)) - {2, 23, 41, 45, 53, 60, 71, 92, 97, 99, 102, 123, 124, 135, 137, 138, 144, 148, 156, 157, 159, 160})),
        "enamel-algo": create_task(name="ENAMEL_Algo", subset=[10, 18, 36, 39, 40, 43, 46, 49, 55, 59, 63, 76, 83, 96, 107, 109, 114, 129, 147, 154]),
        "enamel-impl": create_task(name="ENAMEL_Impl", subset=[1, 5, 8, 9, 11, 12, 15, 16, 17, 19, 21, 22, 24, 25, 26, 27, 31, 33, 37, 38, 44, 48, 49, 50, 51, 52, 56, 57, 58, 59, 61, 64, 66, 69, 70, 72, 73, 74, 75, 78, 80, 82, 85, 87, 89, 91, 93, 94, 95, 96, 98, 100, 104, 105, 108, 110, 111, 112, 113, 116, 117, 118, 121, 122, 125, 127, 128, 131, 140, 142, 143, 150, 152, 155, 161]),
    }
