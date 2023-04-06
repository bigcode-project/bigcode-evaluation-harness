from dataclasses import dataclass
from typing import List, Dict

from datasets import Dataset

from lm_eval.base import Task
from evaluate import load
from transformers import AutoTokenizer, PreTrainedTokenizer


_CITATION = """
@article{allamanis2021self,
  title={Self-supervised bug detection and repair},
  author={Allamanis, Miltiadis and Jackson-Flux, Henry and Brockschmidt, Marc},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={27865--27876},
  year={2021}
}
"""


@dataclass(frozen=True)
class PyPiBugsDatasetFeaturesNames:
    INITIAL_STATE: str = "initial_state"
    FINAL_STATE: str = "final_state"


@dataclass(frozen=True)
class PyPiBugsPrompts:
    COMMIT_BEFORE_TOKEN: str = "<commit_before>"
    COMMIT_MSG_TOKEN: str = "<commit_msg>"
    commit_msg: str = f"{COMMIT_MSG_TOKEN} Fix bug"
    COMMIT_AFTER_TOKEN: str = "<commit_after>"


class BugRepair(Task):
    """
    Generate code to fix a bug in a given code snippet.
    NOTE: The only stop word is
    """

    DATASET_PATH: str = "Nadav-Timor/PyPiBugs"
    # DATASET_NAME = None
    dataset_features_names: PyPiBugsDatasetFeaturesNames = (
        PyPiBugsDatasetFeaturesNames()
    )
    prompts: PyPiBugsPrompts = PyPiBugsPrompts()

    def __init__(self) -> None:
        tokenizer_checkpoint: str = "bigcode/santacoder"
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_checkpoint
        )
        super().__init__(
            stop_words=[tokenizer.eos_token],
            requires_execution=True,
        )

    def get_dataset(self) -> Dataset:
        return self.dataset

    def get_prompt(self, doc: Dict) -> str:
        commit_before: str = doc[self.dataset_features_names.INITIAL_STATE]
        commit_after: str = doc[self.dataset_features_names.FINAL_STATE]
        return (
            self.prompts.COMMIT_BEFORE_TOKEN
            + commit_before
            + self.prompts.commit_msg
            + self.prompts.COMMIT_AFTER_TOKEN
            + commit_after
        )

    def get_reference(self, doc: Dict) -> str:
        return doc[self.dataset_features_names.FINAL_STATE]

    def postprocess_generation(self, generation, idx) -> None:
        pass

    def process_results(
        self, generations: List[List[str]], references: List[str]
    ) -> Dict[str, float]:
        """
        Computes the exact match score for the generations.
        """
        exact_match = load("exact_match")
        return exact_match.compute(
            references=references,
            predictions=generations,
        )
