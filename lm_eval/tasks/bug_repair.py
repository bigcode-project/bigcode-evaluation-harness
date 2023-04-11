from dataclasses import dataclass, asdict
from typing import List, Dict, Set, NewType

from datasets import Dataset, load_dataset

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
class SpecialTokens:
    COMMIT_BEFORE: str = "<commit-before>\n"
    COMMIT_MSG: str = "\n<commit-msg> Fix bug.\n"
    COMMIT_AFTER: str = "<commit-after>\n"


EvaluatedMetric = NewType('EvaluatedMetric', Dict[str, float])


class BugRepair(Task):
    """
    Generate code to fix a bug in a given code snippet.
    """
    DATASET_PATH: str = "Nadav-Timor/PyPiBugs"
    DATASET_SPLIT: str = "train"
    NUM_OF_FEWSHOT_EXAMPLES: int = 8
    TOKENIZER_CHECKPOINT = "bigcode/santacoder"
    dataset_features_names: PyPiBugsDatasetFeaturesNames = PyPiBugsDatasetFeaturesNames()
    special_tokens: SpecialTokens = SpecialTokens()

    def __init__(self) -> None:
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.TOKENIZER_CHECKPOINT
        )
        new_special_tokens: Set[str] = set(asdict(self.special_tokens).values())
        self.stop_words: List[str] = list(set(self.tokenizer.all_special_tokens_extended).union(new_special_tokens))
        # Load original dataset
        self.dataset: Dataset = load_dataset(self.DATASET_PATH, split=self.DATASET_SPLIT)
        # Split the dataset to fewshot and non-fewshot examples
        self.fewshot_prompt: str = self._get_fewshot_examples_prompt(num_of_examples=self.NUM_OF_FEWSHOT_EXAMPLES)
        self.dataset = self.dataset.select(range(self.NUM_OF_FEWSHOT_EXAMPLES, len(self.dataset)))
        self.requires_execution: bool = False

    def get_dataset(self) -> Dataset:
        return self.dataset

    def _get_single_example_prompt(self, doc: Dict, is_fewshot_example: bool) -> str:
        commit_before: str = doc[self.dataset_features_names.INITIAL_STATE]
        commit_after: str = doc[self.dataset_features_names.FINAL_STATE]
        prompt: str = (
            self.tokenizer.bos_token
            + self.special_tokens.COMMIT_BEFORE
            + commit_before
            + self.tokenizer.eos_token
            + self.special_tokens.COMMIT_MSG
            + self.tokenizer.eos_token
            + self.special_tokens.COMMIT_AFTER
        )
        if is_fewshot_example:
            prompt += commit_after + self.tokenizer.eos_token
        return prompt

    def _get_fewshot_examples_prompt(self, num_of_examples: int) -> str:
        """
        Given a DatasetDict, for its first num_of_examples examples, returns a prompt that contains all of them.
        :param num_of_examples:
        :return:
        """
        examples: List[str] = [
            self._get_single_example_prompt(doc, is_fewshot_example=True)
            for doc in self.dataset.select(range(0, num_of_examples))
        ]
        return '\n'.join(examples)

    def get_prompt(self, doc: Dict) -> str:
        return self.fewshot_prompt + '\n' + self._get_single_example_prompt(doc, is_fewshot_example=False)

    def get_reference(self, doc: Dict) -> str:
        return doc[self.dataset_features_names.FINAL_STATE]

    def postprocess_generation(self, generation, idx) -> str:
        """
        Return the generation until a stop token is encountered.
        """
        for stop_word in self.stop_words:
            if stop_word in generation:
                generation = generation[:generation.index(stop_word)]
        return generation

    def process_results(
        self, generations: List[List[str]], references: List[str]
    ) -> EvaluatedMetric:
        """
        For each reference and its corresponding generations, compute the average exact match score. The average score
        is 1 if all generations are equal to the reference. The average score is 0 if none of the generations are equal
        to the reference. The average score is in the range (0, 1) otherwise.
        """
        metric_name: str = "exact_match"
        avg_metric_name: str = f"avg_{metric_name}"
        metric = load(metric_name)
        ret: EvaluatedMetric = EvaluatedMetric({avg_metric_name: 0.0})
        reference: str
        predictions: List[str]
        for reference, predictions in zip(references, generations):
            curr: EvaluatedMetric = metric.compute(predictions=predictions, references=[reference] * len(predictions))
            ret[avg_metric_name] += curr[metric_name]
        ret[avg_metric_name] /= len(references)
        return ret
