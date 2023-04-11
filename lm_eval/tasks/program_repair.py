from dataclasses import dataclass, asdict
from typing import List, Dict, Set, NewType, Optional

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

_ADDITIONAL_CITATION = """
@article{bradley2023diffmodels,
  title   = "Diff Models - A New Way to Edit Code",
  author  = "Bradley, Herbie and Fan, Honglu and Saini, Harry and Adithyan, Reshinth and Purohit, Shivanshu and Lehman, Joel",
  journal = "CarperAI Blog",
  year    = "2023",
  month   = "Jan",
  url     = "https://carper.ai/diff-model/"
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


class ProgramRepair(Task):
    """
    Generate code to fix a bug in a given code snippet, inspired by https://arxiv.org/abs/2105.12787 and
    https://carper.ai/diff-models-a-new-way-to-edit-code/.

    The task supports zero-shot and few-shot evaluation. The task takes the first NUM_OF_FEWSHOT_EXAMPLES examples from
    the dataset and uses them as few-shot examples. Note that the zero-shot examples are always the same (the first
    NUM_OF_FEWSHOT_EXAMPLES examples from the dataset). Set NUM_OF_FEWSHOT_EXAMPLES to 0 to evaluate in zero-shot mode.
    """
    DATASET_PATH: str = "Nadav-Timor/PyPiBugs"
    DATASET_SPLIT: str = "train"
    NUM_OF_FEWSHOT_EXAMPLES: int = 5
    TOKENIZER_CHECKPOINT = "bigcode/santacoder"
    dataset_features_names: PyPiBugsDatasetFeaturesNames = PyPiBugsDatasetFeaturesNames()
    special_tokens: SpecialTokens = SpecialTokens()

    def __init__(self) -> None:
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.TOKENIZER_CHECKPOINT
        )
        new_special_tokens: Set[str] = set(asdict(self.special_tokens).values())
        self.stop_words: List[str] = list(set(self.tokenizer.all_special_tokens_extended).union(new_special_tokens))
        super().__init__(stop_words=self.stop_words, requires_execution=False)
        # Extract few-shot examples from the dataset
        self.dataset: Dataset = self.dataset[self.DATASET_SPLIT]
        self.fewshot_prompt: str = self._get_fewshot_examples_prompt(num_of_examples=self.NUM_OF_FEWSHOT_EXAMPLES)
        self.dataset = self.dataset.select(range(self.NUM_OF_FEWSHOT_EXAMPLES, len(self.dataset)))

    def get_dataset(self) -> Dataset:
        return self.dataset

    def _get_single_example_prompt(self, doc: Dict, is_fewshot_example: bool) -> str:
        commit_before: str = doc[self.dataset_features_names.INITIAL_STATE]
        commit_after: str = doc[self.dataset_features_names.FINAL_STATE]
        ret: str = (
            self.tokenizer.bos_token
            + self.special_tokens.COMMIT_BEFORE
            + commit_before
            + self.special_tokens.COMMIT_MSG
            + self.special_tokens.COMMIT_AFTER
        )
        if is_fewshot_example:
            ret += commit_after
        return ret

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
        ret: str = self.fewshot_prompt + '\n' + self._get_single_example_prompt(doc, is_fewshot_example=False)
        # print('***************')
        # print('Prompt:')
        # print(ret)
        # print('***************')
        return ret

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
        Returns the number of references that has an exact match generation divided by the number of references.
        For each reference and its corresponding generations, compute the maximal exact match score. The maximal
        exact match score is 1 if there is at least one generation that is equal to the reference, and 0 otherwise.
        """
        metric_name: str = "exact_match"
        avg_metric_name: str = f"avg_{metric_name}"
        metric = load(metric_name)
        ret: EvaluatedMetric = EvaluatedMetric({avg_metric_name: 0.0})
        reference: str
        corresponding_generations: List[str]
        for reference, corresponding_generations in zip(references, generations):
            curr: EvaluatedMetric = metric.compute(predictions=corresponding_generations, references=[reference] * len(corresponding_generations))
            ret[avg_metric_name] += curr[metric_name] > 0
        ret[avg_metric_name] /= len(references)
        return ret
