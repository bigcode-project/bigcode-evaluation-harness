from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, NewType, Any

from datasets import Dataset, load_dataset
import datasets

from lm_eval.base import Task
from evaluate import load, EvaluationModule
from transformers import AutoTokenizer, PreTrainedTokenizer

from pathlib import Path

from lm_eval.utils import init_dataclass_from_kwargs

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
    """
    The names of the relevant features in the PyPiBugs dataset.
    (See https://huggingface.co/datasets/Nadav-Timor/PyPiBugs)
    """
    PATH: str = "old_path"
    INITIAL_STATE: str = "initial_state"
    FINAL_STATE: str = "final_state"


@dataclass(frozen=True)
class DatasetConfig:
    """
    The configuration of the dataset.
    :param DATASET_PATH: The path to the dataset on the HuggingFace Hub.
    :param DATASET_SPLIT: The split of the dataset to use. (See HuggingFace's `dataset` library for more info.)
    :param to_shuffle: Whether to shuffle the dataset. The shuffle is done before extracting the few-shot examples.
                       Hence, different seeds will result in different few-shot examples. By default, the dataset is
                       shuffled.
    :param num_of_fewshot_examples: The number of few-shot examples to extract from the dataset. By default, the first
                                    num_of_fewshot_examples examples are used. If the dataset is not shuffled, the
                                    examples will always be the same.
    """
    DATASET_PATH: str = "Nadav-Timor/PyPiBugs"
    DATASET_SPLIT: str = "train"
    to_shuffle: bool = True
    num_of_fewshot_examples: int = 5


@dataclass(frozen=True)
class NewSpecialTokens:
    FILENAME: str = "<NME> "
    COMMIT_BEFORE: str = "<BEF> "
    COMMIT_MSG: str = "<MSG> "
    COMMIT_AFTER: str = "<DFF> "


@dataclass(frozen=True)
class TokenizerConfig:
    tokenizer_checkpoint = "CarperAI/diff-codegen-350m-v2"


EvaluatedMetric = NewType("EvaluatedMetric", Dict[str, Any])


class ProgramRepair(Task):
    """
    Generate code to fix a bug in a given code snippet, inspired by https://arxiv.org/abs/2105.12787 and
    https://carper.ai/diff-models-a-new-way-to-edit-code/.

    The task supports zero-shot and few-shot evaluation. The task takes the first num_of_fewshot_examples examples from
    the dataset and uses them as few-shot examples. Note that the zero-shot examples are always the same (the first
    num_of_fewshot_examples examples from the dataset). Set num_of_fewshot_examples to 0 to evaluate in zero-shot mode.
    """
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        self.args: Dict[str, Any] = kwargs
        # Dataset
        self.dataset_config: DatasetConfig = init_dataclass_from_kwargs(cls=DatasetConfig, kwargs=kwargs)
        self.dataset_features_names: PyPiBugsDatasetFeaturesNames = PyPiBugsDatasetFeaturesNames()
        self.seed: int = self.args.get("seed", 0)
        # Tokenization
        self.tokenizer_config: TokenizerConfig = init_dataclass_from_kwargs(cls=TokenizerConfig, kwargs=kwargs)
        self.new_special_tokens: NewSpecialTokens = NewSpecialTokens()
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_config.tokenizer_checkpoint
        )
        # NOTE: Names were refactored. `self.special_tokens` is now `self.tokenizer_config.new_special_tokens`.
        # new_special_tokens: Set[str] = set(asdict(self.special_tokens).values())
        # self.stop_words: List[str] = list(
        #     set(self.tokenizer.all_special_tokens_extended).union(new_special_tokens)
        # )
        self.stop_words: List[str] = []
        self.requires_execution: bool = False
        # Extract few-shot examples from the dataset
        self.dataset: Dataset = load_dataset(self.dataset_config.DATASET_PATH, split=self.dataset_config.DATASET_SPLIT)
        datasets.disable_caching()
        if self.dataset_config.to_shuffle:
            self.dataset = self.dataset.shuffle(seed=self.seed)
        self.fewshot_prompt: str = self._get_fewshot_examples_prompt(
            num_of_examples=self.dataset_config.num_of_fewshot_examples
        )
        self.dataset = self.dataset.select(
            range(self.dataset_config.num_of_fewshot_examples, len(self.dataset))
        )

    def get_dataset(self) -> Dataset:
        return self.dataset

    def _get_single_example_prompt(self, doc: Dict, is_fewshot_example: bool) -> str:
        filename: str = Path(doc[self.dataset_features_names.PATH]).name
        commit_before: str = doc[self.dataset_features_names.INITIAL_STATE]
        commit_after: str = doc[self.dataset_features_names.FINAL_STATE]
        ret: str = f"""\
{self.new_special_tokens.FILENAME} {filename}

{self.new_special_tokens.COMMIT_BEFORE} {commit_before}

{self.new_special_tokens.COMMIT_MSG} # Fixed a bug.

{self.new_special_tokens.COMMIT_AFTER} """
        if is_fewshot_example:
            ret += f"{commit_after}\n{self.tokenizer.bos_token}"
        return ret

    def _get_fewshot_examples_prompt(self, num_of_examples: int) -> str:
        """
        Given a DatasetDict, for its first num_of_examples examples, returns a prompt that contains all of them.
        :param num_of_examples:
        :return:
        """
        if num_of_examples < 1:
            return ""
        examples: List[str] = [
            self._get_single_example_prompt(doc, is_fewshot_example=True)
            for doc in self.dataset.select(range(0, num_of_examples))
        ]
        return "\n".join(examples)

    def get_prompt(self, doc: Dict) -> str:
        ret: str = self.fewshot_prompt
        if ret != "":
            ret += "\n"
        ret += self._get_single_example_prompt(doc, is_fewshot_example=False)
        return ret

    def get_reference(self, doc: Dict) -> str:
        return doc[self.dataset_features_names.FINAL_STATE]

    def postprocess_generation(self, generation: str, idx: int = -1) -> str:
        """
        Return the generation until a stop token is encountered. Remove blank lines.
        """
        def slice_until_stop_token() -> None:
            """
            Slice the generation until a stop token is encountered.
            """
            nonlocal generation
            for stop_token in self.stop_words:
                if stop_token in generation:
                    generation = generation[: generation.index(stop_token)]
                    break

        slice_until_stop_token()
        return generation

    def process_results(
        self, generations: List[List[str]], references: List[str], to_strip_surrounding_whitespaces: bool = True
    ) -> EvaluatedMetric:
        """
        Returns the number of references that have an exact match generation divided by the number of references.
        For each reference and its corresponding generations, compute the maximal exact match score. The maximal
        exact match score is 1 if there is at least one generation that is equal to the reference, and 0 otherwise.
        Additionally, returns a histogram of the ratio of references that have an exact match score equal to the key.
        The first metric described is the value of the histogram at key 0.0.
        """
        exact_match: str = "exact_match"
        exact_match_avg_max: str = f"ratio_of_references_with_at_least_one_{exact_match}"
        exact_match_hist: str = f"histogram_of_the_ratio_of_references_that_have_{exact_match}_equal_to_key"
        metric: EvaluationModule = load(exact_match)
        ret: EvaluatedMetric = EvaluatedMetric({exact_match_avg_max: 0.0, exact_match_hist: Counter()})
        i: int
        corresponding_generations: List[str]
        for i, corresponding_generations in enumerate(generations):
            # Strip surrounding whitespaces
            reference: str = references[i]
            if to_strip_surrounding_whitespaces:
                reference = reference.strip()
                corresponding_generations = [gen.strip() for gen in corresponding_generations]
            curr: EvaluatedMetric = metric.compute(
                predictions=corresponding_generations,
                references=[reference] * len(corresponding_generations),
            )
            ret[exact_match_avg_max] += curr[exact_match] > 0
            ret[exact_match_hist][curr[exact_match]] += 1
        num_of_samples: int = len(generations)
        ret[exact_match_avg_max] /= num_of_samples
        ret[exact_match_hist] = {key: value / num_of_samples for key, value in ret[exact_match_hist].items()}
        return ret
