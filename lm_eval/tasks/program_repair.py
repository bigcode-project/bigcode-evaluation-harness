from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, NewType, Any, Set, Iterable

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
    tokenizer_checkpoint = "Salesforce/codegen-350M-mono"


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
        self.dataset_config: DatasetConfig = init_dataclass_from_kwargs(
            cls=DatasetConfig, kwargs=kwargs
        )
        self.dataset_features_names: PyPiBugsDatasetFeaturesNames = (
            PyPiBugsDatasetFeaturesNames()
        )
        self.seed: int = self.args.get("seed", 0)
        # Tokenization
        self.tokenizer_config: TokenizerConfig = init_dataclass_from_kwargs(
            cls=TokenizerConfig, kwargs=kwargs
        )
        self.new_special_tokens: NewSpecialTokens = NewSpecialTokens()
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_config.tokenizer_checkpoint
        )
        self.stop_words: List[str] = []
        self.requires_execution: bool = False
        # Extract few-shot examples from the dataset
        self.dataset: Dataset = load_dataset(
            self.dataset_config.DATASET_PATH, split=self.dataset_config.DATASET_SPLIT
        )
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
        return generation

    def process_results(
        self,
        generations: List[List[str]],
        references: List[str],
        to_strip_surrounding_lines_and_leading_substrings_from_generation: bool = True,
        to_strip_surrounding_whitespaces: bool = True,
    ) -> EvaluatedMetric:
        """
        Returns the ratio of references that have an exact match generation divided by the number of references.
        For each reference and its corresponding generations, compute the maximal exact match score. The maximal
        exact match score is 1 if there is at least one generation that is equal to the reference, and 0 otherwise.
        Additionally, returns a histogram of the ratio of references that have an exact match score equal to the key.
        If `to_strip_surrounding_lines_and_leading_substrings_from_generation` is True, extract the generation while
        ignoring the new special tokens at the beginning of lines, based on CarperAI's diff models evaluation method,
        https://carper.ai/diff-models-a-new-way-to-edit-code.
        """

        n: int = len(generations)

        if to_strip_surrounding_whitespaces:
            references: List[str] = [reference.strip() for reference in references]

        if to_strip_surrounding_lines_and_leading_substrings_from_generation:
            new_special_tokens: Set[str] = set(asdict(self.new_special_tokens).values())
            for i in range(n):
                reference: str = references[i]
                for j in range(len(generations[i])):
                    generations[i][j]: str = extract_patch(
                        patch=reference,
                        multiple_patches=generations[i][j],
                        leading_substrings_to_remove=new_special_tokens,
                    )
                    if to_strip_surrounding_whitespaces:
                        generations[i][j]: str = generations[i][j].strip()

        exact_match: str = "exact_match"
        exact_match_avg_max: str = (
            f"ratio_of_references_with_at_least_one_{exact_match}"
        )
        exact_match_hist: str = (
            f"histogram_of_the_ratio_of_references_that_have_{exact_match}_equal_to_key"
        )
        num_of_references: str = "num_of_references"
        ret: EvaluatedMetric = EvaluatedMetric(
            {
                exact_match_avg_max: 0.0,
                exact_match_hist: Counter(),
                num_of_references: n,
            }
        )
        metric: EvaluationModule = load(exact_match)
        i: int
        for i in range(len(generations)):
            # Strip surrounding whitespaces
            reference: str = references[i]
            if to_strip_surrounding_whitespaces:
                reference = reference.strip()
                generations[i] = [
                    gen.strip() for gen in generations[i]
                ]
            curr: EvaluatedMetric = metric.compute(
                predictions=generations[i],
                references=[reference] * len(generations[i]),
            )
            ret[exact_match_avg_max] += curr[exact_match] > 0
            ret[exact_match_hist][curr[exact_match]] += 1
        ret[exact_match_avg_max] /= n
        ret[exact_match_hist] = {
            key: value / n for key, value in ret[exact_match_hist].items()
        }
        return ret


def extract_patch(
    patch: str,
    multiple_patches: str,
    leading_substrings_to_remove: Iterable[str],
) -> str:
    """
    Given a patch and a string that may contain multiple patches, returns the patch that is the most similar to the
    given target patch. The similarity is measured by the number of lines that are equal between the two patches.
    :param patch: The target patch.
    :param multiple_patches: A string.
    :param leading_substrings_to_remove: A list of substrings to remove from the beginning of each line.
    :return: The patch that is the most similar to the given target patch.
    """
    patch_lines: List[str] = patch.split("\n")
    multiple_patches_lines: List[str] = multiple_patches.split("\n")
    multiple_patches_lines = [
        remove_leading_substrings(line=line, substrings=leading_substrings_to_remove)
        for line in multiple_patches_lines
    ]

    max_matching_lines = 0
    matched_patch_start = -1

    for i in range(len(multiple_patches_lines)):
        matching_lines = 0
        for j, patch_line in enumerate(patch_lines):
            if (
                i + j < len(multiple_patches_lines)
                and patch_line == multiple_patches_lines[i + j]
            ):
                matching_lines += 1
        if matching_lines >= max_matching_lines:
            max_matching_lines = matching_lines
            matched_patch_start = i

    matched_patch = multiple_patches_lines[
        matched_patch_start : matched_patch_start + len(patch_lines)
    ]
    return "\n".join(matched_patch)


def remove_leading_substrings(line: str, substrings: Iterable[str]) -> str:
    substring: str
    for substring in substrings:
        if line.startswith(substring):
            line: str = line[len(substring) :].lstrip()
            break
    return line
