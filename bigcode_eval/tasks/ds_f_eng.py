"""Desc
"""
import fcntl
import json
import logging
from pathlib import Path

from bigcode_eval.tasks.custom_metrics.ds_f_eng_eval import tabpfn_average_accuracy
from bigcode_eval.base import Task

_CITATION = """"""


class DsFEng(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "/Users/ajedrosz/Projects/code_gen_bench/bigcode-evaluation-harness/bigcode_eval/tasks/ds_f_eng/dataset_v1.jsonl"
    DATAFRAMES_URL = ""

    def __init__(self, timeout: float = 3.0):
        super().__init__(
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"],
            requires_execution=True,
        )
        self._ds_f_eng_dir = Path(__file__).parent / "ds_f_eng"
        self._ds_f_eng_dir.mkdir(parents=True, exist_ok=True)
        self._dataframes_dir = self._ds_f_eng_dir / "ds_f_eng_dataframes"
        self.timeout = timeout
        # TODO(ajedrosz): fix once move to hub
        # self._download_dataframes()

    def _download_dataframes(self):
        lock = self._dataframes_dir.with_suffix(".lock")
        with open(lock, "w") as f_lock:
            fcntl.flock(f_lock, fcntl.LOCK_EX)
            if not self._ds_f_eng_dir.exists():
                logging.info(f"Downloading DS-F-ENG dataframes to {self._ds_f_eng_dir}")
                request = requests.get(DsFEng.DATAFRAMES_URL, stream=True)
                zip_file = zipfile.ZipFile(io.BytesIO(request.content))
                zip_file.extractall(self._ds_f_eng_dir)
                logging.info(f"Successfuly extracted all DS-F-ENG dataframes to {self._dataframes_dir}")
            fcntl.flock(f_lock, fcntl.LOCK_UN)

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        # TODO(ajedrosz): fix once move to hub
        return self.dataset["train"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        return doc["prompt"]

    def get_reference(self, doc):
        """Builds the reference solution for the doc.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        return doc["dataframe_id"]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        prompt = self.get_prompt(self.dataset[idx])
        generation = generation[len(prompt) :]
        return self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing references
        """
        return tabpfn_average_accuracy(
            predictions=generations,
            dataframe_paths=[self._dataframes_dir / dataframe_id for dataframe_id in references],
            split_column="split",
            train_split="train",
            test_split="test",
            target_column="target",
            timeout=20.0,
        )
