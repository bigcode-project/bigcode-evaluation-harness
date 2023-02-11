"""MCoNaLa: A Benchmark for Code Generation from Multiple Natural Languages
https://arxiv.org/pdf/2203.08388.pdf

MCoNaLa is a Multilingual Code/Natural Language Challenge dataset with 
896 NL-Code pairs in three languages: Spanish, Japanese, and Russian.

https://github.com/zorazrw/multilingual-conala
"""
import re
from typing import List
from lm_eval.base import Task

import evaluate

bleu_eval_metric = evaluate.load("bleu")

_CITATION = """
@article{wang2022mconala,
  title={MCoNaLa: A Benchmark for Code Generation from Multiple Natural Languages},
  author={Zhiruo Wang, Grace Cuenca, Shuyan Zhou, Frank F. Xu, Graham Neubig},
  journal={arXiv preprint arXiv:2203.08388},
  year={2022}
}
"""


def tokenize_for_bleu_eval(code: str) -> List[str]:
    code = re.sub(r"([^A-Za-z0-9_])", r" \1 ", code)
    code = re.sub(r"([a-z])([A-Z])", r"\1 \2", code)
    code = re.sub(r"\s+", " ", code)
    code = code.replace('"', "`")
    code = code.replace("'", "`")
    tokens = [t for t in code.split(" ") if t]
    if not tokens:
        tokens.extend(["", ""])  # len(hyp) > 1 or bleu zero-division error
    return tokens


class GeneralMCoNaLa(Task):

    DATASET_PATH = "neulab/mconala"
    DATASET_NAME = None

    def __init__(self, lang):
        self.DATASET_NAME = lang
        super().__init__(
            stop_words=[],
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["rewritten_intent"]

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["snippet"]

    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        return generation

    def process_results(self, generations, references):
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
        generations = [
            " ".join(tokenize_for_bleu_eval(gen_list[0])) for gen_list in generations
        ]
        references = [[" ".join(tokenize_for_bleu_eval(ref))] for ref in references]
        bleu_score = bleu_eval_metric.compute(
            predictions=generations,
            references=references,
            max_order=4,
            smooth=True,
        )
        return bleu_score


def create_task(lang):
    class MCoNaLa(GeneralMCoNaLa):
        def __init__(self):
            super().__init__(lang)

    return MCoNaLa


def create_all_tasks():
    """Creates a dictionary of tasks from multiple languages
    :return: {language: task}
        e.g. {es: Task, ja: Task, ru: Task}
    """
    return {f"mconala-{lang}": create_task(lang) for lang in ["es", "ja", "ru"]}
