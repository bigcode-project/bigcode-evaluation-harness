"""Execution-Based Evaluation for Open Domain Code Generation
https://arxiv.org/pdf/2212.10481.pdf

The ODEX dataset includes 945 NL-to-Code generation pairs with 1,707 
human-written test cases. ODEX involves NL intents in four natural languages: 
with 439, 90, 164, and 252 samples in English, Spanish, Japanese, and Russian.

Homepage: https://github.com/zorazrw/odex
"""
from evaluate import load
from lm_eval.base import Task

_CITATION = """
@article{wang2022execution,
         title={Execution-Based Evaluation for Open-Domain Code Generation},
         author={Zhiruo Wang, Shuyan Zhou, Daniel Fried, Graham Neubig},
         journal={arXiv preprint arXiv:2212.10481},
         year={2022}
}
"""

def create_task(lang):
    class ODEX(GeneralODEX):
        def __init__(self):
            super().__init__(lang)

    return ODEX


def create_all_tasks():
    """Creates a dictionary of tasks from multiple languages
    :return: {language: task}
        e.g. {en: Task, en: Task, ja: Task, ru: Task}
    """
    return {f"odex-{lang}": create_task(lang) for lang in ["en", "es", "ja", "ru"]}


class GeneralODEX(Task):

    DATASET_PATH = "neulab/odex"
    DATASET_NAME = None

    def __init__(self, lang):
        self.DATASET_NAME = lang
        super().__init__(
            stop_words=["###", "\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"],
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        function_head, function_prefix = doc["prompt"].split("\n")
        docstr = f'    """{doc["intent"]}\n    """'
        code_body = function_prefix.replace("\t", " " * 4)
        return "\n".join([function_head, docstr, code_body])

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(
            [
                doc["test_start"],
                "".join(doc["test"]),
                "",
                f"check({doc['entry_point']})",
            ]
        )

    @staticmethod
    def remove_last_block(string, stop_words):
        """Remove the last block of code containing stop_words for ODEX."""
        for sw in stop_words:
            if sw in string:
                string = string.split(sw)[0]
        return string

    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
            (not used for ODEX)
        :return: str
        """
        return self.remove_last_block(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("code_eval")
        results, _ = code_metric.compute(
            references=references,
            predictions=generations,
        )
        return results