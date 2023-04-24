# This template file is adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/templates/new_task.py
"""
LEARNING PERFORMANCE-IMPROVING CODE EDITS 🥧
https://arxiv.org/pdf/2302.07867.pdf

Homepage: https://pie4perf.com/
"""
import logging
import subprocess
from lm_eval.base import Task
from lm_eval.tasks.custom_metrics.pie_perf_metric.code_exec import compute


_CITATION = """
@article{madaan2023learning,
    title={Learning Performance-Improving Code Edits},
    author={Madaan, Aman and Shypula, Alexander and Alon, Uri and Hashemi, Milad and Ranganathan, Parthasarathy and Yang, Yiming and Neubig, Graham and Yazdanbakhsh, Amir},
    journal={arXiv preprint arXiv:2302.07867},
    year={2023}
}
"""

_COMMAND = '''
accelerate launch  main.py --model "Salesforce/codegen-350M-mono" --tasks "pieperf" --temperature 0.7 --do_sample True --n_samples 100 --batch_size 10 --allow_code_execution
'''


def create_all_tasks():
    task = PiePerf

    return {
        f"{task.__name__.lower()}": create_task(task)
    }


def create_task(cls):
    class PIE(cls):
        def __init__(self):
            super().__init__()

    return PIE


class PiePerf(Task):
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "rootacess/pie-perf"

    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "codegen_1shot_test"

    #todo: Removed the stopwords
    def __init__(self):
        super().__init__(
            stop_words=[],
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["codegen_1shot_test"]

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        prompt = doc['input']
        return prompt

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc['target']

    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        question_sep = "# slower version:"
        answer_sep = "# optimized version of the same code:"

        output_code = generation.replace("\n\n\n\n\n", "")
        output_code = output_code.split(question_sep)[-1]
        output_code = output_code.split(answer_sep)[-1].strip()

        problem_id = self.dataset['codegen_1shot_test']['problem_id'][idx]
        submission_id = self.dataset['codegen_1shot_test']['submission_id_v0'][idx]

        return output_code, problem_id, submission_id

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
        limit = len(references)

        # preparing testcases
        cmd = "git clone https://huggingface.co/datasets/rootacess/pie-perf-testcases lm_eval/tasks/custom_metrics/pie_perf_metric/public_test_cases"
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        logging.error(f'An error occurred: {error}')

        # running evaluations
        res = compute(generations, references, dataset=self.get_dataset()[:limit])

        # cleaning up
        cmd = "rm -rf lm_eval/tasks/custom_metrics/pie_perf_metric/public_test_cases"
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        logging.error(error)
        return res
