import json
from evaluate import load
from lm_eval.base import Task


class Concode(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "code_x_glue_tc_text_to_code"

    def __init__(self):
        super().__init__(
            stop_words=["\n"],
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["validation"]

    @staticmethod
    def two_shot_prompt(entry, text, examples):
        """Two shot prompt format as instructions & solutions"""
        instrcution1 = "\nInstruction:\n" + examples["instruction1"]
        solution1 = "\nSolution:\n" + examples["solution1"]
        instrcution2 = "\nInstruction:\n" + examples["instruction2"]
        solution2 = "\nSolution:\n" + examples["solution2"]
        examples = entry + instrcution1 + solution1 + instrcution2 + solution2
        prompt = examples + "\nInstruction:\n" + text + "\nSolution:\n"
        return prompt

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        with open(
            "few_shot_examples/concode_few_shot_prompts.json", "r"
        ) as file:
            examples = json.load(file)
        text = doc["nl"].split("concode_field_sep")[0].strip()
        if text.endswith("."):
            text = text[:-1].strip()
        entry = "Answer the following instructions in a one line of Java code:\n"
        prompt = self.two_shot_prompt(entry, text, examples)
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["code"]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        output = generation.split("Solution:\n", 3)[-1]
        output = output.split("\n")[0]
        return output

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing references
        """
        bleu = load("bleu")
        gens = [gen[0] for gen in generations]
        results = bleu.compute(
            references=references, predictions=gens, max_order=4, smooth=True
        )["bleu"]
        return results
