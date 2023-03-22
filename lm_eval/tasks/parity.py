"""Parity bug fixing task."""

import re
from evaluate import load
from lm_eval.base import Task
import itertools

def mutate_code(
    n_bugs: int = 5, task: str = "parity", mutate_method="prompt"
):
    """
    From https://github.com/CarperAI/OpenELM/blob/e6402a0696096011572152334ccbe049f89c332e/src/openelm/utils/code_eval.py
    
    Mutate code to create n bugs. Output the prompt in diff format.
    Args:
        n_bugs: number of bugs to introduce (from 1 to 5).
        task: (Optional) the task to be performed.
        mutate_method: (Optional) 'diff' or 'prompt',
        corresponding to diff mutation or prompt mutation.
    Returns:
        mutated_code, function_string
    """
    mutation_templates = {
        "diff": [
            f"<NME> {task}.py\n<BEF> ",
            "",  # placeholder for the context, e.g., the buggy code
            "\n<MSG> Fixed bugs",
        ],
        "prompt": [
            "# A buggy implementation\n#!/usr/bin/python3\n",
            "",  # placeholder for the context, e.g., the buggy code
            "\n# Fixed bugs\ndef parity_fixed(", # Modified to add the function name
        ],
    }
    mutation_template = mutation_templates[mutate_method]
    if task == "parity":
        variables = ["b", "b", "b", "b", 2]
        for i in range(n_bugs):
            variables[i] = "c" if i < 4 else 3
        func_str = (
            'def parity(b1,b2,b3,b4):\n    """Return binary parity of a sequence of input bits.'
            ' Return 0 for even parity, 1 for odd parity."""\n    bit_sum = sum(['
            "{}1,{}2,{}3,{}4])\n    return bit_sum % {}".format(*variables)
        )
        mutation_template[1] = func_str
        return "".join(mutation_template), func_str
    else:
        raise ValueError(f"Unknown task: {task}")

# https://huggingface.co/spaces/evaluate-metric/code_eval
# https://github.com/CarperAI/OpenELM/blob/e6402a0696096011572152334ccbe049f89c332e/src/openelm/utils/code_eval.py#L131
def parity_reference(b1, b2, b3, b4):
    """
    Return binary parity of a sequence of input bits.
    Return 0 for even parity, 1 for odd parity.
    """
    bit_sum = sum([b1, b2, b3, b4])
    return bit_sum % 2


class Parity(Task):
    def __init__(self):
        super().__init__(
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"],
            requires_execution=True,
        )
        self.mutate_method = "prompt"

        if self.mutate_method == "diff":
            self.parity_tests = "assert " + " and ".join([
                f"({parity_reference(*i)} == parity{i})" for i in itertools.product(range(2), repeat=4)
            ])
        else:
            self.parity_tests = "assert " + " and ".join([
                f"({parity_reference(*i)} == parity_fixed{i})" for i in itertools.product(range(2), repeat=4)
            ])

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return [1, 2, 3, 4, 5]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return mutate_code(n_bugs=doc, task="parity", mutate_method=self.mutate_method)[0]

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return []

    @staticmethod
    def remove_last_block(string, stop_words):
        # Remove the last block of the code containing stop_words
        string_list = re.split("(%s)" % "|".join(stop_words), string)
        # last string should be ""
        return "".join(string_list[:-2])

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
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
        out = {}
        bugs = self.get_dataset()
        assert len(generations) == len(bugs)
        for num_bugs in bugs:
            results, _ = code_metric.compute(
                references=[self.parity_tests],
                predictions=[generations[num_bugs - 1]],
            )
            out[num_bugs + "_bugs"] = results
        return out

