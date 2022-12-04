"""CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation
https://arxiv.org/abs/2102.04664

Code to text task from CodeXGlue (documentation generation) for Python subset in a left only setting:
only the function signature is given as prompt

TODO: implement function signature extraction for other languages in the dataset
"""

import re
import os
from mosestokenizer import MosesDetokenizer
from evaluate import load
from lm_eval.base import Task


TRIPLE_QUOTE = '"""'
SINGLE_TRIPLE_QUOTE = "'''"
SPACES4 = " " * 4


class CodeToTextLeft(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "code_x_glue_ct_code_to_text"
    DATASET_NAME = None

    def __init__(self):
        self.DATASET_NAME = "python"
        super().__init__(
            stop_words=["'''", '"""'],
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    @staticmethod
    def standardize_docstring_prompt(prefix):
        """Strips any existing docstring delimiters from the prompt prefix and
        and adds our own delimiter (triple quote) and whitespace.
        Note an edge cases being handled here:
        - codexglue docstring text sometimes contains the docstring delimiters, inconsistently

        source: InCoder evaluation code https://github.com/dpfried/lm-evaluation-harness/
        """

        for delim in [TRIPLE_QUOTE, SINGLE_TRIPLE_QUOTE]:
            if delim in prefix:
                prefix = prefix[: prefix.index(delim)]
                break

        single_single_quote_with_trailing_spaces = re.compile(r'[^\'"][\']\s*$')
        if single_single_quote_with_trailing_spaces.search(prefix):
            prefix = prefix[
                : single_single_quote_with_trailing_spaces.search(prefix).start()
            ]

        single_double_quote_with_trailing_spaces = re.compile(r'[^\'"]["]\s*$')
        if single_double_quote_with_trailing_spaces.search(prefix):
            prefix = prefix[
                : single_double_quote_with_trailing_spaces.search(prefix).start()
            ]

        prefix += TRIPLE_QUOTE
        return prefix

    def get_prompt(self, doc):
        """Generate prompts for Code to text benchmark (documentation generation)
        Prompt =  function signature.
        """
        code = doc["code"]
        # python code includes the docstring
        text = doc["docstring"]
        prompt_prefix = code[: code.index(text)]
        prompt_prefix = self.standardize_docstring_prompt(prompt_prefix)
        return prompt_prefix

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        # deactivate tokenizer parallelism when calling MosesDetokenizer TODO: do it for all refs once
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # docstring_tokens are preprocessed and don't have extra context like variable defs
        docstring = " ".join(doc["docstring_tokens"]).replace("\n", "")
        # some docstrings started with r""" before tokenization but r was kept
        if docstring[0] == "r":
            docstring = docstring[1:]
        with MosesDetokenizer("en") as detokenize:
            docstring = detokenize(docstring.strip().split())
        return docstring

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this Task)
        """
        output = generation.strip().split("\n")[0].strip()
        for delimiter in [TRIPLE_QUOTE, SINGLE_TRIPLE_QUOTE]:
            if delimiter in generation:
                generation = generation[generation.index(delimiter) + 3 :]
                output = generation.strip().split("\n")[0].strip()
                output = output.split(delimiter, 1)[0]
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
