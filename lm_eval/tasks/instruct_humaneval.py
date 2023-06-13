"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests. 
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""

from evaluate import load
from lm_eval.base import Task
from lm_eval.utils import remove_after_return

_CITATION = """
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""


def create_all_tasks():
    """Creates a dictionary of tasks corresponding for the 2 settings currently available
    - instruction with code completion: we provide function signature/imports.. to the model after the instruction
    - instruction to code generation: we only give the instruction without the function signature/imports..
    """
    return {
        "humaneval-with-context": InstructHumanEvalWithContext,
        "humaneval-without-context": InstructHumanEvalWithoutContext,
    }


class InstructHumanEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "codeparrot/instructhumaneval"

    DATASET_NAME = None

    def __init__(self):
        super().__init__(
            stop_words=[
                "if __name__",
                "\nprint",
                "\nclass"
            ],
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        pass

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing references
        """
        code_metric = load("code_eval")
        results, _ = code_metric.compute(
            references=references,
            predictions=generations,
        )
        return results


class InstructHumanEvalWithContext(InstructHumanEval):
    def __init__(self):
        super().__init__()

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return {"instruction": doc["instruction"], "context": doc["context"]}
    
    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        generation = self._stop_at_stop_token(generation, self.stop_words)
        
        function_name = self.get_dataset()["entry_point"][idx]
        func_index = generation.find(f"def {function_name}")
        return generation[0:func_index]+remove_after_return(generation[func_index:])


class InstructHumanEvalWithoutContext(InstructHumanEval):
    def __init__(self):
        super().__init__()

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return {"instruction": doc["instruction"], "context": ""}

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        example = self.get_dataset()[idx]
        prompt, function_name = example["context"], example["entry_point"]
        prefix = prompt[0 : prompt.find(f"def {function_name}")]

        sep_index = generation.find("```")
        if sep_index == -1:
            pass
        else:
            if (
                generation[sep_index + len("```") : sep_index + len("```python")]
                == "python"
            ):
                generation = generation[sep_index + len("```python") :]
            else:
                generation = generation[sep_index + len("```") :]

        generation = self._stop_at_stop_token(generation, self.stop_words)

        func_index = generation.find(f"def {function_name}")
        if func_index == -1:
            func_index = 0
        return_index = generation[func_index:].rfind("  return ")
        if return_index == -1:
            return_index = 0

        j = func_index + return_index
        n = len(generation)

        while j < n and generation[j] != "\n":
            j += 1

        sep_index_2 = generation.find("```")
        if sep_index_2 == -1:
            return prefix.strip() + "\n" + generation[func_index:j]
        else:
            return prefix.strip() + "\n" + generation[func_index : min(j, sep_index_2)]