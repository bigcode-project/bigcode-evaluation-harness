"""Mercury: An Efficiency Benchmark for LLM Code Synthesis
https://arxiv.org/abs/2402.07844

Mercury is the first benchmark to assess the code efficiency of LLM code generation tasks. 
It consists of 1,889 programming tasks covering diverse difficulty levels alongside test case generators generating unlimited cases for comprehensive evaluation. 

Homepage: https://github.com/Elfsong/Mercury
"""


from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.beyond_eval import compute_beyond_eval

_CITATION = """
@article{du2024mercury,
  title={Mercury: An Efficiency Benchmark for LLM Code Synthesis},
  author={Du, Mingzhe and Luu, Anh Tuan and Ji, Bin and Ng, See-Kiong},
  journal={arXiv preprint arXiv:2402.07844},
  year={2024}
}
"""


class Mercury(Task):
    """
    A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "Elfsong/Mercury"

    def __init__(self, prompt):
        super().__init__(
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>", "<｜end▁of▁sentence｜>"],
            requires_execution=True,
        )
        self.prompt = prompt

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["eval"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return f'\'\'\'{doc["pretty_content"][0]}\'\'\'\n{doc["prompt"]}'
    
    def get_reference(self, doc):
        """Builds the reference solutions for the doc (sample from the test dataset)."""
        return doc

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        generation = generation[len(prompt):]
        generation = self._stop_at_stop_token(generation, self.stop_words)
        return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        
        results = compute_beyond_eval(generations, references)
        
        return results
