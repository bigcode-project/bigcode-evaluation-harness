"""Program Synthesis with Large Language Models
https://arxiv.org/abs/2108.07732

The benchmark consists of around 1,000 crowd-sourced Python programming problems, 
designed to be solvable by entry level programmers, covering programming fundamentals, 
standard library functionality, and so on. Each problem consists of a task description, 
code solution and 3 automated test cases. As described in the paper, a subset of the data
has been hand-verified by the authors.

Homepage:: https://github.com/google-research/google-research/tree/master/mbpp
"""

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


_CITATION = """
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
"""


class MBPP(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "/mnt/roma/abhineet/mbpp_withpseudods"

    def __init__(self):
        super().__init__(
            stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"],
            requires_execution=True,
        )
        checkpoint = "/mnt/roma/abhineet/output_dir_p/checkpoint-5640"
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset
        # the wrong split of mbpp can be loaded with old datasets cache
        assert (
            len(dataset) == 500
        ), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        return dataset

    def generate_prompt(self, doc):
        description = doc["text"]
        test_example = doc["test_list"][0]
        prompt = f'"""Task Description: \n\n{description}\n{test_example}\n\nPseudocode:\n\n"""'
        self.model.to(self.device)
        inputs = self.tokenizer.encode(prompt, return_tensors = "pt").to(self.device)
        outputs = self.model.generate(inputs, max_new_tokens = 300)
        output = self.tokenizer.decode(outputs[0])
        start_idx = output.find("Code:")
        return output[:start_idx]
        
    
    
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        description = doc["text"]
        test_example = doc["test_list"][0]
        pseudocodes = doc["Pseudocodes"]
        prompt = f'"""Task Description: \n\n{description}\n{test_example}{pseudocodes}\nCode:\n\n"""'
        
        #prompt_with_pseudocode = self.generate_prompt(doc)
        #prompt = prompt_with_pseudocode + "\n\n\nCode:\n\n"
        
        
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(doc["test_list"])


    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        prompt = self.get_prompt(self.dataset["test"][idx])
        start_idx = generation.find("Code:")
        generation = generation[start_idx + len("Code:"):]
        print(generation)
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        results, _ = compute_code_eval(
            references=references,
            predictions=generations,
        )
        return results
