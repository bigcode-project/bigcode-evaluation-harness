"""Mercury: An Efficiency Benchmark for LLM Code Synthesis
https://arxiv.org/abs/2402.07844

Mercury is the first benchmark to assess the code efficiency of LLM code generation tasks. 
It consists of 1,889 programming tasks covering diverse difficulty levels alongside test case generators generating unlimited cases for comprehensive evaluation. 

Homepage: https://github.com/Elfsong/Mercury
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.beyond_eval import compute_beyond_eval

_CITATION = """
@article{du2024mercury,
  title={Mercury: An Efficiency Benchmark for LLM Code Synthesis},
  author={Du, Mingzhe and Luu, Anh Tuan and Ji, Bin and Qian, Liu and Ng, See-Kiong},
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
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>", "<｜end▁of▁sentence｜>", "\n###", "\n\n\n\n\n", "<|endoftext|>"],
            requires_execution=True,
        )
        self.prompt = prompt

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["eval"]
    
    @staticmethod
    def prompt_generate(question_content, starter_code):
        examples_json = {
            "question": "You are given a 0-indexed array of positive integers nums. Find the number of triplets (i, j, k) that meet the following conditions:\n\n0 <= i < j < k < nums.length\nnums[i], nums[j], and nums[k] are pairwise distinct.\n\t\nIn other words, nums[i] != nums[j], nums[i] != nums[k], and nums[j] != nums[k].\n\n\n\nReturn the number of triplets that meet the conditions.\n \nExample 1:\n\nInput: nums = [4,4,2,4,3]\nOutput: 3\nExplanation: The following triplets meet the conditions:\n- (0, 2, 4) because 4 != 2 != 3\n- (1, 2, 4) because 4 != 2 != 3\n- (2, 3, 4) because 2 != 4 != 3\nSince there are 3 triplets, we return 3.\nNote that (2, 0, 4) is not a valid triplet because 2 > 0.\n\nExample 2:\n\nInput: nums = [1,1,1,1,1]\nOutput: 0\nExplanation: No triplets meet the conditions so we return 0.\n\n \nConstraints:\n\n3 <= nums.length <= 100\n1 <= nums[i] <= 1000\n\n",
            "sample_code": 'class Solution(object):\n    def unequalTriplets(self, nums: List[int]) -> int:\n        """\n\t:type nums: List[int]\n\t:rtype: int\n\t"""\n        \n',
            "answer": 'class Solution(object):\n    def unequalTriplets(self, nums: List[int]) -> int:\n        """\n\t:type nums: List[int]\n\t:rtype: int\n\t"""\n        \n        ans = 0\n        n = len(a)\n        for i in range(n):\n            for j in range(i + 1, n):\n                for k in range(j + 1, n):\n                    ans += len({a[i], a[j], a[k]}) == 3\n        return ans'
        }

        def get_example_prompt(example):
            prompt = ""
            prompt += "### Question\n"
            prompt += example["question"]
            prompt += "\n\n"
            if starter_code:
                prompt += "### Code Prompt\n"
                prompt += example["sample_code"]
                prompt += "\n\n"
            prompt += "### Completion\n"
            prompt += example["answer"]
            if example["answer"]:
                prompt += "\n\n"
            return prompt

        prompt = ""
        # one-shot generation example
        prompt += get_example_prompt(examples_json)
        # code generation
        prompt += get_example_prompt({"question": question_content,"sample_code": starter_code,"answer": ""})
        
        return prompt

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return self.prompt_generate(doc["pretty_content"][0], doc["prompt"])
    
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
