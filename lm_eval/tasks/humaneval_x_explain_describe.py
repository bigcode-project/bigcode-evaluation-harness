from lm_eval.base import Task


LANGUAGES = ["python", "cpp", "js", "java", "go", "rust"]

ERROR_MSG = """
HumanEval-X-Explain should be run with the flag `--generation_only`.
Once generations are done run HumanEval-X-Generate with `--mutate_method path/to/generations.json`
It will load the explanations, generate from them and evaluate.
"""

def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {apps-interview: Task, apps-competitoon: Task}
    """
    return {f"humaneval-x-explain-describe-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class HumanEvalXExplainDescribe(GeneralHumanEvalXExplainDescribe):
        def __init__(self, mutate_method="prompt", language=language):
            super().__init__(mutate_method=mutate_method, language=language)

    return HumanEvalXExplainDescribe


class GeneralHumanEvalXExplainDescribe(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
    DATASET_PATH = "bigcode/humaneval-x-bugs"
    DATASET_NAME = None

    def __init__(self, mutate_method="prompt", language="python"):

        self.DATASET_NAME = language
        self.mutate_method = mutate_method
        
        stop_words = ["<|endoftext|>"]
        if self.mutate_method == "starchat":
            stop_words.append("<|end|>")

        super().__init__(
            stop_words=stop_words,
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt_encoder(self, doc):
        """Encoder input for models with Enc-Dec architecture like CodeT5"""
        prompt_base = self.get_prompt_base(doc)
        docstring_len = len(doc["docstring"])
        instruction = f"Provide a concise natural language description of the code using at most {docstring_len} characters."
        func = prompt_base + doc["canonical_solution"]

        if self.mutate_method == "instructcodet5p":
            # https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/humaneval/generate_codet5p.py#L89
            prompt = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n{func}\n\n### Response:' 
        else:
            raise NotImplementedError
        return prompt
    
    def get_prompt_base(self, doc):
        # See 
        # https://github.com/roG0d/CodeGeeX/blob/f66205b5f615a4eead9c26d7ec297e14738ea18d/codegeex/benchmark/evaluate_humaneval_x.py#L78
        # https://github.com/THUDM/CodeGeeX/pull/76#issuecomment-1500653190
        if self.DATASET_NAME == "rust":
            main = "\nfn main(){ \n } \n"
            prompt_base = main + doc["declaration"]
        else:
            prompt_base = doc["declaration"]
        return prompt_base
    
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt_base = self.get_prompt_base(doc)
        docstring_len = len(doc["docstring"])
        instruction = f"Provide a concise natural language description of the code using at most {docstring_len} characters."
        func = prompt_base + doc["canonical_solution"]

        if self.mutate_method == "instruct":
            prompt = func + "\n" + instruction
        elif self.mutate_method == "instruct-qa":
            prompt = f'Question: {instruction}\n{func}\n\nAnswer:'
        elif self.mutate_method == "instructcodet5p":
            # https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/humaneval/generate_codet5p.py#L89
            prompt = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n{func}\n\n### Response:'       
        elif self.mutate_method == "starchat":
            # https://huggingface.co/HuggingFaceH4/starchat-beta
            prompt = f"<|system|>\n<|end|>\n<|user|>\n{instruction}\n{func}<|end|>\n<|assistant|>"
        elif self.mutate_method == "starcodercommit":
            prompt = f'<commit_before><commit_msg>{instruction}\n{func}<commit_after>'
        elif self.mutate_method == "wizardcoder":
            prompt = f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n{func}\n\n### Response:'
        return prompt

    def remove_last_block(self, text):
        for w in self.stop_words:
            if w in text:
                text = text[:text.find(w)]
        return text

    def remove_code(self, text, canonical_solution):
        for line in canonical_solution.split("\n"):
            line = line.strip()
            if len(line) > 20 and line in text:
                text = text.replace(line, "")
        return text
    
    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)
        docstring_len = len(doc["docstring"])
        gen = self.remove_last_block(generation[len(prompt):].strip()[:docstring_len]).rstrip()
        gen = self.remove_code(gen, doc["canonical_solution"])
        return gen

    def get_reference(self, doc, get_solution=False):
        return None
            
    def process_results(self, generations, references):
        raise ValueError(ERROR_MSG)