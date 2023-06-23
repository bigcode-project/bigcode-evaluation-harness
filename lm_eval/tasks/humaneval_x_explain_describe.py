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

        super().__init__(
            stop_words=stop_words,
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt_encoder(self, doc):
        """Encoder input for models with Enc-Dec architecture like CodeT5"""
        assert self.mutate_method == "instruct", "Only instruct mutation is supported for Enc-Dec models"
        prompt_base = self.get_prompt_base(doc)
        prompt = prompt_base + doc["canonical_solution"]
        docstring_len = len(doc["docstring"])
        prompt += f"\nProvide a concise natural language description of the above function using at most {docstring_len} characters."

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
        # Use declaration instead of prompt to hide the docstring
        prompt_base = self.get_prompt_base(doc)
        if self.mutate_method == "edit":
            prompt = "<commit_before><commit_after>" + prompt_base + doc["canonical_solution"]
            prompt += "<commit_msg>"
        elif self.mutate_method == "instruct":
            prompt = prompt_base + doc["canonical_solution"]
            docstring_len = len(doc["docstring"])
            prompt += f"\nProvide a concise natural language description of the function using at most {docstring_len} characters."
        elif self.mutate_method == "instruct-qa":
            pass


        return prompt

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
        gen = generation[len(prompt):].strip()[:docstring_len].rstrip()
        return gen

    def get_reference(self, doc, get_solution=False):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return None
            
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        raise ValueError(ERROR_MSG)