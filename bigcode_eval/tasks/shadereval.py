# This template file is adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/templates/new_task.py

# TODO: Remove all TODO comments once the implementation is complete.
"""
Paper-Title: Throwing Shaders at Language Models - Evaluating Creative Code Generation
TODO: Paper-URL: unavailable (unpublished)
Description: ShaderEval aims to be a suite of tasks to evaluate generative model on creative code generation. Espeicically GLSL shadercode.
    Task1 is a proof of concept and looks at code completion for returnstatemetns of Shadertoy functions. Exact_match and greedy decoding.
Homepage: https://huggingface.co/spaces/Vipitis/ShaderEval

Paper-Title: Evaluating Language Models for Computer Graphics Code Completion
TODO: Paper-URL: unavailable (unpublished)
Description: Function Completion task for GLSL shadercode. Metric statically compares and then runs generated code to compare rendered frames with the refernece.
Homepage: https://huggingface.co/spaces/Vipitis/Shadermatch
"""
from bigcode_eval.base import Task
import evaluate
import datasets

# TODO: Add the BibTeX citation for the task.
_CITATION = """tbd
"""

def create_all_tasks():
    """assemble all tasks in a dictionary:
    - task1: return completion
    - task2: function generation
    """
    return {
        "shadereval-1": ReturnCompletion,
        "shadereval-2": FunctionCompletion,
    }

# TODO: Replace `NewTask` with the name of your Task.
class ReturnCompletion(Task): #Task1
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "Vipitis/Shadertoys-fine" # now defunct.
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "return_completion"

    def __init__(self):
        super().__init__(
            # TODO: Specify the list of stop words in `stop_words` for the code generation task \
            # and if the evaluation requires executing the generated code in `requires_execution`.
            stop_words=[";"],
            requires_execution=False,
        )

    def get_dataset(self):
        # TODO: retrieve the evaluation subset from the loaded dataset (e.g. `self.dataset["test"]`)
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def fewshot_examples(self):
        # TODO: load few-shot examples (from lm_eval/tasks/fewshot_examples) if they exist
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    def get_prompt(self, doc):
        # TODO: build the prompt for the language model from a sample `doc` from the dataset
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["body"]

    def get_reference(self, doc):
        # TODO: get the reference solution from a sample `doc` from the dataset
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["return_statement"].split(";")[0].strip()

    def postprocess_generation(self, generation, idx):
        # TODO: define the postprocessing for the LM generation
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        generation = generation.split("return")[1]  # this works?
        return generation.split(";")[0].strip()

    def process_results(self, generations, references):
        # TODO: define how the evaluation score is computed from list of \
        # generations and reference solutions
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
        exact_match = evaluate.load("exact_match")
        generations = [
            generation[0] for generation in generations
        ]  # unpack one list for some reason? (we zero shot)
        return exact_match.compute(predictions=generations, references=references)



class FunctionCompletion(Task): #task2 
    DATASET_PATH = "Vipitis/Shadereval-inputs"
    DATASET_NAME = None
    # revision hash: 274eb4d3017d59da2a1f48bc7194be1545de919f (or v0.4 tag - TBD)

    def __init__(self):
        super().__init__(
            stop_words=[], #early stopping via stop words has impacted generations meaningfully so it's not done!
            requires_execution=True, #we run shadercode which can be unsafe!
        )
        self._metric = evaluate.load("Vipitis/shadermatch") #load the metric from the evaluate library

    def get_dataset(self):
        return self.dataset["test"]

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    def get_prompt(self, doc):
        """
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["model_inp"]

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["image_code"] #returns full original code

    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        # these postprocessing steps are implemented in the metric itself: https://huggingface.co/spaces/Vipitis/shadermatch/blob/main/shadermatch.py#L139-L168
        # and rely in additional dependencies: tree-sitter-glsl, [maybe also wgpu-py, glfw]
        row = self.dataset["test"][idx]
        truncated = self._metric.truncate_generation(model_inp="", generation=generation)
        # TODO: the metric methods will be renaming their args to be more broadly useable.
        altered = self._metric.replace_body(ref_code=row["image_code"], altered_body=truncated, end_header_byte=row["func_bytes"][0], end_function_byte=row["func_bytes"][4])
        return altered

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
        # one candidate per generation, as to be unpacked here.
        generations = [generation[0] for generation in generations]
        # the metric is implemented as an evaluate.metric here: https://huggingface.co/spaces/Vipitis/shadermatch/blob/main/shadermatch.py
        # this defenitely requires wgpu-py, glfw, wgpu-shadertoy, tree-sitter-glsl, numpy, Pillow and tqdm
        results = self._metric.compute(predictions=generations, references=references)
        # this also includes a list of all individual labels (in order).
        return results