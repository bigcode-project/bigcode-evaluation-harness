# This template file is adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/templates/new_task.py

# TODO: Remove all TODO comments once the implementation is complete.
"""
Paper-Title: Throwing Shaders at Language Models - Evaluating Creative Code Generation
TODO: Paper-URL: unavailable (unpublished)
Description: ShaderEval aims to be a suite of tasks to evaluate generative model on creative code generation. Espeicically GLSL shadercode.
    Task1 is a proof of concept and looks at code completion for returnstatemetns of Shadertoy functions. Exact_match and greedy decoding.
Homepage: https://huggingface.co/spaces/Vipitis/ShaderEval

Paper-Title: an unknown title for my bachelor thesis (A Comprehensive Evaluation of shadercode generation with language models)
TODO: Paper-URL: unavailable (unapproved)
Description: Doing everything better than before.
    Task-1b a better version of Task1 (Return Completion) using a deduplicated dataset as well as more metrics (notImplemented)
    Task-2: Function Generation - given a function signature and a docstring, generate the function body,
    tested by patching it back into the original shadercode and comparing if the rendered images are the same. (currently in development, open for debate)
    Task-3: Semantic generation given a title and description, recursively generate more shadercode untill it renders, scored by CLIP match (in planing...)
Homepage: https://huggingface.co/spaces/Vipitis/ShaderEval (could be something else...?)
"""
from lm_eval.base import Task
import evaluate
import datasets
from ShaderCoder.utils import parse_functions, construct_model_context, replace_function #where to import this from(via custom metric?)

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
        "shadereval2": FunctionGeneration,
    }

# TODO: Replace `NewTask` with the name of your Task.
class ReturnCompletion(Task): #Task1
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "Vipitis/Shadertoys-fine"
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



# TODO: Replace `NewTask` with the name of your Task.
class FunctionGeneration(Task): #task2 
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "Vipitis/Shadertoys" #maybe fine in the future
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def __init__(self):
        super().__init__(
            # TODO: Specify the list of stop words in `stop_words` for the code generation task \
            # and if the evaluation requires executing the generated code in `requires_execution`.
            stop_words=[],
            requires_execution=True, #we run shadercode - could that be harmful?
        )

    def get_dataset(self):
        # TODO: retrieve the evaluation subset from the loaded dataset (e.g. `self.dataset["test"]`)
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        ds = self.dataset["test"]
        ds = ds.filter(lambda x: not x["has_inputs"] and x["num_passes"] == 1 and x["code"].count("return") >= 1 and x["license"] != "unknown") 
        # TODO: filter for docstrings -- how?
        return ds

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
        func_node = parse_functions(doc["code"])[0] #always 0 for testing?
        model_ctx = construct_model_context(func_node,prompt="")
        return model_ctx

    def get_reference(self, doc):
        # TODO: get the reference solution from a sample `doc` from the dataset
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["code"] #returns full original code

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
        first_func = parse_functions(generation)[0]
        orignal_code = self.dataset[idx]["code"]
        altered_code = replace_function(orignal_code, first_func)    
        return altered_code

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
        shadermatch = evaluate.load("Vipitis/shadermatch")
        generations = [
            generation[0] for generation in generations
        ]  # unpack one list for some reason? (we zero shot)
        return shadermatch.compute(predictions=generations, references=references)
