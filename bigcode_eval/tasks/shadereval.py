# This template file is adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/templates/new_task.py

# TODO: Remove all TODO comments once the implementation is complete.
"""
Paper-Title: Throwing Shaders at Language Models - Evaluating Creative Code Generation
TODO: Paper-URL: unavailable (unpublished)
Description: ShaderEval aims to be a suite of tasks to evaluate generative model on creative code generation. Espeicically GLSL shadercode.
    Task1 is a proof of concept and looks at code completion for returnstatemetns of Shadertoy functions. Exact_match and greedy decoding.
Homepage: https://huggingface.co/spaces/Vipitis/ShaderEval

Paper-Title: Evaluating language models for computer graphics code completion
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
        "shadereval-2": FunctionGeneration,
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



# TODO: Replace `NewTask` with the name of your Task.
class FunctionGeneration(Task): #task2 
    DATASET_PATH = "Vipitis/Shadereval-experiments-dev" #as a temporary solution to reduce current problems
    
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None #this will eventually be a subset for the Shadertoys dataset, but not right now

    def __init__(self):
        super().__init__(
            # TODO: Specify the list of stop words in `stop_words` for the code generation task \
            # and if the evaluation requires executing the generated code in `requires_execution`.
            # stop_words=["\nfloat ", "\nvec", "\nint", "\nvoid", "\nmat"], #new function starts... so all the keywords
            # TODO: stopwords can cause incorrect early stopping, so we don't edn up using them. I am considering using guided generation with tree-sitter to do early stopping.
            stop_words=[], #set it's to Falsy?
            requires_execution=True, #we run shadercode - could that be harmful? (all in the metric)
        )
        self._metric = evaluate.load("Vipitis/shadermatch") #load the metric from the evaluate library

    def get_dataset(self):
        # TODO replace with subset once that is set up
        return self.dataset["test"]

    def fewshot_examples(self):
        # TODO: load few-shot examples (from lm_eval/tasks/fewshot_examples) if they exist
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    def get_prompt(self, doc):
        # TODO: build the prompt for the language model from a sample `doc` from the dataset
        """
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["model_inp"]

    def get_reference(self, doc):
        # TODO: get the reference solution from a sample `doc` from the dataset
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["image_code"] #returns full original code

    def remove_last_block(self, code):
        """
        Adapted from https://github.com/bigcode-project/bigcode-evaluation-harness/blob/be2a44c2faa29c20b5041d7083acb698eb373309/bigcode_eval/tasks/humanevalpack.py#L275C5-L311C20
        """
        # TODO: can be removed
        for w in self.stop_words:
            if w in code:
                code = code[:code.find(w)]

        ### Find the first occassion where a chain of { } is closed??
        open_brackets = 1
        cut = False
        for i, c in enumerate(code.encode("utf-8")):
            c = chr(c)
            if c == '{':
                open_brackets += 1
            elif c == '}':
                open_brackets -= 1
            if open_brackets == 0:
                code = code.encode("utf-8")[:i+1].decode("utf-8", "ignore")
                cut = True
                break
        if not cut:
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
            else:
                code = code + "// incomplete generation! \n"
        return code

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

        row = self.dataset["test"][idx]
        truncated = self._metric.truncate_generation(model_inp="", generation=generation)
        # TODO: the metric methods will be renaming their args to be more broadly useable.. maybe even refactor the bit at the top.
        altered = self._metric.replace_body(ref_code=row["image_code"], altered_body=truncated, end_header_byte=row["func_bytes"][0], end_function_byte=row["func_bytes"][4])
        return altered

        # TODO: remove the old code
        # assemble into the full code with just the function replaced
        ref = self.dataset["test"][idx]
        model_ctx = ref["model_ctx"]
        full_code = ref["full_code"]
        start, end = ref["func_range"]
        before_gen = full_code.encode("utf-8")[:start].decode("utf-8")
        after_gen = full_code.encode("utf-8")[end:].decode("utf-8")

        if self.prompt == "full":
            gen = self.remove_last_block(generation.encode("utf-8")[start + len(model_ctx.encode("utf-8")):].decode("utf-8"))
        else:
            gen = self.remove_last_block(generation.encode("utf-8")[len(model_ctx.encode("utf-8")):].decode("utf-8")) #remove last block to avoid syntax errors
        return before_gen + model_ctx + gen + after_gen #does this patch it together correctly?

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
        # shadermatch = evaluate.load("Vipitis/shadermatch")
        generations = [
            generation[0] for generation in generations
        ]  # unpack one list for some reason? (we zero shot)
        results = self._metric.compute(predictions=generations, references=references)
        # this also includes a list of all individual labels (in order).
        return results