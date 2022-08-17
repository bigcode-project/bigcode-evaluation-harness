import os
import json

from datasets import load_dataset
from evaluate import load

from lm_eval.generation import get_references_humaneval, get_references_mbpp, parallel_generations


_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""

class Evaluator():
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.output_path = args.output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

        # evaluation dataset arguments
        self.level_apps = args.level_apps
        
    def generate_text(self, task):

        if task == "apps":
            dataset = load_dataset("codeparrot/apps", split="test", difficulties=[self.level_apps])
            generations = parallel_generations(self.accelerator, self.model, self.tokenizer, dataset, mode="apps", args=self.args, num_tasks=self.args.num_tasks_apps)
            references = None
            return generations, references

        elif task == "humaneval":
            dataset = load_dataset("openai_humaneval", split="test")
            generations = parallel_generations(self.accelerator, self.model, self.tokenizer, dataset, mode="humaneval", args=self.args, num_tasks=self.args.num_tasks_he)
            references = get_references_humaneval(dataset, self.args.num_tasks_he)
            return generations, references

        elif task == "mbpp":
            dataset = load_dataset("mbpp", split="test", ignore_verifications=True)
            # the evaluation set is task ids 11->510 
            dataset = dataset.select([i for i in range(10,510)])
            generations = parallel_generations(self.accelerator, self.model, self.tokenizer, dataset, mode="mbpp", args=self.args, num_tasks=self.args.num_tasks_mbpp)
            references = get_references_mbpp(dataset, self.args.num_tasks_mbpp)
            return generations, references

        else:
            raise ValueError(f"Task {task} is not supported, please choose from apps, humaneval, or mbpp")

    def evaluate(self, task):

        if not self.allow_code_execution:
            print(_WARNING)
            raise ValueError("Code evaluation is not enabled. Read the warning above carefully and then use `--allow_code_execution=True` flag to enable code evaluation.")
        generations, references = self.generate_text(task)
        if self.accelerator.is_main_process:
            if self.args.save_generations:
                with open("generations.json", "w") as fp:
                    json.dump(generations, fp)
                    print("generations saved")
            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if task == "apps":
                code_metric = load("codeparrot/apps_metric")
                results = code_metric.compute(predictions=generations, k_list=[1, 10, 100], level=self.level_apps)

            else:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
                code_metric = load("code_eval")
                results, _ = code_metric.compute(
                    references=references, predictions=generations, num_workers=self.args.num_workers
                )

            return results
