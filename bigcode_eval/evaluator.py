import inspect
import json
import os
import warnings
from typing import Optional 

from bigcode_eval import tasks
from bigcode_eval.generation import parallel_generations
from bigcode_eval.utils import _make_instruction_prompt


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

class EvaluatorForEndpoint:
    def __init__(self, api_key: str, args, api_base: Optional[str]=None, api_organization: Optional[str]=None):
        try:
            import litellm
        except ImportError as e:
            print('EvaluationForEndpoint requires package litellm to be installed.')
            
        self.args = args
        litellm.api_key = api_key
        
        if api_base:
            litellm.api_base = api_base
        
        if api_organization:
            litellm.organization = api_organization        
    
    def fetch_dataset_from_task(self, task_name: str):
        task = tasks.get_task(task_name, args=self.args) 
        dataset = task.get_dataset()
        n_tasks = self.args.limit if self.args.limit else len(dataset)
        
        # Build the prompts
        prompts = [] 
        
        for sample in range(self.args.limit_start, self.args.limit_start + n_tasks):
            prompt_contents = task.get_prompt(dataset[sample])
            if isinstance(prompt_contents, str):
                prompt = self.args.prefix + self.args.prompt + prompt_contents
            elif isinstance(prompt_contents, dict):
                if set(prompt_contents.keys()) == {"prefix", "suffix"}:
                    print("Infilling mode for API is not supported")
                    continue
                elif set(prompt_contents.keys()) == {"instruction", "context"}:
                    prompt = _make_instruction_prompt(**prompt_contents, prefix=self.args.prefix)
            else:
                raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")
            prompts.append(prompt)

        # Build the references
        references = [
            task.get_reference(dataset[i])
            for i in range(self.config.limit_start, self.config.limit_start + n_tasks)
        ]
    

class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def generate_text(self, task_name):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        n_tasks = self.args.limit if self.args.limit else len(dataset)
        references = [task.get_reference(dataset[i]) for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]

        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [[task.get_reference(dataset[i], get_solution=True)] for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references

        generations = parallel_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
        )
        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references

    def evaluate(self, task_name):
        task = tasks.get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations, references = self.generate_text(task_name)

        if self.accelerator.is_main_process:
            if not self.args.load_generations_path:
                if self.args.save_generations:
                    with open(self.args.save_generations_path, "w") as fp:
                        json.dump(generations, fp)
                        print(
                            f"generations were saved at {self.args.save_generations_path}"
                        )
                if self.args.save_references:
                    with open(self.args.save_references_path, "w") as fp:
                        json.dump(references, fp)
                        print(f"references were saved at {self.args.save_references_path}")

            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            print("Evaluating generations...")
            results = task.process_results(generations, references)
            return results
