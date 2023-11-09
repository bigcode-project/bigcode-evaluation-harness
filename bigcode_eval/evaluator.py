import inspect
import json
import os
import warnings

from typing import Any, Iterable, List

from datasets import Dataset

from bigcode_eval import tasks
from bigcode_eval.generation import parallel_generations

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

def chunk_list(item_list: List[Any], chunk_size: int = 32) -> List[List[Any]]:
    """
    Turn an list of items into a list of item chunks
    Where each chunk is at most of len `chunk_size`

    Args:
        item_list (List[Any]): an list of items to batchify
        chunk_size (int): the size of each chunk

    Returns:
        a List[List[Any]] where each List[Any] is of at most length chunk_size
        and the length of the list is ceiling(len(item_list)/chunk_size)
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if len(item_list) == 0:
        raise ValueError("Must be a non-empty list")
    return [item_list[i : i + chunk_size] for i in range(0, len(item_list), chunk_size)]

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

    # TODO (Max): add in the passed list of generations to start from an intermediate checkpoint
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

        generations = []

        # TODO (Max): if intermediate generations file is passed
        # Then append all the generations from that task to `generations`
        # and only chunk data from generations onward
        # Note: ALSO want to change `parallel_generations` so that we don't use self.args.limit_start if curr_iter isn't 0
                # chunk data for saving intermediate generations and references
        chunk_size = self.args.save_every_k_samples if self.args.save_every_k_samples >= 1 else len(references)
        dataset_chunks = chunk_list(dataset, chunk_size)
        
        intermediate_save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}-intermediate.json"

        for iter, data_chunk in enumerate(dataset_chunks):
            curr_sample_idx = len(generations)  
            generation_chunk = parallel_generations(
                task,
                Dataset.from_dict(data_chunk),
                self.accelerator,
                self.model,
                self.tokenizer,
                n_tasks=n_tasks,
                args=self.args,
                curr_iter=iter,  # Note: this is because we manually change limit_start to 0 if curr_iter > 0
                curr_sample_idx=curr_sample_idx,  # curr_sample_idx will be used in `complete_code` so we don't mess up indexing during post-process
            )
            generations.extend(generation_chunk)

            # save intermediate results
            if self.accelerator.is_main_process:
                self.save_json_files(
                    generations,
                    references[:len(generations)],
                    intermediate_save_generations_path,
                    "references-intermediate.json"
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
                self.save_json_files(generations, references, self.args.save_generations_path, "references.json")

            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            print("Evaluating generations...")
            results = task.process_results(generations, references)
            return results

    def save_json_files(
        self,
        generations: List[str],
        references: List[str],
        save_generations_path: str,
        save_references_path: str,
    ) -> None:
        if self.args.save_generations:
            with open(save_generations_path, "w") as fp:
                json.dump(generations, fp)
                print(f"generations were saved at {save_generations_path}")
        if self.args.save_references:
            with open(save_references_path, "w") as fp:
                json.dump(references, fp)
                print(f"references were saved at {save_references_path}")
