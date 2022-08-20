import fnmatch
import json

import datasets
import transformers
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from arguments import EvalArguments
from lm_eval.evaluator import Evaluator

ALL_TASKS = ["humaneval", "apps", "mbpp"]


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = HfArgumentParser(EvalArguments)
    parser.add_argument(
        "--model",
        required=True,
        help="Model to evaluate, provide repo name Hugging Face hub or local path",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        choices=MultiChoice(ALL_TASKS),
        help=f"evalution tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--allow_code_execution",
        type=bool,
        default=False,
        help="allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation_results.json",
        help="path to save the results",
    )
    parser.add_argument(
        "--save_generations", type=bool, default=True, help="save code generations"
    )
    return parser.parse_args()


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def main():
    args = parse_args()
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    model = AutoModelForCausalLM.from_pretrained(args.model, use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_auth_token=True)
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    tokenizer.pad_token = tokenizer.eos_token

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Selected Tasks: {task_names}")

    evaluator = Evaluator(accelerator, model, tokenizer, args)
    results = {}
    for task in task_names:
        results[task] = evaluator.evaluate(task)

    # add info about the model and few shot config
    results["config"] = {"model": args.model}

    dumped = json.dumps(results, indent=2)
    if accelerator.is_main_process:
        print(dumped)

    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)


if __name__ == "__main__":
    main()
