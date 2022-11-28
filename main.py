import fnmatch
import json

import datasets
import transformers
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from lm_eval.arguments import EvalArguments
from lm_eval.evaluator import Evaluator
from lm_eval.tasks import ALL_TASKS

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
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
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
        default=10,
        help="batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=512,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--postprocess",
        type=bool,
        default=True,
        help="Postprocess model outputs before execution, only off during genration tests",
    )
    parser.add_argument(
        "--allow_code_execution",
        type=bool,
        default=False,
        help="allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--generation_only",
        type=bool,
        default=False,
        help="do code generation but no evaluation",
    )
    parser.add_argument(
        "--generations_path",
        type=str,
        default=None,
        help="path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
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
    parser.add_argument(
        "--save_references",
        type=bool,
        default=False,
        help="save reference solutions/tests",
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

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Selected Tasks: {task_names}")

    results = {}
    if args.generations_path:
        # here we don't generate code but only evaluate previously computed generations
        if accelerator.is_main_process:
            print("evaluation only mode")
        evaluator = Evaluator(accelerator, None, None, args)
        for task in task_names:
            results[task] = evaluator.evaluate(task)

    else:
        # here we generate code and save it (evaluation is optional but True by default)
        print("Loading the model and tokenizer")
        model = AutoModelForCausalLM.from_pretrained(args.model, use_auth_token=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_auth_token=True)
        if not tokenizer.eos_token:
            if tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
                print("bos_token used as eos_token")
            else:
                raise ValueError("No eos_token or bos_token found")
        tokenizer.pad_token = tokenizer.eos_token

        evaluator = Evaluator(accelerator, model, tokenizer, args)
        for task in task_names:
            if args.generation_only:
                if accelerator.is_main_process:
                    print("generation mode only")
                generations, references = evaluator.generate_text(task)
                if accelerator.is_main_process:
                    with open("generations.json", "w") as fp:
                        json.dump(generations, fp)
                        print("generations were saved")
                    if args.save_references:
                        with open("references.json", "w") as fp:
                            json.dump(references, fp)
                            print("references were saved")
            else:
                results[task] = evaluator.evaluate(task)

    results["config"] = {"model": args.model}
    if not args.generation_only:
        dumped = json.dumps(results, indent=2)
        if accelerator.is_main_process:
            print(dumped)

        with open(args.output_path, "w") as f:
            f.write(dumped)


if __name__ == "__main__":
    main()
