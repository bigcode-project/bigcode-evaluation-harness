import fnmatch
import json
import os

import datasets
import torch
import wandb
import transformers
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, AutoConfig
import pandas as pd

from lm_eval.arguments import EvalArguments
from lm_eval.evaluator import Evaluator
from lm_eval.tasks import ALL_TASKS

import sys
from os.path import join, dirname

sys.path.insert(0, join(dirname(__file__), "..",))
from modeling_gpt_bigcode_sd import GPTBigCodeForCausalLMSkipDecode

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
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        choices=MultiChoice(ALL_TASKS),
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=512,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Do code generation but no evaluation",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save reference solutions/tests",
    )

    parser.add_argument('-sd', "--do_skip_decode", action="store_true")
    parser.add_argument("--max_exit_layer", type=int)
    parser.add_argument("--min_exit_layer", type=int)
    parser.add_argument('-wl', "--num_skip_decode_warmup_layers", type=int)
    
    parser.add_argument('-v', "--verbose", action="store_true")

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
    
    if args.verbose:
        os.environ["verbose"] = "1"
        
    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Selected Tasks: {task_names}")

    ### WANDB SET UP
    wandb.init(
    project=f"{args.tasks}_evaluation",
    name=args.model
    )
    wandb.config.update(args)
    wandb.config.num_layers = []
    wandb.define_metric("generation_time_ms", summary="mean")
    wandb.define_metric("num_new_tokens", summary="mean")
    ###################
    
    results = {}
    if args.load_generations_path:
        # here we don't generate code but only evaluate previously computed generations
        if accelerator.is_main_process:
            print("evaluation only mode")
        evaluator = Evaluator(accelerator, None, None, args)
        for task in task_names:
            results[task] = evaluator.evaluate(task)

    else:
        # here we generate code and save it (evaluation is optional but True by default)
        dict_precisions = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if args.precision not in dict_precisions:
            raise ValueError(
                f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
            )
        
        config = AutoConfig.from_pretrained(args.model)
        
        if args.do_skip_decode:
            os.environ["do_skip_decode"] = "1"
            model_cls = GPTBigCodeForCausalLMSkipDecode
            config.do_skip_decode = True
            config.max_exit_layer = args.max_exit_layer
            config.min_exit_layer = args.min_exit_layer
            config.num_skip_decode_warmup_layers = args.num_skip_decode_warmup_layers
            config.max_length = args.max_length_generation
        else:
            model_cls = AutoModelForCausalLM

        if args.load_in_8bit:
            print("Loading model in 8bit")
            current_device = accelerator.process_index
            # the model needs to fit in one GPU``
            model = model_cls.from_pretrained(
                args.model,
                revision=args.revision,
                load_in_8bit=args.load_in_8bit,
                trust_remote_code=args.trust_remote_code,
                use_auth_token=args.use_auth_token,
                device_map={"": current_device},
                config=config,
            )
        elif args.load_in_4bit:
            print("Loading model in 4bit")
            current_device = accelerator.process_index
            # the model needs to fit in one GPU
            model = model_cls.from_pretrained(
                args.model,
                revision=args.revision,
                load_in_4bit=args.load_in_4bit,
                trust_remote_code=args.trust_remote_code,
                use_auth_token=args.use_auth_token,
                device_map={"": current_device},
                config=config,
            )
        else:
            print(f"Loading model in {args.precision}")
            model = model_cls.from_pretrained(
                args.model,
                revision=args.revision,
                torch_dtype=dict_precisions[args.precision],
                trust_remote_code=args.trust_remote_code,
                use_auth_token=args.use_auth_token,
                config=config
            )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
            truncation_side="left",
            padding_side="right",
        )
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
                    with open(args.save_generations_path, "w") as fp:
                        json.dump(generations, fp)
                        print(f"generations were saved at {args.save_generations_path}")
                    if args.save_references:
                        with open("references.json", "w") as fp:
                            json.dump(references, fp)
                            print("references were saved")
            else:
                results[task] = evaluator.evaluate(task)

    results["config"] = {
        "model": args.model,
        "revision": args.revision,
        "temperature": args.temperature,
        "n_samples": args.n_samples,
    }
    if not args.generation_only:
        dumped = json.dumps(results, indent=2)
        if accelerator.is_main_process:
            print(dumped)

        with open(args.metric_output_path, "w") as f:
            f.write(dumped)

    # Extra logging for early exit
    if len(wandb.config.num_layers) > 0:
        df = pd.DataFrame(wandb.config.num_layers).reset_index()
        avg_exit_layer = df.mean().to_dict()["layer"]
        df_agg = df.groupby("cur_token_index").agg({"layer": "mean", "index": "count"})
        example_count = df_agg["index"].max()
        df_agg_to_log = df_agg[df_agg["index"] == example_count]
        token_exit_metrics = df_agg_to_log.reset_index().to_dict(orient="records")

        for token_metric in token_exit_metrics:
            wandb.log(token_metric)
        
        wandb.log({'avg_exit_layer': avg_exit_layer})
    
    wandb_summary = wandb.run.summary._as_dict()
    wandb.log({"mean_latency": wandb_summary["generation_time_ms"]['mean'] / wandb_summary["num_new_tokens"]['mean']})
    
if __name__ == "__main__":
    main()
