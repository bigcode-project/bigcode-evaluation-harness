import json
from math import ceil

from accelerate.utils import set_seed
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

from lm_eval.utils import TokenizedDataset, complete_code


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)


def parallel_generations(task, dataset, accelerator, model, tokenizer, n_tasks, args):
    if args.load_generations_path:
        # load generated code
        with open(args.load_generations_path) as fp:
            generations = json.load(fp)
            if accelerator.is_main_process:
                print(
                    f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
                )
        return generations[:n_tasks]

    set_seed(args.seed, device_specific=True)

    # Setup generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_length": args.max_length_generation,
    }
    if task.stop_words:
        if tokenizer.eos_token:
            task.stop_words.append(tokenizer.eos_token)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [EndOfFunctionCriteria(0, task.stop_words, tokenizer)]
        )
    if args.instruction_tokens:
        instruction_tokens = args.instruction_tokens.split(",")
        if len(instruction_tokens) != 3:
            raise ValueError(
                "Instruction tokens should contain exactly 3 tokens separated by a comma. If a token is empty, represent it as ''"
            )
        for token in instruction_tokens:
            if token.strip() != "":
                task.stop_words.append(token)
    else:
        instruction_tokens = None
    if accelerator.is_main_process:
        print(f"number of problems for this task is {n_tasks}")
    n_copies = ceil(args.n_samples / args.batch_size)

    ds_tokenized = TokenizedDataset(
        task,
        dataset,
        tokenizer,
        num_devices=accelerator.state.num_processes,
        max_length=args.max_length_generation,
        n_tasks=n_tasks,
        n_copies=n_copies,
        prefix=args.prefix,
        instruction_tokens=instruction_tokens,
    )

    # do not confuse args.batch_size, which is actually the num_return_sequences
    ds_loader = DataLoader(ds_tokenized, batch_size=1)
    is_loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    is_loaded_in_4bit = getattr(model, "is_loaded_in_4bit", False)
    if not is_loaded_in_8bit and not is_loaded_in_4bit:
        # we only wrap data loader to avoid extra memory occupation
        model = model.to(accelerator.device)
        ds_loader = accelerator.prepare(ds_loader)
    else:
        # model.to() is not supported for 8bit and 4bit models
        model, ds_loader = accelerator.prepare(model, ds_loader)

    generations = complete_code(
        task,
        accelerator,
        model,
        tokenizer,
        ds_loader,
        n_tasks=n_tasks,
        batch_size=args.batch_size,
        prefix=args.prefix,
        instruction_tokens=instruction_tokens,
        postprocess=args.postprocess,
        is_wrapped=is_loaded_in_8bit or is_loaded_in_4bit,
        **gen_kwargs,
    )
    return generations
