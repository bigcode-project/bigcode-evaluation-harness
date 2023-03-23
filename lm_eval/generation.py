from tqdm import tqdm
import json

from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteria, StoppingCriteriaList
from accelerate.utils import set_seed

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
    if args.generations_path:
        # load generated code
        with open(args.generations_path) as fp:
            generations = json.load(fp)
            if accelerator.is_main_process:
                print(
                    f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
                )
        return generations[:n_tasks]

    # When num_return_sequnces and n_samples are same - device specific seed to be disabled
    # as device placement of a given batch/task could vary during runtime and having device specific seed could introduce 
    # variation in the results.
    # When num_return_sequnces < n_samples, device specific seed is required to
    # generate variations
    seed_specific_to_device = args.num_return_sequences != args.n_samples
    set_seed(args.seed, device_specific=seed_specific_to_device)

    # Setup generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_length": args.max_length_generation,
    }
    if task.stop_words:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [EndOfFunctionCriteria(0, task.stop_words, tokenizer)]
        )

    if accelerator.is_main_process:
        print(f"number of problems for this task is {n_tasks}")
    n_copies = args.n_samples // args.num_return_sequences

    ds_tokenized = TokenizedDataset(
        task,
        dataset,
        tokenizer,
        num_devices=accelerator.state.num_processes,
        max_length=args.max_length_generation,
        n_tasks=n_tasks,
        n_copies=n_copies,
        prefix=args.prefix,
    )

    ds_loader = DataLoader(ds_tokenized, batch_size=args.batch_size)

    model, ds_loader = accelerator.prepare(model, ds_loader)
    generations = complete_code(
        task,
        accelerator,
        model,
        tokenizer,
        ds_loader,
        n_tasks=n_tasks,
        num_return_sequences=args.num_return_sequences,
        prefix=args.prefix,
        postprocess=args.postprocess,
        **gen_kwargs,
    )
    return generations
