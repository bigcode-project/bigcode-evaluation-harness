from accelerate.utils import set_seed
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import re
from transformers import StoppingCriteria, StoppingCriteriaList

from lm_eval.utils import TokenizedDataset, complete_code

EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]


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


def get_references_humaneval(dataset, num_tasks=None):
    n_tasks = num_tasks if num_tasks is not None else len(dataset)
    references = []
    for task in tqdm(range(n_tasks)):
        test_func = dataset[task]["test"]
        entry_point = f"check({dataset[task]['entry_point']})"
        references.append("\n" + test_func + "\n" + entry_point)
    return references


def get_references_mbpp(dataset, num_tasks=None):
    n_tasks = num_tasks if num_tasks is not None else len(dataset)
    references = []
    for task in tqdm(range(n_tasks)):
        asserts = "\n".join(dataset[task]["test_list"])
        references.append(asserts)
    return references


def get_references_code_to_text(dataset, num_tasks=None):
    n_tasks = num_tasks if num_tasks is not None else len(dataset)
    references = []
    for task in tqdm(range(n_tasks)):
        docstring = dataset[task]["docstring"]
        # strip extraneous content such as arguments definition
        reference = re.split('Arguments:|arguments:|Args:|args:|returns:|Returns:', docstring)[0].strip()
        references.append(reference)
    return references

def parallel_generations(
    accelerator, model, tokenizer, dataset, mode, args, num_tasks=None
):

    set_seed(args.seed, device_specific=True)

    # Setup generation settings
    if mode == "code-to-text":
        # use greedy sampling for the dosctring generation task
        gen_kwargs = {
            "do_sample": False,
            "max_length": args.max_length_generation,
            "stopping_criteria": StoppingCriteriaList(
                [EndOfFunctionCriteria(0, ["\n"], tokenizer)]
            )
        }
    else:
        gen_kwargs = {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_length": args.max_length_generation,
        }
        if mode == "humaneval":
            # to check: stoppingcriteria had an issue with InCoder for MBPP
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [EndOfFunctionCriteria(0, EOF_STRINGS, tokenizer)]
            )

    n_tasks = num_tasks if num_tasks is not None else len(dataset)
    n_copies = args.n_samples // args.batch_size

    ds_tokenized = TokenizedDataset(
        tokenizer,
        dataset,
        mode=mode,
        n_tasks=n_tasks,
        n_copies=n_copies,
        max_length_prompt=args.max_length_prompt,
        include_tests_mbpp=args.include_tests_mbpp,
        include_solution_mbpp=args.include_solution_mbpp,
        prompt_type_mbpp=args.prompt_type_mbpp,
        prompt_type_code_to_text=args.prompt_type_code_to_text,
        language=args.language,
        prefix=args.prefix,
    )

    # do not confuse args.batch_size, which is actually the num_return_sequences
    ds_loader = DataLoader(ds_tokenized, batch_size=1)

    model, ds_loader = accelerator.prepare(model, ds_loader)

    generations = complete_code(
        accelerator,
        model,
        tokenizer,
        ds_loader,
        n_tasks=n_tasks,
        batch_size=args.batch_size,
        mode=mode,
        include_tests_mbpp=args.include_tests_mbpp,
        include_solution_mbpp=args.include_solution_mbpp,
        prompt_type_mbpp=args.prompt_type_mbpp,
        prefix=args.prefix,
        **gen_kwargs,
    )
    return generations
