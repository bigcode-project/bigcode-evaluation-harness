from tqdm import tqdm
from mosestokenizer import MosesDetokenizer
import json

from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteria, StoppingCriteriaList
from accelerate.utils import set_seed

from lm_eval.utils import TokenizedDataset, complete_code

EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
EOF_STRINGS_DOCSTRING = ["'''", '"""']
EOF_APPS_FEW_SHOT = ["\nQUESTION", "\n---", "\nANSWER"]


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
        # docstring_tokens are preprocessed and don't have extra context like variable defs
        docstring = " ".join(dataset[task]["docstring_tokens"]).replace("\n", "")
        # some docstrings started with r""" before tokenization but r was kept
        if docstring[0] == "r":
            docstring = docstring[1:]
        with MosesDetokenizer("en") as detokenize:
            docstring = detokenize(docstring.strip().split())
        references.append(docstring)
    return references


def parallel_generations(
    accelerator, model, tokenizer, dataset, mode, args, num_tasks=None
):
    n_tasks = num_tasks if num_tasks is not None else len(dataset)
    if args.evaluation_only:
        # load generated code
        with open(args.generations_path) as fp:
            generations = json.load(fp)
            if accelerator.is_main_process:
                print(f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates")
        return generations[:num_tasks]

    set_seed(args.seed, device_specific=True)

    # Setup generation settings
    if mode == "code-to-text":
        # use greedy sampling for the dosctring generation task
        stop_words = ["'''", '"""']
        if args.language != "python" or args.prompt_type_code_to_text != "left":
            stop_words = ["\n"]
        gen_kwargs = {
            "do_sample": False,
            "max_length": args.max_length_generation,
            "stopping_criteria": StoppingCriteriaList(
                [EndOfFunctionCriteria(0, stop_words, tokenizer)]
            ),
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
        elif mode in ["conala", "spider", "concode","codexglue-tt"]:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [EndOfFunctionCriteria(0, ["\n"], tokenizer)]
            )
        elif mode == "apps" and args.setup_apps != "finetuning":
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [EndOfFunctionCriteria(0, EOF_APPS_FEW_SHOT, tokenizer)]
            )

    if accelerator.is_main_process:
        print(f"ntasks for generation is {n_tasks}")
    n_copies = args.n_samples // args.batch_size

    ds_tokenized = TokenizedDataset(
        tokenizer,
        dataset,
        num_devices = accelerator.state.num_processes,
        mode=mode,
        n_tasks=n_tasks,
        n_copies=n_copies,
        max_length_prompt=args.max_length_generation,
        include_tests_mbpp=args.include_tests_mbpp,
        include_solution_mbpp=args.include_solution_mbpp,
        prompt_type_mbpp=args.prompt_type_mbpp,
        prompt_type_code_to_text=args.prompt_type_code_to_text,
        language=args.language,
        translation_task=args.translation_task_codexglue_tt,
        setup=args.setup_apps,
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
        setup=args.setup_apps,
        postprocess=args.postprocess,
        **gen_kwargs,
    )
    return generations
