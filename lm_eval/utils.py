import math
import warnings
from collections import defaultdict

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

INFILL_MODE = False


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially.
    See compute_code for more details.
    """

    def __init__(
        self,
        task,
        dataset,
        tokenizer,
        num_devices,
        max_length,
        n_tasks=None,
        n_copies=1,
        prefix="",
    ):
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_devices = num_devices
        self.max_length = max_length
        self.n_tasks = n_tasks
        self.n_copies = n_copies
        self.prefix = prefix

    def __iter__(self):
        prompts = []
        infill = []
        for sample in range(self.n_tasks):
            prompt_contents = self.task.get_prompt(self.dataset[sample])
            if isinstance(prompt_contents, str):
                infill.append(False)
                prompt = self.prefix + prompt_contents
            elif isinstance(prompt_contents, dict):
                assert set(prompt_contents.keys()) == {"prefix", "suffix"}
                infill.append(True)
                prompt = self._make_infill_prompt(
                    **prompt_contents, preprefix=self.prefix
                )
            else:
                raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")
            prompts.append(prompt)

        if not len(set(infill)) == 1:
            raise ValueError("Mixed infill and completion prompts are not supported.")
        global INFILL_MODE
        INFILL_MODE = infill[0]
        if INFILL_MODE:
            return_token_type_ids = False
        else:
            return_token_type_ids = None  # default

        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            return_token_type_ids=return_token_type_ids,
        )

        if self.n_copies == 1 and self.n_tasks % self.num_devices != 0:
            self.n_copies = 2
            warnings.warn(
                "n_copies (n_samples/batch_size) was changed from 1 to 2 because n_tasks isn't proportional to num devices"
            )

        for sample in range(self.n_tasks):
            for _ in range(self.n_copies):
                yield {
                    "ids": outputs.input_ids[sample],
                    "task_id": sample,
                    "input_len": outputs.attention_mask[sample].sum(),
                }

    def _make_infill_prompt(self, prefix, suffix, preprefix=""):
        """Make a prompt for infilling.
        Currently supported only for official InCoder and SantaCoder implementations.
        """
        model_id = self.tokenizer.name_or_path
        if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            return f"{preprefix}{prefix}<|mask:0|>{suffix}<|mask:0|>"
        elif model_id in ["bigcode/santacoder"]:
            return f"<fim-prefix>{preprefix}{prefix}<fim-suffix>{suffix}<fim-middle>"
        elif model_id in ["bigcode/large-model", "bigcode/temp-model"]:
            return f"<fim_prefix>{preprefix}{prefix}<fim_suffix>{suffix}<fim_middle>"
        else:
            raise ValueError(f"Infilling not yet supported for: {model_id}")


def complete_code(
    task,
    accelerator,
    model,
    tokenizer,
    dataloader,
    n_tasks,
    batch_size=20,
    prefix="",
    postprocess=True,
    **gen_kwargs,
):
    """Generate multiple codes for each task in the dataset using multiple GPUs with accelerate.
    dataloader sends all the prompts from the evalution dataset to the model as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1] where nc is the number of copies of the prompt,
    and nt is the number of tasks. nc is such that num_samples(for each task)= nc * batch_size
    """

    gen_token_dict = defaultdict(list)  # dict of list of generated tokens
    for step, batch in tqdm(
        enumerate(dataloader),
        total=math.ceil(
            n_tasks * dataloader.dataset.n_copies / accelerator.num_processes
        ),
    ):
        with torch.no_grad():
            if task.stop_words:
                # Set the start_length after which to check for stopping to be the longest input ignoring padding
                gen_kwargs["stopping_criteria"][0].start_length = (
                    batch["input_len"].max().item()
                )
            generated_tokens = model.generate(
                input_ids=batch["ids"][:, : batch["input_len"]],
                num_return_sequences=batch_size,
                **gen_kwargs,
            )
            # each task is generated batch_size times
            generated_tasks = batch["task_id"].repeat(batch_size)
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens, generated_tasks = accelerator.gather(
                (generated_tokens, generated_tasks)
            )
            generated_tokens = generated_tokens.cpu().numpy()
            generated_tasks = generated_tasks.cpu().numpy()

            for sample, generated_tokens in zip(generated_tasks, generated_tokens):
                gen_token_dict[sample].append(generated_tokens)

    def parse_infill(code, tokenizer):
        """Reorder infill code and remove remaining special tokens."""
        model_id = tokenizer.name_or_path
        if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
            prefix, suffix, infill = code.split("<|mask:0|>", 2)
            infill = infill.split("<|endofmask|>")[0]
        elif model_id in ["bigcode/santacoder"]:
            prefix, rest = code.split("<fim-suffix>", 1)
            suffix, infill = rest.split("<fim-middle>", 1)
            infill = infill.split("<|endoftext|>")[0]
        elif model_id in ["bigcode/large-model", "bigcode/temp-model"]:
            prefix, rest = code.split("<fim_suffix>", 1)
            suffix, infill = rest.split("<fim_middle>", 1)
            infill = infill.split("<|endoftext|>")[0]
        else:
            raise ValueError(f"Infilling not yet supported for: {model_id}")
        for k, v in tokenizer.special_tokens_map.items():
            if k == "additional_special_tokens":
                for t in v:
                    infill = infill.replace(t, "")
            else:
                infill = infill.replace(v, "")
        return infill

    code_gens = [[] for _ in range(n_tasks)]
    for sample, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            if INFILL_MODE:
                gen_code = parse_infill(
                    tokenizer.decode(
                        s, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    ),
                    tokenizer,
                )
            elif tokenizer.eos_token in task.stop_words:
                gen_code = tokenizer.decode(
                    s, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
            else:
                gen_code = tokenizer.decode(
                    s, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            if postprocess:
                if INFILL_MODE:
                    code_gens[sample].append(
                        task.postprocess_generation(gen_code, int(sample))
                    )
                else:
                    code_gens[sample].append(
                        task.postprocess_generation(
                            gen_code[len(prefix) :], int(sample)
                        )
                    )
            else:
                warnings.warn(
                    "model output is not postprocessed, this might lower evaluation scores"
                )
                code_gens[sample].append(gen_code[len(prefix) :])

    return code_gens
