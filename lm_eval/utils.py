import re
from collections import defaultdict
import warnings

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from tqdm import tqdm

from lm_eval.prompts import (
    generate_prompt_apps,
    mbpp_incoder_prompt,
    mbpp_google_prompt,
    code_to_text_prompt,
    conala_prompt,
    spider_prompt,
    concode_prompt,
    codexglue_tt_prompt,
)


EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
MBPP_EOF_STRINGS = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/"]
EOF_APPS_FEW_SHOT = ["\nQUESTION", "\n---", "\nANSWER"]
TRIPLE_QUOTE = '"""'
SINGLE_TRIPLE_QUOTE = "'''"


def first_block(string, stop_words):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    return re.split("|".join(stop_words), string)[0].rstrip()


def remove_last_block(string, stop_words):
    """Remove the last block of the code containing stop_words for HumanEval and MBPP"""
    string_list = re.split("(%s)" % "|".join(stop_words), string)
    # last string should be ""
    return "".join(string_list[:-2])


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially.
    See compute_code for more details.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        num_devices,
        mode="humaneval",
        n_tasks=None,
        n_copies=1,
        max_length_prompt=1024,
        include_tests_mbpp=True,
        include_solution_mbpp=False,
        prompt_type_mbpp="incoder",
        prompt_type_code_to_text="left",
        language="python",
        translation_task="zh_en",
        prefix="",
        setup="finetuning",
    ):

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.num_devices = num_devices
        self.mode = mode
        self.n_tasks = len(dataset) if n_tasks is None else n_tasks
        self.n_copies = n_copies
        self.max_length_prompt = max_length_prompt
        self.include_tests_mbpp = include_tests_mbpp
        self.include_solution_mbpp = include_solution_mbpp
        self.prompt_type_mbpp = prompt_type_mbpp
        self.prompt_type_code_to_text = prompt_type_code_to_text
        self.language = language
        self.translation_task = translation_task
        self.prefix = prefix
        self.setup = setup

    def __iter__(self):
        prompts = []
        for task in range(self.n_tasks):
            if self.mode == "apps":
                prompt = generate_prompt_apps(
                    self.dataset[task],
                    self.tokenizer,
                    self.max_length_prompt,
                    prefix=self.prefix,
                    setup=self.setup,
                ).strip()
            elif self.mode == "mbpp":
                if self.prompt_type_mbpp == "incoder":
                    prompt = mbpp_incoder_prompt(
                        self.dataset[task],
                        self.include_solution_mbpp,
                        prefix=self.prefix,
                    )
                else:
                    prompt = mbpp_google_prompt(
                        self.dataset[task], self.include_tests_mbpp, prefix=self.prefix
                    )
            elif self.mode == "humaneval":
                prompt = self.prefix + self.dataset[task]["prompt"].strip()

            elif self.mode == "conala":
                prompt = conala_prompt(self.dataset[task], prefix="")

            elif self.mode == "spider":
                prompt = spider_prompt(self.dataset[task], prefix="")

            elif self.mode == "concode":
                prompt = concode_prompt(self.dataset[task], prefix="")

            elif self.mode == "code-to-text":
                prompt = code_to_text_prompt(
                    self.dataset[task],
                    language=self.language,
                    prompt_type=self.prompt_type_code_to_text,
                    prefix=self.prefix,
                )
            
            elif self.mode == "codexglue-tt":
                prompt = codexglue_tt_prompt(
                    self.dataset[task],
                    trans_task=self.translation_task,
                    prefix=""
                )
    
            prompts.append(prompt)

        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length_prompt,
        )
        if self.n_copies == 1 and self.n_tasks % self.num_devices != 0:
            self.n_copies = 2
            warnings.warn("n_copies (n_samples/batch_size) was changed from 1 to 2 because n_tasks isn't proportional to num devices ")

        for task in range(self.n_tasks):
            for _ in range(self.n_copies):
                yield {
                    "ids": outputs.input_ids[task],
                    "task_id": task,
                    "input_len": outputs.attention_mask[task].sum(),
                }


def complete_code(
    accelerator,
    model,
    tokenizer,
    dataloader,
    n_tasks,
    batch_size=20,
    mode="humaneval",
    include_tests_mbpp=True,
    include_solution_mbpp=False,
    prompt_type_mbpp="incoder",
    prompt_type_code_to_text="left",
    language="python",
    prefix="",
    setup="finetuning",
    postprocess=True,
    **gen_kwargs,
):

    """Generate multiple codes for each task in the dataset using multiple GPUs with accelerate.
    dataloader sends all the prompts from the evalution dataset to the model as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1] where nc is the number of copies of the prompt,
    and nt is the number of tasks. nc is such that num_samples(for each task)= nc * batch_size
    """

    if mode == "mbpp":
        MBPP = load_dataset("mbpp", split="test")

    gen_token_dict = defaultdict(list)  # dict of list of generated tokens
    for step, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            if mode in ["humaneval", "code-to-text", "conala", "spider", "concode"] or (mode == "apps" and setup != "finetuning"):
                gen_kwargs["stopping_criteria"][0].start_length = batch["ids"].shape[-1]
            elif mode == "codexglue-tt":
                gen_kwargs["stopping_criteria"][0].start_length = batch["input_len"]
            generated_tokens = accelerator.unwrap_model(model).generate(
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

            for task, generated_tokens in zip(generated_tasks, generated_tokens):
                gen_token_dict[task].append(generated_tokens)

    code_gens = [[] for _ in range(n_tasks)]
    for task, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            gen_code = tokenizer.decode(
                s, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            if mode == "humaneval":
                if postprocess:
                    code_gens[task].append(
                        remove_last_block(gen_code[len(prefix) :], EOF_STRINGS)
                    )
                else:
                    warnings.warn("model output is not postprocessed, this might lower evaluation scores")
                    code_gens[task].append(gen_code[len(prefix) :])
            elif mode == "apps":
                try:
                    if setup != "finetuning":
                        # we take the third answwer (2 are few shot examples)
                        output = gen_code.split("\nANSWER:", 3)[-1]
                        #output = re.split("|".join(EOF_APPS_FEW_SHOT), output)[0].rstrip()
                        code_gens[task].append(output)
                    else:
                        code_gens[task].append(gen_code.split("\nANSWER:", 1)[1])
                except IndexError:
                    print(f"Index error for task {task}!")
                    code_gens[task].append(gen_code.replace(tokenizer.eos_token, ""))

            elif mode == "mbpp":
                if prompt_type_mbpp == "incoder":
                    prompt = mbpp_incoder_prompt(
                        MBPP[int(task)], include_solution_mbpp, prefix
                    )
                else:
                    prompt = mbpp_google_prompt(
                        MBPP[int(task)], include_tests_mbpp, prefix
                    )
                if postprocess:
                    gen_code = gen_code[len(prompt) :]
                    code_gens[task].append(first_block(gen_code, MBPP_EOF_STRINGS))
                else:
                    warnings.warn("model output is not postprocessed, this might lower evaluation scores")
                    code_gens[task].append(gen_code)

            elif mode == "code-to-text":
                # delimiters used in case the prompt = full function body
                delimiters = {
                    "python": '\n"""Explanation of the code above:\n',
                    "ruby": "\n=begin Explanation of the code above:\n",
                    "other": "\n/* Explanation of the code above:\n",
                }
                if language == "python" and prompt_type_code_to_text == "left":
                    output = gen_code.strip().split("\n")[0].strip()
                    for delimiter in [TRIPLE_QUOTE, SINGLE_TRIPLE_QUOTE]:
                        if delimiter in gen_code:
                            gen_code = gen_code[gen_code.index(delimiter) + 3 :]
                            output = gen_code.strip().split("\n")[0].strip()
                            output = output.split(delimiter, 1)[0]
                else:
                    output = gen_code.split(delimiters[language])[1].strip()
                    output = output.split("\n")[0]
                code_gens[task].append(output)

            elif mode in ["conala", "spider", "concode"]:
                output = gen_code.split("Solution:\n", 3)[-1]
                output = output.split("\n")[0]
                code_gens[task].append(output)
            elif mode == "codexglue-tt":
                output = gen_code.split("\nEnglish:\n", 3)[-1]
                output = output.split("\n")[0]
                code_gens[task].append(output)


    return code_gens
