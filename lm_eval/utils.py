import re
import json
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import IterableDataset

from datasets import load_dataset

EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]


def truncate_prompt(prompt, tokenizer, max_length, call_format):
    # if a prompt is very long we truncate it but keep the end phrases
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    if len(input_ids) > max_length:
        end_phrase = tokenizer(call_format + "\nANSWER:\n", return_tensors="pt").input_ids[0]
        max_length = max_length - len(end_phrase)
        new_ids = torch.cat((input_ids[:max_length], end_phrase))
        prompt = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return prompt


def generate_prompt_apps(sample, tokenizer, max_length=1024):
    """Generate prompts for APPS, they include a question along with some starter code and function name if they exist
    We also specify the type of the prompt, i.e. whether it is call-based or standard input"""

    starter_code = None if len(sample["starter_code"]) == 0 else sample["starter_code"] 
    try:
        input_outpout = json.loads(sample["input_output"])
        fn_name = None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
    except ValueError:
        fn_name = None
    prompt = "\nQUESTION:\n"
    prompt += sample["question"]
    if starter_code:
        prompt += starter_code
    if fn_name:
        call_format = "\nUse Standard Input format"
        prompt += call_format
    else:
        call_format = "\nUse Call-Based format"
        prompt += call_format
    prompt += "\nANSWER:\n"
    prompt = truncate_prompt(prompt, tokenizer, max_length, call_format)
    return prompt


def generate_prompt_mbpp(sample):
    """Generate prompts for MBPP prompt similarily to the original paper:
    prompt = description + 'Your code should
    satisfy these tests:'+ three assert statements"""

    prompt = sample["text"] + " Your code should satisfy these tests:\n"
    for test in sample["test_list"]:
        prompt += "\n" + test
    return prompt


def remove_last_block(string):
    """Remove the last block of the code containing EOF_STRINGS for HumanEval"""
    
    string_list = re.split("(%s)" % "|".join(EOF_STRINGS), string)
    # last string should be ""
    return "".join(string_list[:-2])


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially.
    See compute_code for more details.
    """

    def __init__(self, tokenizer, dataset, mode="humaneval", n_tasks=None, n_copies=1, max_length=1024):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.mode = mode
        self.n_tasks = len(dataset) if n_tasks is None else n_tasks
        self.n_copies = n_copies
        self.max_length = 1024
        
    def __iter__(self):
        prompts = []
        for task in range(self.n_tasks):
            if self.mode == "apps":
                prompt = generate_prompt_apps(self.dataset[task], self.tokenizer, self.max_length)
            elif self.mode == "mbpp":
                prompt = generate_prompt_mbpp(self.dataset[task])
            else:
                prompt = self.dataset[task]["prompt"].strip()
            prompt = self.tokenizer.eos_token + prompt
            prompts.append(prompt)
        outputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
        for task in range(self.n_tasks):
            for _ in range(self.n_copies):
                yield {
                    "ids": outputs.input_ids[task],
                    "task_id": task,
                    "input_len": outputs.attention_mask[task].sum(),
                }


def complete_code(accelerator, model, tokenizer, dataloader, n_tasks, batch_size=20, mode="humaneval", **gen_kwargs):
    """Generate multiple codes for each task in the dataset using multiple GPUs with accelerate.
    dataloader sends all the prompts from the evalution dataset to the model as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1] where nc is the number of copies of the prompt,
    and nt is the number of tasks. nc is such that num_samples(for each task)= nc * batch_size
    """
    if mode == "mbpp":
        MBPP = load_dataset("mbpp", split="test", ignore_verifications=True)
        # the MBPP evaluation set is task ids 11->510 
        MBPP = MBPP.select([i for i in range(10,510)])

    gen_token_dict = defaultdict(list)  # dict of list of generated tokens
    for step, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            if mode == "humaneval":
                gen_kwargs["stopping_criteria"][0].start_length = batch["ids"].shape[-1]
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=batch["ids"][:, : batch["input_len"]], num_return_sequences=batch_size, **gen_kwargs
            )
            # each task is generated batch_size times
            generated_tasks = batch["task_id"].repeat(batch_size)
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens, generated_tasks = accelerator.gather((generated_tokens, generated_tasks))
            generated_tokens = generated_tokens.cpu().numpy()
            generated_tasks = generated_tasks.cpu().numpy()

            for task, generated_tokens in zip(generated_tasks, generated_tokens):
                gen_token_dict[task].append(generated_tokens)

    code_gens = [[] for _ in range(n_tasks)]
    for task, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            gen_code = tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if mode == "humaneval":
                code_gens[task].append(remove_last_block(gen_code))
            elif mode == "apps":
                try:
                    code_gens[task].append(gen_code.split("ANSWER:")[1].replace(tokenizer.eos_token, ""))
                except IndexError:
                    print(f"Index error for task {task}!!!!")
                    code_gens[task].append(gen_code.replace(tokenizer.eos_token, ""))
            else:
                prompt = generate_prompt_mbpp(MBPP[task])
                code_gens[task].append(gen_code[len(prompt):].replace(tokenizer.eos_token, ""))
    return code_gens