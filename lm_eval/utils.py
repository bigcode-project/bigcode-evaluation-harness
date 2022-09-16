import json
import re
from collections import defaultdict

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from tqdm import tqdm

EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
MBPP_EOF_STRINGS = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/"]

def truncate_prompt_apps(prompt, tokenizer, max_length, call_format):
    # if a prompt is very long we truncate it but keep the end phrases
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    if len(input_ids) > max_length:
        end_phrase = tokenizer(
            call_format + "\nANSWER:\n", return_tensors="pt"
        ).input_ids[0]
        max_length = max_length - len(end_phrase)
        new_ids = torch.cat((input_ids[:max_length], end_phrase))
        prompt = tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    return prompt


def first_block(string, stop_words):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    return re.split("|".join(stop_words), string)[0].rstrip()


def remove_last_block(string, stop_words):
    """Remove the last block of the code containing stop_words for HumanEval and MBPP"""
    string_list = re.split("(%s)" % "|".join(stop_words), string)
    # last string should be ""
    return "".join(string_list[:-2])


def generate_prompt_apps(sample, tokenizer, max_length=1024, prefix="", setup="finetuning"):
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
    if setup != "finetuning":
        # few shot mode: this adds 270 tokens in avg to the prompt
        prompt = apps_few_shot_prompt(prompt)
    prompt = truncate_prompt_apps(prompt, tokenizer, max_length, call_format)
    return prefix + prompt


def apps_few_shot_prompt(prompt):
    with open("lm_eval/few_shot_examples/apps_few_shot_prompts.json", "r") as file:
        examples = json.load(file)
    
    # add two examples one for each implementation type: call-based/input-based
    one_shot_prompt = "Implement answers to the following problems:\nProblem:\n" + examples["problem_type1"] + \
        "\nUse Standard Input format\nANSWER:\n" + examples["solution_type1"] + "\n\nProblem:\n" + examples["problem_type2"] \
        + "\nUse Call-Based format\nANSWER:\n\n" + examples["solution_type2"] + "\n\nProblem:\n" + prompt
    return one_shot_prompt


def mbpp_incoder_prompt(sample, include_solution_mbpp=False, prefix=""):
    """Generate prompts for MBPP prompt similarily to InCoder, docstring
    that includes one test"""
    description = sample["text"]
    test_example = sample["test_list"][0]
    prompt = f'"""\n{description}\n{test_example}\n"""\n'

    if include_solution_mbpp:
        prompt += f"{sample['code']}\n"
    return prefix + prompt


def mbpp_google_prompt(sample, include_tests=True, prefix=""):
    """Generate prompts for MBPP similarily to the original google paper
    with an option for including the tests cases or not:
    prompt = description + 'Your code should
    satisfy these tests:'+ three assert statements"""

    prompt = sample["text"]
    if include_tests:
        prompt += " Your code should satisfy these tests:\n"
        for test in sample["test_list"]:
            prompt += "\n" + test
    return prefix + prompt

def code_to_text_prompt(sample, language="python", prompt_type="left", prefix=""):
    """Generate prompts for code-to-text task
    For prompt_type left we include the left code with function signature (only possible for Python now), 
    else we only include the whole body"""
    # TODO implement signature extraction for other languages ?
    code = sample["code"]
    if language == "python":
        splits = code.split('"""')
        if prompt_type == "left":
            prompt = splits[0].strip() + '\n    """\n'
        else:
            prompt = splits[0].strip() + splits[2] + '\n"""Explanation of the code above:\n'            
        return prefix + prompt
    if language == "Ruby":          
        return prefix + prompt + '\n=begin Explanation of the code above:\n' 
    else:
        return prefix + prompt + '\n/* Explanation of the code above:\n' 


def two_shot_prompt(entry, text, examples):
    instrcution1 = "\nInstruction:\n" + examples["instruction1"]
    solution1 = "\nSolution:\n" + examples["solution1"]
    instrcution2 = "\nInstruction:\n" + examples["instruction2"]
    solution2 = "\nSolution:\n" + examples["solution2"]
    examples = entry + instrcution1 + solution1 + instrcution2 + solution2 
    prompt = examples + "\nInstruction:\n" + text + "\nSolution:\n"
    return prompt

def conala_prompt(sample, prefix=""):
    """Generate prompts for CoNaLa text-to-code task in a 2-shot setting"""
    with open("lm_eval/few_shot_examples/conala_few_shot_prompts.json", "r") as file:
        examples = json.load(file)
    text_column = 'rewritten_intent' if sample['rewritten_intent'] else 'intent'
    text = prefix + sample[text_column].strip()
    entry = "Answer the following instructions in one line of Python code:\n"
    prompt = two_shot_prompt(entry, text, examples)
    return prefix + prompt

def spider_prompt(sample, prefix=""):
    """Generate prompts for Spider text-to-code task in a 2-shot setting"""
    with open("lm_eval/few_shot_examples/spider_few_shot_prompts.json", "r") as file:
        examples = json.load(file)
    text = prefix + sample["question"].strip()
    entry = "Answer the following instructions in a one line SQL query:\n"
    prompt = two_shot_prompt(entry, text, examples)
    return prefix + prompt

def concode_prompt(sample, prefix=""):
    """Generate prompts for Spider text-to-code task in a 2-shot setting"""
    with open("lm_eval/few_shot_examples/concode_few_shot_prompts.json", "r") as file:
        examples = json.load(file)
    text = sample["nl"].split("concode_field_sep")[0].strip()
    if text.endswith("."):
        text = text[:-1].strip()
    text = prefix + text
    entry = "Answer the following instructions in a one line of Java code:\n"
    prompt = two_shot_prompt(entry, text, examples)
    return prefix + prompt


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially.
    See compute_code for more details.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        mode="humaneval",
        n_tasks=None,
        n_copies=1,
        max_length_prompt=1024,
        include_tests_mbpp=True,
        include_solution_mbpp=False,
        prompt_type_mbpp="incoder",
        prompt_type_code_to_text="left",
        language="python",
        prefix="",
        setup="finetuning",
    ):

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.mode = mode
        self.n_tasks = len(dataset) if n_tasks is None else n_tasks
        self.n_copies = n_copies
        self.max_length_prompt = max_length_prompt
        self.include_tests_mbpp = include_tests_mbpp
        self.include_solution_mbpp = include_solution_mbpp
        self.prompt_type_mbpp = prompt_type_mbpp
        self.prompt_type_code_to_text = prompt_type_code_to_text
        self.language = language
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
                
            prompts.append(prompt)

        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length_prompt,
        )
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
    **gen_kwargs,
):

    """Generate multiple codes for each task in the dataset using multiple GPUs with accelerate.
    dataloader sends all the prompts from the evalution dataset to the model as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1] where nc is the number of copies of the prompt,
    and nt is the number of tasks. nc is such that num_samples(for each task)= nc * batch_size
    """

    if mode == "mbpp":
        MBPP = load_dataset("mbpp", split="test", ignore_verifications=True)
        # the MBPP evaluation set is task ids 11->510
        MBPP = MBPP.select([i for i in range(10, 510)])

    gen_token_dict = defaultdict(list)  # dict of list of generated tokens
    for step, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            if mode in ["humaneval", "code-to-text", "conala", "spider", "concode"]:
                gen_kwargs["stopping_criteria"][0].start_length = batch["ids"].shape[-1]
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
                code_gens[task].append(
                    remove_last_block(gen_code[len(prefix) :], EOF_STRINGS)
                )

            elif mode == "apps":
                try:
                    if setup != "finetuning":
                        # we take the last answer
                        code_gens[task].append(gen_code.split("\nANSWER:", -1)[-1])
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
                output = gen_code[len(prompt) :]
                code_gens[task].append(first_block(output, MBPP_EOF_STRINGS))

            elif mode == "code-to-text":
                delimiters = {"python": '\n"""Explanation of the code above:\n',
                             "ruby": '\n=begin Explanation of the code above:\n',
                             "other":'\n/* Explanation of the code above:\n'}

                if language == "python" and prompt_type_code_to_text == "left":
                    output = gen_code.split('"""\n')[1].strip()
                    output = output.split("\n")[0]
                else:
                    output = gen_code.split(delimiters[language])[1].strip()
                    output = output.split("\n")[0]
                code_gens[task].append(output)

            elif mode in ["conala", "spider", "concode"]:
                output = gen_code.split("Solution:\n", 3)[-1]
                output = output.split("\n")[0]
                code_gens[task].append(output)
            
    return code_gens
