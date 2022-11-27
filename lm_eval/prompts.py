"""Prompt design for each benchmark"""

import json
import re
import torch


EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
MBPP_EOF_STRINGS = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/"]
TRIPLE_QUOTE = '"""'
SINGLE_TRIPLE_QUOTE = "'''"
SPACES4 = " " * 4
SOURCE_LANG = {
                "dn_en":"danish",
                "zh_en":"chinese",
                "no_en":"norwegian",
                "lv_en":"latvian",
            }


def truncate_prompt_apps(prompt, tokenizer, max_length, call_format):
    # if a prompt is very long we truncate it but keep the end phrases
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    if len(input_ids) > max_length:
        print("max reached:", len(input_ids))
        end_phrase = tokenizer(
            call_format + "\nANSWER:\n", return_tensors="pt"
        ).input_ids[0]
        max_length = max_length - len(end_phrase)
        new_ids = torch.cat((input_ids[:max_length], end_phrase))
        prompt = tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    return prompt


def apps_few_shot_prompt(prompt):
    with open("lm_eval/few_shot_examples/apps_few_shot_prompts.json", "r") as file:
        examples = json.load(file)

    # add two examples one for each implementation type: call-based/input-based
    one_shot_prompt = (
        "Implement answers to the following questions:\nQUESTION:\n"
        + examples["problem_type1"]
        + "\nUse Standard Input format\nANSWER:\n"
        + examples["solution_type1"]
        + "\nQUESTION:\n"
        + examples["problem_type2"]
        + "\nUse Call-Based format\nANSWER:\n"
        + examples["solution_type2"]
        + "\n"
        + prompt
    )
    return one_shot_prompt


def generate_prompt_apps(
    sample, tokenizer, max_length=1024, prefix="", setup="finetuning"
):
    """Generate prompts for APPS
    Finetuning setup: prompt= question  with some starter code and function name if they exist.
    We also specify the type of the prompt, i.e. whether it is call-based or standard input
    2-shot: two examples of input/output are included"""

    starter_code = (
        None if len(sample["starter_code"]) == 0 else sample["starter_code"]
    )
    try:
        input_outpout = json.loads(sample["input_output"])
        fn_name = (
            None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
        )
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
        # add two examples one for each implementation type: call-based/input-based
        prompt = apps_few_shot_prompt(prompt)
    prompt = truncate_prompt_apps(prompt, tokenizer, max_length, call_format)
    return prefix + prompt

# MBPP prompts inspired from https://github.com/dpfried/incoder/blob/d195d64eee055081585d0ecb1f93e2adbe694546/evaluation/mbpp.py
def mbpp_incoder_prompt(sample, include_solution_mbpp=False, prefix=""):
    """Generate prompts for MBPP prompt similarily to InCoder
    prompt = docstringthat includes one test"""
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
    # TODO implement signature extraction for other languages?
    code = sample["code"]

    if language == "python":
        # python code includes the docstring
        text = sample["docstring"]
        prompt_prefix = code[: code.index(text)]
        prompt_prefix = standardize_docstring_prompt(prompt_prefix)
        if prompt_type == "left":
            return prefix + prompt_prefix
        else:
            prompt_suffix = code[code.index(text) + len(text) :]
            prompt_suffix = prompt_suffix.replace(TRIPLE_QUOTE, "")
            prompt_suffix = prompt_suffix.replace(SINGLE_TRIPLE_QUOTE, "")

            prompt_prefix = prompt_prefix.strip().removesuffix(TRIPLE_QUOTE)
            prompt_prefix = prompt_prefix.strip().removesuffix(SINGLE_TRIPLE_QUOTE)
            prompt = (
                prompt_prefix + prompt_suffix + '\n"""Explanation of the code above:\n'
            )
            return prefix + prompt

    elif language == "Ruby":
        return prefix + code + "\n=begin Explanation of the code above:\n"

    else:
        return prefix + code + "\n/* Explanation of the code above:\n"


# source: InCoder evaluation code https://github.com/dpfried/lm-evaluation-harness/
def standardize_docstring_prompt(prefix):
    """Strips any existing docstring delimiters from the prompt prefix and
    and adds our own delimiter (triple quote) and whitespace.
    Note an edge cases being handled here:
    - codexglue docstring text sometimes contains the docstring delimiters, inconsistently
    """

    for delim in [TRIPLE_QUOTE, SINGLE_TRIPLE_QUOTE]:
        if delim in prefix:
            prefix = prefix[: prefix.index(delim)]
            break

    single_single_quote_with_trailing_spaces = re.compile(r'[^\'"][\']\s*$')
    if single_single_quote_with_trailing_spaces.search(prefix):
        prefix = prefix[
            : single_single_quote_with_trailing_spaces.search(prefix).start()
        ]

    single_double_quote_with_trailing_spaces = re.compile(r'[^\'"]["]\s*$')
    if single_double_quote_with_trailing_spaces.search(prefix):
        prefix = prefix[
            : single_double_quote_with_trailing_spaces.search(prefix).start()
        ]

    prefix += TRIPLE_QUOTE
    return prefix


def two_shot_prompt(entry, text, examples):
    """Two shot prompt format as instructions & solutions"""
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
    text_column = "rewritten_intent" if sample["rewritten_intent"] else "intent"
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

def codexglue_tt_prompt(sample,trans_task="zh_en",prefix=""):
    """Generate prompts for CodeXGlue text-to-text task in a 2-shot setting"""
    language = SOURCE_LANG[trans_task]
    with open("lm_eval/few_shot_examples/codexglue_text_to_text_few_shot_prompts.json","r") as file:
        examples = json.load(file)
    text = sample["source"]
    text = prefix + text
    examples = examples[language]
    entry = f"Translate the following documentation from {language.title()} to English:\n"
    src1 = f"\n{language.title()}:\n" + examples["source1"]
    tgt1 = "\nEnglish:\n" + examples["target1"]
    src2 = f"\n{language.title()}:\n" + examples["source2"]
    tgt2 = "\nEnglish:\n" + examples["target2"]
    examples = entry + src1 + tgt1 + src2 + tgt2
    prompt = examples + f"\n{language.title()}:\n" + text + "English:\n"
    return prefix + prompt
