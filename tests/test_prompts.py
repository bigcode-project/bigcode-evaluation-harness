import json

from lm_eval import tasks

TASKS = ["pal-gsm8k-greedy"]

sample_doc = {"pal-gsm8k-greedy": {"question": "test"}}


def load_reference_prompt(task_name):
    with open(f"tests/data/{task_name}_prompt.json") as fp:
        prompts = json.load(fp)
    return prompts["prompt"]


def test_gsm_prompt():
    for task_name in TASKS:
        task = tasks.get_task(task_name)
        task_prompt = task.get_prompt(sample_doc[task_name])
        ref_prompt = load_reference_prompt(task_name)
        assert task_prompt == ref_prompt
