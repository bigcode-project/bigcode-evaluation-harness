import inspect
from pprint import pprint

from . import (apps, codexglue_code_to_text, codexglue_text_to_text, conala,
               concode, ds1000, humaneval, humaneval_x_bugs, mbpp, parity, python_bugs, quixbugs)

TASK_REGISTRY = {
    **apps.create_all_tasks(),
    **codexglue_code_to_text.create_all_tasks(),
    **codexglue_text_to_text.create_all_tasks(),
    "codexglue_code_to_text-python-left": codexglue_code_to_text.LeftCodeToText,
    "conala": conala.Conala,
    "concode": concode.Concode,
    **ds1000.create_all_tasks(),
    "humaneval": humaneval.HumanEval,
    "humaneval_x_bugs": humaneval_x_bugs.HumanEvalXBugs,
    "mbpp": mbpp.MBPP,
    "parity": parity.Parity,
    "python_bugs": python_bugs.PythonBugs,
    "quixbugs": quixbugs.QuixBugs,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, mutate_method=None):
    try:
        # Only if the task takes a mutate_method argument, should we pass it
        if "mutate_method" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            return TASK_REGISTRY[task_name](mutate_method=mutate_method)
        else:
            return TASK_REGISTRY[task_name]()
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
