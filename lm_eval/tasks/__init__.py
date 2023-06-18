import inspect
from pprint import pprint

from . import (apps, codexglue_code_to_text, codexglue_text_to_text, conala,
               concode, ds1000, gsm, humaneval, humaneval_x_bugs, humaneval_x_explain_describe, humaneval_x_explain_generate, humaneval_x_generate, mbpp, multiple, parity, python_bugs, quixbugs)

TASK_REGISTRY = {
    **apps.create_all_tasks(),
    **codexglue_code_to_text.create_all_tasks(),
    **codexglue_text_to_text.create_all_tasks(),
    **multiple.create_all_tasks(),
    "codexglue_code_to_text-python-left": codexglue_code_to_text.LeftCodeToText,
    "conala": conala.Conala,
    "concode": concode.Concode,
    **ds1000.create_all_tasks(),
    "humaneval": humaneval.HumanEval,
    **humaneval_x_bugs.create_all_tasks(),
    **humaneval_x_generate.create_all_tasks(),
    **humaneval_x_explain_describe.create_all_tasks(),
    **humaneval_x_explain_generate.create_all_tasks(),
    "mbpp": mbpp.MBPP,
    "parity": parity.Parity,
    "python_bugs": python_bugs.PythonBugs,
    "quixbugs": quixbugs.QuixBugs,
    **gsm.create_all_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, args=None):
    try:
        ## Only if the task takes a mutate_method argument, should we pass it
        #if "mutate_method" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
        #    return TASK_REGISTRY[task_name](mutate_method=mutate_method)
        #else:
        kwargs = {}
        if "mutate_method" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["mutate_method"] = args.mutate_method
        if "load_data_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["load_data_path"] = args.load_data_path
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
