from pprint import pprint

from . import (apps, codexglue_code_to_text, codexglue_text_to_text, odex, mconala, conala,
               concode, ds1000, gsm, humaneval, mbpp)

TASK_REGISTRY = {
    **apps.create_all_tasks(),
    **codexglue_code_to_text.create_all_tasks(),
    **codexglue_text_to_text.create_all_tasks(),
    **odex.create_all_tasks(),
    **mconala.create_all_tasks(),
    **gsm.create_all_tasks(),
    "codexglue_code_to_text-python-left": codexglue_code_to_text.LeftCodeToText,
    "conala": conala.Conala,
    "concode": concode.Concode,
    **ds1000.create_all_tasks(),
    "humaneval": humaneval.HumanEval,
    "mbpp": mbpp.MBPP,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]()
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
