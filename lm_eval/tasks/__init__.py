from pprint import pprint

from . import (apps, codexglue_code_to_text, codexglue_text_to_text, conala,
               concode, ds1000, gsm, humaneval, mbpp, program_repair)

TASK_REGISTRY = {
    **apps.create_all_tasks(),
    **codexglue_code_to_text.create_all_tasks(),
    **codexglue_text_to_text.create_all_tasks(),
    "codexglue_code_to_text-python-left": codexglue_code_to_text.LeftCodeToText,
    "conala": conala.Conala,
    "concode": concode.Concode,
    **ds1000.create_all_tasks(),
    "humaneval": humaneval.HumanEval,
    "mbpp": mbpp.MBPP,
    **gsm.create_all_tasks(),
    "program_repair": program_repair.ProgramRepair
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


# def get_task(task_name):
#     try:
#         return TASK_REGISTRY[task_name]()
def get_task(task_name, **kwargs):
    try:
        task = TASK_REGISTRY[task_name]
        return task(**kwargs) if kwargs else task()
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
