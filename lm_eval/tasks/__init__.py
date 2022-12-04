from pprint import pprint

from . import humaneval
from . import mbpp
from . import apps
from . import concode
from . import conala
from . import code_to_text
from . import code_to_text_python_left

TASK_REGISTRY = {
    "humaneval": humaneval.HumanEval,
    "mbpp": mbpp.MBPP,
    **apps.create_all_tasks(),
    "concode": concode.Concode,
    "conala": conala.Conala,
    **code_to_text.create_all_tasks(),
    "code_to_text_python_left": code_to_text_python_left.CodeToTextLeft,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]()
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
