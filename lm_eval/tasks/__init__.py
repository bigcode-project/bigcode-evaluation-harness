from pprint import pprint

from . import apps
from . import code_to_text
from . import code_to_text_python_left
from . import conala
from . import concode
from . import humaneval
from . import mbpp

TASK_REGISTRY = {
    **apps.create_all_tasks(),
    **code_to_text.create_all_tasks(),
    "code_to_text_python_left": code_to_text_python_left.CodeToTextLeft,
    "conala": conala.Conala,
    "concode": concode.Concode,
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
