from pprint import pprint

from . import humaneval
from . import mbpp
from . import apps

TASK_REGISTRY = {
    "humaneval": humaneval.HumanEval,
    "mbpp": mbpp.MBPP,
    **apps.create_all_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]()
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
