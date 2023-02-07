import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Optional

from .containerized_eval import eval_string_script

# Get working directory
WORKING_DIR = Path(__file__).parent.parent

# program: str => Result
CACHE = dict()
CACHE_LOCK = Lock()


def cache_get(program: str) -> Optional[dict]:
    if program in CACHE:
        result = CACHE[program]
        return result
    else:
        return None


def cache_set(program: str, result: dict):
    if program in CACHE:
        print("Setting already-existing cache")
    CACHE[program] = result


def cached_eval_script(problem, index) -> dict:
    program = (
        problem["prompt"] + problem["completions"][index] + "\n" + problem["tests"]
    )
    CACHE_LOCK.acquire(True)
    cached = cache_get(program)
    if cached is not None:
        CACHE_LOCK.release()
        return cached
    else:
        result_yaml = dict()
        cache_set(program, result_yaml)
        CACHE_LOCK.release()
        result_dict = eval_string_script(problem["language"], program)
        for k in result_dict.keys():
            result_yaml[k] = result_dict[k]
            result_yaml["timestamp"] = int(time.time())
        return result_yaml


def get_test_results_json_path(
    output_dir: Path, problem_json_path: Path, input_dir: Path
) -> Path:
    suffixes = ".results.json"
    problem_name = problem_json_path.name[: -len(".json")]
    if input_dir:
        return output_dir / (
            problem_json_path.relative_to(input_dir).parent / (problem_name + suffixes)
        )
    return output_dir / (problem_name + suffixes)


def evaluate_problem(
    output_dir: Path, problem_json_path: Path, max_workers: int, input_dir: Path = None
):
    with open(problem_json_path, "r") as f:
        problem = json.load(f)

    # Do not create a blank .results.yaml file if there are no completions ready.
    if len(problem["completions"]) == 0:
        return

    test_results_path = get_test_results_json_path(
        output_dir, problem_json_path, input_dir
    )

    test_results_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)

    if not test_results_path.exists():
        test_results = problem.copy()
        del test_results["completions"]
        test_results["results"] = []
    else:
        with open(test_results_path, "r") as f:
            test_results = json.load(f)

    num_problems = len(problem["completions"])

    if len(test_results["results"]) == num_problems:
        return
    elif len(test_results["results"]) > num_problems:
        print(f"ERROR more results than completions for {problem_json_path}")
        return

    min_problem = len(test_results["results"])

    # In case we have previously computed results, warm the cache with them
    for already_computed in test_results["results"]:
        CACHE[already_computed["program"]] = already_computed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for j in executor.map(
            lambda index: cached_eval_script(problem, index),
            range(min_problem, num_problems),
        ):
            test_results["results"].append(j)
            with open(test_results_path, "w") as f:
                f.write(json.dumps(test_results, indent=2))
