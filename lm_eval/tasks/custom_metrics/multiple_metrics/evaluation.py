import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Optional

from containerized_eval import eval_string_script

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


def get_test_results_json_path(output_dir: Path, problem_json_path: Path) -> Path:
    suffixes = ".results.json"
    problem_name = problem_json_path.name[: -len(".json")]
    return output_dir / (problem_name + suffixes)


def evaluate_problem(output_dir: Path, problem_json_path: Path, max_workers: int):
    with open(problem_json_path, "r") as f:
        problem = json.load(f)

    test_results_path = get_test_results_json_path(output_dir, problem_json_path)

    test_results = problem.copy()
    del test_results["completions"]
    test_results["results"] = []
    num_problems = len(problem["completions"])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for j in executor.map(
            lambda index: cached_eval_script(problem, index),
            range(num_problems),
        ):
            test_results["results"].append(j)
            with open(test_results_path, "r") as f:
                f.write(json.dumps(test_results, indent=2))
