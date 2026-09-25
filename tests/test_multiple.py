import json
import os
import tempfile

import pytest

from bigcode_eval.tasks import multiple
from bigcode_eval.tasks.custom_metrics.multiple_metrics.evaluation import \
    get_test_results_json_path

NAMES = [
    "HumanEval_0_has_close_elements",
    "HumanEval_1_separate_paren_groups",
    "HumanEval_2_truncate_number",
]


class DummyMultiPLE(multiple.GeneralMultiPLE):
    # skips loading the dataset from the hub
    def __init__(self, language, names):
        self.language = language
        self.docs = [
            {"prompt": f"prompt {name}", "name": name, "language": language}
            for name in names
        ]

    def get_dataset(self):
        return self.docs


def fake_evaluate_problem(output_dir, problem_json_path, max_workers):
    # writes a results file in the same format as the real evaluate_problem,
    # a completion passes when it ends with "pass"
    with open(problem_json_path) as f:
        problem = json.load(f)
    results = []
    for completion in problem["completions"]:
        ok = completion.endswith("pass")
        status = "OK" if ok else "Exception"
        results.append({"status": status, "exit_code": 0 if ok else 1})
    test_results = {"name": problem["name"], "results": results}
    path = get_test_results_json_path(output_dir, problem_json_path, None)
    with open(path, "w") as f:
        json.dump(test_results, f)


@pytest.fixture
def system_tmp(tmp_path, monkeypatch):
    # stand-in for the system temp dir, shared by every run on the machine
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(multiple, "evaluate_problem", fake_evaluate_problem)
    return tmp_path


def write_leftover(directory, name, n_ok):
    results = [{"status": "OK", "exit_code": 0}] * n_ok
    with open(os.path.join(directory, f"{name}.results.json"), "w") as f:
        json.dump({"name": name, "results": results}, f)


def run(task, generations):
    references = ["tests"] * len(generations)
    return task.process_results(generations, references)


def test_single_run_mixed(system_tmp):
    task = DummyMultiPLE("js", NAMES[:2])
    assert run(task, [["fail"], ["pass"]]) == {"pass@1": 0.5}


def test_single_run_all_pass_and_all_fail(system_tmp):
    task = DummyMultiPLE("js", NAMES)
    assert run(task, [["pass"]] * 3) == {"pass@1": 1.0}
    assert run(task, [["fail"]] * 3) == {"pass@1": 0.0}


def test_several_samples_per_problem(system_tmp):
    task = DummyMultiPLE("js", NAMES[:2])
    generations = [["pass"] * 3 + ["fail"] * 7, ["fail"] * 10]
    results = run(task, generations)
    assert set(results) == {"pass@1", "pass@10"}
    assert results["pass@1"] == pytest.approx(0.15)
    assert results["pass@10"] == pytest.approx(0.5)


def test_leftover_results_file_is_not_scored(system_tmp):
    write_leftover(system_tmp, "HumanEval_99_leftover", n_ok=20)
    task = DummyMultiPLE("js", NAMES[:1])
    assert run(task, [["fail"]]) == {"pass@1": 0.0}


def test_leftover_results_file_is_left_alone(system_tmp):
    write_leftover(system_tmp, "HumanEval_99_leftover", n_ok=20)
    task = DummyMultiPLE("js", NAMES[:1])
    run(task, [["fail"]])
    assert os.path.exists(
        os.path.join(system_tmp, "HumanEval_99_leftover.results.json")
    )


def test_previous_task_in_same_process_is_not_scored(system_tmp):
    # problem names are shared across languages, js has more problems than d
    js = DummyMultiPLE("js", NAMES)
    d = DummyMultiPLE("d", NAMES[:2])
    assert run(js, [["pass"]] * 3) == {"pass@1": 1.0}
    assert run(d, [["fail"]] * 2) == {"pass@1": 0.0}
