# SPDX-License-Identifier: Apache-2.0

import json
import math
import random
from fractions import Fraction
from types import SimpleNamespace

import pytest

from bigcode_eval import tasks
from bigcode_eval.evaluator import Evaluator
from bigcode_eval.tasks.custom_metrics.bayes_at_n import estimate_bayes_at_n
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval


@pytest.mark.parametrize(
    "correct, mean, variance",
    [
        ([0, 2], Fraction(1, 2), Fraction(3, 160)),
        ([1, 1], Fraction(1, 2), Fraction(1, 40)),
        ([0], Fraction(1, 4), Fraction(3, 80)),
        ([2], Fraction(3, 4), Fraction(3, 80)),
    ],
)
def test_independent_exact_moments(correct, mean, variance):
    assert estimate_bayes_at_n([2] * len(correct), correct) == pytest.approx(
        (float(mean), math.sqrt(variance))
    )


@pytest.mark.parametrize("correct,mean", [(0, 1 / 3), (1, 2 / 3)])
def test_one_trial_has_posterior_uncertainty(correct, mean):
    assert estimate_bayes_at_n([1], [correct]) == pytest.approx(
        (mean, math.sqrt(Fraction(1, 18)))
    )


def test_representative_workload_and_permutations():
    rng = random.Random(213)
    rows = [[rng.randrange(2) for _ in range(64)] for _ in range(164)]
    counts = [sum(row) for row in rows]
    exact_means = [Fraction(c + 1, 66) for c in counts]
    exact_variances = [Fraction((c + 1) * (65 - c), 66**2 * 67) for c in counts]
    actual = estimate_bayes_at_n([64] * 164, counts)
    assert actual == pytest.approx(
        (float(sum(exact_means) / 164), math.sqrt(sum(exact_variances)) / 164),
        rel=1e-14,
        abs=1e-15,
    )
    assert actual == estimate_bayes_at_n([64] * 164, counts[::-1])


@pytest.mark.parametrize(
    "total,correct",
    [
        ([], []),
        ([0], [0]),
        ([2, 3], [0, 1]),
        ([2], [-1]),
        ([2], [3]),
        ([2], [0.5]),
        ([2.0], [1]),
        ([2], [True]),
        ([2], []),
    ],
)
def test_invalid_counts(total, correct):
    with pytest.raises(ValueError, match="Bayes@N"):
        estimate_bayes_at_n(total, correct)


@pytest.mark.parametrize(
    "predictions,references,expected_n",
    [
        ([], [], None),
        ([[]], ["assert True"], None),
        ([["answer = 1"]], [], None),
        ([["answer = 1"], ["answer = 1", "answer = 0"]], ["", ""], None),
        ([["answer = 1"]], [""], 2),
        ([[None]], [""], None),
        (["answer = 1"], [""], None),
        ([["answer = 1"]], [None], None),
        ([["answer = 1"]], [""], True),
        ([["answer = 1"]], [""], 0),
    ],
)
def test_invalid_input_fails_before_program_execution(
    monkeypatch, predictions, references, expected_n
):
    import bigcode_eval.tasks.custom_metrics.code_eval as metric

    def unexpected_execution(*args):
        raise AssertionError("invalid inputs must not execute")

    monkeypatch.setattr(metric, "check_correctness", unexpected_execution)
    with pytest.raises(ValueError, match="Bayes@N"):
        compute_code_eval(
            predictions, references, bayes_at_n=True, expected_n=expected_n
        )


def test_trusted_programs_keep_question_verdicts(monkeypatch):
    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "1")
    predictions = [["answer = 0", "answer = 0"], ["answer = 1", "answer = 1"]]
    scores, details = compute_code_eval(
        predictions,
        ["assert answer == 1"] * 2,
        bayes_at_n=True,
        expected_n=2,
    )
    assert scores == pytest.approx(
        {
            "pass@1": 0.5,
            "bayes_at_n": 0.5,
            "bayes_at_n_std": math.sqrt(3 / 160),
        }
    )
    assert [[entry[1]["passed"] for entry in details[i]] for i in range(2)] == [
        [False, False],
        [True, True],
    ]


def test_duplicate_verdicts_rejected(monkeypatch):
    import bigcode_eval.tasks.custom_metrics.code_eval as metric

    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "1")
    monkeypatch.setattr(
        metric,
        "check_correctness",
        lambda program, timeout, task, completion: {
            "task_id": task,
            "completion_id": 0,
            "passed": True,
        },
    )
    with pytest.raises(ValueError, match="unique binary verdicts"):
        compute_code_eval([["", ""]], [""], bayes_at_n=True)


@pytest.fixture
def local_dataset(monkeypatch):
    import bigcode_eval.base as base

    doc = {
        "prompt": "answer = ",
        "text": "Return one.",
        "entry_point": "answer",
        "test": "def check(candidate):\n    assert candidate == 1",
        "test_list": ["assert answer == 1"],
    }
    monkeypatch.setattr(
        base, "load_dataset", lambda **kwargs: {"test": [dict(doc) for _ in range(500)]}
    )


def evaluator_args(tmp_path):
    path = tmp_path / "generations.json"
    path.write_text(
        json.dumps([["answer = 0", "answer = 0"], ["answer = 1", "answer = 1"]])
    )
    return SimpleNamespace(
        load_generations_path=str(path),
        n_samples=2,
        limit=2,
        limit_start=0,
        check_references=False,
        save_every_k_tasks=-1,
        save_generations_path=str(tmp_path / "unused.json"),
        save_generations=False,
        save_references=False,
        metric_output_path=str(tmp_path / "metrics.json"),
        allow_code_execution=True,
    )


@pytest.mark.parametrize(
    "name",
    [
        "humaneval-bayes-at-n",
        "humaneval-unstripped-bayes-at-n",
        "mbpp-bayes-at-n",
    ],
)
def test_registered_task_through_saved_generation_evaluator(
    tmp_path, local_dataset, name
):
    args = evaluator_args(tmp_path)
    assert name in tasks.ALL_TASKS
    task = tasks.get_task(name, args)
    assert task.bayes_at_n and task.n_samples == 2
    evaluator = Evaluator(SimpleNamespace(is_main_process=True), None, None, args)
    result = json.loads(json.dumps(evaluator.evaluate(name)))
    assert result == pytest.approx(
        {
            "pass@1": 0.5,
            "bayes_at_n": 0.5,
            "bayes_at_n_std": math.sqrt(3 / 160),
        }
    )


@pytest.mark.parametrize("name", ["humaneval", "humaneval-unstripped", "mbpp"])
def test_original_task_outputs_unchanged(tmp_path, local_dataset, name):
    args = evaluator_args(tmp_path)
    evaluator = Evaluator(SimpleNamespace(is_main_process=True), None, None, args)
    assert evaluator.evaluate(name) == {"pass@1": 0.5}


def test_requested_trial_count_is_enforced(tmp_path, local_dataset):
    args = evaluator_args(tmp_path)
    args.n_samples = 3
    evaluator = Evaluator(SimpleNamespace(is_main_process=True), None, None, args)
    with pytest.raises(ValueError, match="exactly expected_n"):
        evaluator.evaluate("humaneval-bayes-at-n")


def test_saved_generation_offset_rejected(tmp_path, local_dataset):
    args = evaluator_args(tmp_path)
    args.limit_start = 1
    evaluator = Evaluator(SimpleNamespace(is_main_process=True), None, None, args)
    with pytest.raises(ValueError, match="existing loader does not apply"):
        evaluator.evaluate("humaneval-bayes-at-n")


@pytest.mark.parametrize("checkpoint", [[], [None, ["answer = 1", "answer = 1"]]])
def test_intermediate_checkpoint_cannot_shift_question_rows(
    tmp_path, local_dataset, checkpoint
):
    args = evaluator_args(tmp_path)
    args.load_generations_path = None
    evaluator = Evaluator(SimpleNamespace(is_main_process=True), None, None, args)
    with pytest.raises(ValueError, match="does not support intermediate checkpoints"):
        evaluator.evaluate("humaneval-bayes-at-n", intermediate_generations=checkpoint)


@pytest.mark.parametrize("matrix", [[], [[]], ["answer = 1"], [["answer = 1"], None]])
def test_invalid_saved_matrix_has_clear_error(tmp_path, local_dataset, matrix):
    args = evaluator_args(tmp_path)
    (tmp_path / "generations.json").write_text(json.dumps(matrix))
    evaluator = Evaluator(SimpleNamespace(is_main_process=True), None, None, args)
    with pytest.raises(ValueError, match="nonempty program list"):
        evaluator.evaluate("humaneval-bayes-at-n")


def test_normal_cli_parser_and_result_file(tmp_path, local_dataset, monkeypatch):
    import main

    args = evaluator_args(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--tasks",
            "humaneval-bayes-at-n",
            "--n_samples",
            "2",
            "--load_generations_path",
            args.load_generations_path,
            "--limit",
            "2",
            "--allow_code_execution",
            "--metric_output_path",
            args.metric_output_path,
        ],
    )
    parsed = main.parse_args()
    assert parsed.tasks == "humaneval-bayes-at-n" and parsed.n_samples == 2
    monkeypatch.setattr(
        main, "Accelerator", lambda: SimpleNamespace(is_main_process=True)
    )
    main.main()
    result = json.loads((tmp_path / "metrics.json").read_text())
    assert result["humaneval-bayes-at-n"]["bayes_at_n_std"] == pytest.approx(
        math.sqrt(3 / 160)
    )
    assert result["humaneval-bayes-at-n"]["bayes_at_n"] == 0.5
