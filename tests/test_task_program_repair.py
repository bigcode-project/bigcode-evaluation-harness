import pytest
from typing import List
from lm_eval.tasks.program_repair import ProgramRepair, EvaluatedMetric


@pytest.fixture
def references() -> List[str]:
    return [
        "first",
        "<|endoftext|>",
        "<|endoftext|>",
        "<|endoftext|>",
        "",
        "",
        "last",
        "last",
    ]


@pytest.fixture
def generations() -> List[List[str]]:
    return [
        ["1st", "irst", "", "not-first"],                   # Does not include an exact match
        ["<|endoftext|>", "", "<special_token>", "first"],  # Includes an exact match
        ["<|endoftext|>", "", "first", "<special_token>"],  # Includes an exact match
        ["", "", "", "<special_token>"],                    # Does not include an exact match
        ["", "", "", "<|endoftext|>"],                      # Includes an exact match
        ["<>", " ", "   ", "<|endoftext|>"],                # Does not include an exact match
        ["last", "", "", ""],                               # Includes an exact match
        ["", "", "", "last"],                               # Includes an exact match
    ]


@pytest.fixture
def expected_score() -> float:
    return 0.625                                            # 6 out of 8 references have an exact match


def test_process_results(references, generations, expected_score):
    evaluated_metric: EvaluatedMetric = ProgramRepair().process_results(generations, references)
    assert evaluated_metric['avg_exact_match'] == expected_score
