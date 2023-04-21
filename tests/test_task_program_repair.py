import pytest
from typing import List, Dict, Iterable, Set
from lm_eval.tasks.program_repair import ProgramRepair, EvaluatedMetric, PyPiBugsDatasetFeaturesNames, SpecialTokens
from dataclasses import asdict
import re


@pytest.fixture
def prompt() -> str:
    dataset_features_names: PyPiBugsDatasetFeaturesNames = PyPiBugsDatasetFeaturesNames()
    doc: Dict[str, str] = {
        dataset_features_names.PATH: "tests/test_task_program_repair.py",
        dataset_features_names.INITIAL_STATE: "This is the initial state.",
        dataset_features_names.FINAL_STATE: "This is the final state.",
    }
    return ProgramRepair().get_prompt(doc)


@pytest.fixture
def new_special_tokens() -> Iterable[str]:
    return asdict(SpecialTokens()).values()


def test_get_prompt(prompt, new_special_tokens):
    print('***************')
    print('Prompt:')
    print(prompt)
    print('***************')

    def get_lines_containing_substrings_not_at_the_beginning(string: str, substrings: Iterable[str]) -> Set[str]:
        """
        Returns a set of all the lines that contain at least one of the substrings such that the substring is not at
        the beginning of the line.
        """
        ret: List[str] = []
        for substring in substrings:
            match = re.search(rf"^.+{substring}", string, flags=re.MULTILINE)
            if match:
                ret.append(match.group())
        return set(ret)

    bad_lines: Set[str] = get_lines_containing_substrings_not_at_the_beginning(prompt, new_special_tokens)
    assert len(bad_lines) == 0, f"Found bad lines: {bad_lines}"


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
        ["<>", " ", "   ", "<|endoftext|>"],                # Does not include an exact match (surrounding whitespace)
        ["last", "", "", ""],                               # Includes an exact match
        ["", "", "", "last"],                               # Includes an exact match
    ]


@pytest.fixture
def expected_score() -> float:
    return 0.625                                            # 5 out of 8 references have an exact match


def test_process_results(references, generations, expected_score):
    evaluated_metric: EvaluatedMetric = ProgramRepair().process_results(
        generations, references, to_strip_surrounding_whitespaces=False
    )
    assert evaluated_metric["avg_exact_match"] == expected_score
    # If we strip the surrounding whitespaces, then the score should be 0.75 (6 out of 8 references have an exact match)
    evaluated_metric: EvaluatedMetric = ProgramRepair().process_results(
        generations, references
    )
    assert evaluated_metric["avg_exact_match"] == 0.75
