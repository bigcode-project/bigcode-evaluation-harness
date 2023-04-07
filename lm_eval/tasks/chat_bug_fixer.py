import os
import openai
import jsonlines
import termcolor

from cdifflib import CSequenceMatcher
from datasets import load_dataset
from typing import List, Dict


messages = [
    {
        "role": "system",
        "content": "You are an AI programming assistant. "
                   "Follow the user's requirements carefully & to the letter."
    },
    {
        "role": "user",
        "content": "I'll provide you code snippet with bug in code. Your should fix it. "
                   "Make sure you leave function signature and it's description unchanged: "
                   "no additional symbols, rephrasing, changing format pr fixing typos. "
                   "Output format should contains only code text without any explanations.",
    },
    {
        "role": "assistant",
        "content": "Yes, my assignment is clear. Please send me a code.",
    },
]


class ParseError(Exception):
    pass


class ChatWrapper:

    def __init__(self,
                 model: str,
                 messages: List[Dict[str, str]],
                 stop_on_parse_error: bool = False):
        self._model = model
        self._messages = messages
        self._stop_on_parse_error = stop_on_parse_error
        self._parse_errors = 0
        self._query_counter = 0

    @staticmethod
    def _parse_content(prompt: str, content: str):
        # NOTE: Model doesn't follow instructions directly:
        # adds description of change and sometimes fixes
        # typos, or other "bugs" in description.
        if "```" in content:
            content = content.split("```")[1]
        matcher = CSequenceMatcher(None, prompt, content)
        tag, _, _, j1, j2 = matcher.get_opcodes()[-1]
        if tag != "insert":
            raise ParseError(f"prompt not found in content:\n{content}")
        return content[j1:j2]

    @property
    def parse_error_rate(self) -> float:
        return self._parse_errors / self._query_counter

    def __call__(self, prompt: str, solution: str) -> str:
        messages = [
            *self._messages,
            {
                "role": "user",
                "content": prompt + solution,
            }
        ]
        while True:
            try:
                self._query_counter += 1
                response = openai.ChatCompletion.create(
                    model=self._model,
                    messages=messages)
                message = response["choices"][0]["message"]
                assert message["role"] == "assistant"
                return self._parse_content(prompt, message["content"])
            except ParseError as e:
                self._parse_errors += 1
                print("PARSE EXCEPTION:", e)
                if self._stop_on_parse_error:
                    return ""
            except Exception as e:
                print("API EXCEPTION:", e)


if __name__ == '__main__':
    TIMES = 1
    VERBOSE = True

    openai.organization = os.getenv("OPENAI_ORGANIZATION")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    samples = [s for s in load_dataset("bigcode/humaneval-x-bugs", "python")["test"]] * TIMES

    chat_wrapper = ChatWrapper("gpt-3.5-turbo", messages)
    for idx, sample in enumerate(samples):
        if VERBOSE:
            print(f"Processing {sample['task_id']} ({idx + 1}/{len(samples)}))...")
            print(termcolor.colored(sample["prompt"], "yellow"))
            print(termcolor.colored(sample["buggy_solution"], "red"))
        sample["completion"] = chat_wrapper(sample["prompt"], sample["buggy_solution"])
        if VERBOSE:
            print(termcolor.colored(sample["completion"], "green"))

    results_filename = "completions.jsonl"
    with jsonlines.open(results_filename, "w") as writer:
        # NOTE: compatibility with humaneval format
        for sample in samples:
            sample["task_id"] = sample["task_id"].replace("Python", "HumanEval")
        writer.write_all(samples)

    # NOTE: sample code to run evaluation
    import subprocess
    res = subprocess.check_output(
        f"evaluate_functional_correctness {results_filename}",
        shell=True, stderr=subprocess.DEVNULL)
    print("parse error rate:", chat_wrapper.parse_error_rate)
    print(res.decode("utf8"))
