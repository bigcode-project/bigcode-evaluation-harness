import os
import openai
import jsonlines
import termcolor

from cdifflib import CSequenceMatcher
from camel_converter import to_snake
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


class ContentParser:

    @staticmethod
    def _entry_point_variations(entry_point: str) -> List[str]:
        # NOTE: workaround dataset's bug with entry point naming
        return [
            entry_point,
            to_snake(entry_point),
            entry_point[0].lower() + entry_point[1:],
        ]

    def __call__(self, prompt: str, content: str, entry_point: str):
        # NOTE: Model doesn't follow instructions directly:
        # adds description of change and sometimes fixes
        # typos, or other "bugs" in description.
        if "```" in content:
            content = content.split("```")[1]
        # first parse with assumption that content has description
        matcher = CSequenceMatcher(None, prompt, content)
        tag, _, _, j1, j2 = matcher.get_opcodes()[-1]
        if tag == "insert":
            return content[j1:j2]
        # second parse content with assumption that model wrote code without description
        for entry_point in self._entry_point_variations(entry_point):
            if entry_point in content:
                content = content.split(entry_point)[-1]
                return "".join(content.splitlines(keepends=True)[1:])
        raise ParseError(f"prompt is not in content:\n{content}")


class ChatWrapper:

    def __init__(self,
                 model: str,
                 messages: List[Dict[str, str]]):
        self._model = model
        self._messages = messages

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
                response = openai.ChatCompletion.create(
                    model=self._model,
                    messages=messages)
                message = response["choices"][0]["message"]
                assert message["role"] == "assistant"
                return message["content"]
            except Exception as e:
                print("API EXCEPTION:", e)


if __name__ == '__main__':
    TIMES = 1
    VERBOSE = True
    LANGUAGE = "python"

    openai.organization = os.getenv("OPENAI_ORGANIZATION")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    samples = [s for s in load_dataset("bigcode/humaneval-x-bugs", LANGUAGE)["test"]] * TIMES

    chat_wrapper = ChatWrapper("gpt-3.5-turbo", messages)
    parse_errors = 0
    parser = ContentParser()
    for idx, sample in enumerate(samples):
        prompt = sample["prompt"] if LANGUAGE not in ["rust"] else sample["prompt"] + sample["declaration"]
        if VERBOSE:
            print(f"Processing {sample['task_id']} ({idx + 1}/{len(samples)}))...")
            print(termcolor.colored(sample["entry_point"], "yellow", attrs=["bold"]))
            print(termcolor.colored(prompt, "yellow"))
            print(termcolor.colored(sample["buggy_solution"], "red"))
        sample["raw_generation"] = chat_wrapper(prompt, sample["buggy_solution"])
        try:
            sample["generation"] = parser(prompt, sample["raw_generation"], sample["entry_point"])
        except ParseError as e:
            parse_errors += 1
            print("PARSE EXCEPTION:", e)
            sample["generation"] = ""
        if VERBOSE:
            print(termcolor.colored(sample["generation"], "green"))
    if VERBOSE:
        print("parse error rate:", parse_errors / len(samples))

    results_filename = f"completions_{LANGUAGE}.jsonl"
    with jsonlines.open(results_filename, "w") as writer:
        writer.write_all(samples)
