import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import openai
from lm_eval import tasks
from dataclasses import dataclass, asdict


Generations = List[List[str]]
References = List[str]


@dataclass(frozen=True)
class OpenAIModelArguments:
    """
    Arguments for generating text using the OpenAI API.
    """

    model: str = "text-davinci-003"
    max_tokens: int = 512
    temperature: float = 0.7


@dataclass(frozen=True)
class TaskArguments:
    """
    Arguments for selecting tasks.
    :param task_name: The name of the task to generate text for.
    :param num_of_predictions: The number of predictions to generate for each sample.
    :param num_of_inputs: The number of samples to generate text for. If num_of_inputs is 0, all samples are used.
    :param store_path: The path to save the generated text to. By default, the text is saved to the current directory.
                       To disable saving, set `store_path=''` (an empty string).
    """

    task_name: str = "program_repair"
    num_of_predictions: int = 1
    num_of_inputs: int = 1
    store_path: str = "."


class OpenAIGenerator:
    """
    A class to generate text using the OpenAI API. The generations could be stored in a file.
    """

    def __init__(self, api_key: str):
        openai.api_key = api_key

    @staticmethod
    def generate_text(
        openai_model_args: OpenAIModelArguments, task_args: TaskArguments
    ) -> Tuple[Generations, References]:
        """
        An alternative to Evaluator.generate_text. Evaluator assumes to be given an Accelerator object and generates
        text in parallel. In contrast, this function generates text using the OpenAI API.
        :return: generations, references
        """

        def get_predictions_from_response() -> List[str]:
            """
            Get all generated predictions from the response.
            :return: A list of predictions.
            """
            nonlocal response
            return [prediction["text"] for prediction in response["choices"]]

        task = tasks.get_task(task_args.task_name)
        dataset = task.get_dataset()
        num_of_inputs = (
            task_args.num_of_inputs if task_args.num_of_inputs > 0 else len(dataset)
        )
        stop: Optional[List[str]] = task.stop_words if task.stop_words else None
        generations = []
        for i in range(num_of_inputs):
            prompt: str = task.get_prompt(dataset[i])
            response = openai.Completion.create(
                model=openai_model_args.model,
                prompt=prompt,
                max_tokens=openai_model_args.max_tokens,
                temperature=openai_model_args.temperature,
                n=task_args.num_of_predictions,
                stop=stop,
            )
            predictions = get_predictions_from_response()
            generations.append(predictions)
        references = [task.get_reference(dataset[i]) for i in range(num_of_inputs)]
        return generations, references


def parse_args() -> argparse.Namespace:
    """
    For each argument of the dataclasses, add an argument to the parser. The name of the argument is the name of the
    field in the dataclass. The type of the argument is the type of the field in the dataclass. The default value of
    the argument is the default value of the field in the dataclass.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    for dc in [OpenAIModelArguments, TaskArguments]:
        for key, value in asdict(dc()).items():
            parser.add_argument(f"--{key}", type=type(value), default=value)
    return parser.parse_args()


def init_data_class_from_kwargs(cls, kwargs):
    """
    Initialize a dataclass from a dictionary of keyword arguments.
    :param cls: The dataclass to initialize.
    :param kwargs: A dictionary of keyword arguments.
    :return: An instance of the dataclass.
    """
    return cls(
        **{key: value for key, value in kwargs.items() if key in asdict(cls()).keys()}
    )


def generate_from_openai() -> Tuple[Generations, References]:
    """
    A function to generate text using the OpenAI API. The generations could be stored in a file.
    To execute from the command line, use:
        python -m lm_eval.generate_from_saas_apis --api_key=<your_api_key>
    To edit the default arguments, use:
        python -m lm_eval.generate_from_saas_apis --api_key=<your_api_key> --<parameter>=<value>
    For example:
        python -m lm_eval.generate_from_saas_apis --api_key=<your_api_key> --num_of_predictions=100 --num_of_inputs=10 --max_tokens=1024 --temperature=0.9
    """
    args: argparse.Namespace = parse_args()
    openai_model_args: OpenAIModelArguments = init_data_class_from_kwargs(
        OpenAIModelArguments, vars(args)
    )
    task_args: TaskArguments = init_data_class_from_kwargs(TaskArguments, vars(args))
    api_key: str = args.api_key
    generator: OpenAIGenerator = OpenAIGenerator(api_key=api_key)
    generations: Generations
    references: References
    generations, references = generator.generate_text(
        openai_model_args=openai_model_args, task_args=task_args
    )
    if task_args.store_path:
        dirpath: Path = Path(task_args.store_path) / "outputs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        dirpath.mkdir(parents=True, exist_ok=False)
        filepath_g: Path = dirpath / "generations.json"
        filepath_r: Path = dirpath / "references.json"
        with open(filepath_g, "w") as f:
            json.dump(generations, f)
        with open(filepath_r, "w") as f:
            json.dump(references, f)
    return generations, references


if __name__ == "__main__":
    generate_from_openai()
