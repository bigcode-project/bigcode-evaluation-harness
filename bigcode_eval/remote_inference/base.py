from abc import abstractmethod
from argparse import Namespace
from typing import Any, Optional

from datasets import Dataset as HfDataset

from bigcode_eval.base import Task
from bigcode_eval.utils import _parse_instruction


Dataset = HfDataset | list[dict[str, Any]]


class RemoteInferenceInterface:
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def _prepare_generation_params(self, args: Namespace) -> dict[str, Any]:
        """Method maps HF generation parameters to platform-specific ones."""

        raise NotImplementedError

    @staticmethod
    def _limit_inputs(
        dataset: Dataset, limit: Optional[int], offset: Optional[int]
    ) -> Dataset:
        """Method limits input dataset based on provided `limit` and `limit_start` args."""

        is_hf = isinstance(dataset, HfDataset)

        if offset:
            dataset = (
                dataset.select(range(offset, len(dataset)))
                if is_hf
                else dataset[offset:]
            )

        if limit:
            dataset = (
                dataset.take(limit)
                if is_hf
                else dataset[:limit]
            )

        return dataset

    @staticmethod
    def _make_instruction_prompt(
        instruction: str,
        context: str,
        prefix: str,
        instruction_tokens: Optional[str],
    ) -> str:
        """Method creates a prompt for instruction-tuning based on a given prefix and instruction tokens."""

        user_token, end_token, assistant_token = "", "", "\n"
        if instruction_tokens:
            user_token, end_token, assistant_token = instruction_tokens.split(",")

        return "".join(
            (
                prefix,
                user_token,
                instruction,
                end_token,
                assistant_token,
                context,
            )
        )

    @staticmethod
    def _make_infill_prompt(prefix: str, content_prefix: str, content_suffix: str) -> str:
        """Method creates a prompt for infilling.
        As it depends on particular models, it may be necessary to implement the method separately for each platform.
        """

        return f"{prefix}{content_prefix}{content_suffix}"

    def _create_prompt_from_dict(
        self, content: dict[str, str], prefix: str, instruction_tokens: Optional[str]
    ) -> str:
        """Method prepares a prompt in similar way to the `TokenizedDataset` class for either instruction or infilling mode."""

        if all(key in ("instruction", "context") for key in content):
            return self._make_instruction_prompt(
                content["instruction"], content["context"], prefix, instruction_tokens
            )

        elif all(key in ("prefix", "suffix") for key in content):
            return self._make_infill_prompt(prefix, content["prefix"], content["suffix"])

        else:
            raise ValueError(f"Unsupported prompt format:\n{content}.")

    def _prepare_prompts(
        self,
        dataset: Dataset,
        task: Task,
        prefix: str,
        instruction_tokens: Optional[str],
    ) -> list[str]:
        """Method creates prompts for inputs based on the task prompt, prefix and instruction tokens (if applicable)."""

        is_string = isinstance(task.get_prompt(dataset[0]), str)

        return [
            prefix + task.get_prompt(instance)
            if is_string
            else self._create_prompt_from_dict(
                task.get_prompt(instance), prefix, instruction_tokens
            )
            for instance in dataset
        ]

    @abstractmethod
    def _infer(
        self, inputs: list[str], params: dict[str, Any], args: Namespace
    ) -> list[list[str]]:
        """Method responsible for inference on a given platform."""

        raise NotImplementedError

    @staticmethod
    def _postprocess_predictions(
        predictions: list[list[str]],
        prompts: list[str],
        task: Task,
        instruction_tokens: Optional[str],
    ) -> list[list[str]]:
        """Method postprocess model's predictions based on a given task and instruction tokens (if applicable)."""

        if instruction_tokens:
            predictions = [
                [_parse_instruction(prediction[0], instruction_tokens.split(","))]
                for prediction in predictions
            ]

        return [
            [
                task.postprocess_generation(
                    prompts[i] + predictions[i][0], i
                )
            ]
            for i in range(len(predictions))
        ]

    def prepare_generations(
        self,
        dataset: Dataset,
        task: Task,
        args: Namespace,
        prefix: str = "",
        postprocess: bool = True,
    ) -> list[list[str]]:
        """Method generates (and postprocess) code using given platform. It follows the same process as HF inference."""

        gen_params = self._prepare_generation_params(args)

        dataset = self._limit_inputs(dataset, args.limit, args.limit_start)

        prompts = self._prepare_prompts(
            dataset, task, prefix, args.instruction_tokens
        )

        predictions = self._infer(prompts, gen_params, args)

        if postprocess:
            return self._postprocess_predictions(
                predictions, prompts, task, args.instruction_tokens
            )

        return predictions
