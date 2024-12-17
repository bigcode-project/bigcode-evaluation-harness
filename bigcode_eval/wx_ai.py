from argparse import Namespace
from typing import Any, Optional

from datasets import Dataset
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from unitxt.inference import WMLInferenceEngineBase

from bigcode_eval.base import Task
from bigcode_eval.utils import _parse_instruction


class WxInference:
    def __init__(self):
        wx_creds = WMLInferenceEngineBase._read_wml_credentials_from_env()
        WMLInferenceEngineBase._verify_wml_credentials(wx_creds)

        self.client = APIClient(
            project_id=wx_creds.pop("project_id", None),
            space_id=wx_creds.pop("space_id", None),
            credentials=wx_creds,
        )

    @staticmethod
    def prepare_wx_gen_params(args: Namespace) -> dict[str, Any]:
        return {
            "decoding_method": "sample" if args.do_sample else "greedy",
            "random_seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "length_penalty": args.length_penalty,
            "stop_sequences": args.stop_sequences,
        }

    def initialize_wx_model(self, model_name: str) -> ModelInference:
        return ModelInference(
            model_id=model_name,
            api_client=self.client,
        )

    @staticmethod
    def create_prompt_from_dict(
        prompt_content: dict[str, str],
        prefix: str,
        instruction_tokens: Optional[list[str]],
    ) -> str:
        if all(key in ("instruction", "context") for key in prompt_content):
            user_token, end_token, assistant_token = "", "", "\n"
            if instruction_tokens:
                user_token, end_token, assistant_token = instruction_tokens

            return "".join(
                (
                    prefix,
                    user_token,
                    prompt_content["instruction"],
                    end_token,
                    assistant_token,
                    prompt_content["context"],
                )
            )

        # Infilling mode is only used for particular models, none of which is available in wx.
        elif all(key in ("prefix", "suffix") for key in prompt_content):
            return f"{prefix}{prompt_content['prefix']}{prompt_content['suffix']}"

        else:
            raise ValueError(
                f"Unsupported prompt format: '{prompt_content}'."
            )

    def prepare_inputs(
        self,
        dataset: Dataset,
        task: Task,
        prefix: str,
        instruction_tokens: Optional[list[str]],
    ) -> list[str]:
        sample = task.get_prompt(dataset[0])

        assert isinstance(sample, (str, dict)), (
            f"Prompt must be either a string, or a dictionary. "
            f"However, '{sample}' was given, which is "
            f"of type: '{type(sample)}'."
        )

        return [
            prefix + task.get_prompt(instance)
            if isinstance(sample, str)
            else self.create_prompt_from_dict(
                task.get_prompt(instance), prefix, instruction_tokens
            )
            for instance in dataset
        ]

    def infer(
        self,
        dataset: Dataset,
        task: Task,
        args: Namespace,
        prefix: str = "",
        postprocess: bool = True,
    ) -> list[list[str]]:
        gen_params = self.prepare_wx_gen_params(args)

        model = self.initialize_wx_model(args.model)

        prompts = self.prepare_inputs(
            dataset, task, prefix, args.instruction_tokens
        )

        predictions = [
            [result["results"][0]["generated_text"]]
            for result in
            model.generate(
                prompt=prompts,
                params=gen_params,
            )
        ]

        if postprocess:
            predictions = self.postprocess_predictions(
                predictions,
                prompts,
                task,
                args.instruction_tokens,
            )

        return predictions

    @staticmethod
    def postprocess_predictions(
        predictions: list[list[str]],
        prompts: list[str],
        task: Task,
        instruction_tokens: Optional[list[str]],
    ) -> list[list[str]]:
        if instruction_tokens:
            predictions = [
                _parse_instruction(prediction, instruction_tokens)
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
