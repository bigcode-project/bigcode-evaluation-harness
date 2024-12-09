from argparse import Namespace
from typing import Any, Iterable, Optional

from datasets import Dataset
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from unitxt.inference import WMLInferenceEngineBase

from bigcode_eval.base import Task


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
    def get_stop_sequences_for_wx(args: Namespace, task: Task) -> Optional[list[str]]:
        # TODO
        stop_sequences = []

        if isinstance(task.stop_words, Iterable):
            stop_sequences.extend(task.stop_words)
        elif task.stop_words is not None:
            stop_sequences.append(task.stop_words)

        if args.instruction_tokens:
            for token in args.instruction_tokens.split(","):
                if token.strip() != "":
                    stop_sequences.append(token)

        return stop_sequences if stop_sequences else None

    @staticmethod
    def prepare_wx_gen_params(args: Namespace) -> dict[str, Any]:
        # TODO: need more generation params?
        # TODO: HF also uses more "sophisticated" stop criteria than stop words
        params = {
            "decoding_method": "sample" if args.do_sample else "greedy",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_length_generation,  # wx only has "max_new_tokens"
        }

        return params

    def initialize_wx_model(self, model_name: str) -> ModelInference:
        return ModelInference(
            model_id=model_name,
            api_client=self.client,
        )

    @staticmethod
    def prepare_inputs(dataset: Dataset, task: Task) -> list[str]:
        return [
            task.get_prompt(sample)
            for sample in dataset
        ]

    def infer(
        self,
        dataset: Dataset,
        task: Task,
        args: Namespace,
        postprocess: bool = True,
    ) -> list[str]:
        gen_params = self.prepare_wx_gen_params(args)
        # gen_params["stop_sequences"] = self.get_stop_sequences_for_wx(
        #     args, task
        # )

        model = self.initialize_wx_model(args.model)

        predictions = [
            result["results"][0]["generated_text"]
            for result in
            model.generate(
                prompt=self.prepare_inputs(dataset, task),
                params=gen_params,
            )
        ]

        if postprocess:
            predictions = self.postprocess_predictions(predictions, task)

        return predictions

    @staticmethod
    def postprocess_predictions(predictions: list[str], task: Task) -> list[str]:
        return [
            task.postprocess_generation(prediction, idx)
            for idx, prediction in enumerate(predictions)
        ]
