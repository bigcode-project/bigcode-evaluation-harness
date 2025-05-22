import json
import os
from argparse import Namespace
from datetime import datetime
from typing import Any, Literal, Optional, TypedDict

from datasets import Dataset as HfDataset
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.experiment import TuneExperiment
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import PeftParameters
from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.helpers.connections.base_connection import BaseConnection
from ibm_watsonx_ai.wml_client_error import ApiRequestFailure, WMLClientError
from unitxt.inference import WMLInferenceEngineBase

from bigcode_eval.base import Task
from bigcode_eval.utils import _parse_instruction


class FineTuningResults(TypedDict):
    tuned_model_id: str
    deployed_model_id: str
    base_model_id: str
    metrics: dict[str, Any]
    creation_time: str
    start_time: str
    finish_time: str


Dataset = HfDataset | list[dict[str, Any]]
TuneMethod = Literal["lora", "qlora", "full"]


class WxInference:
    def __init__(self):
        wx_creds = WMLInferenceEngineBase._read_wml_credentials_from_env()
        WMLInferenceEngineBase._verify_wml_credentials(wx_creds)

        self.client = APIClient(
            project_id=wx_creds.get("project_id"),
            space_id=wx_creds.get("space_id"),
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
            "repetition_penalty": args.repetition_penalty,
        }

    def initialize_wx_model(
        self,
        model_name: str,
        deployed_model_id: Optional[str],
    ) -> ModelInference:
        if deployed_model_id:
            model_name = None

        return ModelInference(
            model_id=model_name,
            deployment_id=deployed_model_id,
            api_client=self.client,
        )

    @staticmethod
    def create_prompt_from_dict(
        prompt_content: dict[str, str],
        prefix: str,
        instruction_tokens: Optional[str],
    ) -> str:
        if all(key in ("instruction", "context") for key in prompt_content):
            user_token, end_token, assistant_token = "", "", "\n"
            if instruction_tokens:
                user_token, end_token, assistant_token = instruction_tokens.split(",")

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

    @staticmethod
    def limit_inputs(
        dataset: Dataset,
        args: Namespace,
    ) -> Dataset:
        limit, offset = args.limit, args.limit_start

        if isinstance(offset, int) and offset > 0:
            dataset = (
                dataset.select(range(offset, len(dataset)))
                if isinstance(dataset, HfDataset)
                else dataset[offset:]
            )

        if isinstance(limit, int) and limit > 0:
            dataset = (
                dataset.take(limit)
                if isinstance(dataset, HfDataset)
                else dataset[:limit]
            )

        return dataset

    def prepare_inputs(
        self,
        dataset: Dataset,
        task: Task,
        prefix: str,
        instruction_tokens: Optional[str],
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
        deployed_model_id: Optional[str] = None,
    ) -> list[list[str]]:
        gen_params = self.prepare_wx_gen_params(args)

        model = self.initialize_wx_model(args.model, deployed_model_id)

        dataset = self.limit_inputs(dataset, args)

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
        instruction_tokens: Optional[str],
    ) -> list[list[str]]:
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


class WxFineTuning:
    def __init__(self, tune_method: Optional[TuneMethod] = None):
        wx_creds = WMLInferenceEngineBase._read_wml_credentials_from_env()
        WMLInferenceEngineBase._verify_wml_credentials(wx_creds)

        self.experiment = TuneExperiment(
            credentials=wx_creds,
            project_id=wx_creds.get("project_id"),
            space_id=wx_creds.get("space_id"),
        )

        self.client = APIClient(
            project_id=wx_creds.get("project_id"),
            space_id=wx_creds.get("space_id"),
            credentials=wx_creds,
        )

        self.tune_method = tune_method or "full"

        self.__fine_tuner = None

        self._assets_ids: set[str] = set()
        self._deployments_ids: set[str] = set()

    @staticmethod
    def get_general_tune_parameters(model_id: str, tune_params: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": "classification",  # this parameter doesn't matter but it's required anyway
            "name": tune_params.get("remote_tune_name"),
            "base_model": model_id,
            "num_epochs": tune_params.get("num_epochs"),
            "learning_rate": tune_params.get("learning_rate"),
            "batch_size": tune_params.get("batch_size"),
            "max_seq_length": tune_params.get("max_seq_length"),
            "accumulate_steps": tune_params.get("accumulate_steps"),
            "verbalizer": tune_params.get("verbalizer"),
            "response_template": tune_params.get("response_template"),
            "auto_update_model": tune_params.get("auto_update_model", True),
            "group_by_name": tune_params.get("group_by_name", False),
            "gpu": tune_params.get(
                "gpu", {"num": 1}
            ),
            "gradient_checkpointing": tune_params.get(
                "gradient_checkpointing", False
            ),
        }

    @staticmethod
    def get_peft_tune_parameters(
        tune_method: TuneMethod,
        tune_params: dict[str, Any],
    ) -> PeftParameters:
        return PeftParameters(
            type=tune_method.lower(),
            rank=tune_params.get("lora_r"),
            lora_alpha=tune_params.get("lora_alpha"),
            target_modules=tune_params.get("lora_target_modules"),
            lora_dropout=tune_params.get("lora_dropout"),
        )

    def prepare_tune_parameters(
        self,
        base_model_id: str,
        raw_tune_params: dict[str, Any],
    ) -> dict[str, Any]:
        tune_params = self.get_general_tune_parameters(
            base_model_id, raw_tune_params
        )

        if self.tune_method in ("lora", "qlora"):
            tune_params["peft_parameters"] = self.get_peft_tune_parameters(
                self.tune_method, raw_tune_params
            )

        return tune_params

    def fine_tune(
        self,
        tune_params: dict[str, Any],
        data_references: list[DataConnection],
    ) -> dict[str, Any]:
        self.__fine_tuner = self.experiment.fine_tuner(**tune_params)

        self.__fine_tuner.run(
            training_data_references=data_references,
            background_mode=False,
        )

        run_details = self.__fine_tuner.get_run_details(include_metrics=True)

        self._assets_ids.add(run_details["entity"]["tuned_model"]["id"])

        return run_details

    def store_base_model(self, base_model_id: str) -> str:
        software_spec_id = self.client.software_specifications.get_id_by_name(
            "watsonx-cfm-caikit-1.1"
        )

        metadata = {
            "name": "Base FT model",
            "type": "base_foundation_model_1.0",
            "software_spec": software_spec_id,
        }

        stored_base_model_details = self.client.repository.store_model(
            model=base_model_id, meta_props=metadata
        )

        stored_base_model_id = self.client.repository.get_model_id(
            stored_base_model_details
        )

        self._assets_ids.add(stored_base_model_id)

        return stored_base_model_id

    def deploy_lora_adapter(
        self,
        tuned_model_id: str,
        base_model_asset_id: str,
        base_model_deployment_meta_properties: dict[str, Any],
    ) -> str:
        base_model_deployment_meta_properties["online"] = {
            "parameters": {
                "foundation_model": {
                    "enable_lora": True,
                },
            },
        }

        base_model_deployment_id = self.client.deployments.create(
            base_model_asset_id, base_model_deployment_meta_properties
        )["metadata"]["id"]

        self._deployments_ids.add(base_model_deployment_id)

        lora_adapter_deployment_meta_properties = {
            "name": f"bigcode-lora-adapter-deployment_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
            "base_deployment_id": base_model_deployment_id,
            "online": {},
        }

        lora_adapter_deployment_id = self.client.deployments.create(
            tuned_model_id, lora_adapter_deployment_meta_properties
        )["metadata"]["id"]

        self._deployments_ids.add(lora_adapter_deployment_id)

        return lora_adapter_deployment_id

    def deploy_tuned_model(self, tuned_model_id: str, base_model_id: str) -> str:
        meta_properties = {
            "name": f"bigcode-ft-deployment_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
            "online": {},
        }

        if self.tune_method in ("lora", "qlora"):
            stored_base_model_asset_id = self.store_base_model(base_model_id)

            deployment_id = self.deploy_lora_adapter(
                tuned_model_id=tuned_model_id,
                base_model_asset_id=stored_base_model_asset_id,
                base_model_deployment_meta_properties=meta_properties.copy(),
            )

        else:
            deployment_id = self.client.deployments.create(
                tuned_model_id, meta_properties
            )["metadata"]["id"]

        self._deployments_ids.add(deployment_id)

        return deployment_id

    def prepare_training_references(
        self,
        name: str,
        input_data: list[str],
        output_data: list[str],
    ) -> [DataConnection]:
        name = name.split('/')[-1]
        path = f"{os.getcwd()}/ft_temp_{name}.json"

        data = [
            {"input": input_sample, "output": output_sample}
            for input_sample, output_sample in zip(input_data, output_data)
        ]

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file)

        asset_details = self.client.data_assets.create(
            name=name,
            file_path=path,
        )

        os.remove(path)

        asset_id = self.client.data_assets.get_id(asset_details)

        self._assets_ids.add(asset_id)

        data_connection = DataConnection(
            data_asset_id=asset_id,
            connection=BaseConnection(),
        )

        return [data_connection]

    def run(
        self,
        dataset: Dataset,
        task: Task,
        raw_tune_params: dict[str, Any],
        base_model_id: str,
    ) -> FineTuningResults:
        tune_params = self.prepare_tune_parameters(
            base_model_id, raw_tune_params
        )

        # TODO: currently only supported for MBPP
        data_references = self.prepare_training_references(
            name=task.DATASET_NAME or task.DATASET_PATH,
            input_data=[task.get_prompt(sample) for sample in dataset],
            output_data=[sample["code"] for sample in dataset],
        )

        tuning_details = self.fine_tune(tune_params, data_references)

        tuned_model_id = tuning_details["entity"]["tuned_model"]["id"]

        deployment_id = self.deploy_tuned_model(
            tuned_model_id=tuned_model_id,
            base_model_id=base_model_id,
        )

        return {
            "tuned_model_id": tuned_model_id,
            "deployed_model_id": deployment_id,
            "base_model_id": base_model_id,
            "metrics": tuning_details["entity"]["status"]["metrics"],
            "creation_time": tuning_details["metadata"]["created_at"],
            "start_time": tuning_details["entity"]["status"]["running_at"],
            "finish_time": tuning_details["entity"]["status"]["completed_at"],
        }

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if self.__fine_tuner is not None:
            try:
                self.__fine_tuner.cancel_run(hard_delete=True)
            except (ApiRequestFailure, WMLClientError) as e:
                print(
                    f"Encountered an exception during fine tuning run deleting:"
                    f"\n{str(e)}"
                )

        for deployment_id in self._deployments_ids:
            try:
                self.client.deployments.delete(deployment_id)
            except (ApiRequestFailure, WMLClientError) as e:
                print(
                    f"Encountered an exception during deployments cleanup:"
                    f"\n{str(e)}"
                )

        for asset_id in self._assets_ids:
            try:
                self.client.data_assets.delete(asset_id)
            except (ApiRequestFailure, WMLClientError) as e:
                print(
                    f"Encountered an exception during assets cleanup:"
                    f"\n{str(e)}"
                )

        return False


def wx_inference(dataset: Dataset, task: Task, args: Namespace) -> list[list[str]]:
    deployment_id, tune_method = None, None
    fine_tuning_params_file_path = args.fine_tuning_params

    if fine_tuning_params_file_path:
        with open(fine_tuning_params_file_path, "r", encoding="utf-8") as file:
            tune_params = json.load(file)

        try:
            tune_method = tune_params.pop("tune_method").lower()
        except KeyError:
            raise ValueError(
                "You must specify the 'tune_method' in your fine tuning parameters."
            )

    with WxFineTuning(tune_method=tune_method) as fine_tuning_engine:
        if tune_method:
            training_data: Dataset = task.get_train_data()
            ft_details = fine_tuning_engine.run(
                training_data, task, tune_params, args.model
            )
            deployment_id = ft_details["deployed_model_id"]

        inference_engine = WxInference()
        results = inference_engine.infer(
            dataset=dataset, task=task, args=args, deployed_model_id=deployment_id
        )

    return results
