import logging
import os
from argparse import Namespace
from typing import Any

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference

from bigcode_eval.remote_inference.base import RemoteInferenceInterface


class WxInference(RemoteInferenceInterface):
    def __init__(self):
        creds = self._read_wx_credentials()

        self.client = APIClient(credentials=creds)

        if "project_id" in creds:
            self.client.set.default_project(creds["project_id"])
        if "space_id" in creds:
            self.client.set.default_space(creds["space_id"])

    @staticmethod
    def _read_wx_credentials() -> dict[str, str]:
        credentials = {}

        url = os.environ.get("WX_URL")
        if not url:
            raise EnvironmentError(
                "You need to specify the URL address by setting the env "
                "variable 'WX_URL', if you want to run watsonx.ai inference."
            )
        credentials["url"] = url

        project_id = os.environ.get("WX_PROJECT_ID")
        space_id = os.environ.get("WX_SPACE_ID")
        if project_id and space_id:
            logging.warning(
                "Both the project ID and the space ID were specified. "
                "The class 'WxInference' will access the project by default."
            )
            credentials["project_id"] = project_id
        elif project_id:
            credentials["project_id"] = project_id
        elif space_id:
            credentials["space_id"] = space_id
        else:
            raise EnvironmentError(
                "You need to specify the project ID or the space id by setting the "
                "appropriate env variable (either 'WX_PROJECT_ID' or 'WX_SPACE_ID'), "
                "if you want to run watsonx.ai inference."
            )

        apikey = os.environ.get("WX_APIKEY")
        username = os.environ.get("WX_USERNAME")
        password = os.environ.get("WX_PASSWORD")
        if apikey and username and password:
            logging.warning(
                "All of API key, username and password were specified. "
                "The class 'WxInference' will use the API key for authorization "
                "by default."
            )
            credentials["apikey"] = apikey
        elif apikey:
            credentials["apikey"] = apikey
        elif username and password:
            credentials["username"] = username
            credentials["password"] = password
        else:
            raise EnvironmentError(
                "You need to specify either the API key, or both the username and "
                "password by setting appropriate env variable ('WX_APIKEY', 'WX_USERNAME', "
                "'WX_PASSWORD'), if you want to run watsonx.ai inference."
            )

        return credentials

    def _prepare_generation_params(self, args: Namespace) -> dict[str, Any]:
        """Method maps generation parameters from args to be compatible with watsonx.ai."""

        return {
            "decoding_method": "sample" if args.do_sample else "greedy",
            "random_seed": None if args.seed == 0 else args.seed,  # seed must be greater than 0
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": None if args.top_k == 0 else args.top_k,  # top_k cannot be 0
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "length_penalty": args.length_penalty,
            "stop_sequences": args.stop_sequences,
            "repetition_penalty": args.repetition_penalty,
        }

    def _infer(
        self, inputs: list[str], params: dict[str, Any], args: Namespace
    ) -> list[list[str]]:
        model = ModelInference(
            model_id=args.model,
            api_client=self.client,
        )

        return [
            [result["results"][0]["generated_text"]]
            for result in
            model.generate(
                prompt=inputs,
                params=params,
            )
        ]
