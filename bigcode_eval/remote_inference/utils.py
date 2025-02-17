from argparse import Namespace
from importlib import import_module

from bigcode_eval.base import Task

from bigcode_eval.remote_inference.base import Dataset, RemoteInferenceInterface
from bigcode_eval.remote_inference.wx_ai import WxInference


required_packages = {
    "wx": ["ibm_watsonx_ai"],
}


def check_packages_installed(names: list[str]) -> bool:
    for name in names:
        try:
            import_module(name)
        except (ImportError, ModuleNotFoundError, NameError):
            return False
    return True


def remote_inference(
    inference_platform: str,
    dataset: Dataset,
    task: Task,
    args: Namespace,
) -> list[list[str]]:
    packages = required_packages.get(inference_platform)
    if packages and not check_packages_installed(packages):
        raise RuntimeError(
            f"In order to run inference with '{inference_platform}', the "
            f"following packages are required: '{packages}'. However, they "
            f"could not be properly imported. Check if the packages are "
            f"installed correctly."
        )

    inference_cls: RemoteInferenceInterface

    if inference_platform == "wx":
        inference_cls = WxInference()

    else:
        raise ValueError(
            f"Unsupported remote inference platform: '{inference_platform}'."
        )

    return inference_cls.prepare_generations(
        dataset=dataset,
        task=task,
        args=args,
        prefix=args.prefix,
        postprocess=args.postprocess,
    )
