"""Desc"""

import itertools
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Set, List, Optional, Union
import pickle
from pathlib import Path
from copy import deepcopy
import statistics

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score

from .execute import unsafe_execute


_CITATION = """"""

_DESCRIPTION = """"""

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
This metric executes untrusted model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network.

Once you have read this disclaimer and taken appropriate precautions,
set the environment variable HF_ALLOW_CODE_EVAL="1". Within Python you can to this
with:

>>> import os
>>> os.environ["HF_ALLOW_CODE_EVAL"] = "1"

################################################################################\
"""

_LICENSE = """"""


def tabpfn_average_accuracy(
    predictions: List[List[str]], 
    dataframe_paths: List[Path], 
    split_column: str,
    train_split: str,
    test_split: str,
    target_column: str, 
    timeout: float
) -> Dict[str, float]:
    if len(predictions) != len(dataframe_paths):
        raise ValueError(f"Expected num. predictions and num. dataframes to be the same, {len(predictions)} != {len(dataframe_paths)}")
    dataframe_path_to_predictions = defaultdict(list)
    for prediction, dataframe_path in zip(predictions, dataframe_paths):
        if len(prediction) != 1:
            raise ValueError
        dataframe_path_to_predictions[dataframe_path].append(prediction.pop())
    accuracies = []
    for dataframe_path, dataframe_predictions in dataframe_path_to_predictions.items():
        dataframe = load_dataframe(dataframe_path=dataframe_path)
        train_dataframe = dataframe[dataframe[split_column] == train_split]
        test_dataframe = dataframe[dataframe[split_column] == test_split]
        for prediction in dataframe_predictions:
            accuracies.append(
                tabpfn_accuracy(
                    prediction=prediction,
                    train_dataframe=remove_dataframe_columns(train_dataframe, columns_to_remove={split_column}),
                    test_dataframe=remove_dataframe_columns(test_dataframe, columns_to_remove={split_column}),
                    target_column=target_column,
                    timeout=timeout,
                )
            )
    return {"tabpfn_avg_accuracy": statistics.mean(accuracies)}


def load_dataframe(dataframe_path: Path) -> pd.DataFrame:
    # TODO(ajedrosz): some ok serialization
    return pd.read_pickle(dataframe_path)


def remove_dataframe_columns(dataframe: pd.DataFrame, columns_to_remove: Set[str]) -> pd.DataFrame:
    return dataframe[[column for column in dataframe.columns if column not in columns_to_remove]]


def tabpfn_accuracy(
    prediction: str, 
    train_dataframe: pd.DataFrame, 
    test_dataframe: pd.DataFrame, 
    target_column: str, 
    timeout: float
) -> float:
    train_target = train_dataframe[target_column]
    test_target = test_dataframe[target_column]
    train_features = remove_dataframe_columns(train_dataframe, columns_to_remove={target_column})
    test_features = remove_dataframe_columns(test_dataframe, columns_to_remove={target_column})
    # TODO(ajedrosz): handle potentially modified order of samples, e.g. id column that's not given in prompt header
    train_features_transformed = _transform_dataframe_inplace(
        prediction=prediction,
        dataframe=train_features,
        timeout=timeout,
    )
    test_features_transformed = _transform_dataframe_inplace(
        prediction=prediction,
        dataframe=test_features,
        timeout=timeout,
    )
    # TODO(ajedrosz): what with hparams
    # TODO(ajedrosz): does this do regression too
    classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
    classifier.fit(train_features_transformed, train_target)
    test_target_hat = classifier.predict(test_features_transformed)
    return accuracy_score(test_target, test_target_hat)


def _transform_dataframe_inplace(prediction: str, dataframe: pd.DataFrame, timeout: float) -> Union[pd.DataFrame, None]:
    if os.getenv("HF_ALLOW_CODE_EVAL", 0) != "1":
        raise ValueError(_WARNING)
    # TODO(ajedrosz): header constant
    df_transformation_with_assignment = f"""{prediction}
dataframe_transformed = transform(dataframe)"""
    # TODO(ajedrosz): need to copy?
    exec_globals = {"dataframe": dataframe}
    result = []
    unsafe_execute(
        check_program=df_transformation_with_assignment,
        result=result,
        timeout=timeout,
        exec_globals=exec_globals,
    )
    if result.pop() != "passed":
        return None
    else:
        return exec_globals["dataframe_transformed"]
