import pytest

import pandas as pd

from bigcode_eval.tasks.custom_metrics.ds_f_eng_eval import tabpfn_accuracy, _transform_dataframe_inplace


def test_transform_dataframe_inplace():
    df = pd.DataFrame(
        [
            ["train", 1.0, 2.0, "c1"],
            ["train", 3.0, 2.0, "c2"],
            ["train", 6.0, 1.0, "c2"],
            ["train", 3.0, 1.0, "c1"],
            ["test", 1.0, 2.0, "c1"],
            ["test", 8.0, 4.0, "c2"],
        ],
        columns=["split", "f1", "f2", "target"]
    )
    prediction = """
def transform(dataframe):
    dataframe['f3'] = dataframe['f1'] * dataframe['f2']
    return dataframe
    """
    out = _transform_dataframe_inplace(
        prediction=prediction, dataframe=df, timeout=1.0
    )
    df['f3'] = df['f1'] * df['f2']
    assert out.equals(df)


@pytest.mark.parametrize(
    "prediction, expected_avg_acc",
    [
        (
            """
def transform(dataframe):
    return dataframe
            """,
            0.8
        ),
        (
            """
def transform(dataframe):
    l = [5.0 for _ in range(len(dataframe))]
    l[-3:] = [7.0, 7.0, 7.0]
    dataframe['f3'] = l
    return dataframe
            """,
            1.0,

        )
    ]
)
def test_tabpfn_accuracy(prediction, expected_avg_acc):
    train_df = pd.DataFrame(
        [
            [-1.0, 2.0, "c1"],
            [-2.0, 2.0, "c1"],
            [-3.0, 1.0, "c1"],
            [-4.0, 1.0, "c2"],
            [-4.0, -41.0, "c2"],
            [4.0, 3.0, "c2"],
        ],
        columns=["f1", "f2", "target"]
    )
    test_df = pd.DataFrame(
        [
            [1.0, 2.0, "c1"],
            [1.0, -2.0, "c1"],
            [8.0, 4.0, "c2"],
            [8.0, -4.0, "c2"],
            [8.0, -5.0, "c2"],
        ],
        columns=["f1", "f2", "target"]
    )
    assert tabpfn_accuracy(
            prediction=prediction, 
            train_dataframe=train_df, 
            test_dataframe=test_df, 
            target_column="target",
            timeout=1.0
        ) == expected_avg_acc
