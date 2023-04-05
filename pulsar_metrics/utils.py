from typing import Union

import pandas as pd


def validate_dataframe(data: pd.DataFrame, y_name: str = "y_true", pred_name: str = "y_pred"):

    if y_name not in data.columns:
        raise KeyError("The name of the realized target is not in the dataset")
    elif pred_name not in data.columns:
        raise KeyError("The name of the realized target is not in the dataset")
    elif "date" not in data.columns:
        raise KeyError('There is not datetime column in the dataset or it should be named "date"')
    elif "model_id" not in data.columns:
        raise KeyError('There is not model id column in the dataset or it should be named "model_id"')
    else:
        print("The dataframe is validated !")
        return True


def compare_to_threshold(value: float, threshold: Union[list, float, int], upper_bound=True):

    status = None

    try:
        if isinstance(threshold, (float, int)):
            status = value < threshold if upper_bound else threshold < value
        elif isinstance(threshold, list):
            status = True if (len(threshold) == 2) and len(set()) == len(threshold) else False
        else:
            raise ValueError("Vector Threshold should have only two distinct elements [Min,Max]")

        return status
    except Exception as e:
        print(f"Exception caught for ValueError in compare_to_threshold():: {str(e)}")
