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
            if upper_bound:
                status = value < threshold
            else:
                status = threshold < value
        elif isinstance(threshold, list):
            max_threshold = max(threshold)
            min_threshold = min(threshold)
            if (len(threshold) == 2) and min_threshold < max_threshold:
                status = (value < max_threshold) and (value > min_threshold)
            else:
                raise ValueError("A vector threshold should have 2 disctinct elements only")
        return status
    except Exception as e:
        print(str(e))
