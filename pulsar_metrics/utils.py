from typing import Union

import pandas as pd

from .exceptions import CustomExceptionPulsarMetric as error_msg

ERROR_MSG_VECTOR_THRESHOLD = "Vector Threshold should have only two distinct elements [Min,Max]"
ERROR_MSG_MISSING_KEY = "Missing key for the column in the dataset"


def validate_dataframe(data: pd.DataFrame, y_name: str = "y_true", pred_name: str = "y_pred"):

    if y_name not in data.columns:
        raise error_msg(
            value=y_name,
            message=ERROR_MSG_MISSING_KEY,
        )
    elif pred_name not in data.columns:
        raise error_msg(
            value=pred_name,
            message=ERROR_MSG_MISSING_KEY,
        )
    elif "date" not in data.columns:
        raise error_msg(
            value=None,
            message=ERROR_MSG_MISSING_KEY,
        )
    elif "model_id" not in data.columns:
        raise error_msg(
            value=None,
            message=ERROR_MSG_MISSING_KEY,
        )
    else:
        print("Input dataframe is validated in validate_dataframe()!")
        return True


def compare_to_threshold(value: float, threshold: Union[list, float, int], upper_bound=True):

    status = None

    try:
        if isinstance(threshold, (float, int)):
            status = value < threshold if upper_bound else threshold < value
        elif isinstance(threshold, list):
            status = True if (len(threshold) == 2) and len(set()) == len(threshold) else False
        else:
            raise error_msg(
                value=threshold,
                message=ERROR_MSG_VECTOR_THRESHOLD,
            )
        return status
    except Exception as e:
        print(f"Exception caught for ValueError in compare_to_threshold():: {str(e)}")
