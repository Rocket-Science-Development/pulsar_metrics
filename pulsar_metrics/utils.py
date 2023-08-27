from numbers import Number
from typing import Union

import pandas as pd

from .exceptions import CustomExceptionPulsarMetric as error_msg

ERROR_MSG_VECTOR_THRESHOLD = "Vector Threshold should have only two distinct elements [Min,Max]"
ERROR_MSG_MISSING_KEY = "Missing key for the column in the dataset"


def validate_dataframe(data: pd.DataFrame, y_name: str = "y_true", pred_name: str = "y_pred"):
    """Validate the attributes of data pandas dataframe.

    Parameters
    ----------
    data : DataFrame
        The input data (pandas DataFrame)
    y_name : str
        The input y_name (pandas DataFrame)
    pred_name : str
        The input pred_name (pandas DataFrame)

    Raises
    ------
        ValueError if input value have validation issues

    Returns
    -------
    bool
        Status of the input dataframe is validated
    """
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


def compare_to_threshold(value: float, threshold: Union[list, Number], upper_bound=True):
    """Get input value status comparing with threshold values[Min,Max]

    Parameters
    ----------
    value : float
        The input value for comparision
    threshold : Union[list, Number]
        Threshold values to validate the input value
    upper_bound : bool, optional
        A flag used to set the upper_bound param

    Raises
    ------
        ValueError if input value do not satisfy between Threshold values [Min,Max]

    Returns
    -------
    bool
        Status of the input value after comparing with threshold values[Min,Max]
    """
    status = None

    if isinstance(threshold, Number):
        status = value < threshold if upper_bound else threshold < value
    elif isinstance(threshold, list) and (len(set(threshold)) == 2) and all(isinstance(i, Number) for i in threshold):
        status = True if (min(threshold) < value < max(threshold)) else False
    elif threshold is None:
        status = None
    else:
        raise error_msg(
            value=None,
            message=ERROR_MSG_VECTOR_THRESHOLD,
        )

    return status
