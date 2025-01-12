from numbers import Number
from typing import Union
import pandas as pd
from .exceptions import CustomExceptionPulsarMetric as error_msg

ERROR_MSG_VECTOR_THRESHOLD = "Vector Threshold should have only two distinct elements [Min,Max]"
ERROR_MSG_MISSING_KEY = "Missing key for the column in the dataset"

'''
- Using a set difference operation to identify all missing columns in a single pass, which reduces the number of membership checks and is faster, especially for dataframes with many columns.
- Simplified Error Reporting by simply identifying all missing columns at once and include them in the error message, which provides more actionable feedback.
- Removed Unnecessary Print Statement since it introduces unnecessary I/O overhead and is bad practice for production code.
Levent Ozbek
'''

def validate_dataframe(data: pd.DataFrame, y_name: str = "y_true", pred_name: str = "y_pred"):
    """Validate the attributes of data pandas dataframe.
    Optimized missing column search.
    Removed redundant print statements.
    """
    required_columns = {y_name, pred_name, "date", "model_id"}
    missing_columns = required_columns - set(data.columns)
    
    if missing_columns:
        raise error_msg(
            value=", ".join(missing_columns),
            message=ERROR_MSG_MISSING_KEY,
        )

    return True

'''
- Removed the status Variable since its value was returned immediately and hence is unnecessary.
- Simplified the Numeric Threshold Check since it used an unnecessary assignment. The result is returned directly.
- Simplified List Threshold Validation by directly returnin the result of the range check, which makes the use of set() and all() unnecessary.
- Retained Exception Handling
Levent Ozbek
'''

def compare_to_threshold(value: float, threshold: Union[list, Number], upper_bound=True):
    """Get input value status comparing with threshold values [Min,Max]."""
    if isinstance(threshold, Number):
        return value < threshold if upper_bound else value > threshold
    
    if (
        isinstance(threshold, list)
        and len(threshold) == 2
        and isinstance(threshold[0], Number)
        and isinstance(threshold[1], Number)
    ):
        return min(threshold) < value < max(threshold)
    
    raise error_msg(
        value=None,
        message=ERROR_MSG_VECTOR_THRESHOLD,
    )
