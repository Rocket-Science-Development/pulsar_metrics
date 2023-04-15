import re
import sys

import pandas as pd
from pytest import raises as pytest_raises

sys.path.append("..")

from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.utils import *

TARGET_COLUMN = "y_true"
PREDICTION_COLUMN = "y_pred"
DATE_COLUMN = "date"
MODELID_COLUMN = "model_id"

df = pd.DataFrame(columns=[TARGET_COLUMN, PREDICTION_COLUMN, DATE_COLUMN, MODELID_COLUMN])

# Testing dataframe validation
# ==============================


def test_dataframe_is_valid():
    print(df.columns)
    assert validate_dataframe(data=df) == True


def test_validate_target_is_missing():
    with pytest_raises(error_msg, match=ERROR_MSG_MISSING_KEY):
        validate_dataframe(data=df.drop(TARGET_COLUMN, axis=1))


def test_validate_prediction_is_missing():
    with pytest_raises(error_msg, match=ERROR_MSG_MISSING_KEY):
        validate_dataframe(data=df.drop(PREDICTION_COLUMN, axis=1))


def test_validate_date_is_missing():
    with pytest_raises(error_msg, match=ERROR_MSG_MISSING_KEY):
        validate_dataframe(data=df.drop(DATE_COLUMN, axis=1))


def test_validate_modelid_is_missing():
    with pytest_raises(error_msg, match=ERROR_MSG_MISSING_KEY):
        validate_dataframe(data=df.drop(MODELID_COLUMN, axis=1))


# Testing comparison to threshold
# ===================================


def test_compare_to_single_threshold():
    value = 3
    threshold = 2
    assert compare_to_threshold(value, threshold) == False


def test_compare_to_interval_threshold():
    value = 3
    threshold = [1, 4]
    assert compare_to_threshold(value, threshold) == True


def test_invalid_threshold_with_three_elements():
    value = 3
    threshold = [1, 4, 6]
    with pytest_raises(ValueError, match=re.escape(ERROR_MSG_VECTOR_THRESHOLD)):
        compare_to_threshold(value, threshold)


def test_invalid_threshold_with_equal_bounds():
    value = 3
    threshold = [1, 1]
    with pytest_raises(ValueError, match=re.escape(ERROR_MSG_VECTOR_THRESHOLD)):
        compare_to_threshold(value, threshold)


def test_invalid_threshold_type():
    value = 3
    threshold = "a"
    with pytest_raises(ValueError, match=re.escape(ERROR_MSG_VECTOR_THRESHOLD)):
        compare_to_threshold(value, threshold)
