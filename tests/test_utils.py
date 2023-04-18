import re
import sys

import pandas as pd
import pytest
import yaml

sys.path.append("..")

from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.utils import *

from . import TestConfiguration

df = pd.DataFrame(
    columns=[
        TestConfiguration.TARGET_COLUMN,
        TestConfiguration.PREDICTION_COLUMN,
        TestConfiguration.DATE_COLUMN,
        TestConfiguration.MODELID_COLUMN,
    ]
)

# Testing dataframe validation
# ==============================


# Dataframe is valid
def test_dataframe_is_valid():
    print(df.columns)
    assert validate_dataframe(data=df) == True


# Dataframe is missing important columns
@pytest.mark.parametrize(
    "missing_column",
    [
        TestConfiguration.TARGET_COLUMN,
        TestConfiguration.PREDICTION_COLUMN,
        TestConfiguration.DATE_COLUMN,
        TestConfiguration.MODELID_COLUMN,
    ],
)
def test_validate_missing_column(missing_column):
    with pytest.raises(error_msg, match=ERROR_MSG_MISSING_KEY):
        validate_dataframe(data=df.drop(missing_column, axis=1))


# Testing comparison to threshold
# ===================================


# Comparing to valid thresholds
@pytest.mark.parametrize("threshold,value,status", [(3, 1, True), (1, 3, False), ([1, 4], 3, True)])
def test_compare_to_valid_threshold(threshold, value, status):
    assert compare_to_threshold(value, threshold) == status


# Comparing to invalid thresholds
@pytest.mark.parametrize("threshold,value", [([1, 4, 6], 3), ([1, 1], 3), ("a", 3), ([1, "2"], 1.5)])
def test_invalid_threshold(threshold, value):
    with pytest.raises(ValueError, match=re.escape(ERROR_MSG_VECTOR_THRESHOLD)):
        compare_to_threshold(value, threshold)
