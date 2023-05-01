import json
import sys

sys.path.append("..")

TARGET_COLUMN = "y_true"
PREDICTION_COLUMN = "y_pred"
DATE_COLUMN = "date"
MODELID_COLUMN = "model_id"
TEST_DATA_FILENAME = "tests/TestInputData.json"
TEST_UTILS_KEY = "utils"


# Getting test input data from json file
def get_test_data():
    with open(TEST_DATA_FILENAME) as f:
        data = json.loads(f.read())
    return data
