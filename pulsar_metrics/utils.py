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
