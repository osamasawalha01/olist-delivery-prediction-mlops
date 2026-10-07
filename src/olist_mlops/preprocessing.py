import pandas as pd


def transform_features(data: pd.DataFrame, preprocessor):
    """Apply the saved fitted preprocessing pipeline."""

    return preprocessor.transform(data)
