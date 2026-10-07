import pandas as pd

from olist_mlops.features import REQUIRED_INPUT_COLUMNS


def validate_input(data: pd.DataFrame) -> None:
    """Validate raw inference input."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Input data is empty.")

    missing_columns = [
        column for column in REQUIRED_INPUT_COLUMNS if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required input columns: {missing_columns}")
