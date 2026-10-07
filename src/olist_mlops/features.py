import pandas as pd

RAW_LOCATION_COLUMNS = [
    "customer_zip_code_prefix_x",
    "customer_city_x",
    "customer_state_x",
    "customer_zip_code_prefix_y",
    "customer_city_y",
    "customer_state_y",
]

REQUIRED_INPUT_COLUMNS = [
    "order_purchase_timestamp",
    *RAW_LOCATION_COLUMNS,
]


def validate_input_columns(data: pd.DataFrame) -> None:
    """Validate that the input contains the columns needed for feature building."""
    missing_columns = [
        column for column in REQUIRED_INPUT_COLUMNS if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required input columns: {missing_columns}")


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the feature engineering used in Notebook 5.

    No fitting happens here. The fitted preprocessing object is applied later.
    """
    validate_input_columns(data)

    features = data.copy()

    # Recreate the date features used during training.
    purchase_date = pd.to_datetime(
        features["order_purchase_timestamp"],
        errors="coerce",
    )

    features["purchase_year"] = purchase_date.dt.year
    features["purchase_month"] = purchase_date.dt.month
    features["purchase_day"] = purchase_date.dt.day
    features["purchase_weekday"] = purchase_date.dt.weekday
    features["purchase_hour"] = purchase_date.dt.hour

    # Keep only the 11 raw features expected by the saved preprocessor.
    model_features = features[
        [
            "customer_zip_code_prefix_x",
            "customer_city_x",
            "customer_state_x",
            "customer_zip_code_prefix_y",
            "customer_city_y",
            "customer_state_y",
            "purchase_year",
            "purchase_month",
            "purchase_day",
            "purchase_weekday",
            "purchase_hour",
        ]
    ].copy()

    return model_features
