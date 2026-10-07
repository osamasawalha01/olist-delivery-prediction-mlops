import pandas as pd
import pytest

from olist_mlops.gx_validation import validate_with_gx


def valid_input():
    return pd.DataFrame(
        {
            "order_purchase_timestamp": ["2018-01-15 14:30:00"],
            "customer_zip_code_prefix_x": [12345],
            "customer_city_x": ["sao paulo"],
            "customer_state_x": ["SP"],
            "customer_zip_code_prefix_y": [54321],
            "customer_city_y": ["rio de janeiro"],
            "customer_state_y": ["RJ"],
        }
    )


def test_valid_data_passes():
    validate_with_gx(valid_input())


def test_invalid_state_fails():
    data = valid_input()
    data["customer_state_x"] = "XX"

    with pytest.raises(
        ValueError,
        match="Great Expectations validation failed",
    ):
        validate_with_gx(data)


def test_invalid_zip_fails():
    data = valid_input()
    data["customer_zip_code_prefix_x"] = 100000

    with pytest.raises(
        ValueError,
        match="Great Expectations validation failed",
    ):
        validate_with_gx(data)


def test_null_value_fails():
    data = valid_input()
    data.loc[0, "customer_city_x"] = None

    with pytest.raises(
        ValueError,
        match="Great Expectations validation failed",
    ):
        validate_with_gx(data)
