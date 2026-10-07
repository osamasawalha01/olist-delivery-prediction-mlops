import pandas as pd
import pytest

from olist_mlops.features import build_features


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


def test_build_features():
    result = build_features(valid_input())

    assert result.shape == (1, 11)

    assert result["purchase_year"].iloc[0] == 2018
    assert result["purchase_month"].iloc[0] == 1
    assert result["purchase_day"].iloc[0] == 15
    assert result["purchase_hour"].iloc[0] == 14


def test_missing_required_column():
    data = valid_input().drop(columns=["customer_city_x"])

    with pytest.raises(
        ValueError,
        match="Missing required input columns",
    ):
        build_features(data)


def test_build_features_excludes_leakage_columns():
    data = valid_input()

    # Columns that would reveal information unavailable
    # at prediction time.
    data["order_delivered_customer_date"] = ["2018-01-20 10:00:00"]
    data["order_estimated_delivery_date"] = ["2018-01-22 00:00:00"]
    data["late_delivery"] = [0]

    result = build_features(data)

    leakage_columns = {
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "late_delivery",
    }

    assert result.shape == (1, 11)
    assert leakage_columns.isdisjoint(result.columns)
