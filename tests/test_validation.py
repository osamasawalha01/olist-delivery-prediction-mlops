import pandas as pd
import pytest

from olist_mlops.validation import validate_input


def test_empty_input():
    data = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Input data is empty",
    ):
        validate_input(data)


def test_invalid_input_type():
    with pytest.raises(
        TypeError,
        match="Input must be a pandas DataFrame",
    ):
        validate_input({"customer_state_x": "SP"})
