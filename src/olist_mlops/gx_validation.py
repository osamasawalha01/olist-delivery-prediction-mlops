import great_expectations as gx
import pandas as pd

REQUIRED_COLUMNS = [
    "order_purchase_timestamp",
    "customer_zip_code_prefix_x",
    "customer_city_x",
    "customer_state_x",
    "customer_zip_code_prefix_y",
    "customer_city_y",
    "customer_state_y",
]

STATE_COLUMNS = [
    "customer_state_x",
    "customer_state_y",
]

ZIP_COLUMNS = [
    "customer_zip_code_prefix_x",
    "customer_zip_code_prefix_y",
]

BRAZIL_STATES = [
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
]


def validate_with_gx(data: pd.DataFrame) -> None:
    """Validate incoming inference data using Great Expectations."""

    context = gx.get_context()

    source = context.data_sources.add_pandas("inference_source")
    asset = source.add_dataframe_asset(name="inference_data")

    batch_definition = asset.add_batch_definition_whole_dataframe("inference_batch")

    batch = batch_definition.get_batch(batch_parameters={"dataframe": data})

    expectations = []

    for column in REQUIRED_COLUMNS:
        expectations.append(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )

    for column in STATE_COLUMNS:
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=column,
                value_set=BRAZIL_STATES,
            )
        )

    for column in ZIP_COLUMNS:
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=0,
                max_value=99999,
            )
        )

    suite = gx.ExpectationSuite(
        name="inference_validation_suite",
        expectations=expectations,
    )

    result = batch.validate(suite)

    if not result.success:
        raise ValueError("Great Expectations validation failed for inference data.")
