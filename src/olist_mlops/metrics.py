from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "olist_api_requests_total",
    "Total number of API prediction requests.",
    ["endpoint", "status"],
)


REQUEST_LATENCY = Histogram(
    "olist_api_request_latency_seconds",
    "Prediction API request latency in seconds.",
    ["endpoint"],
)


PREDICTION_COUNT = Counter(
    "olist_predictions_total",
    "Total number of model predictions.",
    ["label", "model_version"],
)


PREDICTION_ERRORS = Counter(
    "olist_prediction_errors_total",
    "Total number of prediction errors.",
    ["endpoint", "error_type"],
)
