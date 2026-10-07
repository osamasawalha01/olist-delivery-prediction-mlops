from pydantic import BaseModel


class PredictionRequest(BaseModel):
    order_purchase_timestamp: str
    customer_zip_code_prefix_x: int
    customer_city_x: str
    customer_state_x: str
    customer_zip_code_prefix_y: int
    customer_city_y: str
    customer_state_y: str


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    model_name: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    orders: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
