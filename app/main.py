import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from olist_mlops.config import config
from olist_mlops.inference import InferencePipeline
from olist_mlops.metrics import (
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from olist_mlops.monitoring import initialize_monitoring_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize monitoring tables before accepting API requests."""
    initialize_monitoring_db()
    yield


app = FastAPI(
    title="Olist Late Delivery Prediction API",
    version=config["project"]["version"],
    description="Production inference API for predicting late Olist deliveries.",
    lifespan=lifespan,
)

pipeline = InferencePipeline()


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.get("/model")
def model_info():
    return {
        "model_name": pipeline.model_name,
        "model_version": pipeline.model_version,
    }


@app.get("/metrics")
def metrics():
    """Expose Prometheus monitoring metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    endpoint = "/predict"
    start_time = time.perf_counter()

    try:
        data = pd.DataFrame([request.model_dump()])
        result = pipeline.predict(data)

        REQUEST_COUNT.labels(
            endpoint=endpoint,
            status="success",
        ).inc()

        PREDICTION_COUNT.labels(
            label=result["label"],
            model_version=result["model_version"],
        ).inc()

        return result

    except (ValueError, TypeError) as exc:
        REQUEST_COUNT.labels(
            endpoint=endpoint,
            status="error",
        ).inc()

        PREDICTION_ERRORS.labels(
            endpoint=endpoint,
            error_type=type(exc).__name__,
        ).inc()

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        REQUEST_COUNT.labels(
            endpoint=endpoint,
            status="error",
        ).inc()

        PREDICTION_ERRORS.labels(
            endpoint=endpoint,
            error_type=type(exc).__name__,
        ).inc()

        raise

    finally:
        latency = time.perf_counter() - start_time

        REQUEST_LATENCY.labels(
            endpoint=endpoint,
        ).observe(latency)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(request: BatchPredictionRequest):
    endpoint = "/predict/batch"
    start_time = time.perf_counter()

    try:
        predictions = []

        for order in request.orders:
            data = pd.DataFrame([order.model_dump()])
            result = pipeline.predict(data)

            predictions.append(result)

            PREDICTION_COUNT.labels(
                label=result["label"],
                model_version=result["model_version"],
            ).inc()

        REQUEST_COUNT.labels(
            endpoint=endpoint,
            status="success",
        ).inc()

        return {
            "predictions": predictions,
        }

    except (ValueError, TypeError) as exc:
        REQUEST_COUNT.labels(
            endpoint=endpoint,
            status="error",
        ).inc()

        PREDICTION_ERRORS.labels(
            endpoint=endpoint,
            error_type=type(exc).__name__,
        ).inc()

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        REQUEST_COUNT.labels(
            endpoint=endpoint,
            status="error",
        ).inc()

        PREDICTION_ERRORS.labels(
            endpoint=endpoint,
            error_type=type(exc).__name__,
        ).inc()

        raise

    finally:
        latency = time.perf_counter() - start_time

        REQUEST_LATENCY.labels(
            endpoint=endpoint,
        ).observe(latency)
