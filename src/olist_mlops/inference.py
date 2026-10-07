import time

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from olist_mlops.config import config
from olist_mlops.features import build_features
from olist_mlops.gx_validation import validate_with_gx
from olist_mlops.logging import setup_logging
from olist_mlops.monitoring import save_prediction
from olist_mlops.preprocessing import transform_features
from olist_mlops.validation import validate_input

logger = setup_logging()


class InferencePipeline:
    """Production inference pipeline using saved fitted artifacts."""

    def __init__(self):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

        registered_model_name = config["mlflow"]["registered_model_name"]
        model_version = config["mlflow"]["model_version"]

        model_uri = f"models:/{registered_model_name}/{model_version}"

        # Load registered model from MLflow.
        self.model = mlflow.sklearn.load_model(model_uri)

        # Get the run associated with this registered model version.
        client = mlflow.MlflowClient()
        model_version_info = client.get_model_version(
            name=registered_model_name,
            version=model_version,
        )

        # Download fitted preprocessor from the same MLflow run.
        preprocessor_path = mlflow.artifacts.download_artifacts(
            run_id=model_version_info.run_id,
            artifact_path="preprocessing/05_preprocessor.joblib",
        )

        self.preprocessor = joblib.load(preprocessor_path)

        self.model_name = config["model"]["name"]
        self.model_version = model_version

        self.threshold = config["inference"]["probability_threshold"]

        logger.info(
            "Model loaded | name=%s | version=%s",
            self.model_name,
            self.model_version,
        )

    def predict(self, data: pd.DataFrame) -> dict:
        """Generate a prediction for one order."""
        start_time = time.perf_counter()

        request_input = (
            data.to_dict(orient="records")
            if isinstance(data, pd.DataFrame)
            else str(data)
        )

        logger.info(
            "Prediction request | input=%s | model_version=%s",
            request_input,
            self.model_version,
        )

        try:
            validate_input(data)
            validate_with_gx(data)

            features = build_features(data)

            transformed_features = transform_features(
                features,
                self.preprocessor,
            )

            probability = float(self.model.predict_proba(transformed_features)[0, 1])

            prediction = int(probability >= self.threshold)

            result = {
                "prediction": prediction,
                "label": ("late" if prediction == 1 else "on_time"),
                "probability": probability,
                "model_name": self.model_name,
                "model_version": self.model_version,
            }

            latency_ms = (time.perf_counter() - start_time) * 1000
            save_prediction(
                prediction=prediction,
                label=result["label"],
                probability=probability,
                latency_ms=latency_ms,
                model_name=self.model_name,
                model_version=self.model_version,
            )
            logger.info(
                "Prediction completed | rows=%d | "
                "prediction=%d | probability=%.6f | "
                "latency_ms=%.2f | model_version=%s",
                len(data),
                prediction,
                probability,
                latency_ms,
                self.model_version,
            )

            return result

        except (ValueError, TypeError) as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.warning(
                "Invalid prediction input | "
                "error=%s | latency_ms=%.2f | "
                "model_version=%s",
                exc,
                latency_ms,
                self.model_version,
            )

            raise
