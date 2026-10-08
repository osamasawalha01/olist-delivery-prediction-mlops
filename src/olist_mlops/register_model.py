import logging

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from olist_mlops.config import PROJECT_ROOT, config

logger = logging.getLogger(__name__)


def register_model() -> None:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    client = mlflow.MlflowClient()
    model_name = config["mlflow"]["registered_model_name"]

    # Reuse an existing registered version when available.
    existing_versions = client.search_model_versions(f"name='{model_name}'")

    ready_versions = [
        version for version in existing_versions if version.status == "READY"
    ]

    if ready_versions:
        latest_version = max(
            ready_versions,
            key=lambda version: int(version.version),
        )

        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=latest_version.version,
        )

        logger.info(
            "Reusing registered model %s version %s",
            model_name,
            latest_version.version,
        )
        return

    # First-time registration
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    model_path = PROJECT_ROOT / config["paths"]["model_path"]
    preprocessor_path = PROJECT_ROOT / config["paths"]["preprocessor_path"]
    feature_list_path = PROJECT_ROOT / config["paths"]["feature_list_path"]
    results_path = PROJECT_ROOT / config["paths"]["results_summary_path"]

    model = joblib.load(model_path)
    results = pd.read_csv(results_path)

    model_results = results[results["model"] == "LogisticRegression"].iloc[0]

    with mlflow.start_run(run_name="logistic-regression-production"):
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "probability_threshold": config["inference"]["probability_threshold"],
                "dataset": model_results["dataset"],
            }
        )

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ]:
            mlflow.log_metric(metric, float(model_results[metric]))

        mlflow.log_artifact(
            str(preprocessor_path),
            artifact_path="preprocessing",
        )

        mlflow.log_artifact(
            str(feature_list_path),
            artifact_path="features",
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=model_name,
        )

    registered_version = client.search_model_versions(f"run_id='{model_info.run_id}'")

    version = next(
        version for version in registered_version if version.name == model_name
    )

    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=version.version,
    )

    logger.info(
        "Registered model %s version %s",
        model_name,
        version.version,
    )


if __name__ == "__main__":
    register_model()
