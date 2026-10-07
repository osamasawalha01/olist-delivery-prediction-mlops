import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from olist_mlops.config import PROJECT_ROOT, config

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "olist-late-delivery"
REGISTERED_MODEL_NAME = "olist-logistic-regression"


def register_model() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_path = PROJECT_ROOT / config["paths"]["model_path"]
    preprocessor_path = PROJECT_ROOT / config["paths"]["preprocessor_path"]
    feature_list_path = PROJECT_ROOT / config["paths"]["feature_list_path"]
    results_path = PROJECT_ROOT / "artifacts/06_results_summary.csv"

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
            mlflow.log_metric(
                metric,
                float(model_results[metric]),
            )

        mlflow.log_artifact(
            str(preprocessor_path),
            artifact_path="preprocessing",
        )

        mlflow.log_artifact(
            str(feature_list_path),
            artifact_path="features",
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"Registered model: {REGISTERED_MODEL_NAME}")


if __name__ == "__main__":
    register_model()
