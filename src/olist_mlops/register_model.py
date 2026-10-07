import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from olist_mlops.config import PROJECT_ROOT, config

EXPERIMENT_NAME = "olist-late-delivery"


def register_model() -> None:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
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
            registered_model_name=config["mlflow"]["registered_model_name"],
        )

        print(f"Registered model: {config['mlflow']['registered_model_name']}")


if __name__ == "__main__":
    register_model()
