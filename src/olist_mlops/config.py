import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    """Load application configuration from config.yaml."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)

    # Environment variables override connection settings.
    loaded_config["mlflow"]["tracking_uri"] = os.getenv(
        "MLFLOW_TRACKING_URI",
        loaded_config["mlflow"]["tracking_uri"],
    )

    return loaded_config


config = load_config()
