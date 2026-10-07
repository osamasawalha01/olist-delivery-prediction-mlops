import pandas as pd
import pytest

import olist_mlops.inference as inference_module
from olist_mlops.inference import InferencePipeline


@pytest.fixture(scope="module")
def pipeline():
    return InferencePipeline()


@pytest.fixture(autouse=True)
def mock_prediction_persistence(monkeypatch):
    """Prevent inference tests from writing prediction logs to PostgreSQL."""
    monkeypatch.setattr(
        inference_module,
        "save_prediction",
        lambda **kwargs: None,
    )


@pytest.fixture
def sample_input():
    df = pd.read_parquet("artifacts/03_test.parquet")
    return df.iloc[[0]]


def test_model_loaded(pipeline):
    assert pipeline.model is not None
    assert pipeline.preprocessor is not None
    assert pipeline.model.n_features_in_ == 7361


def test_prediction_structure(pipeline, sample_input):
    result = pipeline.predict(sample_input)

    assert "prediction" in result
    assert "label" in result
    assert "probability" in result
    assert "model_name" in result
    assert "model_version" in result


def test_known_prediction(pipeline, sample_input):
    result = pipeline.predict(sample_input)

    assert result["prediction"] == 0
    assert result["label"] == "on_time"

    assert result["probability"] == pytest.approx(
        0.4695051465317476,
        abs=1e-12,
    )

    assert result["model_name"] == "logistic_regression"
    assert result["model_version"] == "1"


def test_probability_range(pipeline, sample_input):
    result = pipeline.predict(sample_input)

    assert 0.0 <= result["probability"] <= 1.0
