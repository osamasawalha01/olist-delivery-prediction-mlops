import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import olist_mlops.monitoring as monitoring
from olist_mlops.monitoring import Base, PredictionLog


@pytest.fixture
def monitoring_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    monkeypatch.setattr(
        monitoring,
        "get_engine",
        lambda: engine,
    )

    return engine


def add_predictions(engine, late_count, on_time_count):
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        for _ in range(late_count):
            session.add(
                PredictionLog(
                    prediction=1,
                    label="late",
                    probability=0.8,
                    latency_ms=100.0,
                    model_name="logistic_regression",
                    model_version="1",
                )
            )

        for _ in range(on_time_count):
            session.add(
                PredictionLog(
                    prediction=0,
                    label="on_time",
                    probability=0.2,
                    latency_ms=100.0,
                    model_name="logistic_regression",
                    model_version="1",
                )
            )

        session.commit()


def test_drift_insufficient_data(monitoring_db):
    add_predictions(
        monitoring_db,
        late_count=2,
        on_time_count=8,
    )

    result = monitoring.check_prediction_drift()

    assert result["status"] == "insufficient_data"
    assert result["total_predictions"] == 10


def test_no_prediction_drift(monitoring_db):
    # 34% late is approximately our 34.197% reference baseline.
    add_predictions(
        monitoring_db,
        late_count=34,
        on_time_count=66,
    )

    result = monitoring.check_prediction_drift()

    assert result["status"] == "ok"
    assert result["drift_detected"] is False
    assert result["production_late_prediction_rate"] == pytest.approx(0.34)


def test_prediction_drift_detected(monitoring_db):
    # 70% late is far above the 34.197% reference baseline.
    add_predictions(
        monitoring_db,
        late_count=70,
        on_time_count=30,
    )

    result = monitoring.check_prediction_drift()

    assert result["status"] == "drift"
    assert result["drift_detected"] is True
    assert result["production_late_prediction_rate"] == pytest.approx(0.70)


def test_drift_emits_warning(monitoring_db, caplog):
    add_predictions(
        monitoring_db,
        late_count=70,
        on_time_count=30,
    )

    with caplog.at_level("WARNING", logger="olist_mlops"):
        result = monitoring.check_prediction_drift()

    assert result["drift_detected"] is True
    assert "Prediction drift detected" in caplog.text
