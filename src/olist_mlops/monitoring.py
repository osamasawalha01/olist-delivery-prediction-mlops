import os
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from olist_mlops.config import config
from olist_mlops.logging import setup_logging

logger = setup_logging()

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://",
        "postgresql+psycopg2://",
        1,
    )


Base = declarative_base()


class PredictionLog(Base):
    """Persistent record of a production prediction."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True)

    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    prediction = Column(Integer, nullable=False)
    label = Column(String(20), nullable=False)
    probability = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)

    actual_delivery_status = Column(
        String(20),
        nullable=True,
    )


def get_engine():
    """Create the database engine."""
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not configured.")

    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )


def initialize_monitoring_db():
    """Create monitoring tables when they do not exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def save_prediction(
    prediction: int,
    label: str,
    probability: float,
    latency_ms: float,
    model_name: str,
    model_version: str,
) -> None:
    """Persist one production prediction."""
    engine = get_engine()

    SessionLocal = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )

    with SessionLocal() as session:
        record = PredictionLog(
            prediction=prediction,
            label=label,
            probability=probability,
            latency_ms=latency_ms,
            model_name=model_name,
            model_version=model_version,
        )

        session.add(record)
        session.commit()


def check_prediction_drift() -> dict:
    """Check drift using the most recent production predictions."""
    engine = get_engine()
    SessionLocal = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )

    minimum_samples = config["monitoring"]["minimum_samples"]
    window_size = config["monitoring"]["window_size"]
    baseline_rate = config["monitoring"]["baseline_late_prediction_rate"]
    drift_threshold = config["monitoring"]["drift_threshold"]

    if window_size < minimum_samples:
        raise ValueError("window_size must be >= minimum_samples.")

    with SessionLocal() as session:
        recent_predictions = (
            session.query(PredictionLog.prediction)
            .order_by(PredictionLog.id.desc())
            .limit(window_size)
            .all()
        )

    total_predictions = len(recent_predictions)
    late_predictions = sum(prediction == 1 for (prediction,) in recent_predictions)

    if total_predictions < minimum_samples:
        return {
            "status": "insufficient_data",
            "total_predictions": total_predictions,
            "minimum_samples": minimum_samples,
        }

    production_rate = late_predictions / total_predictions
    drift_difference = abs(production_rate - baseline_rate)
    drift_detected = drift_difference > drift_threshold

    if drift_detected:
        logger.warning(
            "Prediction drift detected | "
            "baseline_late_rate=%.4f | "
            "production_late_rate=%.4f | "
            "difference=%.4f | "
            "threshold=%.4f | "
            "samples=%d",
            baseline_rate,
            production_rate,
            drift_difference,
            drift_threshold,
            total_predictions,
        )

    return {
        "status": "drift" if drift_detected else "ok",
        "total_predictions": total_predictions,
        "late_predictions": late_predictions,
        "baseline_late_prediction_rate": baseline_rate,
        "production_late_prediction_rate": production_rate,
        "drift_difference": drift_difference,
        "drift_threshold": drift_threshold,
        "drift_detected": drift_detected,
    }
