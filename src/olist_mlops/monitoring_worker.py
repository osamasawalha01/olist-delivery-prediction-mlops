import os
import time

from olist_mlops.logging import setup_logging
from olist_mlops.monitoring import check_prediction_drift

logger = setup_logging()


def main() -> None:
    """Periodically check production prediction drift."""
    interval_seconds = int(os.getenv("MONITORING_INTERVAL_SECONDS", "300"))

    if interval_seconds <= 0:
        raise ValueError("MONITORING_INTERVAL_SECONDS must be positive.")

    logger.info(
        "Monitoring worker started | interval_seconds=%d",
        interval_seconds,
    )

    while True:
        try:
            result = check_prediction_drift()

            if result["status"] == "insufficient_data":
                logger.info(
                    "Drift check skipped | samples=%d | required=%d",
                    result["total_predictions"],
                    result["minimum_samples"],
                )
            elif result["status"] == "drift":
                logger.warning(
                    "DRIFT ALERT | production_late_rate=%.4f | "
                    "baseline_late_rate=%.4f | difference=%.4f",
                    result["production_late_prediction_rate"],
                    result["baseline_late_prediction_rate"],
                    result["drift_difference"],
                )
            else:
                logger.info(
                    "Drift check passed | production_late_rate=%.4f",
                    result["production_late_prediction_rate"],
                )

        except Exception:
            logger.exception("Monitoring worker drift check failed.")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
