import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "production"

MODEL_URL = (
    "https://github.com/osamasawalha01/"
    "olist-delivery-prediction-mlops/releases/download/"
    "model-v1.0.0/olist-model-v1.0.0.zip"
)

EXPECTED_SHA256 = "A89A3B5957BAF18A086D70225AC358C7A1DB0860AE62EEE6425644648DBB23C3"

EXPECTED_FILES = {
    "05_feature_list.csv",
    "05_preprocessor.joblib",
    "06_logistic_regression.joblib",
    "06_results_summary.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def bootstrap_model() -> None:
    if all((MODEL_DIR / name).is_file() for name in EXPECTED_FILES):
        print("Production model files already exist. Skipping download.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        zip_path = temp_path / "model.zip"

        print("Downloading production model bundle...")
        urllib.request.urlretrieve(MODEL_URL, zip_path)

        actual_hash = sha256_file(zip_path)

        if actual_hash != EXPECTED_SHA256:
            raise RuntimeError(
                f"Model checksum mismatch: expected {EXPECTED_SHA256}, "
                f"got {actual_hash}"
            )

        print("SHA-256 checksum verified.")

        with zipfile.ZipFile(zip_path) as archive:
            archive_files = set(archive.namelist())

            if archive_files != EXPECTED_FILES:
                raise RuntimeError(f"Unexpected ZIP contents: {archive_files}")

            archive.extractall(temp_path / "extracted")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        for name in EXPECTED_FILES:
            shutil.copy2(temp_path / "extracted" / name, MODEL_DIR / name)

    print("Production model artifacts are ready.")


if __name__ == "__main__":
    bootstrap_model()
