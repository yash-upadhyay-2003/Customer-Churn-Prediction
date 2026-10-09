import json
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
DEFAULT_MODEL_PATH: Path = ARTIFACTS_DIR / "churn_pipeline.joblib"
DEFAULT_METADATA_PATH: Path = ARTIFACTS_DIR / "model_metadata.json"


def _resolve_model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path).resolve()
    return DEFAULT_MODEL_PATH


def _resolve_model_version(metadata_path: Path = DEFAULT_METADATA_PATH) -> str:
    env_version = os.getenv("MODEL_VERSION")
    if env_version:
        return env_version
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return str(metadata.get("model_version", metadata.get("version", "1.0.0")))
        except Exception:
            return "1.0.0"
    return "1.0.0"


@dataclass
class Settings:
    base_dir: Path = field(default_factory=lambda: BASE_DIR)
    artifacts_dir: Path = field(default_factory=lambda: ARTIFACTS_DIR)
    model_path: Path = field(default_factory=_resolve_model_path)
    model_version: str = field(default_factory=_resolve_model_version)
    api_title: str = field(
        default_factory=lambda: os.getenv("API_TITLE", "Customer Churn Prediction API")
    )
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    @property
    def MODEL_PATH(self) -> Path:
        return self.model_path

    @property
    def MODEL_VERSION(self) -> str:
        return self.model_version

    @property
    def API_TITLE(self) -> str:
        return self.api_title

    @property
    def API_HOST(self) -> str:
        return self.api_host

    @property
    def API_PORT(self) -> int:
        return self.api_port


settings = Settings()

MODEL_PATH: Path = settings.model_path
MODEL_VERSION: str = settings.model_version
API_TITLE: str = settings.api_title
API_HOST: str = settings.api_host
API_PORT: int = settings.api_port
