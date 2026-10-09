# ==============================================================================
# CENTRALIZED CONFIGURATION MODULE
# ------------------------------------------------------------------------------
# Purpose:
# Centralizes all environment variables, filesystem paths, and server settings
# required by the FastAPI backend and inference layer.
#
# Key design decisions:
# 1. Zero hardcoded machine paths: Uses pathlib.Path(__file__) to resolve paths
#    dynamically across Windows, Linux, Docker, and Render.
# 2. Environment variable overrides: Allows containerized deployments to override
#    paths and ports via standard environment variables (e.g., MODEL_PATH, PORT).
# 3. Model metadata alignment: Automatically inspects artifacts/model_metadata.json
#    to maintain consistent model versioning across the API and frontend.
# ==============================================================================

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Base project directories resolved dynamically relative to this file's location.
# app/config.py -> parent is app/ -> parent.parent is project root.
BASE_DIR: Path = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
DEFAULT_MODEL_PATH: Path = ARTIFACTS_DIR / "churn_pipeline.joblib"
DEFAULT_METADATA_PATH: Path = ARTIFACTS_DIR / "model_metadata.json"


def _resolve_model_path() -> Path:
    """
    Resolves the filesystem path to the serialized ML model pipeline.
    Checks for an environment variable override first (useful in cloud/Docker),
    falling back to the default project artifacts directory.
    """
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path).resolve()
    return DEFAULT_MODEL_PATH


def _resolve_model_version(metadata_path: Path = DEFAULT_METADATA_PATH) -> str:
    """
    Resolves the semantic model version string.
    Checks:
    1. MODEL_VERSION environment variable override.
    2. artifacts/model_metadata.json created during the ML training phase.
    3. Safe default fallback ('1.0.0').
    """
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
    """
    Application settings container.
    Uses default_factory to evaluate environment variables at instantiation time,
    allowing clean test isolation and runtime overrides.
    """
    base_dir: Path = field(default_factory=lambda: BASE_DIR)
    artifacts_dir: Path = field(default_factory=lambda: ARTIFACTS_DIR)
    model_path: Path = field(default_factory=_resolve_model_path)
    model_version: str = field(default_factory=_resolve_model_version)
    api_title: str = field(
        default_factory=lambda: os.getenv("API_TITLE", "Customer Churn Prediction API")
    )
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    # Uppercase property aliases for conventional configuration access (e.g. settings.MODEL_PATH)
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


# Global singleton instance of application settings
settings = Settings()

# Convenient module-level exports for direct importing across the backend
MODEL_PATH: Path = settings.model_path
MODEL_VERSION: str = settings.model_version
API_TITLE: str = settings.api_title
API_HOST: str = settings.api_host
API_PORT: int = settings.api_port

