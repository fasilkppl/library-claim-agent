"""Runtime settings, read from environment variables and an optional .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_prefix="CLAIM_", extra="ignore")

    sweeps_dir: Path = ROOT / "sweeps"
    models_dir: Path = ROOT / "models"
    web_dir: Path = ROOT / "web"

    # Locale defaults; the live agent confirms or changes them at the start of every sweep.
    country: str = "IN"
    currency: str = "INR"
    second_country: str = "US"

    # Local models
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3-vl:8b-instruct"
    whisper_model: str = "large-v3-turbo"
    whisper_beam_size: int = 5
    whisper_device: str = "mps"  # Apple GPU; falls back to the small model on the CPU
    whisper_hint: str = "Home library contents claim."
    whisper_threads: int = 6
    item_vision_check: bool = True  # Qwen3-VL names each item crop (fixes detector look-alikes)
    torch_threads: int = 4  # leave CPU for speech recognition during the sweep
    tts_voice: str = "af_heart"
    yolo_books_model: str = "yolo11x.pt"
    yolo_world_model: str = "yolov8x-worldv2.pt"
    device: str = "mps"

    # Price sources. Empty key = source disabled; lines fall back or go to review.
    serpapi_key: str = Field(default="", validation_alias="SERPAPI_API_KEY")
    serper_key: str = Field(default="", validation_alias="SERPER_API_KEY")  # used instead of SerpAPI when set
    ebay_client_id: str = Field(default="", validation_alias="EBAY_CLIENT_ID")
    ebay_client_secret: str = Field(default="", validation_alias="EBAY_CLIENT_SECRET")
    google_books_key: str = Field(default="", validation_alias="GOOGLE_BOOKS_API_KEY")

    # Review rules
    appraisal_threshold_usd: float = 150.0  # converted to local currency; above it a line is not auto-priced
    rare_before_year: int = 1950
    id_accept: float = 0.85  # id_confidence at or above: identified
    id_review: float = 0.65  # between review and accept: identified but sent to review
    scale_disagreement: float = 0.10

    # Capture quality
    blur_min: float = 20.0  # detail sharpness (capture.quality.detail_sharpness); half of it drops a frame
    glare_max: float = 0.06  # share of clipped pixels in the spine region


@lru_cache
def get_settings() -> Settings:
    return Settings()
