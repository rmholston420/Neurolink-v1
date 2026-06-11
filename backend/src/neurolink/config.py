"""Neurolink configuration via pydantic-settings."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All environment-variable-driven settings for Neurolink."""

    model_config = SettingsConfigDict(
        env_prefix="NEUROLINK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Adapter
    adapter_type: str = "mock"           # mock | ble | lsl
    device_model: str = "muse_s_gen1"    # muse_s_gen1 | muse_s_athena
    muse_ble_address: str = ""           # BLE MAC, required for ble mode

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Database
    db_path: str = "./data/neurolink.db"

    # CORS
    cors_origins: str = "http://localhost:5173"

    # Logging
    log_json: bool = False

    # EEG mode (backward compat)
    eeg_mode: str = "ble"

    # Publish cadence (Hz)
    publish_hz: float = 4.0

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def db_url(self) -> str:
        """Return async SQLite URL."""
        return f"sqlite+aiosqlite:///{self.db_path}"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
