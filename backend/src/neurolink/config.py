"""Neurolink application settings.

All configuration via environment variables (NEUROLINK_* prefix).
Pydantic v2 BaseSettings with automatic validation.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

_settings = None


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_prefix="NEUROLINK_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Hardware adapter
    adapter_type: str = "mock"  # mock | ble | lsl
    device_model: str = "muse_s_gen1"  # muse_s_gen1 | muse_s_athena | mock
    muse_ble_address: str = ""  # BLE MAC address for direct BLE mode

    # EEG pump
    publish_hz: float = 4.0

    # Database
    db_path: str = "data/neurolink.db"  # relative path; use :memory: for tests

    # Redis (optional)
    redis_url: str = "redis://localhost:6379"
    redis_enabled: bool = False

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def get_settings() -> Settings:
    """Return the global Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
