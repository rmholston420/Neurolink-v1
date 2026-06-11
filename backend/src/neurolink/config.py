"""Pydantic-settings configuration for Neurolink.

All configuration is read from environment variables with NEUROLINK_ prefix.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEUROLINK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Adapter
    adapter_type: str = Field(default="mock")       # "mock" | "ble" | "lsl"
    device_model: str = Field(default="muse_s_gen1") # "muse_s_gen1" | "muse_s_athena"
    muse_ble_address: str = Field(default="")        # BLE MAC address
    publish_hz: float = Field(default=4.0)

    # Database
    db_path: str = Field(default="./data/neurolink.db")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_enabled: bool = Field(default=False)

    # CORS
    cors_origins: str = Field(default="http://localhost:5173,http://localhost:3000")

    # Logging
    log_json: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # Auth (optional, disabled by default)
    auth_enabled: bool = Field(default=False)

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def db_url(self) -> str:
        """SQLAlchemy async DB URL."""
        path = self.db_path
        if path == ":memory:":
            return "sqlite+aiosqlite:///:memory:"
        return f"sqlite+aiosqlite:///{path}"


# Module-level mutable settings singleton (can be reset in tests)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached Settings instance.

    Can be reset by setting neurolink.config._settings = None in tests.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
