"""Unit tests for logging_config.configure_logging."""

from __future__ import annotations

import logging

import structlog

from neurolink.logging_config import configure_logging


def test_configure_logging_console():
    configure_logging(log_json=False, log_level="WARNING")
    root = logging.getLogger()
    assert root.level == logging.WARNING


def test_configure_logging_json():
    configure_logging(log_json=True, log_level="DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG


def test_configure_logging_default_level():
    configure_logging()
    root = logging.getLogger()
    assert root.level == logging.INFO


def test_configure_logging_clears_handlers():
    # Add a stray handler first
    logging.getLogger().addHandler(logging.NullHandler())
    configure_logging(log_json=False, log_level="INFO")
    # After configure, root should have exactly the one StreamHandler we add
    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)


def test_configure_logging_structlog_bound_logger():
    configure_logging(log_json=False)
    logger = structlog.get_logger("test")
    # Should not raise
    logger.info("test_message", key="value")
