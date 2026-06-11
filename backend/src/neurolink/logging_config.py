"""Structlog configuration for Neurolink.

Call configure_logging() once at startup (in lifespan).
"""
from __future__ import annotations

import logging
import sys
from typing import Any, MutableMapping, Union

import structlog

# Type alias that satisfies structlog's Processor signature for mypy
_Renderer = Union[
    structlog.processors.JSONRenderer,
    structlog.dev.ConsoleRenderer,
]


def configure_logging(log_json: bool = False, log_level: str = "INFO") -> None:
    """Configure structlog for the application.

    Args:
        log_json: If True, emit JSON-formatted log lines.
        log_level: Python logging level name.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: _Renderer
    if log_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,  # type: ignore[arg-type]
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
