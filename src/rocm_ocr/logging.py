"""Logging configuration and filters for ROCm OCR microservice."""

import logging
from typing import Any

from asgi_correlation_id import CorrelationIdFilter
from uvicorn.logging import AccessFormatter


class HealthCheckFilter(logging.Filter):
    """Filter suppressing routine 200 OK access logs for /healthz and /readyz."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for successful health check probes to prevent log spam."""
        msg = record.getMessage()
        return (
            "200" not in msg or "GET /healthz" not in msg and "GET /readyz" not in msg
        )


class CorrelationAccessFormatter(AccessFormatter):
    """AccessFormatter prepending correlation_id to the request line."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize AccessFormatter with correlation_id format string."""
        fmt = (
            "%(levelprefix)s [%(correlation_id)s] "
            '%(client_addr)s - "%(request_line)s" %(status_code)s'
        )
        super().__init__(fmt=fmt, **kwargs)


def setup_logging() -> None:
    """Attach correlation ID filter and health check filter to uvicorn loggers."""
    cid_filter = CorrelationIdFilter()
    health_filter = HealthCheckFilter()

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addFilter(health_filter)
    access_logger.addFilter(cid_filter)

    if not access_logger.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(health_filter)
        access_logger.addHandler(handler)
    for handler in access_logger.handlers:
        handler.addFilter(cid_filter)
        handler.setFormatter(CorrelationAccessFormatter())

    for name in ("uvicorn", "uvicorn.error", ""):
        log = logging.getLogger(name)
        log.addFilter(cid_filter)
        for h in log.handlers:
            h.addFilter(cid_filter)
