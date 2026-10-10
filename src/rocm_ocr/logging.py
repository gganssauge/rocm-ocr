"""Logging configuration and filters for ROCm OCR microservice."""

import logging

from asgi_correlation_id import CorrelationIdFilter


class HealthCheckFilter(logging.Filter):
    """Filter suppressing routine 200 OK access logs for /healthz and /readyz."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for successful health check probes to prevent log spam."""
        msg = record.getMessage()
        return not ("200" in msg and ("GET /healthz" in msg or "GET /readyz" in msg))


def setup_logging() -> None:
    """Attach correlation ID filter and health check filter to uvicorn loggers."""
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addFilter(HealthCheckFilter())

    cid_filter = CorrelationIdFilter()
    access_logger.addFilter(cid_filter)
    logging.getLogger("uvicorn").addFilter(cid_filter)
    logging.getLogger().addFilter(cid_filter)
