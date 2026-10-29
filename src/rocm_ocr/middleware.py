"""HTTP request resource limits."""

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_REQUEST_BYTES = 50 * 1024 * 1024
REQUEST_TOO_LARGE = "HTTP request exceeds the 50 MiB limit"


def _content_length_error(scope: Scope) -> JSONResponse | None:
    content_length = Headers(scope=scope).get("content-length")
    if content_length is None:
        return None
    try:
        size = int(content_length)
    except ValueError:
        size = -1
    if size < 0:
        return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
    if size > MAX_REQUEST_BYTES:
        return JSONResponse({"detail": REQUEST_TOO_LARGE}, status_code=413)
    return None


class RequestSizeLimitMiddleware:
    """Reject HTTP request bodies larger than 50 MiB."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        error = _content_length_error(scope)
        if error is not None:
            await error(scope, receive, send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_REQUEST_BYTES:
                    raise HTTPException(status_code=413, detail=REQUEST_TOO_LARGE)
            return message

        await self.app(scope, limited_receive, send)
