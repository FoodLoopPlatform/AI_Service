import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

request_id_ctx_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_correlation_id() -> str | None:
    """Returns the current correlation request ID."""
    return request_id_ctx_var.get()


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to preserve or generate X-Request-ID for distributed tracing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        header_name = "X-Request-ID"
        request_id = request.headers.get(header_name) or uuid.uuid4().hex
        token = request_id_ctx_var.set(request_id)

        try:
            response = await call_next(request)
            response.headers[header_name] = request_id
            return response
        finally:
            request_id_ctx_var.reset(token)
