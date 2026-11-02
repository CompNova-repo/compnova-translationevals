"""
Structured error handling.

The previous API answered every failure with
    HTTPException(500, detail=str(e))
which forced clients to string-match a raw Python exception in order to know
whether a request was worth retrying - and leaked internals (paths, CUDA
messages) to the caller.

Everything now fails as an ApiError with a stable machine-readable code, an
HTTP status, and an explicit `retryable` flag.
"""
from typing import Any, Optional


class ApiError(Exception):
    """An error with a stable code and a known retry policy."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 500,
        retryable: bool = False,
        details: Optional[dict] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            }
        }
        if self.details:
            body["error"]["details"] = self.details
        return body


# --------------------------------------------------------------------------- #
# constructors for the failures this service actually produces
# --------------------------------------------------------------------------- #

def validation(message: str, **details) -> ApiError:
    return ApiError("VALIDATION_ERROR", message, 422, False, details or None)


def unauthorized(message: str = "missing or invalid X-API-Key") -> ApiError:
    return ApiError("UNAUTHORIZED", message, 401, False)


def not_found(message: str, **details) -> ApiError:
    return ApiError("NOT_FOUND", message, 404, False, details or None)


def conflict(message: str, **details) -> ApiError:
    return ApiError("CONFLICT", message, 409, False, details or None)


def payload_too_large(message: str, **details) -> ApiError:
    return ApiError("PAYLOAD_TOO_LARGE", message, 413, False, details or None)


def unsupported_media(message: str, **details) -> ApiError:
    return ApiError("UNSUPPORTED_AUDIO", message, 415, False, details or None)


def busy(message: str = "the service is busy, try again shortly", **details) -> ApiError:
    return ApiError("BUSY", message, 503, True, details or None)


def out_of_memory(message: str = "the GPU ran out of memory", **details) -> ApiError:
    return ApiError("GPU_OUT_OF_MEMORY", message, 503, True, details or None)


def insufficient_disk(message: str = "not enough free disk space", **details) -> ApiError:
    """The spool or the database volume is full.

    Deliberately NOT retryable: a client that keeps retrying will keep
    failing until an operator frees the disk.
    """
    return ApiError("INSUFFICIENT_DISK", message, 507, False, details or None)


def model_unavailable(message: str, **details) -> ApiError:
    return ApiError("MODEL_UNAVAILABLE", message, 503, True, details or None)


def timeout(message: str = "the evaluation took too long", **details) -> ApiError:
    return ApiError("TIMEOUT", message, 504, True, details or None)


def internal(message: str, **details) -> ApiError:
    return ApiError("INTERNAL_ERROR", message, 500, False, details or None)


# --------------------------------------------------------------------------- #
# mapping arbitrary exceptions onto the codes above
# --------------------------------------------------------------------------- #

_TRYABLE_MARKERS = (
    ("model_unavailable", model_unavailable),
    ("metricx checkout not found", model_unavailable),
    ("out of memory", out_of_memory),
    ("cuda error", busy),
    ("device-side assert", out_of_memory),
    ("no space left on device", out_of_memory),
    ("cannot allocate", out_of_memory),
    ("connection reset", busy),
    ("timed out", timeout),
    ("timeout", timeout),
    ("temporarily unavailable", busy),
    ("rate limit", busy),
)


def classify(exc: BaseException) -> ApiError:
    """Turn an arbitrary exception into an ApiError with a sensible code.

    The original message is kept - operators need it - but it is logged in
    full while the response carries a shortened, safe version.
    """
    if isinstance(exc, ApiError):
        return exc

    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()

    for marker, factory in _TRYABLE_MARKERS:
        if marker in lowered:
            return factory(text[:500])

    if isinstance(exc, FileNotFoundError):
        return validation(f"file not found: {exc}", path=str(exc))

    if isinstance(exc, MemoryError):
        return out_of_memory(str(exc)[:500])

    return internal(text[:500])
