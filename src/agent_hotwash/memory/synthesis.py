"""Bounded generation retries with a cooldown shared by synthesis workers."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

_SAFE_HEADERS = frozenset(
    {
        "x-request-id",
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
    }
)
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class GenerationHTTPError(httpx.HTTPStatusError):
    def __init__(self, original, metadata):
        super().__init__(str(original), request=original.request, response=original.response)
        self.transport_metadata = metadata


class GenerationControl:
    def __init__(self, *, clock=time.monotonic, sleep=time.sleep):
        self.clock = clock
        self.sleep = sleep
        self.lock = threading.Lock()
        self.resume_at = 0.0

    def defer(self, seconds):
        with self.lock:
            self.resume_at = max(self.resume_at, self.clock() + seconds)

    def wait(self):
        while True:
            with self.lock:
                delay = self.resume_at - self.clock()
            if delay <= 0:
                return
            self.sleep(delay)


def _retry_delay(value, fallback, *, now=None):
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=UTC)
                return max(0.0, (date - (now or datetime.now(UTC))).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return fallback


def generate(request, credential, *, timeout_s=180, max_retries=3, control=None):
    control = control or GenerationControl()
    attempts = []
    # Let the guarded caller's hard wall-clock deadline govern total wait time;
    # a short per-chunk read timeout would otherwise abort a slow but active
    # response long before that deadline.
    timeout = httpx.Timeout(
        timeout_s,
        connect=min(30, timeout_s),
        read=timeout_s,
        write=min(30, timeout_s),
    )
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        for attempt in range(max_retries + 1):
            control.wait()
            deadline = time.monotonic() + timeout_s
            try:
                with client.stream(
                    "POST",
                    "https://api.openai.com/v1/responses",
                    json=request,
                    headers={
                        "Authorization": "Bearer " + credential,
                        # Avoid HTTPX streaming decompression failures on long
                        # Responses bodies; JSON payloads are small enough that
                        # transport compression is not needed here.
                        "Accept-Encoding": "identity",
                    },
                ) as streamed:
                    body = bytearray()
                    try:
                        chunks = streamed.iter_bytes()
                        for chunk in chunks:
                            body.extend(chunk)
                            if time.monotonic() > deadline:
                                raise httpx.ReadTimeout(
                                    "OpenAI Responses request exceeded its wall-clock deadline",
                                    request=streamed.request,
                                )
                    except httpx.DecodingError as exc:
                        headers = {
                            key: value
                            for key, value in streamed.headers.items()
                            if key.lower() in {"content-encoding", "content-length"}
                        }
                        raise httpx.DecodingError(
                            f"{str(exc)[:300]}; response headers={headers}; "
                            f"decoded body bytes before error={len(body)}",
                            request=streamed.request,
                        ) from exc
                    received = httpx.Response(
                        status_code=streamed.status_code,
                        headers=streamed.headers,
                        content=bytes(body),
                        request=streamed.request,
                    )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                attempts.append(
                    {
                        "http_status": None,
                        "headers": {},
                        "error_type": type(exc).__name__,
                    }
                )
                if attempt < max_retries:
                    control.defer(min(5.0, 2.0**attempt))
                    continue
                raise
            headers = {key: value for key, value in received.headers.items() if key.lower() in _SAFE_HEADERS}
            attempts.append({"http_status": received.status_code, "headers": headers})
            metadata = {"attempts": attempts, "retries": len(attempts) - 1}
            if received.status_code in _RETRY_STATUS and attempt < max_retries:
                delay = _retry_delay(received.headers.get("retry-after"), min(60.0, 2.0**attempt))
                control.defer(delay)
                continue
            try:
                received.raise_for_status()
            except httpx.HTTPStatusError as exc:
                try:
                    error = received.json().get("error")
                except (ValueError, AttributeError):
                    error = None
                if isinstance(error, dict):
                    metadata["provider_error"] = {
                        key: str(error[key])[:1000] for key in ("code", "type", "message") if key in error
                    }
                raise GenerationHTTPError(exc, metadata) from exc
            result = received.json()
            if isinstance(result, dict):
                result["_transport"] = metadata
            return result
    raise AssertionError("unreachable synthesis retry state")
