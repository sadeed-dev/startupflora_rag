"""Request-local stage timings without logging prompts, answers, or credentials."""



import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps


_current = ContextVar("rag_request_timings", default=None)
_deadline = ContextVar("rag_request_deadline", default=None)
_purpose = ContextVar("rag_model_purpose", default=None)
log = logging.getLogger("uvicorn.error")



@contextmanager
def without_request_timings():
    """Exclude telemetry database writes from the request being measured."""
    token = _current.set(None)
    try:
        yield
    finally:
        _current.reset(token)


def record_answer_cache(status, *, reason=None, details=None):
    """Report cache decisions without recording questions or user identities."""
    trace = _current.get()
    if trace is not None:
        trace["answer_cache"] = status
        if reason is not None:
            trace["answer_cache_reason"] = reason
        if details is not None:
            trace.setdefault("answer_cache_details", {}).update(details)


def record_answer_diagnostic(details):
    """Keep answer decisions inspectable without logging questions or passage text."""
    trace = _current.get()
    if trace is not None:
        trace["answer_diagnostic"] = details
    log.info("rag_answer_diagnostic %s", json.dumps({
        "request_id": trace["request_id"] if trace is not None else None,
        **details,
    }, sort_keys=True))


def record_call_policy(minimum_calls, retry_limit):
    """Make automatic HTTP retry limits explicit beside model invocation counts."""
    trace = _current.get()
    if trace is not None:
        trace["minimum_calls"] = minimum_calls
        trace["automatic_retry_limit"] = retry_limit



def record_token_usage(provider, model, operation, usage, *, status="success"):
    """Log provider-reported tokens; missing usage is unknown, not zero billing."""
    trace = _current.get()
    usage = usage or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if operation == "embedding" and input_tokens is not None:
        output_tokens = 0
    reported = input_tokens is not None and output_tokens is not None
    entry = {
        "request_id": trace["request_id"] if trace is not None else None,
        "provider": provider, "model": model, "operation": operation, "status": status,
        "purpose": _purpose.get() or operation,
        "automatic_retry_limit": trace.get("automatic_retry_limit") if trace is not None else None,
        "usage_reported": reported,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens", input_tokens + output_tokens if reported else None),
    }
    if trace is not None:
        trace["token_usage"].append(entry)
    log.info("rag_token_usage %s", json.dumps(entry, sort_keys=True))


@contextmanager
def request_deadline(seconds):
    """Propagate the HTTP deadline into synchronous worker threads."""
    token = _deadline.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _deadline.reset(token)


def ensure_request_active():
    """Prevent an abandoned request from saving a late answer after its deadline."""
    deadline = _deadline.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Response deadline exceeded")


@contextmanager
def stage(name):
    purpose = _purpose.set(name) if name == "generation" or name == "embedding" or name.startswith("model_") else None
    trace = _current.get()
    if trace is None:
        try:
            yield
        finally:
            if purpose is not None:
                _purpose.reset(purpose)
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        if purpose is not None:
            _purpose.reset(purpose)
        trace["stages_ms"][name] = trace["stages_ms"].get(name, 0) + (time.perf_counter() - start) * 1000
        trace["calls"][name] = trace["calls"].get(name, 0) + 1


def timed(name):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            with stage(name):
                return function(*args, **kwargs)
        return measured
    return decorate


@contextmanager
def request_timings(request_id):
    trace = {"request_id": str(request_id), "stages_ms": {}, "calls": {},
             "token_usage": [], "answer_cache": "bypass"}
    token = _current.set(trace)
    start = time.perf_counter()
    try:
        yield trace
    finally:
        trace["total_ms"] = round((time.perf_counter() - start) * 1000, 1)
        trace["stages_ms"] = {name: round(value, 1) for name, value in trace["stages_ms"].items()}
        _current.reset(token)
        log.info("rag_request_timing %s", json.dumps(trace, sort_keys=True))
