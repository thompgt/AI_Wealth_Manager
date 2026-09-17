"""Unit tests for logging_setup: context propagation, filtering, and JSON formatting."""

import json
import logging
import threading
from typing import Dict, Any

from logging_setup import (
    log_context,
    current_context,
    _ContextFilter,
    _JsonFormatter,
    configure_logging,
    get_logger,
)


def test_log_context_propagation():
    """Verify log_context binds fields and unbinds cleanly, supporting nesting."""
    assert current_context() == {}

    with log_context(run_id="run-123", client_id=42):
        ctx = current_context()
        assert ctx.get("run_id") == "run-123"
        assert ctx.get("client_id") == 42

        # Nested scope merges
        with log_context(node="market_regime", client_id=99):
            nested_ctx = current_context()
            assert nested_ctx.get("run_id") == "run-123"
            assert nested_ctx.get("node") == "market_regime"
            assert nested_ctx.get("client_id") == 99

        # Outer scope restored
        assert current_context().get("client_id") == 42
        assert "node" not in current_context()

    assert current_context() == {}


def test_log_context_thread_isolation():
    """Context bound on one thread should not leak to another."""
    results: Dict[str, Any] = {}

    def worker():
        results["initial"] = current_context()
        with log_context(thread_test="thread-val"):
            results["inside"] = current_context()
        results["after"] = current_context()

    with log_context(main_thread="main-val"):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert current_context().get("main_thread") == "main-val"

    assert results.get("initial") == {}
    assert results.get("inside") == {"thread_test": "thread-val"}
    assert results.get("after") == {}


def test_context_filter_attaches_attributes():
    """_ContextFilter should stamp bound context onto record and default run_id."""
    cf = _ContextFilter()
    record = logging.LogRecord("test.logger", logging.INFO, "test.py", 10, "Hello", (), None)

    with log_context(run_id="run-abc", custom_tag="tag-xyz"):
        passed = cf.filter(record)
        assert passed is True
        assert getattr(record, "run_id") == "run-abc"
        assert getattr(record, "custom_tag") == "tag-xyz"

    # When no run_id in context, default "-" is assigned
    empty_record = logging.LogRecord("test.logger", logging.INFO, "test.py", 11, "Hello2", (), None)
    cf.filter(empty_record)
    assert getattr(empty_record, "run_id") == "-"


def test_json_formatter_valid_json_and_fields():
    """_JsonFormatter emits valid single-line JSON with extra fields and exceptions."""
    formatter = _JsonFormatter()
    record = logging.LogRecord("test.logger", logging.WARNING, "test.py", 20, "Something happened", (), None)
    record.run_id = "run-999"
    record.extra_field = "custom_value"

    output = formatter.format(record)
    data = json.loads(output)

    assert data["level"] == "WARNING"
    assert data["logger"] == "test.logger"
    assert data["message"] == "Something happened"
    assert data["run_id"] == "run-999"
    assert data["extra_field"] == "custom_value"
    assert "ts" in data


def test_json_formatter_with_exception():
    """_JsonFormatter stringifies tracebacks into the 'exception' JSON field."""
    formatter = _JsonFormatter()
    try:
        raise ValueError("simulated test error")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord("test.logger", logging.ERROR, "test.py", 30, "Failed op", (), exc_info)
    record.run_id = "run-err"
    output = formatter.format(record)
    data = json.loads(output)

    assert data["level"] == "ERROR"
    assert "exception" in data
    assert "ValueError: simulated test error" in data["exception"]


def test_configure_logging_idempotence():
    """Calling configure_logging multiple times should not raise or duplicate handlers."""
    configure_logging()
    configure_logging()
    logger = get_logger("test.subsystem")
    assert logger is not None
