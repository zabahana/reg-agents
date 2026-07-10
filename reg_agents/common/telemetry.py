"""OpenTelemetry distributed tracing for A2A hops (best-effort, AIQ-aligned).

Each agent is a FastAPI A2A server; the A2A/MCP clients use `httpx`. By
instrumenting both sides and propagating W3C trace-context over HTTP headers, a
single governance run shows up as one connected trace in Jaeger:

    orchestrator ─► validation-agent ─► (MCP) model-registry
                 ├► fraud-agent      ─► (MCP) fraud scoring
                 ├► retriever-agent  ─► (MCP) regulations search
                 └► report-agent

Tracing activates only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set (the
docker-compose `monitoring` profile points it at Jaeger). It is a silent no-op
otherwise, so bare `run_local.sh` / test runs are unaffected. Everything is
wrapped in try/except so a missing/broken exporter never breaks an agent.

NVIDIA's NeMo Agent Toolkit (AIQ) emits the same OTel spans, so this is the
drop-in seam to later route traces/eval through AIQ.
"""

from __future__ import annotations

import logging
import os
import threading

_lock = threading.Lock()
_setup_done = False


def _enabled() -> bool:
    return bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))


def setup(service_name: str = "reg-agents") -> None:
    """Idempotently configure the tracer provider + OTLP exporter, and
    instrument the outbound httpx client so A2A/MCP calls propagate context."""
    global _setup_done
    if not _enabled():
        return
    with _lock:
        if _setup_done:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            # Keep logs clean when the collector isn't reachable (e.g. compose
            # run without the `monitoring` profile): the batch processor logs a
            # full traceback per failed export otherwise. Traces are non-critical.
            for _name in (
                "opentelemetry.sdk.trace.export",
                "opentelemetry.exporter.otlp.proto.http.trace_exporter",
            ):
                logging.getLogger(_name).setLevel(logging.CRITICAL)

            provider = TracerProvider(
                resource=Resource.create(
                    {"service.name": os.getenv("OTEL_SERVICE_NAME", service_name)}
                )
            )
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            trace.set_tracer_provider(provider)
            HTTPXClientInstrumentor().instrument()
            _setup_done = True
        except Exception:  # noqa: BLE001 - tracing is best-effort
            pass


def instrument_fastapi(app, service_name: str = "reg-agents") -> None:
    """Attach server-side tracing to a FastAPI app (extracts incoming context)."""
    if not _enabled():
        return
    setup(service_name)
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception:  # noqa: BLE001 - tracing is best-effort
        pass
