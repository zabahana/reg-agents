"""A lightweight, faithful implementation of the Agent-to-Agent (A2A) protocol.

A2A (introduced by Google, now a Linux Foundation project) lets independent
agents advertise capabilities via an **Agent Card** and exchange work via a
JSON-RPC `message/send` method that returns a `Task` with `artifacts`.

We implement the subset needed for the demo:
  - GET  /.well-known/agent-card.json  -> AgentCard
  - POST /                             -> JSON-RPC 2.0 { method: "message/send" }

Each of our agents is an A2A **server**; the orchestrator is an A2A **client**
that composes them. Inside an agent, tool calls go out over MCP (see mcp_client).
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool


# --------------------------------------------------------------------------- #
# Data model (A2A-aligned)
# --------------------------------------------------------------------------- #
class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: List[str] = Field(default_factory=list)


class AgentCard(BaseModel):
    name: str
    description: str
    version: str = "0.1.0"
    url: str = ""
    skills: List[AgentSkill] = Field(default_factory=list)
    default_input_modes: List[str] = Field(default_factory=lambda: ["text"])
    default_output_modes: List[str] = Field(default_factory=lambda: ["text"])


class TextPart(BaseModel):
    kind: str = "text"
    text: str


class Message(BaseModel):
    role: str = "user"  # "user" | "agent"
    parts: List[TextPart] = Field(default_factory=list)
    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    @classmethod
    def text(cls, text: str, role: str = "user") -> "Message":
        return cls(role=role, parts=[TextPart(text=text)])

    def as_text(self) -> str:
        return "\n".join(p.text for p in self.parts if p.kind == "text")


class Artifact(BaseModel):
    artifact_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "result"
    parts: List[TextPart] = Field(default_factory=list)

    def as_text(self) -> str:
        return "\n".join(p.text for p in self.parts if p.kind == "text")


class Task(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "completed"  # submitted | working | completed | failed
    artifacts: List[Artifact] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Server helper
# --------------------------------------------------------------------------- #
Handler = Callable[[Message, Dict[str, Any]], "Task | str"]


def build_a2a_app(card: AgentCard, handler: Handler) -> FastAPI:
    """Wrap a handler(message, metadata) -> Task|str in an A2A FastAPI app."""
    app = FastAPI(title=card.name, version=card.version)

    # Prometheus /metrics endpoint (request rate, latency, in-progress, errors)
    # for every agent, scraped by kube-prometheus-stack and shown in Grafana.
    # Optional so local runs work even if the instrumentator isn't installed.
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    except Exception:  # noqa: BLE001 - metrics are best-effort
        pass

    @app.get("/.well-known/agent-card.json")
    def agent_card() -> Dict[str, Any]:
        return card.model_dump()

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "agent": card.name}

    @app.post("/")
    async def rpc(request: Request) -> JSONResponse:
        body = await request.json()
        rpc_id = body.get("id", 1)
        method = body.get("method")
        params = body.get("params", {})
        if method != "message/send":
            return JSONResponse(
                {"jsonrpc": "2.0", "id": rpc_id,
                 "error": {"code": -32601, "message": f"Method not found: {method}"}}
            )
        msg = Message(**params.get("message", {"parts": [{"text": ""}]}))
        metadata = params.get("metadata", {}) or {}
        try:
            # Handlers are sync and make blocking MCP/A2A calls; run them off the
            # event loop so nested asyncio (MCP SSE client) works cleanly.
            result = await run_in_threadpool(handler, msg, metadata)
            task = result if isinstance(result, Task) else Task(
                artifacts=[Artifact(parts=[TextPart(text=str(result))])]
            )
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": task.model_dump()})
        except Exception as exc:  # noqa: BLE001 - surface as JSON-RPC error
            return JSONResponse(
                {"jsonrpc": "2.0", "id": rpc_id,
                 "error": {"code": -32000, "message": str(exc)}}
            )

    return app


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class A2AClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_card(self) -> AgentCard:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(f"{self.base_url}/.well-known/agent-card.json")
            r.raise_for_status()
            return AgentCard(**r.json())

    def send(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Task:
        payload = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "message/send",
            "params": {
                "message": Message.text(text).model_dump(),
                "metadata": metadata or {},
            },
        }
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(f"{self.base_url}/", json=payload)
            r.raise_for_status()
            data = r.json()
        if "error" in data:
            raise RuntimeError(f"A2A error from {self.base_url}: {data['error']}")
        return Task(**data["result"])

    def send_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        task = self.send(text, metadata)
        return "\n".join(a.as_text() for a in task.artifacts)
