"""A2A protocol tests using FastAPI's TestClient (in-process, no network)."""

from fastapi.testclient import TestClient

from reg_agents.common.a2a import AgentCard, Artifact, Message, Task, TextPart, build_a2a_app


def _echo_handler(message: Message, metadata):
    return Task(artifacts=[Artifact(parts=[TextPart(text=f"echo: {message.as_text()}")])])


def test_agent_card_served():
    app = build_a2a_app(AgentCard(name="Test", description="t"), _echo_handler)
    client = TestClient(app)
    r = client.get("/.well-known/agent-card.json")
    assert r.status_code == 200
    assert r.json()["name"] == "Test"


def test_message_send_roundtrip():
    app = build_a2a_app(AgentCard(name="Test", description="t"), _echo_handler)
    client = TestClient(app)
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {"message": Message.text("hello").model_dump()},
    }
    r = client.post("/", json=payload)
    assert r.status_code == 200
    task = r.json()["result"]
    assert "echo: hello" in task["artifacts"][0]["parts"][0]["text"]


def test_unknown_method_errors():
    app = build_a2a_app(AgentCard(name="Test", description="t"), _echo_handler)
    client = TestClient(app)
    r = client.post("/", json={"jsonrpc": "2.0", "id": "1", "method": "bogus", "params": {}})
    assert r.json()["error"]["code"] == -32601
