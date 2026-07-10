# Single image used by every MCP server and A2A agent; the command selects role.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps kept minimal; faiss-cpu ships wheels.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY reg_agents ./reg_agents
COPY data ./data
COPY scripts ./scripts
COPY pyproject.toml .

# Default: orchestrator. Override `command` per service in compose/k8s.
EXPOSE 8100
CMD ["python", "-m", "uvicorn", "reg_agents.agents.orchestrator:app", "--host", "0.0.0.0", "--port", "8100"]
