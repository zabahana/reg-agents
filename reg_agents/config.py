"""Central configuration.

Everything is env-driven so the exact same code runs locally against OpenAI and
in the GCP GPU demo against NVIDIA NIM / NeMo Retriever / Triton.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # --- LLM provider selection -------------------------------------------
    llm_provider: str = Field(default="openai")  # "openai" | "nim"

    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_model: str = Field(default="gpt-4o-mini")
    openai_embed_model: str = Field(default="text-embedding-3-small")

    nim_api_key: str = Field(default="")
    nim_base_url: str = Field(default="https://integrate.api.nvidia.com/v1")
    nim_model: str = Field(default="meta/llama-3.1-8b-instruct")

    # --- Embeddings -------------------------------------------------------
    embedding_provider: str = Field(default="openai")  # "openai" | "nemo"
    nemo_embed_base_url: str = Field(default="https://integrate.api.nvidia.com/v1")
    nemo_embed_model: str = Field(default="nvidia/nv-embedqa-e5-v5")

    # --- Vector store -----------------------------------------------------
    vector_backend: str = Field(default="faiss")  # "faiss" | "milvus"
    milvus_uri: str = Field(default="http://localhost:19530")

    # --- Fraud model serving (Triton). Empty => local heuristic fallback --
    triton_url: str = Field(default="")

    # --- A2A agent endpoints ----------------------------------------------
    retriever_agent_url: str = Field(default="http://localhost:8101")
    validation_agent_url: str = Field(default="http://localhost:8102")
    fraud_agent_url: str = Field(default="http://localhost:8103")
    report_agent_url: str = Field(default="http://localhost:8104")

    # --- MCP tool servers (SSE transport) ---------------------------------
    regulations_mcp_url: str = Field(default="http://localhost:9101/sse")
    model_registry_mcp_url: str = Field(default="http://localhost:9102/sse")
    fraud_mcp_url: str = Field(default="http://localhost:9103/sse")

    @property
    def active_model(self) -> str:
        return self.nim_model if self.llm_provider == "nim" else self.openai_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
