"""Thin MCP (Model Context Protocol) client wrapper.

Agents use this to call tools exposed by our MCP servers over SSE transport.
MCP is Anthropic's open standard for exposing tools/resources to LLM apps;
here it decouples *what a tool does* (regulation search, model-registry lookup,
fraud scoring) from *which agent uses it*. Any MCP-compatible client (Claude
Desktop, Cursor, our agents) can reuse the same servers.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from mcp import ClientSession
from mcp.client.sse import sse_client


async def _call_tool_async(url: str, tool: str, arguments: Dict[str, Any]) -> str:
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments)
            chunks: List[str] = []
            for item in result.content:
                text = getattr(item, "text", None)
                if text is not None:
                    chunks.append(text)
            return "\n".join(chunks)


async def _list_tools_async(url: str) -> List[str]:
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [t.name for t in tools.tools]


def _run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Running inside an event loop (e.g. uvicorn): use a dedicated loop.
    return asyncio.new_event_loop().run_until_complete(coro)


def call_tool(url: str, tool: str, arguments: Dict[str, Any]) -> str:
    """Synchronous convenience wrapper used inside A2A request handlers."""
    return _run(_call_tool_async(url, tool, arguments))


def list_tools(url: str) -> List[str]:
    return _run(_list_tools_async(url))
