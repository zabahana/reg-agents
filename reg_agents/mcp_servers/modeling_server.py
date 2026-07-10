"""MCP server: model development bake-off (first-line data science).

Exposes the candidate-model training + champion-selection workflow as MCP tools
so the Developer agent can "develop" a model the same way it would call any
governed tool. On GPU the estimators map to RAPIDS cuML / XGBoost.

Run:  python -m reg_agents.mcp_servers.modeling_server  (PORT default 9104)
"""

# NOTE: no `from __future__ import annotations` -- FastMCP needs real annotation
# types when registering tools.

import json
import os

from mcp.server.fastmcp import FastMCP

from reg_agents.common import modeling

mcp = FastMCP("modeling", host="0.0.0.0", port=int(os.getenv("PORT", "9104")))


@mcp.tool()
def list_tasks() -> str:
    """List available model-development tasks (id, name, objective)."""
    return json.dumps(
        [
            {"task_id": t.task_id, "name": t.name, "objective": t.objective}
            for t in modeling.TASKS.values()
        ],
        indent=2,
    )


@mcp.tool()
def list_candidate_models(task_id: str = "fraud") -> str:
    """List the candidate model families considered for a task."""
    try:
        return json.dumps(modeling.list_candidate_models(task_id), indent=2)
    except KeyError as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def run_model_bakeoff(task_id: str = "fraud", test_size: float = 0.3) -> str:
    """Train all candidate models, evaluate on a hold-out, return the leaderboard.

    Returns JSON: {task, dataset, primary_metric, leaderboard[], champion,
    selection_rationale}.
    """
    try:
        return json.dumps(modeling.run_bakeoff(task_id, test_size=test_size), indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})


@mcp.tool()
def get_champion(task_id: str = "fraud") -> str:
    """Return only the selected champion model + selection rationale for a task."""
    try:
        result = modeling.run_bakeoff(task_id)
        return json.dumps(
            {
                "task_id": task_id,
                "champion": result["champion"],
                "primary_metric": result["primary_metric"],
                "selection_rationale": result["selection_rationale"],
            },
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})


if __name__ == "__main__":
    mcp.run(transport="sse")
