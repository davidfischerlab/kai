"""Checkpoint export utilities for debugging.

Replaces custom prompt logging (~300 LOC) with structured checkpoint export.
LangGraph checkpoints already capture prompts, responses, and state -
this module provides utilities to query and export them.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING

from kai.utils import setup_logger

if TYPE_CHECKING:
    from langgraph.pregel import CompiledStateGraph

logger = setup_logger(__name__)


class CheckpointDebugExporter:
    """Export debug traces from LangGraph checkpoints.

    Provides structured export of:
    - Full session traces with state at each step
    - Message history (prompts + responses)
    - Task list evolution
    - Generated code history
    - Execution events
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize exporter.

        Args:
            output_dir: Base directory for exported files
        """
        self.output_dir = output_dir or Path.home() / ".kai_agent" / "debug_exports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def export_session_trace(
        self,
        graph: 'CompiledStateGraph',
        session_id: str,
        format: str = "json",
    ) -> Path:
        """Export full session trace for debugging.

        Args:
            graph: Compiled LangGraph with checkpointer
            session_id: Session/thread ID to export
            format: Output format ("json", "markdown", or "html")

        Returns:
            Path to exported file
        """
        config = {"configurable": {"thread_id": session_id}}

        # Collect checkpoint history
        history = []
        try:
            async for state in graph.aget_state_history(config):
                checkpoint = {
                    "checkpoint_id": state.config.get(
                        "configurable", {}
                    ).get("checkpoint_id", ""),
                    "step": state.metadata.get("step", 0),
                    "timestamp": state.metadata.get("ts", ""),
                    "next_nodes": list(state.next) if state.next else [],
                    "state": self._serialize_state(state.values),
                }
                history.append(checkpoint)
        except Exception as e:
            logger.error(f"Failed to get checkpoint history: {e}")
            history = []

        # Create export
        export_data = {
            "session_id": session_id,
            "exported_at": datetime.now().isoformat(),
            "total_steps": len(history),
            "history": list(reversed(history)),  # Chronological order
        }

        # Write to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session_id}_{timestamp}"

        if format == "json":
            output_path = self.output_dir / f"{filename}.json"
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

        elif format == "markdown":
            output_path = self.output_dir / f"{filename}.md"
            md_content = self._format_as_markdown(export_data)
            with open(output_path, "w") as f:
                f.write(md_content)

        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Exported session trace to {output_path}")
        return output_path

    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize state for export, handling non-serializable values."""
        serialized = {}
        for key, value in state.items():
            try:
                # Test if JSON serializable
                json.dumps(value, default=str)
                serialized[key] = value
            except (TypeError, ValueError):
                # Convert to string representation
                serialized[key] = str(value)
        return serialized

    def _format_as_markdown(self, data: Dict[str, Any]) -> str:
        """Format export data as markdown for human reading."""
        lines = [
            f"# Session Trace: {data['session_id']}",
            f"",
            f"Exported: {data['exported_at']}",
            f"Total Steps: {data['total_steps']}",
            f"",
            "---",
            "",
        ]

        for checkpoint in data["history"]:
            lines.extend([
                f"## Step {checkpoint['step']}",
                f"",
                f"**Checkpoint ID:** {checkpoint['checkpoint_id']}",
                f"**Timestamp:** {checkpoint['timestamp']}",
                f"**Next Nodes:** {', '.join(checkpoint['next_nodes']) or 'END'}",
                f"",
                "### State",
                "",
                "```json",
                json.dumps(checkpoint["state"], indent=2, default=str)[:5000],
                "```",
                "",
                "---",
                "",
            ])

        return "\n".join(lines)

    async def get_state_at_step(
        self,
        graph: 'CompiledStateGraph',
        session_id: str,
        step: int,
    ) -> Optional[Dict[str, Any]]:
        """Get state at specific step for time-travel debugging.

        Args:
            graph: Compiled LangGraph with checkpointer
            session_id: Session/thread ID
            step: Step number to retrieve

        Returns:
            State dict at that step, or None if not found
        """
        config = {"configurable": {"thread_id": session_id}}

        try:
            async for state in graph.aget_state_history(config):
                if state.metadata.get("step") == step:
                    return state.values
        except Exception as e:
            logger.error(f"Failed to get state at step {step}: {e}")

        return None

    async def replay_from_checkpoint(
        self,
        graph: 'CompiledStateGraph',
        session_id: str,
        checkpoint_id: str,
    ) -> Dict[str, Any]:
        """Replay graph execution from a specific checkpoint.

        Useful for debugging - re-run from a known good state.

        Args:
            graph: Compiled LangGraph with checkpointer
            session_id: Session/thread ID
            checkpoint_id: Checkpoint to replay from

        Returns:
            Final state after replay
        """
        config = {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_id": checkpoint_id,
            }
        }

        # Resume execution from checkpoint
        result = await graph.ainvoke(None, config)
        return result


async def export_session_trace(
    graph: 'CompiledStateGraph',
    session_id: str,
    output_dir: Optional[Path] = None,
    format: str = "json",
) -> Path:
    """Convenience function for one-off export.

    Args:
        graph: Compiled LangGraph with checkpointer
        session_id: Session/thread ID to export
        output_dir: Output directory (default: ~/.kai_agent/debug_exports)
        format: Output format ("json" or "markdown")

    Returns:
        Path to exported file
    """
    exporter = CheckpointDebugExporter(output_dir)
    return await exporter.export_session_trace(graph, session_id, format)
