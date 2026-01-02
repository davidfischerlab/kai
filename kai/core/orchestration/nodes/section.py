"""Section execution node functions."""

from typing import Any, Dict

from kai.core.orchestration.state import SectionState
from kai.utils import setup_logger

logger = setup_logger(__name__)


def section_check_position(state: SectionState) -> Dict[str, Any]:
    """Check if we've completed all cells in the section."""
    current = state.get("current_cell_index", state.get("start_cell", 0))
    end = state.get("end_cell", 0)
    is_complete = current > end
    logger.debug(f"[SECTION] Position check: cell {current}/{end}, complete={is_complete}")
    return {"execution_complete": is_complete}


def section_route_from_position_check(state: SectionState) -> str:
    """Route based on position check result."""
    if state.get("execution_complete", False):
        return "complete"
    return "execute"


def section_execute_cell(state: SectionState) -> Dict[str, Any]:
    """Execute current cell (placeholder - actual execution via VSCode)."""
    cell_index = state.get("current_cell_index", 0)
    start_cell = state.get("start_cell", 0)
    actual_index = start_cell + cell_index
    logger.info(f"[SECTION] Executing cell {actual_index}")
    return {"current_cell_index": cell_index}


def section_check_execution(state: SectionState) -> Dict[str, Any]:  # noqa: ARG001
    """Check if last execution succeeded or failed."""
    return {}


def section_route_from_execution_check(state: SectionState) -> str:
    """Route based on execution result."""
    if state.get("last_execution_failed", False):
        return "error"
    return "success"


def section_advance_cell(state: SectionState) -> Dict[str, Any]:
    """Advance to next cell after successful execution."""
    current = state.get("current_cell_index", 0)
    new_index = current + 1
    logger.debug(f"[SECTION] Advancing from cell {current} to {new_index}")
    return {
        "current_cell_index": new_index,
        "last_execution_failed": False,
        "current_error": None,
        "error_cell_index": None,
    }


def section_route_fix_operation(state: SectionState) -> str:
    """Route to appropriate fix operation based on section review."""
    fix_attempts = state.get("fix_attempts", [])
    max_attempts = state.get("max_fix_attempts", 3)

    if len(fix_attempts) >= max_attempts:
        logger.warning(f"[SECTION] Max fix attempts ({max_attempts}) exceeded")
        return "fail"

    last_result = state.get("_last_tool_result", {})
    result = last_result.get("result")

    if not result or not result.output_ui:
        logger.error("[SECTION] No fix decision from section_code_review")
        return "fail"

    fix_decision = result.output_ui
    operation = fix_decision.get("operation", "").lower()

    if operation not in ("delete", "replace", "insert"):
        logger.error(f"[SECTION] Invalid operation: {operation}")
        return "fail"

    logger.info(f"[SECTION] Fix operation: {operation}")
    return operation


def section_apply_delete(state: SectionState) -> Dict[str, Any]:
    """Apply delete fix - remove cells from section."""
    last_result = state.get("_last_tool_result", {})
    result = last_result.get("result")
    fix_decision = result.output_ui if result else {}

    position = fix_decision.get("position", [])
    if not isinstance(position, list):
        position = [position]

    section_code = list(state.get("section_code", []))
    end_cell = state.get("end_cell", 0)

    fix_attempts = list(state.get("fix_attempts", []))
    fix_attempts.append({
        "operation": "delete",
        "position": position,
        "intent": fix_decision.get("intent", ""),
        "reasoning": fix_decision.get("reasoning", ""),
    })

    success = True
    for cell_idx in sorted(position, reverse=True):
        if 0 <= cell_idx < len(section_code):
            section_code.pop(cell_idx)
        else:
            success = False

    new_end = max(0, end_cell - len(position))
    logger.info(f"[SECTION] Deleted cells {position}, new section size: {len(section_code)}")

    return {
        "section_code": section_code,
        "end_cell": new_end,
        "fix_attempts": fix_attempts,
        "fix_applied": success,
        "fix_decision": fix_decision,
    }


def section_apply_replace(state: SectionState) -> Dict[str, Any]:
    """Apply replace fix - replace cells with new code."""
    last_result = state.get("_last_tool_result", {})
    result = last_result.get("result")
    fix_decision = result.output_ui if result else {}

    position = fix_decision.get("position", [])
    if not isinstance(position, list):
        position = [position]

    fix_attempts = list(state.get("fix_attempts", []))
    fix_attempts.append({
        "operation": "replace",
        "position": position,
        "intent": fix_decision.get("intent", ""),
        "reasoning": fix_decision.get("reasoning", ""),
    })

    logger.info(f"[SECTION] Replace requested for cells {position}")

    return {
        "fix_attempts": fix_attempts,
        "fix_applied": True,
        "fix_decision": fix_decision,
    }


def section_apply_insert(state: SectionState) -> Dict[str, Any]:
    """Apply insert fix - add new cell at position."""
    last_result = state.get("_last_tool_result", {})
    result = last_result.get("result")
    fix_decision = result.output_ui if result else {}

    position = fix_decision.get("position", 0)
    if isinstance(position, list):
        position = position[0] if position else 0

    fix_attempts = list(state.get("fix_attempts", []))
    fix_attempts.append({
        "operation": "insert",
        "position": position,
        "intent": fix_decision.get("intent", ""),
        "reasoning": fix_decision.get("reasoning", ""),
    })

    logger.info(f"[SECTION] Insert requested at position {position}")

    return {
        "fix_attempts": fix_attempts,
        "fix_applied": True,
        "fix_decision": fix_decision,
    }


def section_check_fix_result(state: SectionState) -> Dict[str, Any]:  # noqa: ARG001
    """Check if fix was successfully applied."""
    return {
        "last_execution_failed": False,
        "current_error": None,
    }


def section_route_from_fix_check(state: SectionState) -> str:
    """Route based on fix application result."""
    if state.get("fix_applied", False):
        return "retry"
    return "fail"


def section_complete_success(state: SectionState) -> Dict[str, Any]:
    """Mark section execution as successful."""
    start = state.get("start_cell", 0)
    end = state.get("end_cell", 0)
    logger.info(f"[SECTION] Successfully executed cells {start} to {end}")
    return {"execution_complete": True, "execution_success": True}


def section_complete_failure(state: SectionState) -> Dict[str, Any]:
    """Mark section execution as failed."""
    current = state.get("current_cell_index", 0)
    fix_attempts = state.get("fix_attempts", [])
    logger.error(
        f"[SECTION] Execution failed at cell {current} after {len(fix_attempts)} fix attempts"
    )
    return {"execution_complete": True, "execution_success": False}
