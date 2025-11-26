"""Run Section Workflow for executing and debugging notebook sections.

This workflow manages the execution of a specific range of cells, handling
errors through intelligent code review and minimal fixes. It focuses on
code maintenance and debugging rather than feature development.
"""

import asyncio
import json
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .deterministic_tools import CellDeletionTool

from .base_tool import BaseTool, ToolResult, ToolOutputType
from .prompt_tools import SectionCodeReviewTool, CodeUpdateTool, CodeGenerationTool
from kai.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class SectionExecutionContext:
    """Context for section execution and debugging."""
    start_cell: int
    end_cell: int  # inclusive
    current_cell_index: int
    section_code: List[str]  # Code content for each cell in the section
    fix_attempts: List[Dict] = None
    conversation_history: List[Dict] = None
    execution_history: List[Dict] = None
    current_error: Optional[str] = None  # Current error message from execution
    
    def __post_init__(self):
        if self.fix_attempts is None:
            self.fix_attempts = []
        if self.conversation_history is None:
            self.conversation_history = []
        if self.execution_history is None:
            self.execution_history = []


class RunSectionWorkflow:
    """Workflow for running a section of notebook cells with error recovery."""
    
    def __init__(self, llm_interface):
        self.llm = llm_interface

        # Initialize tools needed for section execution - pass LLMInterface for class-based routing
        self.section_review_tool = SectionCodeReviewTool(llm_interface)
        self.cell_deletion_tool = CellDeletionTool()
        self.code_fixing_tool = CodeUpdateTool(llm_interface)
        self.code_generation_tool = CodeGenerationTool(llm_interface)
    
    async def execute_section(
        self, 
        start_cell: int, 
        end_cell: int, 
        section_code: List[str],
        conversation_history: List[Dict] = None,
        execution_history: List[Dict] = None
    ) -> ToolResult:
        """Execute a section of cells, handling errors through intelligent recovery.
        
        Args:
            start_cell: Starting cell index (inclusive)
            end_cell: Ending cell index (inclusive) 
            section_code: List of code strings for each cell
            conversation_history: Context about what this code is supposed to do
            execution_history: Previous execution context
            
        Returns:
            ToolResult indicating success/failure of section execution
        """
        context = SectionExecutionContext(
            start_cell=start_cell,
            end_cell=end_cell,
            current_cell_index=start_cell,
            section_code=section_code,
            conversation_history=conversation_history or [],
            execution_history=execution_history or []
        )
        
        self._send_status_update("🔄 Running notebook section", "rerun_debugging")
        
        try:
            # Execute cells one by one until all succeed
            while context.current_cell_index <= context.end_cell:
                success = await self._execute_single_cell(context)
                
                if success:
                    context.current_cell_index += 1
                    if context.current_cell_index <= context.end_cell:
                        self._send_status_update(f"✅ Cell {context.current_cell_index - 1} completed, continuing...", "rerun_debugging")
                else:
                    # Handle error through intelligent recovery
                    recovery_success = await self._handle_cell_error(context)
                    
                    if not recovery_success:
                        # If we can't recover, fail the section
                        self._send_status_update("❌ Section execution failed - unable to recover", "rerun_debugging")
                        return ToolResult(
                            output=f"Section execution failed at cell {context.current_cell_index}",
                            output_type=ToolOutputType.TOOL_USAGE,
                            effects=["section_execution_failed"]
                        )
                    
                    # After successful recovery, retry current cell (don't increment)
            
            # All cells executed successfully
            self._send_status_update("✅ Section executed successfully", "rerun_debugging")
            return ToolResult(
                output=f"Successfully executed cells {start_cell} to {end_cell}",
                output_type=ToolOutputType.TOOL_USAGE,
                effects=["section_executed_successfully"],
                metadata={
                    "start_cell": start_cell,
                    "end_cell": end_cell,
                    "fix_attempts": context.fix_attempts
                }
            )
            
        except Exception as e:
            logger.error(f"Unexpected error in RunSectionWorkflow: {e}")
            self._send_status_update("❌ Section execution failed unexpectedly", "rerun_debugging")
            return ToolResult(
                output=f"Section execution failed: {str(e)}",
                output_type=ToolOutputType.TOOL_USAGE,
                effects=["section_execution_failed"]
            )
    
    async def _execute_single_cell(self, context: SectionExecutionContext) -> bool:
        """Execute a single cell and return whether it succeeded.
        
        Args:
            context: Current execution context
            
        Returns:
            True if cell executed successfully, False if error occurred
        """
        cell_index = context.current_cell_index
        if cell_index >= len(context.section_code):
            return False
        
        cell_code = context.section_code[cell_index]
        
        # Create VSCode command to execute this cell
        vscode_command = {
            "command": "executeCell", 
            "cellIndex": context.start_cell + cell_index,
            "code": cell_code
        }
        
        # Send execution command and wait for result
        # This is a simplified version - in reality we'd need to coordinate with VSCode
        # For now, we simulate execution and assume errors are provided via context
        
        # Check if this execution would result in an error (simulated)
        # In real implementation, this would be determined by actual execution results
        return True  # Placeholder - actual error detection happens through VSCode integration
    
    async def _handle_cell_error(self, context: SectionExecutionContext, error_message: str = None) -> bool:
        """Handle an error in cell execution through intelligent recovery.
        
        Args:
            context: Current execution context with error information
            error_message: The actual error message from execution
            
        Returns:
            True if recovery was successful, False if unable to recover
        """
        # Get error details
        error_cell = context.current_cell_index
        actual_error = error_message or context.current_error or "Error in cell execution"
        context.current_error = actual_error  # Store for later use
        
        self._send_status_update(f"🔍 Analyzing error in cell {error_cell}", "rerun_debugging")
        
        # Use section review tool to determine fix
        review_inputs = {
            "section_code": context.section_code,
            "error_cell": error_cell,
            "error_message": actual_error,
            "fix_attempts": context.fix_attempts,
            "conversation_history": context.conversation_history,
            "execution_history": context.execution_history
        }
        
        review_result = await self.section_review_tool.execute(review_inputs)
        
        if "error" in review_result.effects:
            logger.error(f"Section review failed: {review_result.output_ui}")
            return False
        
        # Extract the fix decision
        fix_decision = review_result.metadata
        operation = fix_decision.get("operation")
        position = fix_decision.get("position")
        intent = fix_decision.get("intent", "")
        reasoning = fix_decision.get("reasoning", "")
        
        # Record this fix attempt
        context.fix_attempts.append({
            "operation": operation,
            "position": position,
            "intent": intent,
            "reasoning": reasoning
        })
        
        self._send_status_update(f"🛠️ Applying fix: {operation} on position {position}", "rerun_debugging")
        
        # Apply the fix
        fix_success = await self._apply_fix(context, fix_decision)
        
        if fix_success:
            self._send_status_update("✅ Fix applied successfully", "rerun_debugging")
        else:
            self._send_status_update("❌ Fix application failed", "rerun_debugging")
        
        return fix_success
    
    async def _apply_fix(self, context: SectionExecutionContext, fix_decision: Dict) -> bool:
        """Apply the fix decision to the notebook section.
        
        Args:
            context: Current execution context
            fix_decision: Fix decision from section review tool
            
        Returns:
            True if fix was applied successfully
        """
        operation = fix_decision.get("operation")
        position = fix_decision.get("position")
        intent = fix_decision.get("intent", "")
        
        try:
            if operation == "delete":
                # Delete specified cells - position is a list of cell indices
                if not isinstance(position, list):
                    return False
                    
                delete_inputs = {"cells_to_delete": position}
                delete_result = await self.cell_deletion_tool.execute(delete_inputs)
                
                # Update section code by removing deleted cells
                for cell_idx in sorted(position, reverse=True):
                    if 0 <= cell_idx < len(context.section_code):
                        context.section_code.pop(cell_idx)
                
                # Adjust end_cell if we deleted cells
                context.end_cell = max(0, context.end_cell - len(position))
                
                return "error" not in delete_result.effects
                
            elif operation == "replace":
                # Replace code in specified cells - position is a list of cell indices
                if not isinstance(position, list) or not position:
                    return False
                
                # Use code fixing tool to generate replacement code
                fix_inputs = {
                    "current_cell": context.section_code[position[0]] if position[0] < len(context.section_code) else "",
                    "error_message": context.current_error or "Error in section execution",
                    "conversation_history": context.conversation_history,
                    "execution_history": context.execution_history,
                    "autonomous_mode": True,
                    "message": intent if intent else "Fix the code based on the error analysis"
                }
                
                fix_result = await self.code_fixing_tool.execute(fix_inputs)
                
                # Extract fixed code
                if "code" in fix_result.metadata:
                    fixed_code = fix_result.metadata["code"]
                    
                    # For replace: remove cells in list, add new cell at position of first deleted cell
                    first_cell_pos = min(position)
                    
                    # Remove cells from back to front to maintain indices
                    for cell_idx in sorted(position, reverse=True):
                        if 0 <= cell_idx < len(context.section_code):
                            context.section_code.pop(cell_idx)
                    
                    # Insert fixed code at the position of the first cell that was removed
                    context.section_code.insert(first_cell_pos, fixed_code)
                    
                    # Adjust end_cell: we removed len(position) cells and added 1
                    context.end_cell = context.end_cell - len(position) + 1
                
                return "error" not in fix_result.effects
                
            elif operation == "insert":
                # Insert new code at specified position - position is an integer
                if not isinstance(position, int) or position < 0:
                    return False
                
                # Use code generation tool to create new code
                gen_inputs = {
                    "message": intent if intent else "Generate code to fix the error based on the section review",
                    "conversation_history": context.conversation_history,
                    "execution_history": context.execution_history,
                    "autonomous_mode": True,
                    "error_context": context.current_error  # Provide error context for better code generation
                }
                
                gen_result = await self.code_generation_tool.execute(gen_inputs)
                
                # Extract generated code
                if "code" in gen_result.metadata:
                    new_code = gen_result.metadata["code"]
                    
                    # Insert into section code
                    context.section_code.insert(position, new_code)
                    
                    # Adjust end_cell since we added a cell
                    context.end_cell += 1
                
                return "error" not in gen_result.effects
                
            else:
                logger.error(f"Unknown operation: {operation}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying fix: {e}")
            return False
    
    def _send_status_update(self, message: str, status_type: str = "rerun_debugging"):
        """Send status update without disrupting main workflow todo lists."""
        status_message = {
            "type": "temporary_status",
            "message": message,
            "status_type": status_type,
            "timestamp": f"{asyncio.get_event_loop().time():.3f}"
        }
        print(json.dumps(status_message), file=sys.stdout, flush=True)