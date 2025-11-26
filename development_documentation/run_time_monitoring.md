# Runtime Cell Execution Monitoring

## Overview

The runtime monitoring system detects and terminates long-running cells that are stuck in infinite loops or non-progressing computations. It uses periodic LLM-based analysis of partial outputs to make intelligent termination decisions, then feeds the termination context back into the autonomous workflow for automatic recovery.

## Architecture

### Components

1. **ExecutionMonitorTool** ([prompt_tools.py:914-943](../kai/core/orchestration/prompt_tools.py#L914-L943))
   - Structured prompt tool using `ExecutionMonitor` schema
   - Analyzes cell code, elapsed time, and partial outputs
   - Returns: `{action: "continue"|"terminate", feedback: "reasoning"}`
   - Uses: `large_llm` with `medium` reasoning level

2. **TypeScript Progress Monitoring** ([notebook-operations.ts:795-831](../UI/vscode/src/providers/notebook-operations.ts#L795-L831))
   - Checks progress every 5 minutes during cell execution
   - Captures partial outputs via `getCellOutputsAsText()`
   - Sends to Python for LLM analysis
   - Interrupts cell if LLM decides to terminate

3. **Termination State Tracking** ([notebook-operations.ts:392-417](../UI/vscode/src/providers/notebook-operations.ts#L392-L417))
   - `_lastTerminatedCellIndex`: Tracks which cell was terminated
   - `_terminationReason`: Stores LLM's feedback for why
   - Cleared after autonomous execution handles it

4. **Autonomous Integration** ([autonomous-execution.ts:390-403](../UI/vscode/src/providers/autonomous-execution.ts#L390-L403))
   - Detects termination via `lastTerminatedCellIndex`
   - Sets `_lastExecutionFailed = true` (triggers retry branch)
   - Creates error message with feedback + partial outputs

## Communication Flow

```
Cell runs 5+ min → TypeScript monitors progress
                ↓
TypeScript → Python: {code, elapsed_time, partial_outputs}
                ↓
ExecutionMonitorTool analyzes with LLM
                ↓
LLM decides: "terminate" + feedback explaining why
                ↓
TypeScript interrupts cell, sets termination flags
                ↓
Autonomous execution detects → treats as error
                ↓
Next iteration: AutonomousMarkCompletionTool sees termination
                ↓
Sets retry_objective → standard retry branch
                ↓
RAG retrieval with feedback + error message
                ↓
Code generation fixes the issue
```

## Error Message Format

When a cell is terminated, the error message passed to the next iteration is:

```
[EXECUTION TERMINATED BY MONITORING AGENT]
{LLM feedback explaining why termination occurred}

Partial outputs:
{Actual outputs captured before termination}
```

**Example:**
```
[EXECUTION TERMINATED BY MONITORING AGENT]
Cell appears stuck in infinite loop - no progress in outputs for 5+ minutes.
The loop variable is not being updated correctly.

Partial outputs:
Processing item 1...
Processing item 2...
Processing item 2...
Processing item 2...
```

## Data Flow Through System

### TypeScript → Python

```typescript
// 1. TypeScript creates termination message
this._lastExecutionOutput =
  `[EXECUTION TERMINATED BY MONITORING AGENT]\n` +
  `${this.notebookOps.terminationReason}\n\n` +
  `Partial outputs:\n${this.notebookOps.formatCellOutputToString(notebookCell)}`;

// 2. Sent to Python as
context.executionResult = this._lastExecutionOutput;
context.lastExecutionFailed = true;
```

### Python Processing

```python
# 3. agent.py:212-215 - Preserves full termination message
if context_data['execution_result'].startswith("[EXECUTION TERMINATED BY MONITORING AGENT]"):
    context_data['error_message'] = context_data['execution_result']  # Keep everything!

# 4. prompt_manager.py:1719-1728 - Builds error section
error_message = exec_context.inputs.context["error_message"]
# Includes: termination marker + feedback + partial outputs

# 5. AutonomousMarkCompletionTool sees full context
# LLM analyzes termination and sets:
{
  "tasks": [...],  # Active task stays active
  "retry_objective": "Fix infinite loop - add iteration limit"
}

# 6. workflow_orchestrator.py:464-468 - RAG with both
snippet_retrieval_query = [
    context['retry_objective'],      # High-level goal
    context['error_message']          # Detailed feedback + outputs
]
```

## LLM Prompt Configuration

### System Prompt ([prompt_manager.py:536-552](../kai/core/prompt_manager.py#L536-L552))

```
A cell has been executing for an extended period. Analyze the partial outputs
and code to determine if execution should continue or be terminated.

Indicators that execution should CONTINUE:
- Outputs show clear progress (incrementing values, progress bars)
- Known long-running operations (large matrix computations)
- Normal processing patterns (no errors, sensible values)

Indicators that execution should be TERMINATED:
- No new outputs for extended period (stuck/frozen)
- Repeated identical outputs suggesting infinite loop
- Memory warnings or resource exhaustion messages
- Error messages in outputs
- Non-progressing computation (same values repeating)

Be conservative - only terminate if there's clear evidence the cell is stuck.
```

### User Template ([prompt_manager.py:1234-1244](../kai/core/prompt_manager.py#L1234-L1244))

Uses `execution_monitor_section` built by `_build_execution_monitor_section()`:

```
=== Cell code currently executing:
```python
{code}
```

=== Execution information:
- Elapsed time: {elapsed_time} seconds
- Active task: {active_task}

=== Partial outputs so far:
{partial_outputs}

{notebook_structure_section}
```

## Configuration

### Monitoring Parameters

- **Check Interval**: Every 5 minutes ([notebook-operations.ts:778](../UI/vscode/src/providers/notebook-operations.ts#L778))
- **Max Cell Execution Time**: 30 minutes ([notebook-operations.ts:776](../UI/vscode/src/providers/notebook-operations.ts#L776))
- **Output Capture Limit**: 5000 characters ([notebook-operations.ts:897](../UI/vscode/src/providers/notebook-operations.ts#L897))
- **Progress Check Timeout**: 30 seconds ([agent-provider.ts:531](../UI/vscode/src/providers/agent-provider.ts#L531))

### LLM Pool Settings ([llm_pool.py:41,68](../kai/core/llm_pool.py#L41))

```python
"tool_llm_mapping": {
    "ExecutionMonitorTool": "large_llm"  # Uses gpt-oss:20b (local) or gpt-oss:120b (turbo)
}

"tool_reasoning_mapping": {
    "ExecutionMonitorTool": "medium"  # Balance between analysis depth and speed
}
```

## Message Types

### Python Subprocess ([python-subprocess.py:214-237](../UI/vscode/src/python-subprocess.py#L214-L237))

**Request:**
```json
{
  "type": "execution_progress_check",
  "request_id": "check_123456",
  "context": {
    "current_cell": "while True:\n    process()",
    "elapsed_time": 300,
    "partial_outputs": "Processing...\nProcessing...",
    "active_task": "Process the dataset"
  }
}
```

**Response:**
```json
{
  "type": "execution_progress_check_response",
  "request_id": "check_123456",
  "action": "terminate",
  "feedback": "Cell stuck in infinite loop - no progress detected"
}
```

## Why This Design Works

1. **Rich Context**: Full termination feedback + partial outputs flow through error message
2. **No Information Loss**: Both LLM's analysis AND actual outputs preserved
3. **Standard Retry Path**: Uses existing error recovery infrastructure
4. **Conservative Default**: Only terminates with clear evidence of stuck execution
5. **Double RAG**: Both `retry_objective` (high-level) and `error_message` (detailed) used for retrieval
6. **Clean State Management**: Termination flags cleared after use, no pollution

## Example Scenario

**Cell Code:**
```python
for item in data:
    if item > 0:
        process(item)
    # Missing: item iterator advancement
```

**After 5 Minutes:**
1. TypeScript captures outputs: `"Processing item 1...\nProcessing item 1...\nProcessing item 1..."`
2. Sends to Python for analysis
3. LLM analyzes: "Identical outputs repeating, no progress, infinite loop detected"
4. Returns: `{action: "terminate", feedback: "Loop variable not advancing - add iterator"}`
5. TypeScript interrupts cell
6. Next iteration: AutonomousMarkCompletionTool sees full context
7. Sets: `retry_objective: "Fix infinite loop by advancing iterator"`
8. RAG finds examples of loop iteration
9. Code generation produces fixed code:
```python
for i, item in enumerate(data):
    if item > 0:
        process(item)
    if i > MAX_ITERATIONS:
        break
```

## Fallback Behavior

- **Monitoring Failure**: Defaults to "continue" (conservative approach)
- **Agent Not Initialized**: Returns "continue" with explanation
- **No Outputs Yet**: Allows execution to continue
- **Non-text Outputs**: Notes "plots/images only" but doesn't terminate
