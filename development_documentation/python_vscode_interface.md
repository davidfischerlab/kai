# Python-VSCode Communication Interface

This document describes the unified communication architecture between the Python agent and VSCode extension.

## Architecture Overview

The system uses a **dual-channel communication pattern**:
- **Promise-based requests**: VSCode → Python → Promise resolution (for request completion confirmation)
- **Real-time messaging**: Python → VSCode → UI updates (for workflow data and progress)

### Critical Timing Architecture

**The Core Problem**: VSCode notebook cell execution is asynchronous, but autonomous workflows need synchronous behavior to properly handle error recovery.

**The Solution**: **Message Queue with Async Coordination**
- Messages trigger async notebook execution chains
- The **MessageQueue** acts as a central completion register
- Messages are only removed after actual notebook execution completes
- The autonomous loop waits for an empty queue before proceeding

## Key Entry Points

### VSCode → Python (Outgoing)

**Starting Point**: `UI/vscode/src/providers/agent-provider.ts`

- **`sendRequest(type, payload)`**: Universal method for all Python communication
- **`sendRegularRequest(message, context)`**: Regular chat interactions
- **`handleAutonomousInitiation(message, context)`**: Start autonomous workflows
- **`handleAutonomousContinuation(context)`**: Continue autonomous execution
- **`stopAutonomousExecution()`**: Stop autonomous workflows

**Message Format**: `{"type": "chat"|"stop_autonomous", "request_id": "...", ...payload}`

### Python → VSCode (Incoming)

**Starting Point**: `UI/vscode/src/providers/agent-provider.ts:handleResponse()`

**Promise Resolution Messages** (confirm request processed):
- `{"type": "response", "request_id": "...", "response": {"status": "processed"}}`

**Real-time Messages** (actual workflow data):
- `{"type": "display", "response": {...}}` → Chat UI updates
- `{"type": "task_list_display", "response": {...}}` → Task list UI
- `{"type": "execute_code", "response": {...}}` → **Async chain**: Notebook execution
- `{"type": "workflow_result", "auto_loop_update": "LOOP_COMPLETE|LOOP_INCOMPLETE"}` → **Async chain**: Workflow control

## Message Flow & Async Coordination

### 1. Request Processing
```
VSCode Component → KaiAgentProvider.sendRequest() → Python stdin → Promise resolution
```

### 2. Real-time Updates (Sync Messages)
```
Python stdout → KaiAgentProvider.handleResponse() → messageCallback → ChatViewProvider._handleMessage() → Component delegation
```

### 3. Async Execution Chain (execute_code/auto_loop_update)
```
Python stdout → queueAndProcessMessage() → _handleMessage() → handleAutonomousCodeExecution() → executeCell()
     ↓
MessageQueue.addMessage()              ↓ (awaits VSCode execution completion)
     ↓                                 ↓
Autonomous loop waits              MessageQueue.removeMessage()
     ↓
Loop continues with execution results
```

## Component Integration

### VSCode Side

**`MessageQueue`** (`UI/vscode/src/providers/message-queue.ts`)
- **Central completion register** for async operations
- Tracks notebook execution completion, not just message processing start
- Enables synchronous autonomous loop behavior via async coordination

**`KaiAgentProvider`** (`UI/vscode/src/providers/agent-provider.ts`)
- **Async message processing chain** for execution-triggering messages
- Promise-based request management
- Message queue integration for completion tracking

**`ChatViewProvider`** (`UI/vscode/src/providers/chat-view-provider.ts`)
- Main coordinator for all chat interactions
- **Async message handlers** for execution operations
- Routes webview messages to appropriate components

**`AutonomousExecution`** (`UI/vscode/src/providers/autonomous-execution.ts`)
- **Synchronous behavior via async coordination**: Waits for empty message queue
- Handles `execute_code` messages with proper notebook execution waiting
- Updates UI buttons when workflows complete

**`ChatCore`** (`UI/vscode/src/providers/chat-core.ts`)
- Processes `display` and `task_list_display` messages
- Manages conversation history and context preparation

### Python Side

**Communication Layer**: `core/orchestration/vscode_communicator.py`
- **`send_tool_result()`**: Sends display/execute messages
- **`send_workflow_result()`**: Sends completion signals

**Workflow Layer**: `core/orchestration/workflow_orchestrator.py`
- All `_handle_*` methods return `None` (no direct return values)
- Results sent via `VSCodeCommunicator` messages

## Key Architectural Principles

### Synchronous Autonomous Loop via Async Coordination
- **Problem**: Notebook execution is async, but autonomous workflows need sync behavior
- **Solution**: The autonomous loop waits for the MessageQueue (central completion register) to be empty
- **Result**: Error recovery gets proper execution results because the loop doesn't proceed until execution completes

### Async Message Processing Chain
- **execute_code/auto_loop_update messages** trigger async chains that wait for notebook completion
- **display/task_list_display messages** are processed synchronously
- **Promise resolution messages** handle request lifecycle independently

### Single Source of Truth
- **MessageQueue** is the authoritative register of pending async operations
- **KaiAgentProvider** manages all Python communication and message routing
- **ChatViewProvider** coordinates components but doesn't process messages directly

## Adding New Message Types

1. **Python Side**: Use `VSCodeCommunicator.send_tool_result()` or add new method
2. **VSCode Side**: Add handler in `ChatViewProvider._handleMessage()`
3. **If async execution needed**: Ensure proper await chain and message queue integration
4. **Update Documentation**: Add to `KaiAgentProvider` class documentation

## Key Files

- `UI/vscode/src/providers/message-queue.ts` - Central completion register
- `UI/vscode/src/providers/agent-provider.ts` - Communication core & async coordination
- `UI/vscode/src/providers/chat-view-provider.ts` - Message routing & async handlers
- `UI/vscode/src/providers/autonomous-execution.ts` - Sync behavior via async coordination
- `core/orchestration/vscode_communicator.py` - Python message sending
- `core/orchestration/workflow_orchestrator.py` - Workflow execution