# Core Module Architecture

## Overview

The core module implements a **LangGraph-based architecture** for the bioinformatics agent. The architecture uses state machines for tool orchestration and autonomous workflow execution.

## Architecture Layers

```
Layer 1: Public Interface
└── KaiAgent (agent.py)
    └── Main entry point, manages orchestrator

Layer 2: Graph-Based Orchestration
└── LangGraphOrchestrator (orchestration/langgraph_orchestrator.py)
    ├── Builds autonomous and regular execution graphs
    ├── Routes based on state conditions
    ├── Manages tool execution through graph nodes
    └── Returns VSCode-ready responses

Layer 3: Consolidated Tools
└── Tools (tools/*.py)
    ├── 15 consolidated MCP-compatible tools
    └── Each tool handles multi-step operations internally
```

## Complete Data Flow

```
VSCode Interface → KaiAgent → LangGraphOrchestrator → Graph Nodes → Tools → VSCode Response
```

## Current Implementation Files

```
core/
├── agent.py                           # KaiAgent
├── llm_interface.py                   # LLM provider interface
├── prompt_manager.py                  # Centralized prompt generation
├── tools/                             # Consolidated tools
│   ├── __init__.py                    # Tool registry
│   ├── code_generation.py             # Code generation tools
│   ├── task_management.py             # Task planning and progress
│   ├── error_handling.py              # Error recovery and backtracking
│   ├── workflow_search.py             # Reference workflow search
│   ├── rag.py                         # Code snippet retrieval
│   ├── execution.py                   # Cell execution tools
│   ├── interaction.py                 # User interaction tools
│   └── notebook.py                    # Notebook operations
└── orchestration/
    ├── base_tool.py                   # Base tool interface
    ├── langgraph_orchestrator.py      # Graph-based orchestrator
    ├── state.py                       # KaiState schema
    ├── tool_adapter.py                # Tool-to-graph adapter
    ├── prompt_tools.py                # LLM-based tools
    └── deterministic_tools.py         # Rule-based tools
```

## Component Responsibilities

### KaiAgent (`agent.py`)
- **Purpose**: Main entry point and system coordinator
- **Responsibilities**:
  - Initialize LangGraphOrchestrator with LLM and knowledge base
  - Provide unified chat interface
  - Parse context from VSCode interface
  - Manage system status
- **Dependencies**: LangGraphOrchestrator, LLMInterface

### LangGraphOrchestrator (`orchestration/langgraph_orchestrator.py`)
- **Purpose**: Graph-based request processing and tool orchestration
- **Responsibilities**:
  - Build execution graphs for autonomous and regular modes
  - Route between tools using conditional edges
  - Manage state propagation through graph
  - Execute tool sequences based on state conditions
- **Dependencies**: All consolidated tools, LangGraph
- **Key Methods**:
  - `_build_autonomous_graph()` - Creates graph for autonomous mode
  - `process_request()` - Main entry point for all requests
  - `_route_autonomous_action()` - Conditional routing logic

### Tool Adapter (`orchestration/tool_adapter.py`)
- **Purpose**: Bridge between BaseTool interface and LangGraph nodes
- **Responsibilities**:
  - Convert KaiState to ExecutionContext for tools
  - Convert ToolResult back to state updates
  - Wrap BaseTool.execute() as async graph node function

## Consolidated Tools Architecture

### Code Generation (`tools/code_generation.py`)
- **GenerateCodeTool**: Generates code with positioning and guidance
- **UpdateCodeTool**: Updates existing code cells

### Task Management (`tools/task_management.py`)
- **PlanTasksTool**: Creates and critiques task lists (internal loop)
- **ManageProgressTool**: Assesses progress, updates tasks, critiques, advances (2-3 LLM calls)

### Error Handling (`tools/error_handling.py`)
- **HandleErrorTool**: Analyzes errors and generates recovery code
- **BacktrackTool**: Removes problematic cells and repositions (2 LLM calls)

### Workflow Search (`tools/workflow_search.py`)
- **SearchWorkflowsTool**: 4-step pipeline for reference workflow retrieval

### RAG (`tools/rag.py`)
- **SearchCodeSnippetsTool**: Retrieves relevant code documentation

### Execution (`tools/execution.py`)
- **ExecuteCellTool**: Executes notebook cells
- **RestartAndRerunTool**: Restarts kernel and reruns cells

### Interaction (`tools/interaction.py`)
- **ClassifyIntentTool**: Classifies user intent
- **AnswerQuestionTool**: Answers user questions
- **ReviewCodeTool**: Reviews code quality
- **RespondWithReasoningTool**: Generates reasoning responses

### Notebook (`tools/notebook.py`)
- **NotebookOperationsTool**: Handles notebook metadata operations

## State Management

### KaiState Schema (`orchestration/state.py`)
TypedDict containing:
- `messages`: Conversation history
- `task_list`: Current task list
- `active_task`: Currently executing task
- `notebook_cells`: Notebook state
- `execution_history`: Execution results
- `error_context`: Error information
- `reference_workflow_content`: Retrieved workflows
- `autonomous_mode`: Mode flags
- And more...

## Request Processing Flow

```
1. User Input → KaiAgent.chat()
   ├── Parse VSCode context
   └── Build initial KaiState

2. Graph Execution → LangGraphOrchestrator.process_request()
   ├── Select graph (autonomous or regular)
   └── Execute graph with initial state

3. Conditional Routing → Graph edges
   ├── Check state conditions
   └── Route to next tool node

4. Tool Execution → Individual tools
   ├── Adapter converts state to ExecutionContext
   ├── Tool executes (possibly with internal LLM loops)
   ├── ToolResult converted back to state updates
   └── State propagates to next node

5. Graph Completion → Back to VSCode
   ├── Extract final response from state
   └── Return VSCode-ready format
```

## Autonomous Mode Flow

### Initial Planning
```
1. agent_router node
   ├── Checks state flags
   └── Routes to plan_tasks

2. plan_tasks node (PlanTasksTool)
   ├── Generates task list with internal critique loop
   └── Updates state with task_list

3. Conditional edge
   ├── autonomous_mode_continue = False
   └── Routes to complete (END)
```

### Autonomous Execution Loop
```
1. agent_router node
   ├── Checks active_task
   └── Routes based on task type and state

2. Possible routes:
   ├── generate_code → GenerateCodeTool
   ├── handle_error → HandleErrorTool
   ├── backtrack → BacktrackTool
   ├── manage_progress → ManageProgressTool
   └── complete → END

3. Loop continues until:
   ├── All tasks completed
   └── OR error requires user intervention
```

## Design Principles

1. **Graph-Based Orchestration**: LangGraph manages control flow
2. **Consolidated Tools**: Tools handle multi-step operations internally
3. **State Propagation**: KaiState flows through graph automatically
4. **Conditional Routing**: Graph edges make decisions based on state
5. **Tool Independence**: Tools don't call each other directly
6. **MCP Compatibility**: Tools follow industry standards

## Supporting Components

### LLMInterface (`llm_interface.py`)
- **Purpose**: Unified interface for different LLM providers
- **Supports**: OpenAI, Ollama, with provider-specific configurations

### PromptManager (`prompt_manager.py`)
- **Purpose**: Centralized prompt generation for all LLM tools
- **Scenarios**: Code generation, task lists, intent classification, etc.
- **Context Integration**: Combines notebook structure, execution history, RAG content

This architecture provides powerful orchestration capabilities through graph-based execution while maintaining clean separation of concerns.
