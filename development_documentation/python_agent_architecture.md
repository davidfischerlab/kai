# Core Module Architecture

## Overview

The core module implements a clean, **orchestrated architecture** for the bioinformatics agent. The architecture follows a direct flow from VSCode interface through a unified orchestrator to specialized tools.

## Simplified Architecture (2 Layers)

```
Layer 1: Public Interface
└── BioinformaticsAgent (agent.py)
    └── Main entry point, manages orchestrator directly

Layer 2: Unified Orchestration & Tools  
└── WorkflowOrchestrator (orchestration/workflow_orchestrator.py)
    ├── Handles intent classification internally
    ├── Routes to appropriate tool workflows
    ├── Manages tool execution and piping
    └── Returns VSCode-ready responses directly
```

## Complete Data Flow

```
VSCode Interface → BioinformaticsAgent → WorkflowOrchestrator → Tools → VSCode Response
```

## Current Implementation Files

```
core/
├── agent.py                           # BioinformaticsAgent
├── llm_interface.py                   # LLM provider interface
├── prompt_manager.py                  # Centralized prompt generation
└── orchestration/
    ├── base_tool.py                   # Base tool interface
    ├── workflow_orchestrator.py       # Main orchestration logic
    ├── prompt_tools.py               # LLM-based tools
    └── deterministic_tools.py        # Rule-based tools
```

## Component Responsibilities

### BioinformaticsAgent (`agent.py`)
- **Purpose**: Main entry point and system coordinator
- **Responsibilities**:
  - Initialize WorkflowOrchestrator with LLM and knowledge base
  - Provide unified chat interface
  - Parse context from VSCode interface (execution history, notebook structure)
  - Manage system status
- **Dependencies**: WorkflowOrchestrator, LLMInterface

### WorkflowOrchestrator (`orchestration/workflow_orchestrator.py`)
- **Purpose**: Unified request processing and tool orchestration
- **Responsibilities**:
  - **Intent Classification**: Classify user requests using IntentClassificationTool
  - **Workflow Routing**: Route to appropriate tool workflows based on intent
  - **Autonomous Mode Handling**: Handle TODO generation and autonomous execution
  - **Tool Piping**: Execute tool sequences (e.g., RAG → Code Generation)
  - **Response Formatting**: Return VSCode-ready responses
- **Dependencies**: All tools, LLMInterface
- **Key Method**: `process_request()` - main entry point for all requests

## Tool Architecture

### LLM-Based Tools (`orchestration/prompt_tools.py`)
All tools that require LLM reasoning:

- **IntentClassificationTool**: Classifies user intent (question, code generation, etc.)
- **TodoGenerationTool**: Generates TODO lists for autonomous mode
- **CodeGenerationTool**: Generates code with RAG integration  
- **CellPositioningTool**: Determines cell placement (addition vs replacement)
- **ErrorRecoveryTool**: Generates fixing code for execution errors
- **AutonomousContinueTool**: Continues autonomous execution
- **QuestionAnsweringTool**: Answers questions about code/concepts

### Deterministic Tools (`orchestration/deterministic_tools.py`)
Tools that use algorithmic/rule-based logic:

- **RAGRetrievalTool**: Retrieves relevant documentation
- **ExecutionMonitorTool**: Monitors code execution results

### Base Tool Interface (`orchestration/base_tool.py`)
- **BaseTool**: Abstract base class for all tools
- **ToolResult**: Standardized tool output with metadata and effects
- **ToolOutputType**: Response type classification (RESPONSE, DISPLAY_ONLY, EXECUTE_ONLY, TOOL_USAGE)

**Key Design**: Tools are standalone and do not handle workflow piping - all workflow orchestration is managed by WorkflowOrchestrator.

## Request Processing Flow

```
1. **User Input** → BioinformaticsAgent.chat()
   ├── Parse VSCode context (execution history, notebook structure, autonomous flags)
   └── Extract conversation history

2. **Direct Orchestration** → WorkflowOrchestrator.process_request()
   ├── Check for autonomous mode (TODO generation, continuation, error recovery)
   ├── OR classify intent using IntentClassificationTool
   └── Route to appropriate workflow

3. **Tool Workflow Execution** → WorkflowOrchestrator.execute_workflow()
   ├── Question Answering: RAG → Question Answering
   ├── Code Generation: [Cell Positioning] → RAG → Code Generation  
   ├── Code In-Place: RAG → Code Generation
   ├── Autonomous Initiation: TODO Generation → Cell Positioning → Code Generation
   └── Autonomous Continuation: Autonomous Continue → (Error Recovery OR Continue Analysis)

4. **Tool Execution** → Individual tools
   ├── LLM-based tools use PromptManager for context-aware prompts
   ├── RAG tools query knowledge base
   ├── Tools return ToolResult with VSCode-ready responses
   └── Tools are standalone - no internal piping logic

5. **Direct Response** → Back to VSCode
   ├── No response reformatting (tools output VSCode-ready format)
   ├── Metadata preserved (positioning info, RAG usage, autonomous flags)
   └── Workflow orchestrator handles all tool piping sequentially
```

## Autonomous Mode Flow

### Initial Autonomous Workflow (`auto_mode_initiation=true`)
```
1. **Single Orchestrated Workflow**: 
   TODO Generation → Cell Positioning → Code Generation
   
   ├── TodoGenerationTool: Outputs TODO list (RESPONSE - shown in chat)
   ├── CellPositioningTool: Determines placement (addition mode)
   └── CodeGenerationTool: Outputs code (EXECUTE_ONLY - executed silently)
```

### Continuation Autonomous Workflow (`auto_mode_continuation=true`)
```
1. **Decision Phase**: AutonomousContinueTool
   ├── Updates TODO list (DISPLAY_ONLY - shown in chat)
   └── Returns decision intent (CONTINUE_ANALYSIS, ERROR_RECOVERY, TASK_COMPLETE)

2. **Action Phase**: Based on decision intent
   ├── ERROR_RECOVERY: Error Recovery workflow (EXECUTE_ONLY)
   ├── CONTINUE_ANALYSIS: Cell Positioning → Code Generation (EXECUTE_ONLY)
   └── TASK_COMPLETE: Return completion message
```

**Key Design**: No tool-level piping - orchestrator manages complete workflows based on context flags.

## Design Principles

1. **Direct Communication**: Minimal layers between VSCode and tools
2. **Tool Autonomy**: Tools control their own output format (standalone design)
3. **Unified Interface**: All tools follow BaseTool pattern
4. **Context Preservation**: Full context passed to all tools
5. **Orchestrated Workflows**: Logical tool sequences for complex tasks
6. **Clean Separation**: LLM-based vs deterministic tool organization

## Supporting Components

### LLMInterface (`llm_interface.py`)
- **Purpose**: Unified interface for different LLM providers
- **Supports**: OpenAI, Ollama, with provider-specific configurations
- **Dependencies**: None (standalone utility)

### PromptManager (`prompt_manager.py`)  
- **Purpose**: Centralized prompt generation for all LLM tools
- **Scenarios**: Code generation, TODO lists, intent classification, cell positioning, etc.
- **Context Integration**: Combines notebook structure, execution history, RAG content
- **Dependencies**: None (standalone utility)

This architecture eliminates redundancy while providing powerful orchestration capabilities for complex bioinformatics workflows.