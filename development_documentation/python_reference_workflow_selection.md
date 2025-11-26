# Reference Workflow Selection Information Flow

## Overview
This document describes the complete workflow for reference workflow selection. The system retrieves relevant Jupyter notebooks from a knowledge base, selects specific cells from each notebook, and provides this context to task generation.

## Key Features

### 1. Automatic ID Conversion
The system handles two ID formats transparently:
- **Full ID** (user-facing): `"scverse/scanpy-tutorials/pbmc3k.ipynb"`
- **Internal ID** (storage): `"scverse_scanpy_tutorials_pbmc3k"`

Selection tools automatically convert between formats - LLM sees and returns full paths, which are converted to internal IDs for storage operations.

### 2. Efficient Cell Selection
Cell selection is optimized to minimize LLM calls:
- Detects which workflows are new, unchanged, or removed
- Reuses content for unchanged workflows from previous context
- Runs LLM cell selection only on new workflows
- Merges kept and new content
- **Performance**: O(new_workflows) LLM calls instead of O(total_workflows)

### 3. Loading State UX
When retrieval queries are generated, the system shows a loading indicator (`"⏳ Retrieving reference workflows..."`) between task list generation and the final workflow list, providing clear feedback during processing.

### 4. Thin Orchestrator Design
The orchestrator maintains simple sequential workflow execution. Tools own their business logic including change detection, filtering, and optimization decisions.

## Key Data Structures

### Context Fields
```python
# Current selection (updated by selection tools)
context["reference_workflow_internal_ids"]: List[str]  # ["org_repo_file", ...]
context["reference_workflow_ids"]: str  # "org/repo/file.ipynb, ..."

# Cell-selected content (updated by cell selection)
context["reference_workflow_content"]: str  # Formatted notebook content
context["reference_workflow_percentages"]: Dict[str, float]  # {full_id: percentage}

# Excluded workflows (accumulated across iterations)
context["excluded_workflows"]: List[str]  # ["org_repo_file", ...]
```

### ID Formats

**Full ID** (user-facing):
- Format: `"scverse/scanpy-tutorials/pbmc3k.ipynb"`
- Used in: UI messages, percentages keys, reference_workflow_ids
- Built from: `{source_repository}/{workflow_filename}`

**Internal ID** (storage):
- Format: `"scverse_scanpy_tutorials_pbmc3k"`
- Used in: Storage keys, reference_workflow_internal_ids, excluded_workflows
- Derived by: Replacing `/`, `-`, `.ipynb` with `_`

**Conversion**: Happens automatically in selection tools' `_process_structured_result()`

## Tools Involved

1. **ReferenceWorkflowQueryPreparationTool** - Searches for workflow summaries based on queries
2. **ReferenceWorkflowSelectionTool** - Initial selection with retrieval queries
3. **ReferenceWorkflowSelectionOnlyTool** - Updates selection without new queries
4. **ReferenceWorkflowCellSelectionTool** - Selects relevant cells, handles efficiency
5. **FilterUnusedReferenceWorkflowsTool** - Removes workflows not cited in tasks

## Workflow Flow

### Initial Planning (`_handle_autonomous_planning`)

```python
# Line 240-256: Initial retrieval
workflow = [
    "reference_workflow_query_preparation",  # Search database
    "reference_workflow_selection",          # LLM selects workflows + generates queries
    "reference_workflow_cell_selection"      # LLM selects cells from each workflow
]
```

### Task Iteration Loop

```python
# Line 265-266: Generate task list
workflow = ["task_list_generation"]
exec_context = await self.execute_workflow(workflow, exec_context)
retrieval_queries = exec_context.inputs.context.get("retrieval_queries", [])

# Line 270-293: Update references if enabled
if rag_enabled:
    # Show loading message if queries present
    if has_retrieval_queries:
        send_loading_message("⏳ Retrieving reference workflows...")

    # Always run full pipeline (tools decide if work needed)
    workflow = [
        "reference_workflow_query_preparation",  # Add new summaries from queries
        "reference_workflow_selection_only",     # Update selection (merge cited + new)
        "reference_workflow_cell_selection"      # Process changes only
    ]
    exec_context = await self.execute_workflow(workflow, exec_context)

    # Continue iteration if we updated workflows
    if has_retrieval_queries:
        continue
```

## Detailed Tool Behavior

### 1. ReferenceWorkflowQueryPreparationTool

**Input**:
- `context["retrieval_queries"]`: List of search queries

**Process**:
- Searches ChromaDB with semantic similarity
- Returns top N candidates per query
- Deduplicates against existing summaries

**Output**:
```python
output_workflow = {
    "putative_reference_workflow_summaries": existing + new_summaries
}
```

**Format**:
```
> Notebook ID: 'scverse/scanpy-tutorials/pbmc3k.ipynb' (similarity: 0.64)
Repository: scverse/scanpy-tutorials
>> Summary:
[summary text]
```

### 2. ReferenceWorkflowSelectionTool / SelectionOnlyTool

**Input** (from prompt):
- Putative workflow summaries (showing full IDs)
- Excluded workflows list (internal IDs)
- Current workflows (if SelectionOnly)
- Task list (if SelectionOnly)

**LLM Sees**:
```
=== Excluded workflows (do not select):
- scverse_scanpy_tutorials_old_tutorial

=== Putative reference workflows:
> Notebook ID: 'scverse/scanpy-tutorials/pbmc3k.ipynb' (similarity: 0.64)
```

**LLM Returns**:
```python
{
    "selected_notebooks": [
        "scverse/scanpy-tutorials/pbmc3k.ipynb",  # Full path format
        "scverse/decoupler-tutorials/rna.ipynb"
    ]
}
```

**Processing** (`_process_structured_result`):
```python
# Convert full IDs to internal IDs
internal_ids = []
for notebook_id in structured_result.selected_notebooks:
    if "/" in notebook_id or ".ipynb" in notebook_id:
        internal_id = notebook_id.replace("/", "_").replace("-", "_").replace(".ipynb", "")
        internal_ids.append(internal_id)
    else:
        internal_ids.append(notebook_id)  # Already internal

# Fetch notebook content
selected_notebooks = self.selector.get_selected_notebook_content(internal_ids)

# Build full IDs from metadata
full_ids = []
for internal_id, notebook_data in selected_notebooks.items():
    metadata = notebook_data.get("metadata", {})
    full_id = f"{metadata['source_repository']}/{metadata['workflow_filename']}"
    full_ids.append(full_id)
```

**Output**:
```python
output_workflow = {
    "reference_workflow_ids": "org/repo/file1.ipynb, org/repo/file2.ipynb",
    "reference_workflow_internal_ids": ["org_repo_file1", "org_repo_file2"],
    "reference_workflow_content": formatted_content
}
```

**UI**: SelectionTool sends list, SelectionOnlyTool sends nothing (cell selection will)

### 3. ReferenceWorkflowCellSelectionTool (Optimized)

**Input**:
```python
current_ids = context["reference_workflow_internal_ids"]  # From selection tool
previous_percentages = context["reference_workflow_percentages"]  # From last iteration
previous_content = context["reference_workflow_content"]
```

**Change Detection**:
```python
# Derive previous IDs from percentages (no extra state needed!)
full_to_internal = {full_id: internal_id for ...}
previous_ids = {full_to_internal[fid] for fid in previous_percentages.keys()}

# Calculate changes
kept_ids = current_ids & previous_ids     # Reuse content
new_ids = current_ids - previous_ids      # Run LLM
removed_ids = previous_ids - current_ids  # Drop from output
```

**Early Return (No Changes)**:
```python
if not new_ids and not removed_ids:
    # Still send UI message to replace loading state
    return ToolResult(
        output_ui={"text": format_percentages(previous_percentages)},
        output_workflow={},  # Don't update context
        output_type=ToolOutputType.REFERENCE_WORKFLOWS
    )
```

**Processing (Changes Detected)**:
```python
# Extract content for kept workflows
kept_sections = extract_from_previous_content(kept_ids)

# Run LLM cell selection ONLY on new workflows
for notebook_id in new_ids:
    llm_result = await call_llm_for_cell_selection(notebook_id)
    selected_ranges[notebook_id] = llm_result.selected_cells

# Format new content
new_content = format_notebook_context(new_ids, selected_ranges)

# Merge: header + kept_sections + new_content
combined_content = build_header(all_ids) + kept_sections + new_content

# Build percentages (kept + new)
percentages = {
    **{fid: previous_percentages[fid] for fid in kept_full_ids},
    **{fid: calculate_percentage(nid) for nid in new_ids}
}
```

**Output**:
```python
return ToolResult(
    output_ui={"text": "📚 file1.ipynb (50% of file)\n📚 file2.ipynb (40% of file)"},
    output_workflow={
        "reference_workflow_content": combined_content,
        "reference_workflow_percentages": percentages,
        "excluded_workflows": [iid for iid in new_ids if no_cells_selected]
    },
    output_type=ToolOutputType.REFERENCE_WORKFLOWS
)
```

**Performance**:
- **Before**: O(total_workflows) LLM calls
- **After**: O(new_workflows) LLM calls
- **Example**: 5 existing + 1 new = 1 LLM call instead of 6 (83% reduction)

### 4. FilterUnusedReferenceWorkflowsTool

**Input**:
```python
task_list = context["task_list"]
reference_workflow_content = context["reference_workflow_content"]
reference_workflow_percentages = context["reference_workflow_percentages"]
```

**Process**:
- Extracts workflow citations from task list (regex on full IDs)
- Filters content to only include cited workflows
- Updates percentages dict

**Output**:
```python
output_workflow = {
    "reference_workflow_ids": filtered_full_ids_str,
    "reference_workflow_internal_ids": filtered_internal_ids,
    "reference_workflow_content": filtered_content,
    "reference_workflow_percentages": filtered_percentages
}
```

## Excluded Workflows Mechanism

### How Workflows Get Excluded

1. **Cell selection returns 0 cells** for a workflow
2. Cell selection adds internal ID to `excluded_workflows` list
3. Orchestrator extends `context.inputs.excluded_workflows` (same object as `state.excluded_workflows`)
4. Next iteration includes these in prompt

### Prompt Format
```
=== Excluded workflows (do not select these):
- scverse_scanpy_tutorials_old_tutorial
- bayraktarlab_cell2location_empty_demo
```

### LLM Instruction
```
These workflows were previously selected but had no relevant content.
DO NOT select these workflows again.
```

## Loading State UX

### Timeline

1. **Task List Generated**
   ```
   Chain-of-thought:
   ⏳ 1. Verify AnnData...
   ⏳ 9. Run TF activity using decoupler...
        [adapted from: 'scverse/decoupler-tutorials/rna.ipynb']
   ```

2. **Loading Message** (if `has_retrieval_queries`)
   ```
   ⏳ Retrieving reference workflows...
   ```

3. **Final List** (from cell selection)
   ```
   Retrieved reference workflows
   📚 scverse/scanpy-tutorials/pbmc3k.ipynb (32% of file)
   📚 scverse/decoupler-tutorials/rna.ipynb (40% of file)
   ```

### Edge Cases

| Scenario | Loading Shown? | Cell Selection | Result |
|----------|---------------|----------------|--------|
| New workflows | ✓ | Runs LLM | Shows final list with % |
| No changes | ✓ | Returns previous % | Replaces loading |
| First iteration | ✓ | Runs LLM on all | Shows all with % |
| No queries | ✗ | May skip or refresh | No loading to clear |

## State Management

### Context Flow
```
ExecutionContext
├─ inputs
│  ├─ context: Dict[str, Any]
│  │  ├─ reference_workflow_ids          # Updated by selection
│  │  ├─ reference_workflow_internal_ids # Updated by selection
│  │  ├─ reference_workflow_content      # Updated by cell selection
│  │  ├─ reference_workflow_percentages  # Updated by cell selection
│  │  └─ retrieval_queries               # From task generation
│  └─ excluded_workflows: List[str]      # Accumulated (shared with state)
└─ ...

OrchestratorState
├─ task_list
├─ excluded_workflows  # Same list object as context.inputs.excluded_workflows
└─ ...
```

### Update Mechanism
```python
# After each tool execution
if result.output_workflow:
    context.inputs.context.update(result.output_workflow)

    if "excluded_workflows" in result.output_workflow:
        context.inputs.excluded_workflows.extend(result.output_workflow["excluded_workflows"])
```

## Architecture Principles

1. **Thin Orchestrator**: Simple sequential workflow, no business logic
2. **Smart Tools**: Tools own their logic (change detection, filtering, etc.)
3. **Single Source of Truth**: `reference_workflow_percentages` defines previous state
4. **Lazy Evaluation**: Work only done when needed (early returns)
5. **Consistent IDs**: Automatic conversion at tool boundaries
6. **Clear UX**: Loading states, progress indication, no orphaned messages

## Testing Strategy

### Unit Tests
- ID conversion (full ↔ internal)
- Change detection (new/kept/removed)
- Content extraction from previous context
- Percentage calculation

### Integration Tests
- Full workflow with retrieval queries
- Excluded workflows accumulation
- Loading state replacement
- Empty cell handling

### Performance Tests
- LLM call count (should be O(new_workflows))
- Context size growth
- Cache hit rate for kept workflows
