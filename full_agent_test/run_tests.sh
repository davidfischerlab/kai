#!/bin/bash
#
# Full Agent Test Runner - Shell Orchestrator
#
# Runs each test case in an isolated Python subprocess to prevent resource leaks.
# Each subprocess exit cleans up all ChromaDB file descriptors automatically.
#
# Usage:
#   ./run_tests.sh --api-key YOUR_KEY --env-agent /path/to/agent_env/bin/python --env-notebook /path/to/notebook_env/bin/python

# Increase file descriptor limit to prevent "Too many open files" errors
# ChromaDB opens many file handles during RAG queries
ulimit -n 10240
#
#   Options:
#     --api-key KEY          API key for LLM provider (required)
#     --env-agent PATH       Python executable for agent environment (required)
#     --env-notebook PATH    Python executable for notebook kernels (required)
#     --config PATH          Config file (default: configs/scenario1_config.yaml)
#     --cases CASE1 CASE2    Run specific cases (default: from config)
#     --replicates N-M       Run specific replicate range (default: from config)
#     --max-iterations N     Max autonomous iterations (default: from config)
#     --llm-provider NAME    LLM provider (default: from config)
#     --llm-model NAME       LLM model name (default: from config)
#     --output-dir PATH      Output directory (default: from config)
#     --no-rag               Disable RAG (default: from config)
#

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default config file
CONFIG_FILE="$SCRIPT_DIR/configs/scenario1_config.yaml"

# Save original arguments
ORIGINAL_ARGS=("$@")

# Parse arguments first to get config file if specified
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Reset argument parsing with original arguments
set -- "${ORIGINAL_ARGS[@]}"

# We need to load config but AGENT_PYTHON isn't set yet (comes from CLI args)
# So we'll use whatever python is in PATH for initial config reading, then use AGENT_PYTHON for everything else
echo "Loading configuration from: $CONFIG_FILE"
eval "$(python "$SCRIPT_DIR/read_config.py" "$CONFIG_FILE")"

# Set defaults from config
CASES="$DEFAULT_CASES"
REPLICATES="$DEFAULT_REPLICATES"
MAX_ITERATIONS="$DEFAULT_MAX_ITERATIONS"
LLM_PROVIDER="$DEFAULT_LLM_PROVIDER"
LLM_MODEL="$DEFAULT_LLM_MODEL"
OUTPUT_DIR="$ROOT_DIR/$DEFAULT_OUTPUT_DIR"
BASE_NOTEBOOKS_DIR="$ROOT_DIR/$DEFAULT_BASE_NOTEBOOKS_DIR"

# Set RAG flag from config
if [[ "$DEFAULT_RAG_ENABLED" == "true" ]]; then
  RAG_ENABLED="--rag-enabled"
else
  RAG_ENABLED="--no-rag"
fi

# Parse arguments (override config defaults)
while [[ $# -gt 0 ]]; do
  case $1 in
    --api-key)
      API_KEY="$2"
      shift 2
      ;;
    --env-agent)
      AGENT_PYTHON="$2"
      shift 2
      ;;
    --env-notebook)
      NOTEBOOK_PYTHON="$2"
      shift 2
      ;;
    --config)
      # Already handled above
      shift 2
      ;;
    --cases)
      CASES="$2"
      shift 2
      ;;
    --replicates)
      # Parse range like "1-2" into "1 2"
      if [[ "$2" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        START="${BASH_REMATCH[1]}"
        END="${BASH_REMATCH[2]}"
        REPLICATES=$(seq $START $END)
      else
        REPLICATES="$2"
      fi
      shift 2
      ;;
    --max-iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --llm-provider)
      LLM_PROVIDER="$2"
      shift 2
      ;;
    --llm-model)
      LLM_MODEL="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --no-rag)
      RAG_ENABLED="--no-rag"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run with --help for usage"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "$API_KEY" ]]; then
  echo "Error: --api-key is required"
  exit 1
fi

if [[ -z "$AGENT_PYTHON" ]]; then
  echo "Error: --env-agent is required"
  exit 1
fi

if [[ -z "$NOTEBOOK_PYTHON" ]]; then
  echo "Error: --env-notebook is required"
  exit 1
fi

# Normalize agent Python path (handle both /path/to/env and /path/to/env/bin/python)
if [[ -d "$AGENT_PYTHON" ]]; then
  # It's a directory - look for bin/python
  if [[ -f "$AGENT_PYTHON/bin/python" ]]; then
    AGENT_PYTHON="$AGENT_PYTHON/bin/python"
  else
    echo "Error: $AGENT_PYTHON is a directory but $AGENT_PYTHON/bin/python does not exist"
    exit 1
  fi
elif [[ ! -f "$AGENT_PYTHON" ]]; then
  echo "Error: --env-agent path does not exist: $AGENT_PYTHON"
  exit 1
fi

# Normalize notebook Python path (handle both /path/to/env and /path/to/env/bin/python)
if [[ -d "$NOTEBOOK_PYTHON" ]]; then
  # It's a directory - look for bin/python
  if [[ -f "$NOTEBOOK_PYTHON/bin/python" ]]; then
    NOTEBOOK_PYTHON="$NOTEBOOK_PYTHON/bin/python"
  else
    echo "Error: $NOTEBOOK_PYTHON is a directory but $NOTEBOOK_PYTHON/bin/python does not exist"
    exit 1
  fi
elif [[ ! -f "$NOTEBOOK_PYTHON" ]]; then
  echo "Error: --env-notebook path does not exist: $NOTEBOOK_PYTHON"
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "==============================================="
echo "Full Agent Test Runner"
echo "==============================================="
echo "Agent Python:    $AGENT_PYTHON"
echo "Agent version:   $($AGENT_PYTHON --version)"
echo "Notebook Python: $NOTEBOOK_PYTHON"
echo "Config:          $CONFIG_FILE"
echo "LLM Provider:    $LLM_PROVIDER"
if [[ -n "$LLM_MODEL" ]]; then
  echo "LLM Model:       $LLM_MODEL"
fi
echo "Max Iterations:  $MAX_ITERATIONS"
echo "RAG:             $(if [[ "$RAG_ENABLED" == "--rag-enabled" ]]; then echo "Enabled"; else echo "Disabled"; fi)"
echo "Output Dir:      $OUTPUT_DIR"
echo "Cases:           $CASES"
echo "Replicates:      $REPLICATES"
echo "==============================================="
echo ""

# Track successes and failures
TOTAL_TESTS=0
SUCCESSFUL_TESTS=0
FAILED_TESTS=0

# Run tests
for CASE in $CASES; do
  for REPLICATE in $REPLICATES; do
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    # Load case-specific config from YAML
    eval "$("$AGENT_PYTHON" "$SCRIPT_DIR/read_config.py" "$CONFIG_FILE" "$CASE")"

    # Set up paths
    OUTPUT_NB="$OUTPUT_DIR/${CASE}_r${REPLICATE}.ipynb"
    METADATA="$OUTPUT_DIR/${CASE}_r${REPLICATE}_metadata.json"

    echo "-----------------------------------------------"
    echo "Test $TOTAL_TESTS: $CASE - Replicate $REPLICATE"
    echo "-----------------------------------------------"

    # Build command
    CMD=(
      "$AGENT_PYTHON" "$SCRIPT_DIR/run_single_test.py"
      --case "$CASE"
      --replicate "$REPLICATE"
      --base-notebook "$BASE_NOTEBOOKS_DIR/$BASE_NOTEBOOK"
      --output-notebook "$OUTPUT_NB"
      --metadata-file "$METADATA"
      --task "$TASK"
      --description "$DESCRIPTION"
      --api-key "$API_KEY"
      --notebook-python "$NOTEBOOK_PYTHON"
      --llm-provider "$LLM_PROVIDER"
      --max-iterations "$MAX_ITERATIONS"
      $RAG_ENABLED
    )

    if [[ -n "$LLM_MODEL" ]]; then
      CMD+=(--llm-model "$LLM_MODEL")
    fi

    # Run test in isolated subprocess
    if "${CMD[@]}"; then
      SUCCESSFUL_TESTS=$((SUCCESSFUL_TESTS + 1))
      echo "✅ Success"
    else
      FAILED_TESTS=$((FAILED_TESTS + 1))
      echo "❌ Failed"
    fi

    echo ""
  done
done

# Summary
echo "==============================================="
echo "Test Run Complete"
echo "==============================================="
echo "Total:      $TOTAL_TESTS"
echo "Successful: $SUCCESSFUL_TESTS"
echo "Failed:     $FAILED_TESTS"
echo "==============================================="

# Analyze results
echo ""
echo "Generating analysis..."
"$AGENT_PYTHON" "$SCRIPT_DIR/analyze_results.py" \
    --test-dir "$OUTPUT_DIR" \
    --api-key "$API_KEY" \
    --output "$SCRIPT_DIR/analysis"

exit 0
