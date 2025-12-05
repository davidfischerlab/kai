"""
Config reader for shell scripts.
Reads YAML config and outputs shell-compatible variable assignments.

Usage:
    eval "$(python read_config.py configs/scenario1_config.yaml blood 1)"
    # Sets: BASE_NOTEBOOK, TASK, DESCRIPTION
"""

import sys
import yaml
from pathlib import Path


def read_test_config(config_file, case_name=None, replicate=None):
    """Read config and output shell variables."""

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # If no case specified, output defaults
    if case_name is None:
        # Get test case names
        test_cases = config.get('test_cases', [])
        case_names = [tc['name'] for tc in test_cases]

        # Get execution parameters
        execution = config.get('execution', {})
        num_replicates = execution.get('num_replicates', 3)
        max_iterations = execution.get('max_iterations', 100)

        # Get LLM config
        llm = config.get('llm', {})
        llm_provider = llm.get('provider', 'ollama')
        llm_model = llm.get('model', None)
        if llm_model is None:
            llm_model = ''  # Convert None to empty string for shell
        rag_enabled = llm.get('rag_enabled', True)

        # Get output config
        output = config.get('output', {})
        output_dir = output.get('base_dir', 'full_agent_test/test_outputs')

        # Get base notebooks dir
        base_notebooks_dir = config.get('base_notebooks_dir', 'full_agent_test/base_notebooks')

        # Generate replicate list (1 to num_replicates)
        replicates = ' '.join(str(i) for i in range(1, num_replicates + 1))

        print(f"DEFAULT_CASES='{' '.join(case_names)}'")
        print(f"DEFAULT_REPLICATES='{replicates}'")
        print(f"DEFAULT_MAX_ITERATIONS={max_iterations}")
        print(f"DEFAULT_LLM_PROVIDER='{llm_provider}'")
        print(f"DEFAULT_LLM_MODEL='{llm_model}'")
        print(f"DEFAULT_RAG_ENABLED={str(rag_enabled).lower()}")
        print(f"DEFAULT_OUTPUT_DIR='{output_dir}'")
        print(f"DEFAULT_BASE_NOTEBOOKS_DIR='{base_notebooks_dir}'")
        return

    # Find the test case
    test_cases = config.get('test_cases', [])
    case = None
    for tc in test_cases:
        if tc['name'] == case_name:
            case = tc
            break

    if case is None:
        print(f"Error: Case '{case_name}' not found", file=sys.stderr)
        sys.exit(1)

    # Output shell variables (escape quotes in strings)
    base_nb = case['base_notebook']
    task = case['task'].replace("'", "'\\''")  # Escape single quotes for shell
    desc = case.get('description', '').replace("'", "'\\''")

    print(f"BASE_NOTEBOOK='{base_nb}'")
    print(f"TASK='{task}'")
    print(f"DESCRIPTION='{desc}'")


def main():
    if len(sys.argv) < 2:
        print("Usage: read_config.py <config.yaml> [case_name] [replicate]", file=sys.stderr)
        sys.exit(1)

    config_file = sys.argv[1]
    case_name = sys.argv[2] if len(sys.argv) > 2 else None
    replicate = sys.argv[3] if len(sys.argv) > 3 else None

    read_test_config(config_file, case_name, replicate)


if __name__ == '__main__':
    main()
