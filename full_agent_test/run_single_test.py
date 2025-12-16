"""
Single test executor for full agent tests.

Runs ONE test case in isolation and exits.
Resources are cleaned up by OS on process exit, preventing file descriptor leaks.

Usage:
    python run_single_test.py \\
        --case blood \\
        --replicate 1 \\
        --base-notebook base_notebooks/blood_base.ipynb \\
        --task "Perform clustering analysis" \\
        --output-notebook test_outputs/blood_r1.ipynb \\
        --metadata-file test_outputs/blood_r1_metadata.json \\
        --api-key YOUR_KEY \\
        --notebook-python /path/to/env/bin/python
"""

# Set environment variables BEFORE any imports
import os
import sys
from pathlib import Path

# Debug: Print which Python is being used
print(f"DEBUG: Running with Python: {sys.executable}")
print(f"DEBUG: Python version: {sys.version}")
sys.stdout.flush()

os.environ['TQDM_DISABLE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['KAI_DEBUG_PROMPTS'] = 'true'  # Enable prompt debugging for full agent tests

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Monkey-patch tqdm BEFORE other imports
class _DummyTqdm:
    def __init__(self, *args, **kwargs):
        self.iterable = kwargs.get('iterable', args[0] if args else None)
    def __iter__(self):
        if self.iterable is not None:
            return iter(self.iterable)
        return iter([])
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def update(self, *args, **kwargs):
        pass
    def set_description(self, *args, **kwargs):
        pass
    def close(self):
        pass

# Import tqdm normally first, THEN patch the tqdm class
import tqdm
import tqdm.auto
tqdm.tqdm = _DummyTqdm
tqdm.auto.tqdm = _DummyTqdm
tqdm.std.tqdm = _DummyTqdm

# Now safe to import other modules
import argparse
import asyncio
import json
import shutil
import time
import logging
from datetime import datetime


# Set up logging with custom filters
class QuietModeFilter(logging.Filter):
    """Filter for quiet mode - shows only high-level progress, not every action."""
    def filter(self, record):
        msg = record.getMessage()

        # Always show errors and warnings
        if record.levelname in ['ERROR', 'WARNING', 'CRITICAL']:
            return True

        # Suppress verbose backend loggers
        suppressed_loggers = [
            'kai.retrieval.snippets.storage.chromadb_manager',
            'kai.retrieval.workflow_summaries.notebook_storage',
            'kai.retrieval.workflow_summaries.summary_search',
            'kai.retrieval.workflow_summaries.workflow_extractor',
            'kai.retrieval.snippets.extractors',
            'httpx',
            'httpcore',
            'sentence_transformers',
        ]

        for logger_prefix in suppressed_loggers:
            if record.name.startswith(logger_prefix):
                return False

        # For UI.jupyter loggers, only show iteration summaries
        if record.name.startswith('UI.jupyter'):
            # Show iteration markers and summaries
            if any(marker in msg for marker in ['[ITERATION', '[RAG]', 'Pre-execution', 'Autonomous mode']):
                return True
            # Suppress detailed action logs
            return False

        # For other loggers, show everything
        return True


# Import set_global_filter first (doesn't trigger kai module imports)
from kai.utils.logger import set_global_filter

# Set global filter BEFORE importing any kai modules (quiet mode by default)
quiet_filter = QuietModeFilter()
set_global_filter(quiet_filter)

# NOW import JupyterInterface (which will trigger kai module imports with filter applied)
from UI.jupyter import JupyterInterface

# Set up basic logging (after global filter is set)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Apply filter to root logger for any loggers that don't use setup_logger()
logging.root.addFilter(quiet_filter)
for handler in logging.root.handlers:
    handler.addFilter(quiet_filter)


async def run_single_test(args):
    """
    Execute a single test case.

    Args:
        args: Parsed command-line arguments

    Returns:
        Metadata dictionary with test results
    """
    print(f"Starting: {args.case} - Replicate {args.replicate}")
    print(f"Base notebook: {args.base_notebook}")
    print(f"Output notebook: {args.output_notebook}")
    print(f"Task: {args.task}")

    # Copy base notebook to output location
    shutil.copy(args.base_notebook, args.output_notebook)

    # Track execution metadata
    start_time = time.time()
    metadata = {
        'case_name': args.case,
        'replicate': args.replicate,
        'base_notebook': args.base_notebook,
        'output_notebook': args.output_notebook,
        'task': args.task,
        'description': args.description,
        'start_time': datetime.now().isoformat(),
        'llm_provider': args.llm_provider,
        'llm_model': args.llm_model,
        'rag_enabled': args.rag_enabled,
        'max_iterations': args.max_iterations,
    }

    try:
        # Determine turbo mode based on provider
        # If using ollama-turbo, we must keep turbo enabled
        turbo_enabled = (args.llm_provider == "ollama-turbo")

        # Run test using Jupyter interface
        with JupyterInterface(
            notebook_path=args.output_notebook,
            notebook_python=args.notebook_python,
            llm_provider=args.llm_provider,
            model=args.llm_model,
            api_key=args.api_key,
            rag_enabled=args.rag_enabled,
            turbo_enabled=turbo_enabled
        ) as interface:

            result = await interface.run_autonomous(
                initial_message=args.task,
                max_iterations=args.max_iterations,
                graph_recursion_limit=args.graph_recursion_limit
            )

            # Save modified notebook
            interface.save(args.output_notebook)

        # Record success
        end_time = time.time()
        duration = end_time - start_time

        metadata.update({
            'success': result['success'],
            'iterations': result['iterations'],
            'final_state': result['final_state'],
            'end_time': datetime.now().isoformat(),
            'duration_seconds': duration,
            'duration_minutes': duration / 60,
            'error': result.get('error', None)
        })

        print(f"✅ Completed: {args.case} - Replicate {args.replicate}")
        print(f"   Iterations: {result['iterations']}, Duration: {duration/60:.1f} min")

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        metadata.update({
            'success': False,
            'error': str(e),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': duration,
            'duration_minutes': duration / 60,
        })

        print(f"❌ Failed: {args.case} - Replicate {args.replicate}: {e}")

    # Save metadata
    with open(args.metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Force garbage collection to cleanup resources
    import gc
    gc.collect()

    # Give system time to close file descriptors
    time.sleep(2)

    # Force another GC pass
    gc.collect()

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Run a single full agent test case'
    )

    # Test identification
    parser.add_argument('--case', required=True, help='Test case name (e.g., blood, lung)')
    parser.add_argument('--replicate', type=int, required=True, help='Replicate number (1-3)')

    # Paths
    parser.add_argument('--base-notebook', required=True, help='Path to base notebook')
    parser.add_argument('--output-notebook', required=True, help='Path for output notebook')
    parser.add_argument('--metadata-file', required=True, help='Path for metadata JSON file')

    # Test configuration
    parser.add_argument('--task', required=True, help='Task description for the agent')
    parser.add_argument('--description', default='', help='Test case description')

    # Agent configuration
    parser.add_argument('--api-key', required=True, help='API key for LLM provider')
    parser.add_argument('--notebook-python', required=True, help='Python executable for notebook kernel')
    parser.add_argument('--llm-provider', default='ollama', help='LLM provider (default: ollama)')
    parser.add_argument('--llm-model', default=None, help='LLM model name')
    parser.add_argument('--rag-enabled', action='store_true', default=True, help='Enable RAG')
    parser.add_argument('--no-rag', action='store_false', dest='rag_enabled', help='Disable RAG')
    parser.add_argument('--max-iterations', type=int, default=40, help='Max autonomous iterations')
    parser.add_argument('--graph-recursion-limit', type=int, default=None, help='Max graph steps per iteration (default: None = use orchestrator default of 100)')

    args = parser.parse_args()

    # Run test
    result = asyncio.run(run_single_test(args))

    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
