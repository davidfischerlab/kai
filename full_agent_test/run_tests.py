#!/usr/bin/env python3
"""
Full agent test runner for Kai.

Runs Scenario 1 test cases in triplicate using the Jupyter CLI interface.
Generates notebooks, logs, and metadata for each execution.

IMPORTANT: This script must be run with the Python from the Kai agent environment.
Notebook kernels will run in the environment specified by --env-notebook.

Usage:
    # Run with agent environment Python
    /path/to/kai_env/bin/python full_agent_test/run_tests.py \
        --api-key YOUR_OLLAMA_API_KEY \
        --env-notebook /path/to/notebook_env

    # Run specific cases
    /path/to/kai_env/bin/python full_agent_test/run_tests.py \
        --api-key YOUR_KEY \
        --env-notebook /path/to/notebook_env \
        --cases blood lung

    # Run specific replicates
    /path/to/kai_env/bin/python full_agent_test/run_tests.py \
        --api-key YOUR_KEY \
        --env-notebook /path/to/notebook_env \
        --replicates 1-2
"""

# Set environment variables BEFORE any imports
import os

# Suppress tqdm and tokenizers
os.environ['TQDM_DISABLE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Disable prompt debugging for test runs (prevents file descriptor exhaustion)
# Writes every LLM prompt/response to disk - thousands of files per test run
os.environ['KAI_DEBUG_PROMPTS'] = 'false'

# Monkey-patch tqdm to disable all progress bars BEFORE any other imports
# This must happen before sentence_transformers is imported
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now monkey-patch tqdm
class _DummyTqdm:
    """Dummy tqdm that does nothing."""
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

    def update(self, n=1):
        pass

    def close(self):
        pass

    def set_description(self, desc):
        pass

# Patch tqdm before it gets imported by sentence_transformers
import tqdm
import tqdm.auto
tqdm.tqdm = _DummyTqdm
tqdm.auto.tqdm = _DummyTqdm

# Also patch std (the module-level name used in `from tqdm import tqdm`)
# and ensure it's in sys.modules so later imports get our dummy
tqdm.std.tqdm = _DummyTqdm
sys.modules['tqdm'].tqdm = _DummyTqdm
sys.modules['tqdm.auto'].tqdm = _DummyTqdm

# Now safe to import other modules
import argparse
import asyncio
import json
import logging
import shutil
import time
import yaml
from datetime import datetime
from typing import List, Dict, Any

# Set up logging with custom filters
class SuppressChromaDBFilter(logging.Filter):
    def filter(self, record):
        # Suppress HNSW segment reader errors - they're non-critical
        if "hnsw segment reader" in record.getMessage().lower():
            return False
        if "Error searching" in record.getMessage() and "hnsw" in record.getMessage().lower():
            return False
        return True


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


class TestRunner:
    """
    Test runner for executing Scenario 1 cases in triplicate.

    Manages test execution, logging, and metadata collection.
    """

    def __init__(self, config_path: str, api_key: str, output_dir: str = None, notebook_python: str = None):
        """
        Initialize test runner.

        Args:
            config_path: Path to YAML configuration file
            api_key: Ollama API key
            output_dir: Override output directory from config
            notebook_python: Path to Python executable for notebook kernel (optional)
        """
        self.config_path = Path(config_path)
        self.api_key = api_key
        self.notebook_python = notebook_python
        self.logger = logging.getLogger(__name__)

        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.config['output']['base_dir'])

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Expand base notebooks directory
        base_dir = Path(self.config['base_notebooks_dir']).expanduser()
        self.base_notebooks_dir = base_dir

        # Execution parameters
        self.num_replicates = self.config['execution']['num_replicates']
        self.max_iterations = self.config['execution']['max_iterations']
        self.timeout_minutes = self.config['execution']['timeout_minutes']

        # LLM configuration
        self.llm_provider = self.config['llm']['provider']
        self.llm_model = self.config['llm']['model']
        self.rag_enabled = self.config['llm']['rag_enabled']

        # Test results
        self.results = []

    def get_output_path(self, case_name: str, replicate: int, suffix: str = '') -> Path:
        """
        Generate output path for a test case.

        Args:
            case_name: Test case name (blood, breastcancer, lung)
            replicate: Replicate number (1-based)
            suffix: File suffix (e.g., '.ipynb', '.log', '.json')

        Returns:
            Path to output file
        """
        pattern = self.config['output']['naming_pattern']
        basename = pattern.format(case=case_name, replicate=replicate)
        return self.output_dir / f"{basename}{suffix}"

    async def run_single_test(
        self,
        case: Dict[str, Any],
        replicate: int
    ) -> Dict[str, Any]:
        """
        Run a single test case.

        Args:
            case: Test case configuration
            replicate: Replicate number (1-based)

        Returns:
            Test result metadata
        """
        case_name = case['name']
        base_notebook = self.base_notebooks_dir / case['base_notebook']
        task = case['task']

        self.logger.info(f"Starting: {case_name} - Replicate {replicate}")
        self.logger.info(f"Base notebook: {base_notebook}")
        self.logger.info(f"Task: {task}")

        # Set up paths
        output_notebook = self.get_output_path(case_name, replicate, '.ipynb')
        log_file = self.get_output_path(case_name, replicate, '.log')
        metadata_file = self.get_output_path(case_name, replicate, '_metadata.json')

        # Copy base notebook to output location
        shutil.copy(base_notebook, output_notebook)

        # Track execution metadata
        start_time = time.time()
        metadata = {
            'case_name': case_name,
            'replicate': replicate,
            'base_notebook': str(base_notebook),
            'output_notebook': str(output_notebook),
            'task': task,
            'description': case['description'],
            'start_time': datetime.now().isoformat(),
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'rag_enabled': self.rag_enabled,
            'max_iterations': self.max_iterations,
        }

        try:
            # Run test using Jupyter interface
            with JupyterInterface(
                notebook_path=str(output_notebook),
                notebook_python=self.notebook_python,
                llm_provider=self.llm_provider,
                model=self.llm_model,
                api_key=self.api_key,
                rag_enabled=self.rag_enabled
            ) as interface:

                result = await interface.run_autonomous(
                    initial_message=task,
                    max_iterations=self.max_iterations
                )

                # Save modified notebook
                interface.save(str(output_notebook))

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

            self.logger.info(f"✅ Completed: {case_name} - Replicate {replicate}")
            self.logger.info(f"   Iterations: {result['iterations']}, Duration: {duration/60:.1f} min")

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

            self.logger.error(f"❌ Failed: {case_name} - Replicate {replicate}: {e}")

        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Force garbage collection to cleanup resources
        import gc
        gc.collect()

        # Give system time to close file descriptors
        time.sleep(2)

        # Force another GC pass
        gc.collect()

        return metadata

    async def run_all_tests(
        self,
        case_filter: List[str] = None,
        replicate_range: tuple = None
    ):
        """
        Run all configured test cases.

        Args:
            case_filter: List of case names to run (None = all)
            replicate_range: Tuple of (start, end) replicate numbers (None = all)
        """
        # Filter test cases
        test_cases = self.config['test_cases']
        if case_filter:
            test_cases = [c for c in test_cases if c['name'] in case_filter]

        # Determine replicate range
        if replicate_range:
            start_rep, end_rep = replicate_range
        else:
            start_rep, end_rep = 1, self.num_replicates

        replicates = range(start_rep, end_rep + 1)

        self.logger.info("="*80)
        self.logger.info("Full Agent Test Runner")
        self.logger.info("="*80)
        self.logger.info(f"Test cases: {[c['name'] for c in test_cases]}")
        self.logger.info(f"Replicates: {list(replicates)}")
        self.logger.info(f"Max iterations per case: {self.max_iterations}")
        self.logger.info(f"LLM provider: {self.llm_provider}")
        self.logger.info(f"RAG enabled: {self.rag_enabled}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("="*80)

        # Run tests sequentially (to avoid resource conflicts)
        for case in test_cases:
            for replicate in replicates:
                metadata = await self.run_single_test(case, replicate)
                self.results.append(metadata)

        # Save combined results
        results_file = self.output_dir / 'test_results_summary.json'
        with open(results_file, 'w') as f:
            json.dump({
                'config': str(self.config_path),
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.results),
                'results': self.results
            }, f, indent=2)

        self.logger.info("="*80)
        self.logger.info("All tests completed!")
        self.logger.info(f"Results saved to: {results_file}")
        self._print_summary()

    def _print_summary(self):
        """Print summary of test results."""
        total = len(self.results)
        successful = sum(1 for r in self.results if r['success'])
        failed = total - successful

        self.logger.info("="*80)
        self.logger.info("Test Summary")
        self.logger.info("="*80)
        self.logger.info(f"Total tests: {total}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info("")

        # Summary by case
        cases = {}
        for r in self.results:
            case_name = r['case_name']
            if case_name not in cases:
                cases[case_name] = {'success': 0, 'failed': 0, 'total_iterations': 0}

            if r['success']:
                cases[case_name]['success'] += 1
                cases[case_name]['total_iterations'] += r.get('iterations', 0)
            else:
                cases[case_name]['failed'] += 1

        for case_name, stats in cases.items():
            self.logger.info(f"{case_name}:")
            self.logger.info(f"  Success: {stats['success']}/3")
            if stats['success'] > 0:
                avg_iter = stats['total_iterations'] / stats['success']
                self.logger.info(f"  Avg iterations: {avg_iter:.1f}")
            self.logger.info("")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Kai full agent tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python full_agent_test/run_tests.py --api-key YOUR_KEY

  # Run specific cases
  python full_agent_test/run_tests.py --api-key YOUR_KEY --cases blood lung

  # Run specific replicates
  python full_agent_test/run_tests.py --api-key YOUR_KEY --replicates 1-2

  # Custom output directory
  python full_agent_test/run_tests.py --api-key YOUR_KEY --output my_test_run
        """
    )

    parser.add_argument(
        '--api-key',
        required=True,
        help='Ollama API key'
    )

    parser.add_argument(
        '--config',
        default='full_agent_test/configs/scenario1_config.yaml',
        help='Path to configuration file (default: scenario1_config.yaml)'
    )

    parser.add_argument(
        '--cases',
        nargs='+',
        choices=['blood', 'breastcancer', 'lung'],
        help='Specific test cases to run (default: all)'
    )

    parser.add_argument(
        '--replicates',
        help='Replicate range to run (e.g., "1-2", "2-3", default: all)'
    )

    parser.add_argument(
        '--output',
        help='Override output directory from config'
    )

    parser.add_argument(
        '--env-notebook',
        type=str,
        required=True,
        help='REQUIRED. Python executable or conda/mamba environment for notebook kernel. '
             'Examples: /path/to/env/bin/python OR /path/to/env'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        help='Override max iterations from config'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        default=True,
        help='Quiet mode - show only high-level progress per iteration (default)'
    )

    parser.add_argument(
        '--verbose-logs',
        action='store_true',
        help='Show all detailed logs (disables quiet mode)'
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Import set_global_filter first (doesn't trigger kai module imports)
    from kai.utils.logger import set_global_filter

    # Set global filter BEFORE importing any kai modules
    if args.quiet and not args.verbose_logs:
        quiet_filter = QuietModeFilter()
        set_global_filter(quiet_filter)  # This ensures ALL future loggers get the filter

    # NOW import JupyterInterface (which will trigger kai module imports with filter applied)
    # Import it globally so TestRunner can access it
    global JupyterInterface
    from UI.jupyter import JupyterInterface

    # Now set up basic logging (after global filter is set)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Apply filter to root logger for any loggers that don't use setup_logger()
    if args.quiet and not args.verbose_logs:
        logging.root.addFilter(quiet_filter)
        for handler in logging.root.handlers:
            handler.addFilter(quiet_filter)

        # Environment variables already set at top of file to suppress tqdm/tokenizers
        logger.info("Quiet mode enabled - showing only high-level progress")
    elif args.verbose_logs:
        logger.info("Verbose logging enabled - showing all detailed logs")

    # Parse replicate range
    replicate_range = None
    if args.replicates:
        parts = args.replicates.split('-')
        if len(parts) == 2:
            replicate_range = (int(parts[0]), int(parts[1]))
        else:
            logger.error("Invalid replicate range format. Use: 1-2")
            sys.exit(1)

    # Create test runner
    runner = TestRunner(
        config_path=args.config,
        api_key=args.api_key,
        output_dir=args.output,
        notebook_python=args.env_notebook
    )

    # Override max iterations if provided
    if args.max_iterations:
        runner.max_iterations = args.max_iterations

    # Run tests
    try:
        await runner.run_all_tests(
            case_filter=args.cases,
            replicate_range=replicate_range
        )
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
