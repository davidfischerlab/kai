#!/usr/bin/env python3
"""
Command-line interface for Kai Jupyter interface.

Enables running Kai autonomous agent from the shell without VSCode.

IMPORTANT: This script must be run with the Python from the Kai agent environment.
The notebook code will run in a separate environment specified by --env-notebook.

Usage:
    # Run with agent environment Python
    /path/to/kai_env/bin/python -m UI.jupyter \
        --notebook analysis.ipynb \
        --env-notebook /path/to/notebook_env \
        --task "Load and analyze data"

    # With output file
    /path/to/kai_env/bin/python -m UI.jupyter \
        --notebook input.ipynb \
        --env-notebook /path/to/notebook_env \
        --task "Fix errors" \
        --output output.ipynb

    # Interactive mode (single message)
    /path/to/kai_env/bin/python -m UI.jupyter \
        --notebook analysis.ipynb \
        --env-notebook /path/to/notebook_env \
        --task "What does this code do?" \
        --no-autonomous
"""

import argparse
import asyncio
import sys
import os
import logging
from pathlib import Path

# Disable prompt debugging for Jupyter CLI (prevents file descriptor exhaustion in long runs)
# Writes every LLM prompt/response to disk - can create thousands of files in autonomous mode
os.environ['KAI_DEBUG_PROMPTS'] = 'false'

from UI.jupyter import JupyterInterface
from kai.utils import setup_logger


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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Kai agent with Jupyter notebooks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run autonomous mode to complete a task
  python -m kai.ui.jupyter --notebook analysis.ipynb --task "Load data and perform QC"

  # Save to different output file
  python -m kai.ui.jupyter --notebook input.ipynb --task "Fix errors" --output fixed.ipynb

  # Use specific kernel
  python -m kai.ui.jupyter --notebook analysis.ipynb --task "Analyze" --kernel ir

  # Interactive mode (single message)
  python -m kai.ui.jupyter --notebook analysis.ipynb --task "Explain this code" --no-autonomous

  # With detailed logging
  python -m kai.ui.jupyter --notebook analysis.ipynb --task "Analyze" --log-file session.log --verbose
        """
    )

    # Required arguments
    parser.add_argument(
        '--notebook',
        type=str,
        required=True,
        help='Path to Jupyter notebook file (.ipynb)'
    )

    parser.add_argument(
        '--task',
        type=str,
        required=True,
        help='Task description for Kai agent'
    )

    # Optional arguments
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output notebook path (default: overwrite input notebook)'
    )

    parser.add_argument(
        '--env-notebook',
        type=str,
        required=True,
        help='REQUIRED. Python executable or conda/mamba environment for notebook kernel. '
             'Examples: /path/to/env/bin/python OR /path/to/env'
    )

    parser.add_argument(
        '--llm-provider',
        type=str,
        default='ollama',
        choices=['ollama', 'ollama-turbo', 'openai'],
        help='LLM provider (default: ollama)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Specific model to use (default: provider default)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key for LLM provider (default: from environment)'
    )

    parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Disable retrieval-augmented generation'
    )

    parser.add_argument(
        '--turbo',
        action='store_true',
        help='Enable turbo mode for faster iteration'
    )

    parser.add_argument(
        '--no-autonomous',
        action='store_true',
        help='Run in interactive mode (single message) instead of autonomous'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=50,
        help='Maximum autonomous iterations (default: 50)'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (default: no file logging)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
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

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(__name__, level=log_level)

    # Apply quiet mode filter (default) unless verbose-logs is requested
    if args.quiet and not args.verbose_logs:
        quiet_filter = QuietModeFilter()

        # Apply to root logger and its handlers
        logging.root.addFilter(quiet_filter)
        for handler in logging.root.handlers:
            handler.addFilter(quiet_filter)

        # Apply to ALL existing loggers (critical because kai.* loggers have propagate=False)
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            existing_logger = logging.getLogger(logger_name)
            existing_logger.addFilter(quiet_filter)
            # Apply to all handlers of each logger
            for handler in existing_logger.handlers:
                handler.addFilter(quiet_filter)

        # Suppress tokenizers parallelism warning
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Suppress tqdm progress bars from sentence-transformers
        os.environ['TQDM_DISABLE'] = '1'

        logger.info("Quiet mode enabled - showing only high-level progress")
    elif args.verbose_logs:
        logger.info("Verbose logging enabled - showing all detailed logs")

    # Add file logging if requested
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger('kai').addHandler(file_handler)
        logger.info(f"Logging to file: {args.log_file}")

    # Validate notebook path
    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        logger.error(f"Notebook not found: {notebook_path}")
        sys.exit(1)

    # Initialize interface
    logger.info(f"Initializing Kai Jupyter interface")
    logger.info(f"Notebook: {notebook_path}")
    logger.info(f"Agent Python: {sys.executable}")
    logger.info(f"Notebook environment: {args.env_notebook}")
    logger.info(f"LLM Provider: {args.llm_provider}")
    logger.info(f"RAG: {'disabled' if args.no_rag else 'enabled'}")
    logger.info(f"Turbo: {'enabled' if args.turbo else 'disabled'}")

    try:
        with JupyterInterface(
            notebook_path=str(notebook_path),
            notebook_python=args.env_notebook,
            llm_provider=args.llm_provider,
            model=args.model,
            api_key=args.api_key,
            rag_enabled=not args.no_rag,
            turbo_enabled=args.turbo
        ) as interface:

            if args.no_autonomous:
                # Interactive mode - single message
                logger.info("Running in interactive mode")
                logger.info(f"Task: {args.task}")

                response = await interface.run_interactive(args.task)

                logger.info("Response received:")
                logger.info(f"{response}")

            else:
                # Autonomous mode
                logger.info("Running in autonomous mode")
                logger.info(f"Task: {args.task}")
                logger.info(f"Max iterations: {args.max_iterations}")

                result = await interface.run_autonomous(
                    initial_message=args.task,
                    max_iterations=args.max_iterations
                )

                # Log results
                if result['success']:
                    logger.info("✅ Autonomous execution completed successfully")
                    logger.info(f"Iterations: {result['iterations']}")
                    logger.info(f"Final state: {result['final_state']}")
                else:
                    logger.error("❌ Autonomous execution failed")
                    logger.error(f"Error: {result.get('error', 'Unknown error')}")
                    logger.error(f"Iterations completed: {result['iterations']}")
                    sys.exit(1)

            # Save notebook
            output_path = args.output or args.notebook
            logger.info(f"Saving notebook to: {output_path}")
            interface.save(output_path)

            logger.info("✅ Complete! Notebook saved successfully")

    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
