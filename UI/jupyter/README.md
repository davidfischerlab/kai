# Kai Jupyter Interface

Run Kai's autonomous agent directly with Jupyter notebooks from the command line, **without VSCode**.

## Quick Start

```bash
# Basic usage
python -m UI.jupyter \
  --notebook analysis.ipynb \
  --task "Load data and perform quality control"

# Save to different file
python -m UI.jupyter \
  --notebook input.ipynb \
  --task "Fix clustering errors" \
  --output output.ipynb

# With logging
python -m UI.jupyter \
  --notebook analysis.ipynb \
  --task "Complete the analysis" \
  --log-file session.log
```

## Use Cases

- **Reproducible benchmarking** - Shell-scriptable with deterministic behavior
- **Automated testing** - Integrate into CI/CD pipelines
- **Batch processing** - Process multiple notebooks programmatically

## Installation

Requires `jupyter_client` in addition to standard Kai installation:

```bash
pip install jupyter_client
```

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--notebook PATH` | Path to notebook | **Required** |
| `--task TEXT` | Task for agent | **Required** |
| `--output PATH` | Output notebook | Overwrites input |
| `--kernel NAME` | Jupyter kernel | `python3` |
| `--max-iterations N` | Max iterations | 50 |
| `--log-file PATH` | Log file | No logging |
| `--verbose` | Verbose output | Info level |
| `--no-rag` | Disable RAG | RAG enabled |

See `python -m UI.jupyter --help` for all options.

## Programmatic Usage

```python
from UI.jupyter import JupyterInterface
import asyncio

async def run():
    with JupyterInterface(
        notebook_path='analysis.ipynb',
        kernel_name='python3'
    ) as interface:
        result = await interface.run_autonomous(
            initial_message="Analyze scRNA-seq data",
            max_iterations=50
        )
        interface.save('output.ipynb')
    return result

result = asyncio.run(run())
```

## Features

The Jupyter interface implements **all** autonomous capabilities from VSCode:
- Cell execution with full output capture (stdout, stderr, plots)
- Add/replace/delete cells
- Error detection and recovery
- Kernel restart and re-execution
- Task list tracking
- Progress monitoring for long-running cells

Context format is **identical** to VSCode, so agent behavior is the same.

## Testing

```bash
# Run basic tests
python UI/jupyter/tests/test_basic.py

# Full test suite (requires kai_reproducibility scenarios)
python full_agent_test/run_tests.py --api-key YOUR_KEY
```

## Limitations

- No visual interface (terminal output only)
- No interactive chat (single task per run)
- No real-time cell preview (check log file or output notebook)

## Architecture

```
Shell → JupyterInterface → KaiAgent (core)
            ↓
        NotebookExecutor (jupyter_client)
        ContextBuilder (context formatting)
            ↓
        Jupyter Kernel ↔ .ipynb file
```

See module docstrings for implementation details:
- [jupyter_interface.py](jupyter_interface.py) - Main entry point
- [notebook_executor.py](notebook_executor.py) - Kernel management
- [context_builder.py](context_builder.py) - Context formatting

## Troubleshooting

**Kernel issues:**
```bash
jupyter kernelspec list  # Check available kernels
```

**Import errors:**
```bash
python -c "from UI.jupyter import JupyterInterface; print('OK')"
```

**See logs:**
```bash
python -m UI.jupyter --notebook test.ipynb --task "test" --verbose
```

## Recent Bug Fixes

### Kernel Environment Configuration (Dec 2024)
**Issue**: Incorrect environment passing method - code was trying to set `km.kernel_spec.env` before starting the kernel, which doesn't actually work.

**Root Cause**: The code attempted to modify `kernel_spec.env` directly (line 88), but this attribute is not directly modifiable. The correct approach is to pass the `env` parameter to `start_kernel()`.

**Fix**: Changed from:
```python
env = os.environ.copy()
env['PATH'] = f"{python_dir}:{env.get('PATH', '')}"
self.km.kernel_spec.env = env  # Wrong - doesn't work
self.km.start_kernel()
```

To:
```python
env = os.environ.copy()
env['PATH'] = f"{python_dir}:{env.get('PATH', '')}"
self.km.start_kernel(env=env)  # Correct - env parameter
```

**Files Modified**:
- [UI/jupyter/notebook_executor.py](notebook_executor.py) - Pass env to start_kernel() instead of setting kernel_spec.env

**Verification**: Created test that verifies:
1. Kernel uses the specified Python executable (sys.executable matches)
2. Kernel has access to packages from the specified environment

**Impact**: Kernel now properly uses the specified Python environment and has access to all packages from that environment.

---

### Autonomous Loop Completion Detection (Dec 2024)
**Issue**: Jupyter interface ran until `max_iterations` even when agent reported all tasks complete (`LOOP_COMPLETE`).

**Root Cause**: The Jupyter interface didn't capture or check `workflow_result` messages with `LOOP_COMPLETE` signals. VSCode TypeScript code handles these signals to stop the loop, but Jupyter had no equivalent logic.

**Fix**:
1. Added monkey-patch to capture `send_workflow_result()` calls (same as `send_tool_result`)
2. Modified `_process_tool_messages()` to detect `auto_loop_update: LOOP_COMPLETE`
3. Updated autonomous loop to break when `LOOP_COMPLETE` is detected

**Files Modified**:
- [UI/jupyter/jupyter_interface.py](jupyter_interface.py) - Capture workflow results, detect LOOP_COMPLETE, break loop
- [UI/jupyter/tests/test_basic.py](tests/test_basic.py) - Added test for LOOP_COMPLETE detection

**Impact**: Autonomous mode now stops as soon as all tasks are complete, matching VSCode behavior exactly.

---

### VSCode Message Suppression (Dec 2024)
**Issue**: VSCode JSON messages (`{"type": "console_log", ...}`) were appearing in Jupyter CLI output despite `suppress_vscode_messages=True`.

**Root Cause**: When starting a new autonomous session, `agent.chat()` called `vscode.enable_communication()` which re-enabled messages even though suppression was requested during initialization.

**Fix**: Added `_suppress_vscode_messages` flag to remember the suppression preference. Now `enable_communication()` is only called if suppression was not requested.

**Files Modified**:
- [kai/core/agent.py](../../kai/core/agent.py) - Remember suppression preference, conditionally call enable_communication()
- [UI/jupyter/tests/test_basic.py](tests/test_basic.py) - Added test for suppression persistence

**Impact**: Clean output for Jupyter CLI and automated testing. No VSCode JSON messages in logs or stdout.
