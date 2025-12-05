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
