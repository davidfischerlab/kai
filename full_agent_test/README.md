# Full Agent Test

Automated benchmarking for Kai agent based on kai_reproducibility Scenario 1.

## Run Tests

```bash
./full_agent_test/run_tests.sh \
    --api-key YOUR_KEY \
    --env-agent /path/to/kai_agent \
    --env-notebook /path/to/notebook_env
```

Options:
- `--cases "blood lung"` - Run specific cases
- `--replicates 1-2` - Run specific replicates
- `--max-iterations 150` - Override iteration limit

## Analyze Results

```bash
python full_agent_test/analyze_results.py \
    --test-dir full_agent_test/test_outputs \
    --api-key YOUR_KEY
```

Generates:
- Progress heatmap (stage completion by case/replicate)
- Review JSON files (LLM evaluation per notebook)

## Configuration

Edit `configs/scenario1_config.yaml` to modify:
- Test cases and prompts
- Number of replicates
- Max iterations
- LLM provider

## Directory Structure

```
full_agent_test/
├── run_tests.sh              # Test orchestrator
├── run_single_test.py        # Single test executor
├── analyze_results.py        # LLM-based evaluation
├── configs/
│   └── scenario1_config.yaml # Test configuration
├── test_outputs/             # Test results (gitignored)
└── analysis/                 # Analysis outputs (gitignored)
```

## Requirements

- Base notebooks from kai_reproducibility
- Data files (`.h5ad`) prepared via kai_reproducibility preparation notebooks
