# Full Agent Test Module

Reproducible benchmarking system for Kai based on [kai_reproducibility](https://github.com/davidfischerlab/kai_reproducibility) scenarios.

## Overview

This module provides automated testing of Kai's autonomous agent capabilities by:
1. Running Scenario 1 test cases in triplicate using the Jupyter CLI interface
2. Parsing generated notebooks and execution metadata
3. Generating automated analysis with visualizations

## Quick Start

### 1. Run Tests

Execute all three Scenario 1 cases (blood, breast cancer, lung) in triplicate:

```bash
# Use shell script (runs each test in isolated subprocess)
./full_agent_test/run_tests.sh \
    --api-key YOUR_API_KEY \
    --env-notebook /path/to/notebook_env/bin/python
```

This will:
- Load configuration from `configs/scenario1_config.yaml`
- Copy base notebooks for each test
- Run Kai autonomously on each case in isolated subprocess
- Execute 3 replicates per case (9 total tests)
- Save notebooks and metadata to `full_agent_test/test_outputs/`

**Why subprocess isolation?** Each test runs in a fresh Python process, preventing ChromaDB file descriptor leaks that cause "Too many open files" errors.

### 2. Analyze Results

Generate summary statistics and visualizations:

```bash
python full_agent_test/analyze_results.py
```

This creates:
- **Progress heatmap** - Completion status for each case × replicate
- **Metrics summary** - Iterations, duration, success rates
- **Text report** - Detailed statistics per case
- **CSV data** - Raw summary statistics

Output saved to `full_agent_test/analysis/`

## Directory Structure

```
full_agent_test/
├── __init__.py                      # Package initialization
├── README.md                        # This file
├── run_tests.sh                     # Shell script for test orchestration
├── run_single_test.py               # Single test executor (called by shell)
├── read_config.py                   # YAML config reader helper
├── analyze_results.py               # Analysis script
│
├── configs/
│   └── scenario1_config.yaml        # Scenario 1 configuration
│
├── base_notebooks/                  # Symlink to kai_reproducibility base notebooks
│
├── test_outputs/                    # Generated test outputs (gitignored)
│   ├── .gitkeep
│   ├── full_agent_test_scenario1_blood_repeat1.ipynb
│   ├── full_agent_test_scenario1_blood_repeat1.log
│   ├── full_agent_test_scenario1_blood_repeat1_metadata.json
│   └── ...
│
└── analysis/                        # Analysis outputs (gitignored)
    ├── .gitkeep
    ├── progress_heatmap.png         # Completion heatmap
    ├── metrics_summary.png          # Metrics visualizations
    ├── summary_statistics.csv       # Raw data
    └── summary_report.txt           # Text summary
```

## Usage

### Running Tests

#### Run All Tests (Default)

```bash
./full_agent_test/run_tests.sh \
    --api-key YOUR_KEY \
    --env-notebook /path/to/env/bin/python
```

Runs all 3 cases × 3 replicates = 9 tests

#### Run Specific Cases

```bash
# Just blood and lung
./full_agent_test/run_tests.sh \
    --api-key YOUR_KEY \
    --env-notebook /path/to/env/bin/python \
    --cases "blood lung"

# Just breast cancer
./full_agent_test/run_tests.sh \
    --api-key YOUR_KEY \
    --env-notebook /path/to/env/bin/python \
    --cases "breastcancer"
```

#### Run Specific Replicates

```bash
# Run replicates 1-2 only
./full_agent_test/run_tests.sh \
    --api-key YOUR_KEY \
    --env-notebook /path/to/env/bin/python \
    --replicates 1-2

# Run replicate 3 only
./full_agent_test/run_tests.sh \
    --api-key YOUR_KEY \
    --env-notebook /path/to/env/bin/python \
    --replicates 3-3
```

#### Custom Output Directory

```bash
./full_agent_test/run_tests.sh \
    --api-key YOUR_KEY \
    --env-notebook /path/to/env/bin/python \
    --output-dir my_test_run
```

#### Override Max Iterations

```bash
# Allow more iterations for complex cases
./full_agent_test/run_tests.sh \
    --api-key YOUR_KEY \
    --env-notebook /path/to/env/bin/python \
    --max-iterations 150
```

### Analyzing Results

#### Analyze Default Test Directory

```bash
python full_agent_test/analyze_results.py
```

#### Analyze Custom Test Directory

```bash
python full_agent_test/analyze_results.py --test-dir my_test_run
```

#### Custom Analysis Output

```bash
python full_agent_test/analyze_results.py --output my_analysis
```

## Configuration

Configuration is defined in `configs/scenario1_config.yaml`:

```yaml
# Three Scenario 1 test cases
test_cases:
  - name: "blood"
    base_notebook: "scenario1_blood_base.ipynb"
    task: "Annotate the cell types in this PBMC dataset..."

  - name: "breastcancer"
    base_notebook: "scenario1_breastcancer_base.ipynb"
    task: "Annotate the cell types in this breast cancer tissue..."

  - name: "lung"
    base_notebook: "scenario1_lung_base.ipynb"
    task: "Annotate the cell types in this lung tissue..."

# Execution parameters
execution:
  num_replicates: 3
  max_iterations: 100
  timeout_minutes: 120

# LLM configuration
llm:
  provider: "ollama"
  rag_enabled: true
  turbo_mode: false
```

## Test Outputs

Each test execution generates:

### 1. Generated Notebook (`.ipynb`)

Modified notebook with all cells added/executed by Kai:
- `full_agent_test_scenario1_blood_repeat1.ipynb`
- `full_agent_test_scenario1_blood_repeat2.ipynb`
- `full_agent_test_scenario1_blood_repeat3.ipynb`
- ... (3 replicates × 3 cases = 9 notebooks)

### 2. Execution Log (`.log`)

Detailed log of agent execution (matches `--log-file` output from Jupyter CLI)

### 3. Metadata JSON (`_metadata.json`)

Execution metadata including:
```json
{
  "case_name": "blood",
  "replicate": 1,
  "success": true,
  "iterations": 42,
  "duration_minutes": 87.3,
  "llm_provider": "ollama",
  "rag_enabled": true,
  "start_time": "2025-12-03T10:15:00",
  "end_time": "2025-12-03T11:42:23"
}
```

### 4. Combined Results (`test_results_summary.json`)

All test results in one file for easy parsing

## Analysis Outputs

### 1. Progress Heatmap (`progress_heatmap.png`)

Matrix visualization showing completion status:
- **Rows**: Test cases (blood, breastcancer, lung)
- **Columns**: Replicates (repeat1, repeat2, repeat3)
- **Colors**:
  - Green (1.0) = Full success, no errors
  - Yellow (0.5) = Partial success, some errors
  - Red (0.0) = Failed

### 2. Metrics Summary (`metrics_summary.png`)

Four-panel visualization:
- **Panel 1**: Average iterations by case (with std dev)
- **Panel 2**: Average duration by case (with std dev)
- **Panel 3**: Cell execution success rate by case
- **Panel 4**: Workflow step completion (clustering, UMAP, annotation)

### 3. Summary Statistics (`summary_statistics.csv`)

Raw data table with columns:
```
case, replicate, success, iterations, duration_min, total_cells, code_cells,
executed_cells, successful_executions, failed_executions, execution_success_rate,
has_plots, has_clustering, has_umap, has_cell_type_annotation, num_errors
```

### 4. Text Report (`summary_report.txt`)

Human-readable summary:
```
==================================================
KAI FULL AGENT TEST - SUMMARY REPORT
==================================================

OVERALL STATISTICS
--------------------------------------------------
Total tests: 9
Successful: 8
Failed: 1
Success rate: 88.9%

PER-CASE STATISTICS
--------------------------------------------------

BLOOD:
  Tests: 3
  Success: 3/3
  Avg iterations: 38.7
  Avg duration: 76.2 min
  ...
```

## Scenario 1 Test Cases

### Blood (PBMC)

**Dataset**: Tabula Sapiens - peripheral blood mononuclear cells
**Task**: Cell type annotation using marker genes and automated tools
**Expected cell types**:
- CD4+ T cells
- CD8+ T cells
- B cells
- NK cells
- Monocytes

### Breast Cancer

**Dataset**: Breast cancer tissue
**Task**: Cell type annotation distinguishing tumor, immune, and stromal
**Expected cell types**:
- Epithelial cells
- T cells
- B cells
- Myeloid cells
- Fibroblasts
- Endothelial cells

### Lung

**Dataset**: Lung tissue
**Task**: Cell type annotation of epithelial, immune, and stromal populations
**Expected cell types**:
- Epithelial cells
- T cells
- B cells
- Macrophages
- Fibroblasts
- Endothelial cells

## Requirements

### Base Notebooks

Requires `kai_reproducibility` repository cloned somewhere accessible:
```
/path/to/kai_reproducibility
```

With base notebooks in:
```
/path/to/kai_reproducibility/base_notebooks/
  - full_agent_test_scenario1_blood_base.ipynb
  - full_agent_test_scenario1_breastcancer_base.ipynb
  - full_agent_test_scenario1_lung_base.ipynb
```

### Data Files

Base notebooks load `.h5ad` files from:
```
/path/to/kai_reproducibility/h5ads/
```

**IMPORTANT**: You must prepare the data files before running tests!

#### Prepare Data Files

The test runner expects preprocessed `.h5ad` files. To prepare them:

```bash
cd /path/to/kai_reproducibility

# Run the preparation notebooks (downloads from CELLxGENE census)
jupyter notebook scenario1_blood_preparation.ipynb  # Run all cells
jupyter notebook scenario1_breastcancer_preparation.ipynb  # Run all cells
jupyter notebook scenario1_lung_preparation.ipynb  # Run all cells
```

This will create:
- `h5ads/983d5ec9-40e8-4512-9e65-a572a9c486cb_scenario.h5ad` (PBMC, ~1.3GB)
- `h5ads/6c87755e-a671-41a8-9c7e-4e43b850a57b_scenario.h5ad` (Breast cancer, ~251MB)
- `h5ads/2ac76f1b-43ef-4271-8686-2f165570989f_scenario.h5ad` (Lung, ~526MB)

**The test runner automatically creates a symlink** `full_agent_test/h5ads` → `/path/to/kai_reproducibility/h5ads` to avoid copying large files.

### Python Dependencies

All dependencies from main Kai installation:
- `jupyter-client` (for Jupyter interface)
- `nbformat` (for notebook parsing)
- `pandas`, `numpy` (for analysis)
- `matplotlib`, `seaborn` (for visualization)
- `pyyaml` (for config parsing)

## Example Workflow

### Complete Test Run

```bash
# 1. Run all tests (takes ~8-12 hours for 9 tests)
./full_agent_test/run_tests.sh \
    --api-key YOUR_KEY \
    --env-notebook /path/to/env/bin/python

# Monitor progress
tail -f full_agent_test/test_outputs/full_agent_test_scenario1_blood_repeat1.log

# 2. Generate analysis
python full_agent_test/analyze_results.py

# 3. View results
open full_agent_test/analysis/progress_heatmap.png
open full_agent_test/analysis/metrics_summary.png
cat full_agent_test/analysis/summary_report.txt
```

### Quick Test (Single Case, Single Replicate)

```bash
# Run just one test for quick validation
./full_agent_test/run_tests.sh \
    --api-key YOUR_KEY \
    --env-notebook /path/to/env/bin/python \
    --cases "blood" \
    --replicates 1-1 \
    --max-iterations 50

# Analyze
python full_agent_test/analyze_results.py
```

### Partial Rerun

```bash
# Rerun just replicate 2 for all cases
./full_agent_test/run_tests.sh \
    --api-key YOUR_KEY \
    --env-notebook /path/to/env/bin/python \
    --replicates 2-2 \
    --output-dir full_agent_test/test_outputs  # Same dir to merge results

# Reanalyze all results
python full_agent_test/analyze_results.py
```

## Comparison with kai_reproducibility

| Aspect | kai_reproducibility | full_agent_test |
|--------|-------------------|-----------------|
| **Execution** | Manual VSCode runs | Automated CLI runs |
| **Analysis** | Jupyter notebook | Python script |
| **Replicates** | 3 per case | Configurable (default: 3) |
| **Output** | Manual review | Automated metrics |
| **Reproducibility** | Manual steps | Fully scripted |
| **Visualizations** | Manual plotting | Auto-generated |

## Troubleshooting

### Test Fails to Start

**Problem**: `ModuleNotFoundError: No module named 'UI.jupyter'`

**Solution**: Install Kai in editable mode:
```bash
pip install -e .
```

### Base Notebooks Not Found

**Problem**: `FileNotFoundError: scenario1_blood_base.ipynb`

**Solution**: Clone kai_reproducibility:
```bash
git clone https://github.com/davidfischerlab/kai_reproducibility.git
```

### Data Files Missing

**Problem**: Error loading `.h5ad` file in notebook

**Solution**: Download data files (see kai_reproducibility README):
```bash
cd /path/to/kai_reproducibility
# Run data preparation notebooks
```

### Test Times Out

**Problem**: Test exceeds 2 hour timeout

**Solution**: Increase timeout in config or CLI:
```bash
# Edit configs/scenario1_config.yaml
execution:
  timeout_minutes: 180  # 3 hours

# Or use --max-iterations to allow more time
python full_agent_test/run_tests.py --api-key KEY --max-iterations 150
```

### Analysis Script Fails

**Problem**: No metadata files found

**Solution**: Check test output directory:
```bash
ls full_agent_test/test_outputs/*.json
```

If empty, tests didn't complete - check logs.

## Development

### Adding New Test Cases

1. Add case to `configs/scenario1_config.yaml`:
```yaml
test_cases:
  - name: "newcase"
    base_notebook: "scenario1_newcase_base.ipynb"
    task: "Task description..."
    description: "Case description"
```

2. Create base notebook in kai_reproducibility

3. Run tests:
```bash
./full_agent_test/run_tests.sh \
    --api-key KEY \
    --env-notebook /path/to/env/bin/python \
    --cases "newcase"
```

### Custom Analysis Metrics

Add custom metrics in `analyze_results.py`:

```python
def analyze_notebook(self, notebook_path: Path) -> Dict[str, Any]:
    # ... existing code ...

    # Add custom metric
    analysis['custom_metric'] = self._compute_custom_metric(nb)

    return analysis
```

### Custom Visualizations

Add plots in `analyze_results.py`:

```python
def create_custom_plot(self, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Your plotting code
    plt.savefig(self.output_dir / 'custom_plot.png', dpi=300)
```

## Future Enhancements

Potential additions:
- [ ] Scenario 2 and 3 test configurations
- [ ] Parallel test execution (multiple cases simultaneously)
- [ ] Automated comparison with kai_reproducibility results
- [ ] CI/CD integration for automated testing
- [ ] Web dashboard for results visualization
- [ ] Time-series tracking of agent performance
- [ ] Automated slack/email notifications on completion

## See Also

- [kai_reproducibility](https://github.com/davidfischerlab/kai_reproducibility) - Original benchmark repository
- [Jupyter Interface README](../UI/jupyter/README.md) - Jupyter CLI documentation
- [Main Kai README](../README.md) - Kai installation and setup
