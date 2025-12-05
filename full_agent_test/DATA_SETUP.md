# Data Setup for Full Agent Tests

## Overview

The test runner needs access to preprocessed `.h5ad` data files. These files should be available in the `h5ads/` directory in the repo root.

## Data Files Location

Data files are expected in:
```
full_agent_test/h5ads/
```

You can either:
1. **Symlink to kai_reproducibility** (recommended to save space):
   ```bash
   ln -s /path/to/kai_reproducibility/h5ads full_agent_test/h5ads
   ```

2. **Copy the data files directly**:
   ```bash
   cp -r /path/to/kai_reproducibility/h5ads full_agent_test/h5ads
   ```

**Note**: The `full_agent_test/h5ads/` directory is git-ignored, so you can use either approach.

## Required Files

### Scenario 1 (Cell Type Annotation)

**Blood (PBMC)**
- File: `983d5ec9-40e8-4512-9e65-a572a9c486cb_scenario.h5ad`
- Size: ~1.3 GB
- Cells: 82,785 cells × 61,888 genes
- Source: Tabula Sapiens blood dataset
- Preparation: `scenario1_blood_preparation.ipynb`

**Breast Cancer**
- File: `6c87755e-a671-41a8-9c7e-4e43b850a57b_scenario.h5ad`
- Size: ~251 MB
- Cells: Breast cancer biopsy samples
- Source: CELLxGENE census
- Preparation: `scenario1_breastcancer_preparation.ipynb`

**Lung (COVID-19 patient)**
- File: `2ac76f1b-43ef-4271-8686-2f165570989f_scenario.h5ad`
- Size: ~526 MB
- Cells: Lung tissue from COVID-19 patient
- Source: CELLxGENE census
- Preparation: `scenario1_lung_preparation.ipynb`

## Verification

Check if data files exist:

```bash
ls -lh /path/to/kai_reproducibility/h5ads/*_scenario.h5ad
```

Should show 3-4 files (Scenario 1 + Scenario 3).

## If Data Files Are Missing

If the data files don't exist, you need to run the preparation notebooks:

```bash
cd /path/to/kai_reproducibility

# Option 1: Run in Jupyter
jupyter notebook scenario1_blood_preparation.ipynb
jupyter notebook scenario1_breastcancer_preparation.ipynb
jupyter notebook scenario1_lung_preparation.ipynb

# Option 2: Run from command line
jupyter nbconvert --execute --to notebook --inplace scenario1_blood_preparation.ipynb
jupyter nbconvert --execute --to notebook --inplace scenario1_breastcancer_preparation.ipynb
jupyter nbconvert --execute --to notebook --inplace scenario1_lung_preparation.ipynb
```

**Note**: Preparation notebooks download data from CELLxGENE census API and may take 10-30 minutes each.

## Data Preparation Process

Each preparation notebook:
1. Downloads raw data from CELLxGENE census using `cellxgene_census` package
2. Filters to specific assays (10x 3' v3, 10x 5' v2)
3. Normalizes counts (total count normalization + log1p)
4. Removes ground truth cell type labels (for benchmarking)
5. Saves preprocessed `.h5ad` file

## Base Notebooks

Base notebooks (in `kai_reproducibility/base_notebooks/`) contain only:
1. Kernel configuration (OMP thread settings)
2. Data loading code (expects `./h5ads/` directory)
3. No analysis code - agent generates everything

Example from `scenario1_blood_base.ipynb`:
```python
import anndata
import scanpy as sc

DIR_H5ADS = "./h5ads"
DATASET_ID = "983d5ec9-40e8-4512-9e65-a572a9c486cb"
FN = os.path.join(DIR_H5ADS, DATASET_ID + "_scenario.h5ad")

adata = anndata.read_h5ad(FN)
```

## Setup Instructions

Before running tests, ensure data is available:

```bash
# Navigate to repo root
cd /path/to/kai

# Create symlink (recommended)
ln -s /path/to/kai_reproducibility/h5ads full_agent_test/h5ads

# Verify symlink works
ls -lh h5ads/*_scenario.h5ad
```

This approach:
- Saves disk space (no file copying)
- Works for both development and production
- Allows others to copy files directly if they prefer

## Troubleshooting

### Error: "Data directory not found"

```
FileNotFoundError: Data directory not found: /path/to/kai_reproducibility/h5ads
```

**Solution**: Clone kai_reproducibility repository and run preparation notebooks.

### Error: "FileNotFoundError" when notebook loads data

```
FileNotFoundError: [Errno 2] No such file or directory: './h5ads/983d5ec9-40e8-4512-9e65-a572a9c486cb_scenario.h5ad'
```

**Solution**: The symlink wasn't created. Check that:
1. `/path/to/kai_reproducibility/h5ads/` exists
2. Data files exist in that directory
3. Symlink exists: `ls -la full_agent_test/test_outputs/h5ads`

### Symlink shows "broken"

If the symlink is broken:

```bash
# Remove broken symlink
rm full_agent_test/test_outputs/h5ads

# Verify source exists
ls /path/to/kai_reproducibility/h5ads/

# Re-run test - it will recreate symlink
python full_agent_test/run_tests.py --api-key YOUR_KEY
```

## Disk Space Requirements

Total space for all Scenario 1 data files: **~2.1 GB**

Breakdown:
- PBMC: 1.3 GB
- Breast cancer: 251 MB
- Lung: 526 MB

Plus raw downloaded files (retained by preparation notebooks): **~8-10 GB**

To free space, you can delete the raw files (keep only `*_scenario.h5ad`):
```bash
cd /path/to/kai_reproducibility/h5ads
# Keep only scenario files
ls | grep -v "_scenario.h5ad" | xargs rm
```

## Summary

✅ **Setup is automatic** - just ensure kai_reproducibility has the data files

✅ **No manual copying needed** - symlink handles file access

✅ **Data already exists** on your system (verified)

✅ **Test runner handles everything** - creates symlink on first run

Just run the tests and it will work! 🎉
