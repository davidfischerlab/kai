#!/usr/bin/env python3
"""
Evaluate Jupyter notebooks against abstract stage criteria using Ollama LLM.

This script reads each notebook, sends it to an Ollama LLM with evaluation criteria,
and collects the results into a stage_cell_map dictionary.

Usage:
    python evaluate_notebooks_ollama.py <OLLAMA_AUTH_KEY>
"""

import json
import glob
import os
import sys
import requests
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Original task prompts that generated the notebooks
ORIGINAL_TASK_PROMPTS = {
    'blood': """You are given single-cell RNA-seq data from human PBMC as adata. Cells are already filtered and ready for analysis. Gene expression data is given as raw counts in adata.layer["counts"].X and as log1p-normalized in adata.X. There are multiple samples from different donors in this dataset, you can find the assignments in .obs["donor_id"] - consider this information in the analyses below: correct for this batch effect with harmony if you find that it is necessary but avoid batch effect statistics that depend on per-cell KNN statistics because those take long to compute.

Perform a leiden clustering to use as a basis for cell type annotation. Show the downloaded models in celltypist (don't download any models). Based on that list and any information that you can find about these models, choose one that is generally applicable to immune cells and one that is specific for human PBMC. Use those models to annotate the leiden clusters with cell type labels. Then, assemble a marker gene panel from literature sources for the cell types predicted by the PBMC-specific model, matching its cell type granularity, taking care to assemble a comprehensive panel with several genes per cell type. Use that panel to compute gene expression scores for each cell type. Interpret these scores per cluster and use these scores to assign cell type labels to clusters. Conclude to what degree one can use the results from this celltypist analysis for downstream analyses by working through the agreement of celltypist with the marker gene approach - take care to account for differences in cell type naming and granularity between the different annotations. Generate one final cell type annotation for downstream analyses.""",

    'breastcancer': """You are given single-cell RNA-seq data from human breast cancer biopsies as adata. Cells are already filtered and ready for analysis. Gene expression data is given as raw counts in adata.layer["counts"].X and as log1p-normalized in adata.X. There are multiple samples from different donors in this dataset, you can find the assignments in .obs["donor_id"] - consider this information in the analyses below: correct for this batch effect with harmony if you find that it is necessary but avoid batch effect statistics that depend on per-cell KNN statistics because those take long to compute.

Use tutorials on the following concepts:
- Basic celltypist usage from the celltypist repository.
- Batch effects & harmony from scverse if available.

Perform a leiden clustering to use as a basis for cell type annotation. Show the downloaded models in celltypist (don't download any models). Based on that list and any information that you can find about these models, choose two that are applicable to this tissue. Use those models to annotate the leiden clusters with cell type labels. Then, assemble a marker gene panel from literature sources for the cell types predicted by the models, matching their cell type granularity, taking care to assemble a comprehensive panel with several genes per cell type. Use that panel to compute gene expression scores for each cell type. Interpret these scores per cluster and use these scores to assign cell type labels to clusters. Conclude to what degree one can use the results from this celltypist analysis for downstream analyses by working through the agreement of celltypist with the marker gene approach - take care to account for differences in cell type naming and granularity between the different annotations. Generate one final cell type annotation for downstream analyses.""",

    'lung': """You are given single-cell RNA-seq data from human lung tissue of a COVID-19 patient as adata. Cells are already filtered and ready for analysis. Gene expression data is given as raw counts in adata.layer["counts"].X and as log1p-normalized in adata.X.

Use tutorials on the following concepts:
- Basic celltypist usage from the celltypist repository.
- Basic single-cell analysis tutorial for leiden clustering.

Perform a leiden clustering to use as a basis for cell type annotation. Show the downloaded models in celltypist (don't download any models). Based on that list and any information that you can find about these models, choose two that are applicable to this tissue. Use those models to annotate the leiden clusters with cell type labels. Then, assemble a marker gene panel from literature sources for the cell types predicted by the models, matching their cell type granularity, taking care to assemble a comprehensive panel with several genes per cell type. Use that panel to compute gene expression scores for each cell type. Interpret these scores per cluster and use these scores to assign cell type labels to clusters. Conclude to what degree one can use the results from this celltypist analysis for downstream analyses by working through the agreement of celltypist with the marker gene approach - take care to account for differences in cell type naming and granularity between the different annotations. Generate one final cell type annotation for downstream analyses."""
}

# Stage definitions as dictionary
STAGE_DEFINITIONS_DICT = {
    'blood': {
        'batcheffect_implementation': 'Quantitative assessment of donor batch effects is performed.',
        'batcheffect_interpretation': {
            'criterion': 'The batch effect assessment results are interpreted and an explicit decision is made about whether batch correction is necessary.',
            'reasoning_check': 'Batch effects exist between donors in this dataset - was this recognized?'
        },
        'optional_harmony_implementation': 'Based on the decision, batch correction is either applied or explicitly skipped, and the appropriate embeddings are used for downstream clustering.',
        'leiden_clustering': 'Unsupervised clustering is performed to identify discrete cell populations.',
        'celltypist_selection': {
            'criterion': 'Appropriate cell typist models are selected.',
            'reasoning_check': 'Were models selected that are specific to immune/PBMC cells (tissue-appropriate)?'
        },
        'celltypist_implementation': 'Celltypist models are loaded and annotation is performed.',
        'gene_set_scoring': 'Marker gene expression scores are computed for relevant cell type signatures.',
        'gene_set_interpretation': {
            'criterion': 'Gene set scores are aggregated per cluster and interpreted with respect to infer cell type identities.',
            'reasoning_check': 'Were gene sets defined for meaningful cell types present in PBMC (e.g., T cells, B cells, monocytes, NK cells)?'
        },
        'discussion': {
            'criterion': 'Final predictions are assembled and results from celltypist and gene-set-based scoring compared.',
            'reasoning_check': 'Does the discussion compare both celltypist and gene set results, weighing evidence from both approaches?'
        }
    },
    'breastcancer': {
        'batcheffect_implementation': 'Quantitative assessment of donor batch effects is performed.',
        'batcheffect_interpretation': {
            'criterion': 'The batch effect assessment results are interpreted and an explicit decision is made about whether batch correction is necessary.',
            'reasoning_check': 'Batch effects exist between donors in this dataset - was this recognized?'
        },
        'optional_harmony_implementation': 'Based on the decision, batch correction is either applied or explicitly skipped, and the appropriate embeddings are used for downstream clustering.',
        'leiden_clustering': 'Unsupervised clustering is performed to identify discrete cell populations.',
        'celltypist_selection': {
            'criterion': 'Appropriate cell typist models are selected.',
            'reasoning_check': 'Were models selected that are appropriate for breast cancer tissue (tumor + immune)?'
        },
        'celltypist_implementation': 'Celltypist models are loaded and annotation is performed.',
        'gene_set_scoring': 'Marker gene expression scores are computed for relevant cell type signatures.',
        'gene_set_interpretation': {
            'criterion': 'Gene set scores are aggregated per cluster and interpreted with respect to infer cell type identities.',
            'reasoning_check': 'Were gene sets defined for meaningful cell types in breast cancer (e.g., epithelial/tumor cells, immune cells, stromal cells)?'
        },
        'discussion': {
            'criterion': 'Final predictions are assembled and results from celltypist and gene-set-based scoring compared.',
            'reasoning_check': 'Does the discussion compare both celltypist and gene set results, weighing evidence from both approaches?'
        }
    },
    'lung': {
        'leiden_clustering': 'Unsupervised clustering is performed.',
        'celltypist_selection': {
            'criterion': 'Appropriate cell typist models are selected.',
            'reasoning_check': 'Were models selected that are appropriate for lung tissue?'
        },
        'celltypist_implementation': 'Celltypist models are loaded and annotation is performed.',
        'gene_set_scoring': 'Marker gene expression scores are computed for relevant cell type signatures.',
        'gene_set_interpretation': {
            'criterion': 'Gene set scores are aggregated per cluster and interpreted with respect to infer cell type identities.',
            'reasoning_check': 'Were gene sets defined for meaningful cell types in lung (e.g., epithelial cells, immune cells, endothelial cells)?'
        },
        'discussion': {
            'criterion': 'Final predictions are assembled and results from celltypist and gene-set-based scoring compared.',
            'reasoning_check': 'Does the discussion compare both celltypist and gene set results, weighing evidence from both approaches?'
        }
    }
}


def format_stage_definitions(tissue):
    """Format stage definitions for the given tissue as a text string."""
    stages = STAGE_DEFINITIONS_DICT[tissue]
    original_task = ORIGINAL_TASK_PROMPTS[tissue]

    lines = ["## Original Task Objective\n"]
    lines.append("The notebooks you are evaluating were generated by an AI agent attempting to address the following task:\n")
    lines.append(f"```\n{original_task}\n```\n")

    lines.append("\n## Stage Definitions for Analysis\n")
    lines.append(f"### Tissue: {tissue}\n")
    lines.append("Evaluate whether the notebook accomplished the following stages:\n")

    for i, (stage_name, stage_info) in enumerate(stages.items(), 1):
        if isinstance(stage_info, dict):
            # Stage with reasoning check
            lines.append(f"{i}. **{stage_name}**: {stage_info['criterion']}\n")
            lines.append(f"   - **Reasoning Quality Check**: {stage_info['reasoning_check']}\n")
        else:
            # Simple stage
            lines.append(f"{i}. **{stage_name}**: {stage_info}\n")

    return '\n'.join(lines)


def read_notebook_as_text(notebook_path):
    """Read a Jupyter notebook and convert it to readable text format."""
    try:
        with open(notebook_path, 'r') as f:
            content = f.read()

        # Check if it's a text-only file (not a notebook)
        if not content.strip().startswith('{'):
            raise ValueError("Not a valid Jupyter notebook - appears to be a text file")

        nb = json.loads(content)

        # Verify it has the notebook structure
        if 'cells' not in nb:
            raise ValueError("Not a valid Jupyter notebook - missing 'cells' key")

    except json.JSONDecodeError as e:
        raise ValueError(f"Cannot parse notebook JSON: {e}")

    text_parts = []
    text_parts.append(f"# NOTEBOOK: {os.path.basename(notebook_path)}\n")

    for i, cell in enumerate(nb['cells']):
        cell_type = cell['cell_type']
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        text_parts.append(f"\n## CELL {i} ({cell_type.upper()})")

        if cell_type == 'code':
            has_output = 'outputs' in cell and len(cell.get('outputs', [])) > 0
            text_parts.append(f"HAS_OUTPUT: {has_output}")
            text_parts.append(f"\nSOURCE:\n{source}")

            if has_output:
                text_parts.append(f"\nOUTPUTS:")
                for output in cell['outputs']:
                    # Check for error outputs
                    if output.get('output_type') == 'error':
                        error_name = output.get('ename', 'Unknown')
                        error_value = output.get('evalue', '')
                        text_parts.append(f"ERROR: {error_name}: {error_value}")
                    elif 'text' in output:
                        output_text = ''.join(output['text']) if isinstance(output['text'], list) else output['text']
                        # Truncate very long outputs
                        if len(output_text) > 1000:
                            output_text = output_text[:1000] + "\n... (truncated)"
                        text_parts.append(output_text)
                    elif 'data' in output:
                        text_parts.append(str(output['data'])[:500] + "... (truncated)")
        else:  # markdown
            text_parts.append(f"\nCONTENT:\n{source}")

    return '\n'.join(text_parts)


def evaluate_notebook_with_ollama(notebook_text, notebook_name, tissue, ollama_url, model_name, auth_key):
    """Send notebook to Ollama for evaluation."""

    # Derive expected stages from the dictionary
    expected_stages = list(STAGE_DEFINITIONS_DICT[tissue].keys())

    # Get stages with reasoning checks
    reasoning_stages = [
        stage_name for stage_name, stage_info in STAGE_DEFINITIONS_DICT[tissue].items()
        if isinstance(stage_info, dict) and 'reasoning_check' in stage_info
    ]

    # Format stage definitions for this tissue
    stage_definitions_text = format_stage_definitions(tissue)

    # Build the JSON template with proper stage keys
    stage_impl_template = ',\n        '.join(f'"{stage}": cell_index or "NA"' for stage in expected_stages)
    stage_comp_template = ',\n        '.join(f'"{stage}": cell_index or "NA"' for stage in expected_stages)
    reasoning_template = ',\n        '.join(f'"{stage}": "success"/"failure"' for stage in reasoning_stages)

    prompt = f"""{stage_definitions_text}

---

You are evaluating a Jupyter notebook against the stage definitions above.

**Notebook:** {notebook_name}
**Tissue type:** {tissue}
**Expected stages:** {', '.join(expected_stages)}

Below is the complete notebook content with cell indices, source code, and outputs.

YOUR TASK:
1. Read and understand the notebook completely
2. Score stage_implementation, error, stage_completion, and reasoning_completion as defined below
3. Return ONLY a JSON object mapping stage names to cell indices

IMPORTANT:
First assess "stage_implementation" and "error" to gain an overview of the notebook and then use that to score stage_completion and reasoning_completion.
For stage_implementation:
- For each stage, identify if the stage was implemented and yield the index of the first cell in which it was implemented, otherwise yield "NA"
- A stage counts as successfully implemented if there are cells that would fully address it if they ran without error
- Note that cells do not need to be error free and do not have to be executed in the notebook version to count as implemented

For error: 
- Check if any code cell has an error output (output_type: "error")
- If an error occurred, identify which stage it happened in

For stage_completion:
- Use the original task objective above to understand the context and intent of what the notebook was trying to accomplish
- For each stage, identify if the stage was completed and if completed, yield the index of the last cell that was part of this stage, otherwise yield "NA"
- If an error occurred, ALL stages at or after the error cell must be marked as "NA" in stage_completion (notebook execution stops at errors)
- Completion is defined as follows: the stage was implemented and the cells that correspond to that stage were executed successfully. Specifically, this means:
    - Code cells: Code cells that belong to a stage must have been executed without error for the stage to be completed
                  Timeouts or errors indicating that a notebook was aborted because of excessive run time should be treated as failures like any other errors
    - Markdown cells (reasoning): markdown cells count as executed if present (they don't have outputs) and they are not positioned after cells that contain errors

For reasoning_completion:
- For all stages that have reasoning checks defined above: score whether reasonable conclusions were reached
- The reasoning consists of text that addresses the problem or code that implements decision making on the problem - it may be within code cells or in separate markdown cells
- If an error occurred, ALL reasoning stages at or after the error cell must be marked as "failure" (reasoning in unexecuted cells cannot be valid)
- "success": The reasoning problem presented in the reasoning check was correctly addressed in cells that were successfully executed (or for markdown cells: not positioned after a cell with an error)
- "failure": The reasoning problem was incorrectly addressed, OR the reasoning is in cells positioned after an error (unexecuted cells)

OUTPUT FORMAT (respond with ONLY this JSON, no other text).
stage_implementation must contain ALL these keys: {', '.join(expected_stages)}
stage_completion must contain ALL these keys: {', '.join(expected_stages)}
reasoning_completion must contain ALL these keys: {', '.join(reasoning_stages)}
{{
    "stage_implementation": {{
        {stage_impl_template}
    }},
    "error": {{
        "has_error": true/false,
        "error_cell": cell_index (if has_error is true),
        "error_stage": "stage_name" (if identifiable)
    }},
    "stage_completion": {{
        {stage_comp_template}
    }},
    "reasoning_completion": {{
        {reasoning_template}
    }}
}}


---

NOTEBOOK CONTENT:

{notebook_text}

---

RESPOND WITH ONLY THE JSON OBJECT:"""

    try:
        headers = {}
        if auth_key:
            headers['Authorization'] = f'Bearer {auth_key}'

        response = requests.post(
            f"{ollama_url}/api/generate",
            headers=headers,
            json={
                "model": model_name,
                "system": "Reasoning: high\n\nYou are an expert code reviewer. Carefully analyze the notebook against the evaluation criteria. Think through each stage systematically before making your assessment.",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent evaluation
                    "num_ctx": 32768,    # Large context window
                }
            },
            timeout=300  # 5 minute timeout
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '{}')

            # Extract JSON from response
            # Handle case where LLM adds explanation before/after JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                stages_found = json.loads(json_text)

                # Debug: Print what keys we actually got
                print(f"  DEBUG: LLM returned keys: {list(stages_found.keys())}")

                # Validate expected structure
                if 'stage_implementation' not in stages_found or 'stage_completion' not in stages_found:
                    print(f"  ❌ LLM response missing required fields")
                    print(f"     Expected: stage_implementation, stage_completion, reasoning_completion, error")
                    print(f"     Got keys: {list(stages_found.keys())}")
                    print(f"     Full response (first 1000 chars):")
                    print(f"     {response_text[:1000]}")
                    raise ValueError(f"Invalid LLM response format - missing required fields")

                return stages_found, prompt  # Return both results and prompt
            else:
                print(f"  ❌ Could not find JSON in LLM response")
                print(f"     Response: {response_text[:500]}")
                raise ValueError(f"Could not extract JSON from LLM response")
        else:
            print(f"  ❌ HTTP {response.status_code}: {response.text[:500]}")
            raise ValueError(f"HTTP error {response.status_code}")

    except json.JSONDecodeError as e:
        print(f"  ❌ JSON decode error: {e}")
        print(f"     Attempted to parse: {json_text[:500] if 'json_text' in locals() else 'N/A'}")
        raise
    except Exception as e:
        print(f"  ❌ Error evaluating notebook: {e}")
        raise


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate notebooks using Ollama LLM')
    parser.add_argument('--test-dir', required=True, help='Directory containing test notebooks')
    parser.add_argument('--api-key', required=True, help='API key for Ollama')
    parser.add_argument('--output', default='full_agent_test/analysis', help='Output directory')
    parser.add_argument('--model', default='gpt-oss:120b-cloud', help='Model name (default: gpt-oss:120b-cloud)')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Ollama URL')
    args = parser.parse_args()

    # Configuration
    TEST_DIR = Path(args.test_dir)
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OLLAMA_URL = args.ollama_url
    MODEL_NAME = args.model
    AUTH_KEY = args.api_key

    # Create results folder
    RESULTS_DIR = OUTPUT_DIR / 'notebook_reviews'
    RESULTS_DIR.mkdir(exist_ok=True)

    print("="*80)
    print("NOTEBOOK EVALUATION USING OLLAMA")
    print("="*80)
    print(f"Test directory: {TEST_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Results folder: {RESULTS_DIR}")
    print()

    # Get all notebooks
    all_notebooks = sorted(TEST_DIR.glob('*.ipynb'))

    print(f"Found {len(all_notebooks)} notebooks to evaluate\n")

    # Track evaluation statistics
    successful_evals = 0
    failed_reads = 0

    for i, notebook_path in enumerate(all_notebooks, 1):
        notebook_name = notebook_path.name

        # Determine tissue
        if 'blood' in notebook_name:
            tissue = 'blood'
        elif 'breastcancer' in notebook_name:
            tissue = 'breastcancer'
        else:
            tissue = 'lung'

        print(f"[{i}/{len(all_notebooks)}] Evaluating {notebook_name} ({tissue})...")

        # Read notebook
        try:
            notebook_text = read_notebook_as_text(notebook_path)
        except Exception as e:
            print(f"  ⚠️  Not a valid notebook: {e}")
            # Save standardized JSON output indicating corrupted notebook
            result_file = os.path.join(RESULTS_DIR, f"{notebook_name}.json")
            corrupted_result = {
                "stage_implementation": {},
                "stage_completion": {},
                "reasoning_completion": {},
                "error": {
                    "has_error": True,
                    "error_type": "corrupted_notebook",
                    "error_message": str(e)
                }
            }
            with open(result_file, 'w') as f:
                json.dump(corrupted_result, f, indent=2)
            failed_reads += 1
            continue

        # Evaluate with Ollama (with retry logic for transient errors)
        max_retries = 3
        retry_count = 0
        stages_found = None
        prompt_used = None

        while retry_count < max_retries:
            try:
                stages_found, prompt_used = evaluate_notebook_with_ollama(
                    notebook_text, notebook_name, tissue, OLLAMA_URL, MODEL_NAME, AUTH_KEY
                )
                break  # Success, exit retry loop

            except ValueError as e:
                error_msg = str(e)
                # Check if it's a retryable error (HTTP 500, timeout, connection error)
                if 'HTTP error 500' in error_msg or 'timeout' in error_msg.lower() or 'connection' in error_msg.lower():
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"  ⚠️  Transient error (attempt {retry_count}/{max_retries}): {error_msg}")
                        print(f"  Retrying in 10 seconds...")
                        import time
                        time.sleep(10)
                        continue
                    else:
                        print(f"  ❌ Failed after {max_retries} attempts: {error_msg}")
                        print(f"  ABORTING - Check network/API and restart")
                        raise
                else:
                    # Non-retryable error (e.g., wrong format, validation error)
                    print(f"  ❌ LLM evaluation failed: {e}")
                    print(f"  ABORTING - Fix the issue and restart")
                    raise
            except Exception as e:
                print(f"  ❌ Unexpected error: {e}")
                print(f"  ABORTING")
                raise

        # Save individual JSON file and prompt for this notebook
        if stages_found:
            # Save JSON results
            result_file = RESULTS_DIR / f"{notebook_name}.json"
            with open(result_file, 'w') as f:
                json.dump(stages_found, f, indent=2)

            # Save the prompt used for evaluation
            if prompt_used:
                prompt_file = RESULTS_DIR / f"{notebook_name}.prompt.txt"
                with open(prompt_file, 'w') as f:
                    f.write(prompt_used)

            num_completed = len(stages_found.get('stage_completion', {}))
            num_implemented = len(stages_found.get('stage_implementation', {}))
            has_error = stages_found.get('error', {}).get('has_error', False)
            print(f"  ✓ Implemented: {num_implemented}, Completed: {num_completed}, Error: {has_error}")
            successful_evals += 1
        else:
            print(f"  - No stages found")

    print("="*80)
    print(f"EVALUATION COMPLETE")
    print(f"Successfully evaluated: {successful_evals}/{len(all_notebooks)}")
    print(f"Failed to read: {failed_reads}")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("="*80)
    print()

    # Generate progress heatmap
    print("Generating progress heatmap...")
    create_progress_heatmap(RESULTS_DIR, OUTPUT_DIR)
    print(f"\n✅ Analysis complete! Results in: {OUTPUT_DIR}")


def create_progress_heatmap(results_dir, output_dir):
    """Create heatmaps showing stage-by-stage completion progress.

    Creates one heatmap per tissue showing:
    - Rows: replicates
    - Columns: analysis stages (tissue-specific)
    - Cell states:
        0 (white): Not implemented
        1 (light gray): Implemented but not completed
        2 (dark gray): Completed successfully
        4 (red): Error occurred in this stage
    """
    print("Creating progress heatmaps...")

    # Load all review JSON files
    review_files = list(Path(results_dir).glob('*.ipynb.json'))

    if not review_files:
        print("⚠️  No review JSON files found - skipping heatmap")
        return

    # Parse results grouped by tissue
    tissue_data = {}
    for json_file in review_files:
        with open(json_file, 'r') as f:
            review = json.load(f)

        # Parse filename to get case and replicate
        # Format: blood_r1.ipynb.json
        filename = json_file.stem.replace('.ipynb', '')  # Remove .ipynb from stem
        parts = filename.split('_r')
        if len(parts) == 2:
            tissue = parts[0]
            replicate = int(parts[1])
        else:
            continue

        if tissue not in tissue_data:
            tissue_data[tissue] = []

        tissue_data[tissue].append({
            'replicate': replicate,
            'stage_implementation': review.get('stage_implementation', {}),
            'stage_completion': review.get('stage_completion', {}),
            'error': review.get('error', {})
        })

    if not tissue_data:
        print("⚠️  No valid review data found - skipping heatmap")
        return

    # Create one heatmap per tissue
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    for tissue, data_list in tissue_data.items():
        print(f"  Creating heatmap for {tissue}...")

        # Get stage definitions for this tissue
        if tissue not in STAGE_DEFINITIONS_DICT:
            print(f"    ⚠️  No stage definitions for {tissue} - skipping")
            continue

        stages = list(STAGE_DEFINITIONS_DICT[tissue].keys())
        n_stages = len(stages)

        # Sort by replicate
        data_list = sorted(data_list, key=lambda x: x['replicate'])
        n_replicates = len(data_list)

        # Build state matrix
        # State 0: Not implemented (white)
        # State 1: Implemented but not completed (light gray)
        # State 2: Completed successfully (dark gray)
        # State 4: Error occurred in this stage (red)
        state_matrix = np.zeros((n_replicates, n_stages), dtype=int)

        for row_idx, data in enumerate(data_list):
            impl_stages = data['stage_implementation']
            completion_stages = data['stage_completion']
            error_info = data['error']
            error_stage = error_info.get('error_stage') if error_info.get('has_error') else None

            for col_idx, stage in enumerate(stages):
                # Check if error occurred in this stage (overrides all other states)
                if error_stage == stage:
                    state_matrix[row_idx, col_idx] = 4
                # Check if completed
                elif stage in completion_stages and completion_stages[stage] != "NA":
                    state_matrix[row_idx, col_idx] = 2
                # Check if implemented but not completed
                elif stage in impl_stages and impl_stages[stage] != "NA":
                    state_matrix[row_idx, col_idx] = 1
                # Otherwise: not implemented (state 0)

        # Create figure
        fig, ax = plt.subplots(figsize=(0.7 * n_stages + 2, n_replicates * 0.5 + 1))

        # Colors: white (0), light gray (1), dark gray (2), red (4)
        cmap = ListedColormap(['white', '#d3d3d3', '#333333', 'white', '#d62728'])

        # Create heatmap
        sns.heatmap(state_matrix,
                    cmap=cmap,
                    vmin=0,
                    vmax=4,
                    cbar=False,
                    linewidths=0.5,
                    linecolor='black',
                    ax=ax,
                    square=False,
                    xticklabels=stages,
                    yticklabels=[f"Replicate {d['replicate']}" for d in data_list])

        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        ax.set_xlabel('Analysis Stage', fontsize=10)
        ax.set_ylabel('Replicate', fontsize=10)

        # Add title
        ax.set_title(f'Stage Completion Progress: {tissue}', fontsize=12, fontweight='bold', pad=10)

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='white', edgecolor='black', label='Not implemented'),
            mpatches.Patch(facecolor='#d3d3d3', edgecolor='black', label='Implemented'),
            mpatches.Patch(facecolor='#333333', edgecolor='black', label='Completed'),
            mpatches.Patch(facecolor='#d62728', edgecolor='black', label='Error')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                  frameon=True, fontsize=9)

        plt.tight_layout()

        # Save
        output_file = output_dir / f'progress_heatmap_{tissue}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved to: {output_file}")
        plt.close()


if __name__ == '__main__':
    main()
