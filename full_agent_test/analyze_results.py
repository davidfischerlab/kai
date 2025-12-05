#!/usr/bin/env python3
"""
Analysis script for full agent test results.

Parses test output notebooks and metadata to generate summary statistics,
visualizations, and comparison with kai_reproducibility results.

Usage:
    python full_agent_test/analyze_results.py
    python full_agent_test/analyze_results.py --test-dir full_agent_test/test_outputs
    python full_agent_test/analyze_results.py --output full_agent_test/analysis
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import nbformat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """
    Analyzer for full agent test results.

    Parses notebooks, extracts metrics, and generates visualizations.
    """

    def __init__(self, test_dir: str, output_dir: str):
        """
        Initialize results analyzer.

        Args:
            test_dir: Directory containing test outputs
            output_dir: Directory for analysis outputs
        """
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.metadata_list = []
        self.notebook_analyses = []

    def load_metadata(self):
        """Load all metadata JSON files from test directory."""
        metadata_files = list(self.test_dir.glob('*_metadata.json'))

        logger.info(f"Found {len(metadata_files)} metadata files")

        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.metadata_list.append(metadata)

        logger.info(f"Loaded {len(self.metadata_list)} test results")

    def analyze_notebook(self, notebook_path: Path) -> Dict[str, Any]:
        """
        Analyze a single generated notebook.

        Extracts:
        - Number of cells (code and markdown)
        - Cell execution status
        - Errors encountered
        - Key outputs (plots, analysis results)

        Args:
            notebook_path: Path to notebook file

        Returns:
            Analysis dictionary
        """
        logger.info(f"Analyzing notebook: {notebook_path.name}")

        try:
            with open(notebook_path, 'r') as f:
                nb = nbformat.read(f, as_version=4)
        except Exception as e:
            logger.error(f"Failed to read notebook {notebook_path}: {e}")
            return {'error': str(e)}

        analysis = {
            'notebook_path': str(notebook_path),
            'total_cells': len(nb.cells),
            'code_cells': 0,
            'markdown_cells': 0,
            'executed_cells': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'errors': [],
            'has_plots': False,
            'has_clustering': False,
            'has_umap': False,
            'has_cell_type_annotation': False,
        }

        for cell_idx, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                analysis['code_cells'] += 1

                # Check if executed
                if hasattr(cell, 'execution_count') and cell.execution_count is not None:
                    analysis['executed_cells'] += 1

                # Check execution success
                if hasattr(cell, 'outputs') and cell.outputs:
                    has_error = False
                    for output in cell.outputs:
                        if output.output_type == 'error':
                            has_error = True
                            analysis['errors'].append({
                                'cell_index': cell_idx,
                                'ename': output.get('ename', 'Unknown'),
                                'evalue': output.get('evalue', 'Unknown error')
                            })

                    if has_error:
                        analysis['failed_executions'] += 1
                    else:
                        analysis['successful_executions'] += 1

                    # Check for plots
                    for output in cell.outputs:
                        if output.output_type in ['display_data', 'execute_result']:
                            data = output.get('data', {})
                            if 'image/png' in data or 'image/jpeg' in data:
                                analysis['has_plots'] = True

                # Analyze cell content for key operations
                source = cell.source.lower()

                if any(keyword in source for keyword in ['sc.tl.leiden', 'sc.tl.louvain', 'cluster']):
                    analysis['has_clustering'] = True

                if 'sc.tl.umap' in source or 'sc.pl.umap' in source:
                    analysis['has_umap'] = True

                if any(keyword in source for keyword in ['cell_type', 'celltype', 'annotation', 'marker']):
                    analysis['has_cell_type_annotation'] = True

            elif cell.cell_type == 'markdown':
                analysis['markdown_cells'] += 1

        # Calculate success rate
        if analysis['executed_cells'] > 0:
            analysis['execution_success_rate'] = (
                analysis['successful_executions'] / analysis['executed_cells']
            )
        else:
            analysis['execution_success_rate'] = 0.0

        return analysis

    def analyze_all_notebooks(self):
        """Analyze all notebooks in test directory."""
        notebook_files = list(self.test_dir.glob('scenario1_*.ipynb'))

        logger.info(f"Found {len(notebook_files)} notebooks to analyze")

        for notebook_path in notebook_files:
            analysis = self.analyze_notebook(notebook_path)

            # Match with metadata
            for metadata in self.metadata_list:
                if Path(metadata['output_notebook']) == notebook_path:
                    analysis['metadata'] = metadata
                    break

            self.notebook_analyses.append(analysis)

    def generate_summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics from all test results.

        Returns:
            DataFrame with summary statistics
        """
        logger.info("Generating summary statistics")

        rows = []
        for analysis in self.notebook_analyses:
            metadata = analysis.get('metadata', {})

            row = {
                'case': metadata.get('case_name', 'unknown'),
                'replicate': metadata.get('replicate', 0),
                'success': metadata.get('success', False),
                'iterations': metadata.get('iterations', 0),
                'duration_min': metadata.get('duration_minutes', 0),
                'total_cells': analysis.get('total_cells', 0),
                'code_cells': analysis.get('code_cells', 0),
                'executed_cells': analysis.get('executed_cells', 0),
                'successful_executions': analysis.get('successful_executions', 0),
                'failed_executions': analysis.get('failed_executions', 0),
                'execution_success_rate': analysis.get('execution_success_rate', 0),
                'has_plots': analysis.get('has_plots', False),
                'has_clustering': analysis.get('has_clustering', False),
                'has_umap': analysis.get('has_umap', False),
                'has_cell_type_annotation': analysis.get('has_cell_type_annotation', False),
                'num_errors': len(analysis.get('errors', [])),
            }

            rows.append(row)

        df = pd.DataFrame(rows)

        # Save to CSV
        csv_path = self.output_dir / 'summary_statistics.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Summary statistics saved to: {csv_path}")

        return df

    def create_progress_heatmap(self, df: pd.DataFrame):
        """
        Create progress heatmap similar to kai_reproducibility.

        Shows completion status for each case x replicate combination.

        Args:
            df: Summary statistics DataFrame
        """
        logger.info("Creating progress heatmap")

        # Prepare data for heatmap
        cases = sorted(df['case'].unique())
        replicates = sorted(df['replicate'].unique())

        # Create matrix for heatmap
        matrix = np.zeros((len(cases), len(replicates)))

        for i, case in enumerate(cases):
            for j, rep in enumerate(replicates):
                case_data = df[(df['case'] == case) & (df['replicate'] == rep)]

                if len(case_data) > 0:
                    row = case_data.iloc[0]

                    # Scoring: 0 = failed, 0.5 = partial success, 1.0 = full success
                    if not row['success']:
                        score = 0.0
                    elif row['num_errors'] > 0:
                        score = 0.5
                    else:
                        score = 1.0

                    matrix[i, j] = score

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Completion Score'},
            xticklabels=[f'Repeat {r}' for r in replicates],
            yticklabels=cases,
            vmin=0,
            vmax=1,
            ax=ax
        )

        ax.set_title('Test Progress Heatmap\n(0=Failed, 0.5=Partial, 1.0=Success)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Replicate', fontsize=12)
        ax.set_ylabel('Test Case', fontsize=12)

        plt.tight_layout()

        # Save figure
        fig_path = self.output_dir / 'progress_heatmap.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Progress heatmap saved to: {fig_path}")

        plt.close()

    def create_metrics_plots(self, df: pd.DataFrame):
        """
        Create various metric visualizations.

        Args:
            df: Summary statistics DataFrame
        """
        logger.info("Creating metrics plots")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Iterations by case
        ax = axes[0, 0]
        case_means = df.groupby('case')['iterations'].mean()
        case_std = df.groupby('case')['iterations'].std()

        ax.bar(range(len(case_means)), case_means, yerr=case_std, capsize=5, alpha=0.7)
        ax.set_xticks(range(len(case_means)))
        ax.set_xticklabels(case_means.index, rotation=45, ha='right')
        ax.set_ylabel('Iterations')
        ax.set_title('Average Iterations by Test Case')
        ax.grid(axis='y', alpha=0.3)

        # 2. Duration by case
        ax = axes[0, 1]
        duration_means = df.groupby('case')['duration_min'].mean()
        duration_std = df.groupby('case')['duration_min'].std()

        ax.bar(range(len(duration_means)), duration_means, yerr=duration_std, capsize=5, alpha=0.7, color='orange')
        ax.set_xticks(range(len(duration_means)))
        ax.set_xticklabels(duration_means.index, rotation=45, ha='right')
        ax.set_ylabel('Duration (minutes)')
        ax.set_title('Average Duration by Test Case')
        ax.grid(axis='y', alpha=0.3)

        # 3. Execution success rate
        ax = axes[1, 0]
        success_means = df.groupby('case')['execution_success_rate'].mean() * 100
        success_std = df.groupby('case')['execution_success_rate'].std() * 100

        ax.bar(range(len(success_means)), success_means, yerr=success_std, capsize=5, alpha=0.7, color='green')
        ax.set_xticks(range(len(success_means)))
        ax.set_xticklabels(success_means.index, rotation=45, ha='right')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Cell Execution Success Rate by Test Case')
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)

        # 4. Workflow completeness
        ax = axes[1, 1]
        workflow_metrics = df.groupby('case')[['has_clustering', 'has_umap', 'has_cell_type_annotation']].mean() * 100

        x = np.arange(len(workflow_metrics.index))
        width = 0.25

        ax.bar(x - width, workflow_metrics['has_clustering'], width, label='Clustering', alpha=0.7)
        ax.bar(x, workflow_metrics['has_umap'], width, label='UMAP', alpha=0.7)
        ax.bar(x + width, workflow_metrics['has_cell_type_annotation'], width, label='Cell Type Annotation', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(workflow_metrics.index, rotation=45, ha='right')
        ax.set_ylabel('Completion (%)')
        ax.set_title('Workflow Step Completion by Test Case')
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save figure
        fig_path = self.output_dir / 'metrics_summary.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics summary saved to: {fig_path}")

        plt.close()

    def generate_text_summary(self, df: pd.DataFrame):
        """
        Generate text summary report.

        Args:
            df: Summary statistics DataFrame
        """
        logger.info("Generating text summary")

        summary_path = self.output_dir / 'summary_report.txt'

        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("KAI FULL AGENT TEST - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")

            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total tests: {len(df)}\n")
            f.write(f"Successful: {df['success'].sum()}\n")
            f.write(f"Failed: {(~df['success']).sum()}\n")
            f.write(f"Success rate: {df['success'].mean() * 100:.1f}%\n")
            f.write(f"\n")

            # Per-case statistics
            f.write("PER-CASE STATISTICS\n")
            f.write("-"*80 + "\n")

            for case in sorted(df['case'].unique()):
                case_df = df[df['case'] == case]

                f.write(f"\n{case.upper()}:\n")
                f.write(f"  Tests: {len(case_df)}\n")
                f.write(f"  Success: {case_df['success'].sum()}/{len(case_df)}\n")

                if case_df['success'].sum() > 0:
                    successful = case_df[case_df['success']]
                    f.write(f"  Avg iterations: {successful['iterations'].mean():.1f}\n")
                    f.write(f"  Avg duration: {successful['duration_min'].mean():.1f} min\n")
                    f.write(f"  Avg code cells: {successful['code_cells'].mean():.1f}\n")
                    f.write(f"  Cell execution success: {successful['execution_success_rate'].mean() * 100:.1f}%\n")

                f.write(f"  Workflow completion:\n")
                f.write(f"    - Clustering: {case_df['has_clustering'].sum()}/{len(case_df)}\n")
                f.write(f"    - UMAP: {case_df['has_umap'].sum()}/{len(case_df)}\n")
                f.write(f"    - Cell type annotation: {case_df['has_cell_type_annotation'].sum()}/{len(case_df)}\n")

            f.write("\n" + "="*80 + "\n")

        logger.info(f"Text summary saved to: {summary_path}")

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        logger.info("="*80)
        logger.info("Starting full analysis")
        logger.info("="*80)

        # Load data
        self.load_metadata()
        self.analyze_all_notebooks()

        # Generate outputs
        df = self.generate_summary_statistics()
        self.create_progress_heatmap(df)
        self.create_metrics_plots(df)
        self.generate_text_summary(df)

        logger.info("="*80)
        logger.info("Analysis complete!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze Kai full agent test results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze default test directory
  python full_agent_test/analyze_results.py

  # Analyze custom test directory
  python full_agent_test/analyze_results.py --test-dir my_test_run

  # Custom output directory
  python full_agent_test/analyze_results.py --output my_analysis
        """
    )

    parser.add_argument(
        '--test-dir',
        default='full_agent_test/test_outputs',
        help='Directory containing test outputs'
    )

    parser.add_argument(
        '--output',
        default='full_agent_test/analysis',
        help='Output directory for analysis results'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    analyzer = ResultsAnalyzer(
        test_dir=args.test_dir,
        output_dir=args.output
    )

    try:
        analyzer.run_full_analysis()
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
