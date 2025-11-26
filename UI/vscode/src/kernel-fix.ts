import * as vscode from 'vscode';

export class KernelFix {
    /**
     * Check if the active notebook has the kernel fix cell and add it if missing
     * Returns true if a fix cell was added, false otherwise
     */
    static async ensureFixCellExists(): Promise<boolean> {
        // Only apply on macOS
        if (process.platform !== 'darwin') {
            return false;
        }

        const editor = vscode.window.activeNotebookEditor;
        if (!editor) {
            return false;
        }

        // For empty notebooks, just add the fix cell
        if (editor.notebook.cellCount === 0) {
            return await this.addFixCell(editor);
        }

        // For non-empty notebooks, check if first cell is already the fix cell
        const firstCell = editor.notebook.cellAt(0);
        if (this.isFixCell(firstCell)) {
            return false;
        }

        // Add the fix cell at the top
        return await this.addFixCell(editor);
    }

    /**
     * Check if a cell is the kernel fix cell
     */
    private static isFixCell(cell: vscode.NotebookCell): boolean {
        const content = cell.document.getText();
        return content.includes('MUST RUN FIRST - Kernel Fix for macOS') ||
               (content.includes('OMP_NUM_THREADS') && content.includes('MKL_NUM_THREADS'));
    }

    /**
     * Add the kernel fix cell at the top of the notebook
     */
    private static async addFixCell(editor: vscode.NotebookEditor): Promise<boolean> {
        const fixCellContent = `# CELL 1: MUST RUN FIRST - Kernel Fix for macOS in VSCode when used in combination with agent
import os
import sys

# Set environment variables BEFORE any other imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', message='.*omp_set_nested.*deprecated.*')

# Check matplotlib backend
import matplotlib

print("✅ Kernel fix applied")
print(f"Platform: {sys.platform}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
print(f"Matplotlib backend: {matplotlib.get_backend()}")`;

        try {
            const newCell = new vscode.NotebookCellData(
                vscode.NotebookCellKind.Code,
                fixCellContent,
                'python'
            );

            const edit = new vscode.WorkspaceEdit();
            const notebookEdit = vscode.NotebookEdit.insertCells(0, [newCell]);
            edit.set(editor.notebook.uri, [notebookEdit]);
            
            const success = await vscode.workspace.applyEdit(edit);
            
            if (success) {
                return true;
            } else {
                return false;
            }
        } catch (error) {
            return false;
        }
    }
}