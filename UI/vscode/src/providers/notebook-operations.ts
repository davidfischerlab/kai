import * as vscode from 'vscode';

/**
 * NotebookHistoryTracker - Abstract base class for notebook activity tracking
 * 
 * Core Responsibilities:
 * - Execution history: Formatted string records of cell executions (most recent first)
 * - Modification history: Structured tracking of cell changes with hashing
 * - Cell formatting: Comprehensive string representation with metadata and outputs
 * - History management: Automatic size limits to prevent memory growth
 * 
 * Key Features:
 * - formatCellToString(): Rich cell representation with execution data and outputs
 * - formatCellOutputToString(): Structured output parsing with ==TYPE> headers
 * - Queue position tracking: Integration with cell execution state
 * - Content change detection: Hash-based modification tracking
 * 
 * Architecture Pattern:
 * - Abstract base class requiring queue info and reveal implementations
 * - Provides unified cell formatting and history infrastructure
 * - Designed for inheritance by concrete notebook operation classes
 */
abstract class NotebookHistoryTracker {
    /**
     * Execution history - formatted cell strings (most recent first)
     * Each entry contains: metadata header, code, and structured outputs
     */
    protected _executionHistory: string[] = [];

    /**
     * Modification history tracking - tracks cell content changes, additions, and deletions
     * Format: [{cell_index, timestamp, modification_type, old_content_hash, new_content_hash}, ...]
     */
    protected _modificationHistory: Array<{
        cell_index: number;
        timestamp: string;
        modification_type: 'created' | 'modified' | 'deleted' | 'moved';
        old_content_hash?: string;
        new_content_hash?: string;
        content_preview?: string;
    }> = [];
    
    protected readonly MAX_EXECUTION_HISTORY_ENTRIES = 5;
    protected readonly MAX_MODIFICATION_HISTORY_ENTRIES = 5;

    // Getters for access to internal state
    get executionHistory(): string[] { return this._executionHistory; }
    get modificationHistory() { return this._modificationHistory; }

    // Abstract methods that must be implemented by inheriting class
    protected abstract _getQueueInfoForCell(cellIndex: number): string;
    protected abstract revealCell(cellIndex: number): void;

    /**
     * Adds a cell execution record to history (most recent first).
     * Creates formatted string with metadata, code, and outputs.
     * 
     * @param cell - Executed notebook cell with summary and outputs
     * @interaction Called after cell execution completion
     */
    public addToExecutionHistory(cell: vscode.NotebookCell): void {
        const entry = this.formatCellToString(cell)
        // Add entry to execution history list (most recent first):
        this._executionHistory.unshift(entry);
        // Keep number of entries in limit:
        this._trimExecutionHistory();
    }

    /**
     * Maintains execution history size limit by removing oldest entries.
     * Since new entries are added at start, removes from end.
     */
    private _trimExecutionHistory(): void {
        while (this._executionHistory.length > this.MAX_EXECUTION_HISTORY_ENTRIES) {
            // Remove oldest entries from the end
            this._executionHistory.pop();
        }
    }

    /**
     * Creates comprehensive string representation of a notebook cell.
     * Includes execution metadata, queue info, code, and formatted outputs.
     * 
     * @param cell - Notebook cell to format
     * @returns Complete formatted cell string
     * @interaction Used by addToExecutionHistory and getNotebookStructure
     */
    public formatCellToString(cell: vscode.NotebookCell): string {
        const timestamp = new Date().toLocaleTimeString();
        const cellIndex = cell?.index;
        const content = cell.document.getText();

        // Check if this is a markdown cell
        const isMarkdown = cell.kind === vscode.NotebookCellKind.Markup;

        if (isMarkdown) {
            // Format markdown cells differently - no execution metadata
            let entry = `> MARKDOWN CELL at index ${cellIndex ?? '?'}\nAdded at ${timestamp}\n>>Content of markdown cell at index ${cellIndex ?? '?'}:\n${content.trim()}`;
            return entry;
        }

        // Code cell formatting (original logic)
        const executionOrder = cell?.executionSummary?.executionOrder || 0;
        const success = cell?.executionSummary?.success;
        const timing = cell?.executionSummary?.timing;
        const duration = timing ? ((timing.endTime - timing.startTime) / 1000).toFixed(3) + 's' : 'unknown';

        // Get queue position information at time of execution
        const queueInfo = cellIndex !== undefined ? this._getQueueInfoForCell(cellIndex) : '';

        // Determine status and error type
        const status = success === false ? 'FAILED' : (success === true ? 'SUCCESS' : 'UNKNOWN');

        // Start with enhanced header including queue information
        let entry = `> CELL at index ${cellIndex ?? '?'} and execution order [${executionOrder}]${queueInfo}: ${status}\nExecuted at ${timestamp}, took ${duration}\n>>Content of cell at index ${cellIndex ?? '?'}:\n${content.trim()}`;

        // Add output if available
        if (cell.outputs.length > 0) {
            const outputText = this.formatCellOutputToString(cell);
            entry += `\n>> Outputs of cell at index ${cellIndex ?? '?'}:\n${outputText}`;
        }

        return entry
    }

    /**
     * Formats cell outputs into structured string with type headers.
     * Uses ==TYPE> format (ERROR, STDOUT, PLOT, etc.) for easy parsing.
     * Handles error parsing, image detection, and output truncation.
     * 
     * @param cell - Notebook cell with outputs to format
     * @returns Formatted output string with type headers
     * @interaction Called by formatCellToString
     */
    public formatCellOutputToString(cell: vscode.NotebookCell): string {
        let allOutput = '';

        // First pass: count total outputs (excluding skipped ones)
        let totalOutputs = 0;
        for (const output of cell.outputs) {
            for (const item of output.items) {
                // Count only outputs that we actually process (not skipped mime types)
                if (
                    item.mime === 'text/plain' ||
                    item.mime === 'application/vnd.code.notebook.stdout' ||
                    item.mime === 'application/vnd.code.notebook.error' ||
                    item.mime.startsWith('image/')
                ) {
                    totalOutputs++;
                }
                // Skip stderr and other unsupported types in counting
            }
        }

        // Second pass: process and number outputs
        let currentOutputNumber = 0;
        for (const output of cell.outputs) {
            for (const item of output.items) {
                let outputType: string;
                let content = '';

                // Categorize output types and extract content
                if (
                    item.mime === 'text/plain' ||
                    item.mime === 'application/vnd.code.notebook.stdout'
                ) {
                    currentOutputNumber++;
                    outputType = `>>> Text output ${currentOutputNumber}/${totalOutputs}`;
                    // Extract text content
                    try {
                        content = new TextDecoder().decode(item.data);
                    } catch (e) {
                        content = '[Failed to decode text/plain or stdout output]';
                    }
                } else if (
                    item.mime === 'application/vnd.code.notebook.stderr'
                ) {
                    // These are warning messages and this parsing works, but ignore for now to keep prompts small.
                    // outputType = '>>> ${currentOutputNumber}/${totalOutputs} Warning output';
                    // Parse content:
                    // try {
                    //     content = new TextDecoder().decode(item.data);
                    // } catch (e) {
                    //     content = '[Failed to decode stderr output]';
                    // }
                    continue;
                } else if (
                    item.mime === 'application/vnd.code.notebook.error'
                ) {
                    currentOutputNumber++;
                    outputType = `>>> Error output ${currentOutputNumber}/${totalOutputs}`;
                    // Parse content:
                    try {
                        const text = new TextDecoder().decode(item.data);
                        const error = JSON.parse(text);

                        // Handle two different error formats
                        let ename: string;
                        let evalue: string;
                        let traceback: string[] | undefined;

                        if ('ename' in error && 'evalue' in error) {
                            // Standard Jupyter format: ename, evalue, traceback
                            ename = error.ename || 'Error';
                            evalue = error.evalue || 'Unknown error occurred';
                            traceback = error.traceback;
                        } else if ('name' in error && 'message' in error) {
                            // Format encountered in VSCode: name, message, stack
                            ename = error.name || 'Error';
                            evalue = error.message || 'Unknown error occurred';
                            // Convert stack string to array if needed
                            if (error.stack) {
                                traceback = error.stack.split('\n');
                            }
                        } else {
                            // Unknown format
                            ename = 'Error';
                            evalue = JSON.stringify(error);
                        }
                        content = `${ename}: ${evalue}`;
                        // Add traceback if available
                        if (traceback && Array.isArray(traceback) && traceback.length > 0) {
                            const cleanedTraceback = traceback
                                .map((line: string) => line.replace(/\x1b\[[0-9;]*m/g, ''))
                                .join('\n');
                            content += `\n${cleanedTraceback}`;
                        }
                    } catch (e) {
                        content = '[Failed to decode error output]';
                    }
                } else if (item.mime.startsWith('image/')) {
                    currentOutputNumber++;
                    outputType = `>>> Plot output ${currentOutputNumber}/${totalOutputs}`;
                    content = '[Image output]';
                } else {
                    // Skip unsupported mime types: 'text/html', 'application/json'
                    continue;
                }

                // Add output with heading if there's content
                if (content) {
                    // Limit to last 50 lines per output type
                    const lines = content.split('\n');
                    const trimmedContent = lines.length > 50
                        ? `... (truncated, showing last 50 lines)\n${lines.slice(-50).join('\n')}`
                        : content;

                    allOutput += `${outputType}\n${trimmedContent}\n\n`;
                }
            }
        }

        return allOutput.trim();
    }

    public addToModificationHistory(
        modificationType: 'created' | 'modified' | 'deleted' | 'moved',
        cellIndex: number,
        content?: string,
        oldContentHash?: string,
        newContentHash?: string
    ): void {
        const entry = {
            cell_index: cellIndex,
            timestamp: new Date().toISOString(),
            modification_type: modificationType,
            old_content_hash: oldContentHash,
            new_content_hash: newContentHash,
            content_preview: content ? (content.length > 100 ? content.substring(0, 100) + '...' : content) : undefined
        };

        this._modificationHistory.push(entry);
    }

    /**
     * Tracks cell content modifications (text changes).
     */
    public trackCellModification(cell: vscode.NotebookCell, modificationType: 'modified'): void {
        const cellIndex = cell.index;
        const content = cell.document.getText();
        const timestamp = new Date().toISOString();
        const contentHash = this._simpleHash(content);
        const contentPreview = content.length > 100 ? content.substring(0, 100) + '...' : content;
        
        // Find previous modification of same cell to get old hash
        const lastModification = this._modificationHistory
            .slice()
            .reverse()
            .find(mod => mod.cell_index === cellIndex && mod.modification_type !== 'deleted');
            
        const entry = {
            cell_index: cellIndex,
            timestamp,
            modification_type: modificationType,
            old_content_hash: lastModification?.new_content_hash,
            new_content_hash: contentHash,
            content_preview: contentPreview
        };
        
        this._modificationHistory.push(entry);
        this._trimModificationHistory();
        
        // Auto-follow: reveal the modified cell
        this.revealCell(cellIndex);
    }

    /**
     * Tracks structural changes (add/remove/move cells).
     */
    public trackStructuralChange(cellIndex: number, modificationType: 'created' | 'deleted', content: string): void {
        const timestamp = new Date().toISOString();
        const contentHash = this._simpleHash(content);
        const contentPreview = content.length > 100 ? content.substring(0, 100) + '...' : content;
        
        const entry = {
            cell_index: cellIndex,
            timestamp,
            modification_type: modificationType,
            new_content_hash: modificationType === 'created' ? contentHash : undefined,
            old_content_hash: modificationType === 'deleted' ? contentHash : undefined,
            content_preview: contentPreview
        };
        
        this._modificationHistory.push(entry);
        this._trimModificationHistory();
        
        // Auto-follow: reveal created cells (not deleted ones)
        if (modificationType === 'created') {
            this.revealCell(cellIndex);
        }
    }

    /**
     * Simple hash function for content comparison.
     */
    private _simpleHash(str: string): string {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return hash.toString();
    }

    /**
     * Keeps modification history within size limits.
     */
    private _trimModificationHistory(): void {
        while (this._modificationHistory.length > this.MAX_MODIFICATION_HISTORY_ENTRIES) {
            this._modificationHistory.shift();
        }
    }

}

/**
 * NotebookOperations - Complete notebook manipulation and execution management system
 * 
 * Primary Responsibilities:
 * - Cell operations: Add, replace, execute, delete cells with full error handling
 * - Execution queue management: Track queued and executing cells with position info
 * - VSCode integration: Event handling for notebook document changes and execution
 * - Auto-tracking: Automatic detection and logging of user-initiated cell operations
 * - Cell navigation: Auto-reveal and positioning for user experience
 * 
 * Key Operations:
 * - addCode(): Insert new code cells with optional execution
 * - replaceCode(): Update existing cells while preserving metadata  
 * - executeCell(): Execute cells with comprehensive error handling and status tracking
 * - deleteCell(): Remove cells with structural change tracking
 * - setupExecutionTracking(): Initialize VSCode event listeners for automatic tracking
 * 
 * Architecture Features:
 * - Inherits complete history tracking from NotebookHistoryTracker
 * - Implements abstract methods for queue info and cell revelation
 * - Self-contained with no external dependencies except VSCode API
 * - Provides clean public interface for all notebook operations
 * 
 * Usage Pattern:
 * - Created by ChatViewProvider as the primary notebook interface
 * - Used by AutonomousExecution for programmatic notebook manipulation
 * - Automatically tracks all user and programmatic notebook changes
 */
export class NotebookOperations extends NotebookHistoryTracker {
    /**
     * Execution queue tracking - monitors cell execution states
     */
    private _queuedCells: Set<number> = new Set(); // Cell indices queued for execution
    private _executingCells: Set<number> = new Set(); // Cell indices currently executing

    /**
     * Termination tracking - tracks cells that were terminated by execution monitor
     */
    private _lastTerminatedCellIndex: number = -1; // Index of last cell terminated by monitor
    private _terminationReason: string = ""; // Reason for termination

    /**
     * Tracked notebook - persists notebook reference when user clicks away
     */
    private _trackedNotebook: vscode.NotebookDocument | undefined;

    constructor(
        private revealCellCallback: (cellIndex: number) => void,
        private agentProvider: any
    ) {
        super();
    }

    // Getters for access to internal state
    get queuedCells() { return this._queuedCells; }
    get executingCells() { return this._executingCells; }
    get lastTerminatedCellIndex() { return this._lastTerminatedCellIndex; }
    get terminationReason() { return this._terminationReason; }

    /**
     * Set tracked notebook for autonomous mode (call when autonomous mode starts)
     */
    public setTrackedNotebook(notebook: vscode.NotebookDocument | undefined): void {
        this._trackedNotebook = notebook;
        if (notebook) {
            console.log(`[KAI] setTrackedNotebook: Tracking notebook with ${notebook.cellCount} cells`);
        } else {
            console.log('[KAI] setTrackedNotebook: Cleared tracked notebook');
        }
    }

    /**
     * Get notebook editor - prefers active editor, falls back to tracked notebook.
     * Use this instead of vscode.window.activeNotebookEditor to handle cases where
     * another tab (e.g., a code file) is focused but the notebook is still visible.
     */
    public getNotebookEditor(): vscode.NotebookEditor | undefined {
        const activeEditor = vscode.window.activeNotebookEditor;
        if (activeEditor) {
            return activeEditor;
        }
        // Fall back to tracked notebook
        if (this._trackedNotebook) {
            for (const editor of vscode.window.visibleNotebookEditors) {
                if (editor.notebook.uri.toString() === this._trackedNotebook.uri.toString()) {
                    return editor;
                }
            }
        }
        return undefined;
    }

    /**
     * Clear termination tracking state.
     * Should be called after termination has been handled by the workflow.
     */
    public clearTerminationState(): void {
        this._lastTerminatedCellIndex = -1;
        this._terminationReason = "";
    }

    /**
     * Creates notebook structure with all cells as formatted strings.
     * Uses formatCellToString() to provide comprehensive cell representations.
     * 
     * @returns Object with totalCells count and allCells as formatted strings
     * @interaction Called by ChatCore._getNotebookStructure
     */
    public getNotebookStructure(): any {
        const editor = this.getNotebookEditor();
        if (!editor) {
            return {
                totalCells: 0,
                allCells: []
            };
        }

        // Format to list of cells where each is represented as a string that contains all relevant cell data.
        const cells = editor.notebook.getCells();
        const allCells = cells.map(cell => this.formatCellToString(cell));
        return {
            totalCells: cells.length,
            allCells: allCells
        };
    }

    // Implementation of abstract methods from base class
    protected _getQueueInfoForCell(cellIndex: number): string {
        if (this._queuedCells.has(cellIndex)) {
            const queuePosition = Array.from(this._queuedCells).sort((a, b) => a - b).indexOf(cellIndex) + 1;
            const totalQueued = this._queuedCells.size;
            return ` [Queue: ${queuePosition}/${totalQueued}]`;
        } else if (this._executingCells.has(cellIndex)) {
            return ' [Was executing]';
        }
        return '';
    }

    protected revealCell(cellIndex: number): void {
        this.revealCellCallback(cellIndex);
    }

    public setupExecutionTracking(): void {
        // Track execution state changes (queued, executing, completed)
        // Use the same document change listener to also update queues
        vscode.workspace.onDidChangeNotebookDocument(() => {
            this._updateExecutionQueues();
        });

        // Track VSCode native executions and modifications via workspace events
        vscode.workspace.onDidChangeNotebookDocument(event => {
            // Track cell content/output changes
            for (const cellChange of event.cellChanges) {
                const cell = cellChange.cell;
                
                // Track cell content modifications
                if (cellChange.document) {
                    this.trackCellModification(cell, 'modified');
                }
            }
            
            // Track structural changes (add/remove/move cells)
            for (const contentChange of event.contentChanges) {
                if (contentChange.addedCells) {
                    contentChange.addedCells.forEach((cell, index) => {
                        const cellIndex = contentChange.range.start + index;
                        this.trackStructuralChange(cellIndex, 'created', cell.document.getText());
                    });
                }
                if (contentChange.removedCells) {
                    contentChange.removedCells.forEach((cell, index) => {
                        const cellIndex = contentChange.range.start + index;
                        this.trackStructuralChange(cellIndex, 'deleted', cell.document.getText());
                    });
                }
            }
        });
    }

    private _updateExecutionQueues(): void {
        /**
         * Updates the execution queue tracking by scanning all cells for their execution states.
         * This method identifies which cells are queued, currently executing, or completed.
         */
        const editor = this.getNotebookEditor();
        if (!editor) return;

        const newQueuedCells = new Set<number>();
        const newExecutingCells = new Set<number>();

        for (let i = 0; i < editor.notebook.cellCount; i++) {
            const cell = editor.notebook.cellAt(i);
            
            // Skip non-code cells
            if (cell.kind !== vscode.NotebookCellKind.Code) continue;

            // Check execution state based on execution summary
            if (cell.executionSummary) {
                // Cell has execution summary
                if (cell.executionSummary.success !== undefined) {
                    // Execution completed (success or failure)
                    continue; 
                } else {
                    // Execution in progress (has summary but no success status yet)
                    newExecutingCells.add(i);
                }
            } else {
                // Check if cell appears to be queued by examining notebook execution state
                // VSCode queues cells when "Run All" or similar commands are used
                // We can infer queued state from context (this is a heuristic)
                const hasContent = cell.document.getText().trim().length > 0;
                
                // Simple heuristic: if notebook is in execution mode and cell has content but no execution summary,
                // it's likely queued. We can enhance this logic later with more sophisticated detection.
                if (hasContent && this._isNotebookExecuting()) {
                    newQueuedCells.add(i);
                }
            }
        }

        // Update internal state
        this._queuedCells = newQueuedCells;
        this._executingCells = newExecutingCells;
    }

    private _isNotebookExecuting(): boolean {
        /**
         * Helper method to determine if the notebook is currently in execution mode.
         * This is a heuristic based on whether any cells are currently executing.
         */
        return this._executingCells.size > 0;
    }

    // NOTEBOOK OPERATIONS

    /**
     * Add new markdown cell after specified cell.
     * @param content Markdown content to add
     * @param afterCellNumber Cell to insert after (0-based, use -1 for beginning)
     */
    public async replaceMarkdown(content: string, cellNumber: number): Promise<vscode.NotebookCell> {
        const editor = this.getNotebookEditor();
        if (!editor) {
            throw new Error('No active notebook');
        }

        if (cellNumber < 0 || cellNumber >= editor.notebook.cellCount) {
            throw new Error(`Invalid cell number: ${cellNumber}`);
        }

        const targetCell = editor.notebook.cellAt(cellNumber);
        if (targetCell.kind !== vscode.NotebookCellKind.Markup) {
            throw new Error(`Cell ${cellNumber} is not a markdown cell`);
        }

        try {
            const oldContent = targetCell.document.getText();
            const newCell = new vscode.NotebookCellData(vscode.NotebookCellKind.Markup, content, 'markdown');
            const edit = new vscode.WorkspaceEdit();
            const notebookEdit = vscode.NotebookEdit.replaceCells(new vscode.NotebookRange(cellNumber, cellNumber + 1), [newCell]);
            edit.set(editor.notebook.uri, [notebookEdit]);

            const success = await vscode.workspace.applyEdit(edit);
            if (success) {
                editor.selection = new vscode.NotebookRange(cellNumber, cellNumber + 1);

                // Auto-follow: reveal the replaced cell
                this.revealCell(cellNumber);

                // Get updated cell
                const cell = editor.notebook.cellAt(cellNumber);

                // Add to execution history
                this.addToExecutionHistory(cell);

                // Add to modification history
                this.addToModificationHistory(
                    'modified',
                    cellNumber,
                    content,
                    oldContent
                );

                return cell;
            } else {
                throw new Error(`❌ Failed to replace markdown at cell ${cellNumber}`);
            }
        } catch (error) {
            throw new Error(`Failed to replace markdown: ${error}`);
        }
    }

    public async addMarkdown(content: string, afterCellNumber: number): Promise<vscode.NotebookCell> {
        const editor = this.getNotebookEditor();
        if (!editor) {
            throw new Error('No active notebook');
        }

        // Validate afterCellNumber and calculate insertion position
        // Handle special case: -1 means "insert at beginning" for empty notebooks
        const insertPosition = afterCellNumber === -1 ? 0 : Math.min(afterCellNumber + 1, editor.notebook.cellCount);

        try {
            const newCell = new vscode.NotebookCellData(vscode.NotebookCellKind.Markup, content, 'markdown');
            const edit = new vscode.WorkspaceEdit();
            const notebookEdit = vscode.NotebookEdit.insertCells(insertPosition, [newCell]);
            edit.set(editor.notebook.uri, [notebookEdit]);

            const success = await vscode.workspace.applyEdit(edit);
            if (success) {
                editor.selection = new vscode.NotebookRange(insertPosition, insertPosition + 1);

                // Auto-follow: reveal the newly added cell
                this.revealCell(insertPosition);

                // Markdown cells don't need execution, but should be tracked
                const cell = editor.notebook.cellAt(insertPosition);

                // Add to execution history (even though not "executed", it's part of the flow)
                this.addToExecutionHistory(cell);

                // Add to modification history
                this.addToModificationHistory(
                    'created',
                    insertPosition,
                    content,
                    undefined
                );

                return cell;
            } else {
                throw new Error(`❌ Failed to add markdown after cell ${insertPosition}`);
            }
        } catch (error) {
            throw new Error(`Error adding markdown after cell ${afterCellNumber}: ${error}`);
        }
    }
    
    /**
     * Add new code cell after specified cell.
     * @param code Code content to add
     * @param afterCellNumber Cell to insert after (0-based, use -1 for beginning)
     * @param execute Whether to execute the new cell
     */
    public async addCode(code: string, afterCellNumber: number, execute: boolean = false): Promise<vscode.NotebookCell> {
        const editor = this.getNotebookEditor();
        if (!editor) {
            throw new Error('No active notebook');
        }

        // Validate afterCellNumber and calculate insertion position
        // Handle special case: -1 means "insert at beginning" for empty notebooks
        const insertPosition = afterCellNumber === -1 ? 0 : Math.min(afterCellNumber + 1, editor.notebook.cellCount);
        
        try {
            const newCell = new vscode.NotebookCellData(vscode.NotebookCellKind.Code, code, 'python');
            const edit = new vscode.WorkspaceEdit();
            const notebookEdit = vscode.NotebookEdit.insertCells(insertPosition, [newCell]);
            edit.set(editor.notebook.uri, [notebookEdit]);
            
            const success = await vscode.workspace.applyEdit(edit);
            if (success) {
                editor.selection = new vscode.NotebookRange(insertPosition, insertPosition + 1);
                
                // Auto-follow: reveal the newly added cell
                this.revealCell(insertPosition);
                
                if (execute) {
                    const cell = await this.executeCell(insertPosition);
                    return cell;
                } else {
                    const cell = editor.notebook.cellAt(insertPosition);
                    return cell;
                }
            } else {
                throw new Error(`❌ Failed to add code after cell ${insertPosition}`);
            }
        } catch (error) {
            throw new Error(`Error adding code after cell ${afterCellNumber}: ${error}`);
        }
    }

    /**
     * Replace code in specified cell.
     * @param code New code content
     * @param cellNumber Cell to replace (0-based)
     * @param execute Whether to execute after replacement
     */
    public async replaceCode(code: string, cellNumber: number, execute: boolean = false): Promise<vscode.NotebookCell> {
        const editor = this.getNotebookEditor();
        if (!editor) {
            throw new Error('No active notebook');
        }

        // Validate cell number
        if (cellNumber < 0 || cellNumber >= editor.notebook.cellCount) {
            throw new Error(`Invalid cell number ${cellNumber}. Notebook has ${editor.notebook.cellCount} cells.`);
        }
        
        try {
            const targetCell = editor.notebook.cellAt(cellNumber);
            const edit = new vscode.WorkspaceEdit();
            const fullRange = new vscode.Range(
                targetCell.document.lineAt(0).range.start,
                targetCell.document.lineAt(targetCell.document.lineCount - 1).range.end
            );
            edit.replace(targetCell.document.uri, fullRange, code);
            
            const success = await vscode.workspace.applyEdit(edit);
            if (success) {
                editor.selection = new vscode.NotebookRange(cellNumber, cellNumber + 1);
                
                // Auto-follow: reveal the replaced cell
                this.revealCell(cellNumber);
                
                if (execute) {
                    const cell = await this.executeCell(cellNumber);
                    return cell;
                } else {
                    const cell = editor.notebook.cellAt(cellNumber);
                    return cell;
                }
            } else {
                throw new Error(`❌ Failed to replace code in cell ${cellNumber}`);
            }
        } catch (error) {
            throw new Error(`Error adding code after cell ${cellNumber}: ${error}`);
        }
    }

    /**
     * Unified method for executing notebook cells with robust completion waiting.
     * 
     * This function appears synchronous to callers by:
     * 1. Starting execution via VSCode command (immediate)
     * 2. Polling cell.executionSummary until Jupyter kernel completes (up to 30 minutes)
     * 3. Returning success/failure only after actual completion
     * 
     * This eliminates race conditions in autonomous mode where the next iteration
     * could start before cell execution completes and error detection occurs.
     * 
     * @param cellNumber Cell number to execute (0-based)
     * @param silent If true, suppresses success messages (for autonomous mode)
     * @returns For cell number: boolean success. For cell object: execution order number or undefined
     */
    public async executeCell(cellNumber: number, _silent: boolean = false): Promise<vscode.NotebookCell> {
        let cell: vscode.NotebookCell;
        
        // Sanity checks
        const editor = this.getNotebookEditor();
        if (!editor) {
            throw new Error;
        }

        if (cellNumber < 0 || cellNumber >= editor.notebook.cellCount) {
            throw new Error(`Invalid cell number ${cellNumber}. Notebook has ${editor.notebook.cellCount} cells.`);
        }

        // Define cell to execute
        cell = editor.notebook.cellAt(cellNumber);
        
        // Check that it s a code cell
        if (cell.kind !== vscode.NotebookCellKind.Code) {
            const message = cellNumber !== undefined ? `Cell ${cellNumber} is not a code cell` : 'Can only execute code cells';
            console.error(message);
            return cell;
        }

        // Execute cell directly via VSCode command
        await vscode.commands.executeCommand('notebook.cell.execute', {
            ranges: [{ start: cell.index, end: cell.index + 1 }],
            document: cell.notebook.uri
        });
        
        // Wait for execution to complete (30 minutes for both modes)
        const maxWaitTime = 30 * 60 * 1000; // 30 minutes
        const checkInterval = 500; // Check every 0.5 seconds
        const progressCheckInterval = 5 * 60 * 1000; // Progress check every 5 minutes
        let waited = 0;
        let lastProgressCheck = 0;

        while (waited < maxWaitTime) {
            // Check if execution has completed:
            // The endTime property of the cell is set after completion.
            // Note: endTime is reset at each execution.
            const completedExecution = cell.executionSummary?.timing?.endTime !== undefined;

            if (completedExecution) {
                // Track execution in history now that it's complete
                this.addToExecutionHistory(cell);
                return cell;
            }

            // Progress monitoring for long-running cells
            if (waited - lastProgressCheck >= progressCheckInterval && waited > 0) {
                lastProgressCheck = waited;

                // Get current outputs from the cell
                const partialOutputs = this.getCellOutputsAsText(cell);

                // Send progress check message to Python
                const monitorResult = await this.agentProvider.checkExecutionProgress(
                    cell.document.getText(),
                    Math.floor(waited / 1000), // Convert to seconds
                    partialOutputs
                );

                if (!monitorResult.shouldContinue) {
                    console.log(`Execution monitor decided to terminate cell ${cellNumber} after ${waited}ms`);
                    console.log(`Termination reason: ${monitorResult.feedback}`);

                    // Track termination
                    this._lastTerminatedCellIndex = cellNumber;
                    this._terminationReason = monitorResult.feedback;

                    // Interrupt the cell execution
                    await this.interruptCell(cell);

                    // Add to execution history (with termination marker)
                    this.addToExecutionHistory(cell);

                    return cell;
                }
            }

            // Wait and check again
            await new Promise(resolve => setTimeout(resolve, checkInterval));
            waited += checkInterval;
        }

        // Timeout handling
        console.error(`Execution timeout for cell ${cellNumber} after ${maxWaitTime}ms`);
        return cell;
    }

    public async deleteCell(targetCellIndex: number): Promise<void> {
        const editor = this.getNotebookEditor();
        if (!editor) {
            return;
        }

        const notebook = editor.notebook;
        if (targetCellIndex < 0 || targetCellIndex >= notebook.cellCount) {
            return;
        }

        try {
            // Create workspace edit to delete the cell
            const edit = new vscode.WorkspaceEdit();
            const notebookEdit = vscode.NotebookEdit.deleteCells(new vscode.NotebookRange(targetCellIndex, targetCellIndex + 1));
            edit.set(notebook.uri, [notebookEdit]);
            
            await vscode.workspace.applyEdit(edit);
            
            // Move selection to the cell after the deleted one (or the last cell if deleting the last one)
            const newCellCount = notebook.cellCount - 1;
            let newSelectedIndex = targetCellIndex;
            if (newSelectedIndex >= newCellCount && newCellCount > 0) {
                newSelectedIndex = newCellCount - 1;
            }
            
            if (newCellCount > 0) {
                editor.selection = new vscode.NotebookRange(newSelectedIndex, newSelectedIndex + 1);
                // Don't call revealRange() as it interferes with the kernel
                // The cell will still be visible and selected
            }
            
        } catch (error: any) {
            console.error(`Failed to delete cell ${targetCellIndex}: ${error.message}`);
        }
    }

    private getCellOutputsAsText(cell: vscode.NotebookCell): string {
        /**
         * Extract text representation of all outputs from a cell.
         * Includes stdout, stderr, error messages, and display data.
         */
        if (!cell.outputs || cell.outputs.length === 0) {
            return "No outputs yet";
        }

        const outputTexts: string[] = [];

        for (const output of cell.outputs) {
            for (const item of output.items) {
                // Handle different MIME types
                if (item.mime === 'application/vnd.code.notebook.stdout' ||
                    item.mime === 'text/plain') {
                    const text = new TextDecoder().decode(item.data);
                    outputTexts.push(text);
                } else if (item.mime === 'application/vnd.code.notebook.stderr') {
                    const text = new TextDecoder().decode(item.data);
                    outputTexts.push(`[stderr] ${text}`);
                } else if (item.mime === 'application/vnd.code.notebook.error') {
                    const text = new TextDecoder().decode(item.data);
                    outputTexts.push(`[error] ${text}`);
                }
            }
        }

        if (outputTexts.length === 0) {
            return "Outputs exist but contain no text (possibly plots/images only)";
        }

        // Limit output size to prevent overwhelming the LLM
        const combinedOutput = outputTexts.join('\n');
        const maxLength = 5000; // 5000 characters should be enough for progress analysis
        if (combinedOutput.length > maxLength) {
            return combinedOutput.substring(0, maxLength) + '\n... [output truncated]';
        }

        return combinedOutput;
    }

    private async interruptCell(cell: vscode.NotebookCell): Promise<void> {
        /**
         * Interrupt execution of a running cell.
         */
        try {
            // Use VSCode's built-in interrupt command
            await vscode.commands.executeCommand('notebook.cell.cancelExecution', {
                ranges: [{ start: cell.index, end: cell.index + 1 }],
                document: cell.notebook.uri
            });
            console.log(`Interrupted execution of cell ${cell.index}`);
        } catch (error: any) {
            console.error(`Failed to interrupt cell ${cell.index}: ${error.message}`);
        }
    }
}