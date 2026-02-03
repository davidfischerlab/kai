import * as vscode from 'vscode';
import { KaiAgentProvider } from './agent-provider';
import { ChatCore } from './chat-core';
import { NotebookOperations } from './notebook-operations';
import { AutonomousExecution } from './autonomous-execution';
import { KernelFix } from '../kernel-fix';

/**
 * ChatViewProvider - Main coordinator for chat UI and component interactions
 *
 * **Architecture:**
 * This class serves as the main coordinator between all chat-related components and the VSCode webview.
 * It delegates functionality to specialized components while managing UI updates and message routing.
 *
 * **Component Responsibilities:**
 * - `ChatCore`: Message state, conversation history, context management
 * - `AutonomousExecution`: Autonomous mode state and execution flow control
 * - `NotebookOperations`: Jupyter notebook interactions (cell creation, execution, deletion)
 * - `KaiAgentProvider`: Python subprocess communication and message streaming
 *
 * **Message Flow:**
 * KaiAgentProvider → messageCallback → _handleMessage() → Component delegation
 * Webview → onDidReceiveMessage → Switch-case → Component methods
 * Component updates → _updateWebview() → Webview UI refresh
 * ```
 *
 * **Real-time Message Types Handled:**
 * - `display`: Standard chat messages → ChatCore.handleDisplayMessage()
 * - `task_list_display`: Task list updates → ChatCore.handleTaskListMessage()
 * - `execute_code`: Autonomous code execution → AutonomousExecution.handleAutonomousCodeExecution()
 * - `auto_loop_update`: Autonomous workflow completion signals (LOOP_COMPLETE/LOOP_INCOMPLETE)
 * - `regular_chat_complete`: Regular chat completion logging
 */
export class ChatViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'kai_agent_Chat';
    
    private _view?: vscode.WebviewView;
    
    // Component instances
    private chatCore: ChatCore;
    private notebookOps: NotebookOperations;
    private autonomousExecution: AutonomousExecution;
    
    // State variables needed for chat message handling
    private kernelFixChecked = false;
    
    constructor(
        private readonly _extensionUri: vscode.Uri,
        private agentProvider: KaiAgentProvider
    ) {
        // Initialize components with proper dependencies
        this.chatCore = new ChatCore(
            this._updateWebview.bind(this),
            this._storeToolUsage.bind(this),
            () => this.autonomousExecution?.autonomousMode || false
        );
        
        this.notebookOps = new NotebookOperations(
            this._revealCell.bind(this),
            this.agentProvider
        );
        
        // Set circular reference after both components are created
        this.chatCore.setNotebookOperations(this.notebookOps);

        // Set up targeted message sending for ChatCore
        this.chatCore.setSendToWebview((msg: any) => {
            if (this._view) {
                this._view.webview.postMessage(msg);
            }
        });

        // Initialize agent.
        this.agentProvider.initializePythonAgent(this.chatCore.turboEnabled);
        
        this.autonomousExecution = new AutonomousExecution(
            this.agentProvider,
            this.notebookOps,
            this.chatCore,
            this._updateAutoModeButton.bind(this)
        );
        
        // Set up message callback
        this.agentProvider.setMessageCallback(this._handleMessage.bind(this));

        // Listen for cell executions to track history
        this.notebookOps.setupExecutionTracking();

        // Listen for notebook open/close to update selector
        vscode.workspace.onDidOpenNotebookDocument(() => this._updateNotebookSelector());
        vscode.workspace.onDidCloseNotebookDocument(() => this._updateNotebookSelector());
        vscode.window.onDidChangeActiveNotebookEditor((editor) => {
            // Auto-select newly activated notebook if no notebook is tracked yet
            if (editor && !this.notebookOps.getTrackedNotebook()) {
                this.notebookOps.setTrackedNotebook(editor.notebook);
            }
            this._updateNotebookSelector();
        });
    }
    
    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;
        
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
            enableCommandUris: true
        };
        
        webviewView.webview.html = this.chatCore.getHtmlForWebview(webviewView.webview);
        
        // Handle messages from the webview - delegate to appropriate components
        webviewView.webview.onDidReceiveMessage(async data => {
            switch (data.type) {
                case 'chat':
                    await this._handleChatMessage(data.message);
                    break;
                case 'toggleRag':
                    this.chatCore.ragEnabled = !this.chatCore.ragEnabled;
                    console.log('RAG toggled:', this.chatCore.ragEnabled ? 'enabled' : 'disabled');
                    this._updateRagToggle();
                    break;
                case 'toggleTurbo':
                    this.chatCore.turboEnabled = !this.chatCore.turboEnabled;
                    console.log('Turbo toggled:', this.chatCore.turboEnabled ? 'enabled' : 'disabled');
                    this._updateTurboToggle();
                    break;
                case 'setMode':
                    // Handle mode switching from the 3-mode selector
                    const newMode = data.mode as 'chat' | 'guided' | 'autonomous';
                    this.chatCore.interactionMode = newMode;
                    // Sync autonomous mode state with the mode selector
                    this.autonomousExecution.autonomousMode = (newMode === 'autonomous' || newMode === 'guided');
                    // Tutorial mode (guided) automatically enables learning mode for explanations
                    this.chatCore.learningMode = (newMode === 'guided');
                    console.log('Mode switched to:', newMode, 'learning mode:', this.chatCore.learningMode);
                    this._updateModeIndicator();
                    this._updateAutoFollowToggle();
                    break;
                case 'jumpToCell':
                    // Navigate to a specific cell in the notebook
                    this._jumpToCell(data.cellIndex);
                    break;
                case 'openReferenceNotebook':
                    // Open a reference notebook file in VS Code
                    this._openReferenceNotebook(data.sourcePath, data.notebookId);
                    break;
                case 'viewReferenceCell':
                    // Show cell content in a peek view or quick info
                    this._viewReferenceCell(data.notebookId, data.cellIndex);
                    break;
                case 'taskAction':
                    // Handle task card action buttons
                    this._handleTaskAction(data.action, data.taskId);
                    break;
                case 'toggleAutoFollow':
                    // Allow toggle always, but only effective during autonomous mode
                    this.autonomousExecution.autoFollowEnabled = !this.autonomousExecution.autoFollowEnabled;
                    console.log('Auto-follow toggled:', this.autonomousExecution.autoFollowEnabled ? 'enabled' : 'disabled');
                    this._updateAutoFollowToggle();
                    break;
                case 'toggleLearningMode':
                    // Toggle learning mode (explain-as-you-go)
                    this.chatCore.learningMode = !this.chatCore.learningMode;
                    console.log('Learning mode toggled:', this.chatCore.learningMode ? 'enabled' : 'disabled');
                    this._updateLearningModeToggle();
                    break;
                case 'clear':
                    // Use ChatCore method for state management instead of direct array manipulation
                    this.chatCore.clearMessages();
                    this._updateWebview();
                    break;
                case 'insertCode':
                    // Insert code without executing - requires explicit cell number
                    if (data.cellNumber !== null && data.cellNumber !== undefined) {
                        await this.notebookOps.addCode(data.code, data.cellNumber, false);
                    } else {
                        console.error('insertCode: No cell number provided in button metadata');
                    }
                    break;
                case 'replaceCode':
                    // Replace current cell without executing - requires explicit cell number
                    console.log('🔄 Replace code button clicked');
                    if (data.cellNumber !== null && data.cellNumber !== undefined) {
                        const replaceSuccess = await this.notebookOps.replaceCode(data.code, data.cellNumber, false);
                        if (!replaceSuccess) {
                            console.log(`❌ Replace failed for cell ${data.cellNumber}`);
                        }
                    } else {
                        console.error('replaceCode: No cell number provided in button metadata');
                    }
                    break;
                case 'insertAndExecuteCode':
                    // Insert and execute code - requires explicit cell number
                    if (data.cellNumber !== null && data.cellNumber !== undefined) {
                        await this.notebookOps.addCode(data.code, data.cellNumber, true);
                    } else {
                        console.error('insertAndExecuteCode: No cell number provided in button metadata');
                    }
                    break;
                case 'replaceAndExecuteCode':
                    // Replace current cell and execute - requires explicit cell number
                    if (data.cellNumber !== null && data.cellNumber !== undefined) {
                        const replaceSuccess = await this.notebookOps.replaceCode(data.code, data.cellNumber, true);
                        if (!replaceSuccess) {
                            console.log(`❌ Replace failed for cell ${data.cellNumber}`);
                        }
                    } else {
                        console.error('replaceAndExecuteCode: No cell number provided in button metadata');
                    }
                    break;
                case 'markButtonsUsed':
                    // Use ChatCore method for state management
                    if (typeof data.messageIndex === 'number') {
                        if (this.chatCore.markButtonsUsed(data.messageIndex)) {
                            this._updateWebview();
                        }
                    }
                    break;
                case 'storeToolUsage':
                    this._storeToolUsage(data.tool);
                    break;
                case 'startAutonomousExecution':
                    const autonomousContext = await this.chatCore.getContextForMessage(data.message);
                    await this.autonomousExecution.runAutonomousLoop(data.message, autonomousContext);
                    break;
                case 'stopAutonomousExecution':
                    await this.autonomousExecution.terminateAutonomousExecution();
                    break;
                case 'feedbackWithMessage':
                    await this.autonomousExecution.handleFeedbackInterupt(data.message);
                    break;
                case 'continueAutonomous':
                    // Continue autonomous execution without additional feedback (from continue button)
                    // Remove checkpoint messages from ChatCore to prevent accumulation
                    this.chatCore.removeCheckpointMessages();
                    this._updateWebview();
                    await this.autonomousExecution.handleFeedbackInterupt('');
                    break;
                case 'notebookControl':
                    // Handle notebook control operations (restart, run cells, delete)
                    await this._handleNotebookControl(data);
                    break;
                case 'selectNotebook':
                    // Handle notebook selection from dropdown
                    if (data.notebookUri) {
                        const notebook = vscode.workspace.notebookDocuments.find(
                            doc => doc.uri.toString() === data.notebookUri
                        );
                        if (notebook) {
                            this.notebookOps.setTrackedNotebook(notebook);
                            console.log(`[KAI] Selected notebook: ${notebook.uri.path.split('/').pop()}`);
                        }
                    } else {
                        this.notebookOps.setTrackedNotebook(undefined);
                        console.log('[KAI] Cleared notebook selection');
                    }
                    this._updateNotebookSelector();
                    break;
                // ... other cases will be handled by appropriate components
            }
        });
        
        // Send initial states and restore chat history
        setTimeout(() => {
            this._updateRagToggle();
            this._updateTurboToggle();
            this._updateModeIndicator();
            this._updateAutoFollowToggle();
            this._updateNotebookSelector();

            // Auto-select currently active notebook if none tracked
            if (!this.notebookOps.getTrackedNotebook()) {
                const activeEditor = vscode.window.activeNotebookEditor;
                if (activeEditor) {
                    this.notebookOps.setTrackedNotebook(activeEditor.notebook);
                    this._updateNotebookSelector();
                }
            }

            // Restore chat messages if they exist
            if (this.chatCore.messages.length > 0) {
                this._updateWebview();
            } else {
                // Only send welcome message on first load (no existing messages)
                this.chatCore.addMessage('assistant', 'Hello! I can help you with: explaining code, fixing errors, building code for analyses, and answering any other bioinformatics questions! What would you like to work on today?');
            }
        }, 100);
    }
    
    // Private helper methods that coordinate between components
    private async _handleChatMessage(message: string) {
        // Check for kernel fix cell on first message (when we know a notebook is active)
        if (!this.kernelFixChecked) {
            this.kernelFixChecked = true;
            try {
                const fixAdded = await KernelFix.ensureFixCellExists();
                if (fixAdded) {
                    this.chatCore.addMessage('assistant', '🔧 Added kernel stability fix cell to the top of your notebook (macOS VSCode compatibility). Please run this cell first before your analysis to prevent kernel crashes.');
                }
            } catch (error) {
                console.log('Kernel fix check failed (no active notebook):', error);
            }
        }
        
        // Add user message
        this.chatCore.addMessage('user', message);
        
        // Get context based on message type (used for autonomous/regular mode routing)
        const context = await this.chatCore.getContextForMessage(message);

        // Check if in guided or autonomous mode (both use the autonomous loop)
        const mode = this.chatCore.interactionMode;
        if (mode === 'guided' || mode === 'autonomous') {
            if (!this.autonomousExecution.loopRunning) {
                // Guided/Autonomous mode enabled but no loop running - start the loop
                await this.autonomousExecution.runAutonomousLoop(message, context);
                return;
            } else {
                // Loop is running - handle all feedback through interrupt mechanism
                await this.autonomousExecution.handleFeedbackInterupt(message);
                return;
            }
        }
        
        try {
            // Button actions now bypass this method entirely and go directly to insertion methods
            
            // Show activity indicator
            const indicatorText = this.agentProvider.fullyReady ? 'thinking' : 'initializing';
            this.chatCore.showIndicator(indicatorText);
            
            // Generate request ID for streaming coordination
            const requestId = `chat_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
            
            const result = await this.agentProvider.sendRegularRequest(message, context);

            // Hide activity indicator and add complete response
            this.chatCore.hideIndicator();
            
            // Handle response based on type
            if (result.response && typeof result.response === 'object' && 'text' in result.response) {
                // Structured response with text
                this.chatCore.addMessage('assistant', (result.response as any).text, false);
            } else if (result.response) {
                // Plain text response
                this.chatCore.addMessage('assistant', String(result.response), false);
            }
            
        } catch (error: any) {
            // Hide activity indicator if it's still there
            this.chatCore.hideIndicator();
            
            // Add more detailed error information for debugging
            console.error('Chat error:', error);
            let errorMessage = `Error: ${error.message}`;
            
            if (error.message.includes('Failed to start Python')) {
                errorMessage += '\n\nTip: Make sure Python is installed and the kai_agent.pythonPath setting points to the correct Python executable.';
            } else if (error.message.includes('timed out')) {
                errorMessage += '\n\nTip: The LLM is taking longer than usual. Try a simpler question or restart the extension.';
            }
            
            this.chatCore.addMessage('assistant', errorMessage);
        }
    }

    private _updateAutoModeButton() {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'updateAutoModeButton',
                isAutonomousRunning: this.autonomousExecution.autonomousMode,
                isWaitingForFeedback: this.autonomousExecution.waitingForFeedback,
                isTutorialMode: this.chatCore.interactionMode === 'guided',
            });
        }
    }

    /**
     * Handle real-time messages from KaiAgentProvider.
     *
     * This method processes all real-time messages sent by the Python agent via VSCodeCommunicator.
     * Messages are routed to appropriate components based on their type.
     *
     * @param type Message type identifier
     * @param data Message payload data
     */
    private async _handleMessage(type: string, data: any): Promise<boolean> {
        if (type === 'display') {
            // Handle standard display message types from agent
            this.chatCore.handleDisplayMessage(data);
        } else if (type === 'task_list_display') {
            // Handle real-time task list updates
            this.chatCore.handleTaskListMessage(data);
        } else if (type === 'reference_workflows') {
            // Handle reference workflow IDs storage for task list display
            this.chatCore.storeReferenceWorkflows(data);
        } else if (type === 'execute_code') {
            // Handle execute code messages for autonomous mode
            // Only process execute_code messages if autonomous mode is still active
            // and NOT waiting for feedback (checkpoint shown)
            if (!this.autonomousExecution.autonomousMode) {
                console.log('🛑 Skipping execute_code message - autonomous mode stopped');
            } else if (this.autonomousExecution.waitingForFeedback) {
                console.log('🛑 Skipping execute_code message - waiting for user feedback');
            } else {
                // Pass to autonomous execution for processing
                await this.autonomousExecution.handleAutonomousCodeExecution(data);
            }
        } else if (type === 'auto_loop_update') {
            // Workflow state update from Python - controls loop flow
            // @see langgraph_orchestrator.py send_workflow_result()
            this.autonomousExecution.setWorkflowState(data.status);

            // Signal that workflow has completed - this allows autonomous loop to proceed
            this.autonomousExecution.signalWorkflowCompletion();

            // Route based on state:
            if (data.status === 'LOOP_COMPLETE') {
                // All tasks done - terminate autonomous session
                this.autonomousExecution.terminateAutonomousExecution();
            } else if (data.status === 'LEARNING_MODE_PENDING') {
                // Learning mode iteration - Python signals that VSCode should request
                // learning explanation AFTER confirming execution succeeded.
                // autonomous-execution.ts will handle this based on actual execution result.
            }
            // LOOP_INCOMPLETE is handled by the autonomous loop based on interaction mode
        } else if (type === 'no_output') {
            // No output messages are queued for completion tracking only
            // No processing needed - just return true to mark as complete
        } else {
            console.error(`Unhandled message type: ${type}`, data);
        }

        return true; // All messages complete immediately
    }

    private _updateWebview() {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'updateMessages',
                messages: this.chatCore.messages
            });
        }
    }
    
    private _updateRagToggle() {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'updateRagToggle',
                enabled: this.chatCore.ragEnabled
            });
        }
    }
    
    private _updateTurboToggle() {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'updateTurboToggle',
                enabled: this.chatCore.turboEnabled
            });
        }
    }
    
    private _updateModeIndicator() {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'updateModeIndicator',
                mode: this.chatCore.interactionMode
            });
        }
    }

    private _updateNotebookSelector() {
        if (this._view) {
            // Get all open notebook documents
            const notebooks = vscode.workspace.notebookDocuments
                .filter(doc => doc.uri.path.endsWith('.ipynb'))
                .map(doc => ({
                    uri: doc.uri.toString(),
                    name: doc.uri.path.split('/').pop() || 'Unknown',
                    path: doc.uri.fsPath
                }));

            // Get currently tracked notebook URI
            const trackedNotebook = this.notebookOps.getTrackedNotebook();
            const selectedUri = trackedNotebook ? trackedNotebook.uri.toString() : '';

            this._view.webview.postMessage({
                type: 'updateNotebookList',
                notebooks: notebooks,
                selectedUri: selectedUri
            });
        }
    }

    private _jumpToCell(cellIndex: number): void {
        const editor = this.notebookOps.getNotebookEditor();
        if (editor && cellIndex >= 0 && cellIndex < editor.notebook.cellCount) {
            const range = new vscode.NotebookRange(cellIndex, cellIndex + 1);
            editor.revealRange(range, vscode.NotebookEditorRevealType.InCenter);
            editor.selection = range;
            console.log(`Jumped to cell ${cellIndex}`);
        } else {
            console.log(`Cannot jump to cell ${cellIndex}: invalid index or no notebook`);
        }
    }

    private async _openReferenceNotebook(sourcePath: string, notebookId: string, scrollToCellIndex?: number): Promise<void> {
        if (!sourcePath) {
            vscode.window.showWarningMessage(`Reference notebook path not available for ${notebookId}`);
            return;
        }

        try {
            const uri = vscode.Uri.file(sourcePath);
            await vscode.commands.executeCommand('vscode.openWith', uri, 'jupyter-notebook');
            console.log(`Opened reference notebook: ${sourcePath}`);

            // If a cell index is provided, scroll to it after a short delay
            if (scrollToCellIndex !== undefined && scrollToCellIndex >= 0) {
                // Wait for the notebook to load
                setTimeout(async () => {
                    const editor = vscode.window.visibleNotebookEditors.find(
                        e => e.notebook.uri.fsPath === sourcePath
                    );
                    if (editor && scrollToCellIndex < editor.notebook.cellCount) {
                        const range = new vscode.NotebookRange(scrollToCellIndex, scrollToCellIndex + 1);
                        editor.revealRange(range, vscode.NotebookEditorRevealType.AtTop);
                        editor.selection = range;
                        console.log(`Scrolled to cell ${scrollToCellIndex}`);
                    }
                }, 500);  // Small delay to let notebook render
            }
        } catch (error) {
            console.error(`Failed to open reference notebook: ${error}`);
            vscode.window.showErrorMessage(`Could not open notebook: ${sourcePath}`);
        }
    }

    private _viewReferenceCell(notebookId: string, cellIndex: number): void {
        // Open the notebook and scroll to the specified cell
        const notebooks = this.chatCore.storedReferenceNotebooks;
        if (!notebooks) {
            console.log('No reference notebooks stored');
            return;
        }

        const notebook = notebooks.find(nb => nb.id === notebookId);
        if (!notebook) {
            console.log(`Notebook ${notebookId} not found in stored references`);
            return;
        }

        if (notebook.source_path) {
            // Open notebook and scroll to cell
            this._openReferenceNotebook(notebook.source_path, notebookId, cellIndex);
        } else {
            // No source path available, show info message
            const cell = notebook.cells.find(c => c.index === cellIndex);
            if (cell) {
                vscode.window.showInformationMessage(
                    `Cell ${cellIndex} (${cell.type}): ${cell.preview}`
                );
            }
        }
    }

    private _updateLearningModeToggle() {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'updateLearningModeToggle',
                enabled: this.chatCore.learningMode
            });
        }
    }

    private _handleTaskAction(action: string, taskId: string): void {
        console.log(`Task action: ${action} on task ${taskId}`);

        switch (action) {
            case 'explain':
                // Send a chat message requesting explanation of this task
                const explainMessage = `Explain what task ${taskId} is doing and why it's necessary for the analysis.`;
                this.chatCore.addMessage('user', explainMessage);
                this.autonomousExecution.handleFeedbackInterupt(explainMessage);
                break;
            case 'skip':
                // Request the agent to skip this task
                const skipMessage = `Please skip task ${taskId} and move on to the next task.`;
                this.chatCore.addMessage('user', skipMessage);
                this.autonomousExecution.handleFeedbackInterupt(skipMessage);
                vscode.window.showInformationMessage(`Requested to skip task ${taskId}`);
                break;
        }
    }

    private _updateAutoFollowToggle() {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'updateAutoFollowToggle',
                enabled: this.autonomousExecution.autoFollowEnabled,
                autonomousMode: this.autonomousExecution.autonomousMode
            });
        }
    }
    
    private _revealCell(cellIndex: number): void {
        this.autonomousExecution.revealCell(cellIndex);
    }
    
    private _storeToolUsage(tool: {name: string, query?: string, status: string, collections?: string[]}) {
        // Find the most recent assistant message and add tool usage to it
        let targetMessageIndex = this.chatCore.messages.length - 1;
        
        if (targetMessageIndex < 0) {
            for (let i = this.chatCore.messages.length - 1; i >= 0; i--) {
                if (this.chatCore.messages[i].role === 'assistant') {
                    targetMessageIndex = i;
                    break;
                }
            }
        }
        
        if (targetMessageIndex >= 0 && targetMessageIndex < this.chatCore.messages.length) {
            const message = this.chatCore.messages[targetMessageIndex];
            if (!message.toolUsage) {
                message.toolUsage = [];
            }
            
            // Update existing tool usage or add new one
            const existingIndex = message.toolUsage.findIndex(t => t.name === tool.name);
            if (existingIndex >= 0) {
                message.toolUsage[existingIndex] = tool;
            } else {
                message.toolUsage.push(tool);
            }
            
            // Update the webview to show the stored tool usage
            this._updateWebview();
        }
    }
    
    private async _handleNotebookControl(data: any): Promise<void> {
        /**
         * Handle notebook control operations like restart kernel, run cells, delete cells
         */
        const operation = data.operation;
        
        this.chatCore.addMessage('assistant', `🔧 Notebook control operation`, false, false);
        
        switch (operation) {
            case 'restart_kernel':
                try {
                    await vscode.commands.executeCommand('jupyter.restartkernel');
                    if (!this.autonomousExecution.autonomousMode) {
                        this.chatCore.addMessage('assistant', '✅ Kernel restarted successfully', false, false);
                    }
                } catch (error: any) {
                    if (!this.autonomousExecution.autonomousMode) {
                        this.chatCore.addMessage('assistant', `❌ Failed to restart kernel: ${error.message}`, false, false);
                    }
                }
                break;
                
            case 'delete_cell':
                await this.notebookOps.deleteCell(data.cellIndex || 0);
                break;
                
            case 'run_all_cells_up_to':
                try {
                    // Use tracked notebook editor to handle cases where another tab is focused
                    const editor = this.notebookOps.getNotebookEditor();
                    if (editor && data.cellIndex < editor.notebook.cellCount) {
                        // Execute cells up to the specified index
                        for (let i = 0; i <= data.cellIndex; i++) {
                            const cell = editor.notebook.cellAt(i);
                            if (cell.kind === vscode.NotebookCellKind.Code) {
                                await this.notebookOps.executeCell(i, true);
                            }
                        }
                        this.chatCore.addMessage('assistant', `✅ Executed cells 0-${data.cellIndex}`, false, false);
                    }
                } catch (error: any) {
                    this.chatCore.addMessage('assistant', `❌ Failed to run cells: ${error.message}`, false, false);
                }
                break;
                
            default:
                this.chatCore.addMessage('assistant', `❌ Unknown notebook operation: ${operation}`, false, false);
        }
        
        // Continue autonomous execution if still in progress
        if (this.autonomousExecution.autonomousMode) {
            const progressMessage = `🔄 Continuing autonomous execution after ${operation}...`;
            this.chatCore.addMessage('assistant', progressMessage, true, false);
        }
    }
}