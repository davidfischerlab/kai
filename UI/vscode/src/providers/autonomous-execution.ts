import * as vscode from 'vscode';
import { KaiAgentProvider } from './agent-provider';
import { NotebookOperations } from './notebook-operations';
import { ChatCore } from './chat-core';

/**
 * AutonomousExecution - Manages unified autonomous workflow execution with natural language feedback
 *
 * **Core Responsibilities:**
 * - Orchestrates autonomous execution loops using unified Python-side intent classification
 * - Manages autonomous mode state and UI updates via Python-controlled UI messages
 * - Handles natural language user feedback through unified planning handler
 * - Processes autonomous code execution and notebook cell management
 * - Controls auto-follow functionality for cell navigation during autonomous mode
 *
 * **Unified Autonomous Flow:**
 * ```
 * User starts autonomous → runAutonomousLoop() → Unified intent classification
 *                      ↗ User messages → handleAutonomousPlanning() → Python routes via intent
 *                      ↗ Empty messages → handleAutonomousExecution() → Continue execution
 *                      ↗ Real-time: execute_code messages → handleAutonomousCodeExecution()
 *                      ↗ UI control: ui_control messages → Python controls feedback prompts
 *                      ↗ Completion: LOOP_COMPLETE → stopAutonomousViaAgent()
 * ```
 *
 * **State Management:**
 * - `autonomousMode`: Whether autonomous execution is active
 * - `autoFollowEnabled`: Whether to auto-navigate to cells being executed
 *
 * **Unified Framework Integration:**
 * - All user messages route through `handleAutonomousPlanning()` with Python intent classification
 * - Python controls UI state via `ui_control` messages (PROMPT_USER_FEEDBACK, etc.)
 * - Natural language feedback processed using AutoLoopIntentClassificationTool
 * - Intent types: TASK_LIST_MODIFICATION, CODE_IMPLEMENTATION_FEEDBACK, APPROVAL
 *
 * **Integration Points:**
 * - `KaiAgentProvider`: Unified communication with Python agent (planning + execution)
 * - `NotebookOperations`: Executes code cells and manages notebook state
 * - `ChatCore`: Updates chat UI and manages conversation context
 */
export class AutonomousExecution {
    // Autonomous mode toggle state - enabled by default
    private _autonomousMode: boolean = true;
    // Flag for whether autonomous loop is actively running
    private _loopRunning: boolean = false;
    
    // Auto-follow toggle state - starts as disabled
    private _autoFollowEnabled: boolean = true;
    
    // Current request ID
    private _lastExecutionFailed: boolean = false;
    private _errorCellIndex: number = -1;
    private _lastExecutionOutput: string = "";
    private _lastCellModifiedInAutoMode: number = -1;

    // Pending feedback from interruption
    private _pendingInterruptFeedback: string | null = null;

    // Workflow state tracking for feedback loops
    private _currentWorkflowState: string = "STARTING";
    private _waitingForFeedback: boolean = false;
    private _feedbackPromise: Promise<string> | null = null;
    private _feedbackResolver: ((feedback: string) => void) | null = null;

    // Workflow completion tracking
    private _waitingForWorkflowCompletion: boolean = false;
    private _workflowCompletionPromise: Promise<void> | null = null;
    private _workflowCompletionResolver: (() => void) | null = null;

    constructor(
        private agentProvider: KaiAgentProvider,
        private notebookOps: NotebookOperations,
        private chatCore: ChatCore,
        private updateAutoModeButton: () => void
    ) {}

    get autonomousMode() { return this._autonomousMode; }
    set autonomousMode(value: boolean) { this._autonomousMode = value; }
    
    get autoFollowEnabled() { return this._autoFollowEnabled; }
    set autoFollowEnabled(value: boolean) { this._autoFollowEnabled = value; }

    get loopRunning() { return this._loopRunning; }
    get waitingForFeedback() { return this._waitingForFeedback; }

    /**
     * Called by ChatViewProvider when Python sends workflow state updates.
     * States: LOOP_COMPLETE, LOOP_INCOMPLETE, LOOP_INCOMPLETE_REQUIRE_FEEDBACK
     * @see ChatViewProvider._handleMessage() for state routing
     */
    setWorkflowState(state: string) {
        console.log('WORKFLOW STATE UPDATED:', state);
        this._currentWorkflowState = state;
    }

    /**
     * Called by ChatViewProvider when user provides feedback during autonomous mode.
     * Resolves the promise created by waitForUserFeedback(), allowing the loop to continue.
     * @see ChatViewProvider._handleChatMessage() for routing logic
     */
    provideFeedback(feedback: string) {
        if (this._feedbackResolver) {
            this._feedbackResolver(feedback);
            this._feedbackResolver = null;
            this._feedbackPromise = null;
            this._waitingForFeedback = false;
        }
    }

    /** Wait for user feedback within the autonomous loop */
    private async waitForUserFeedback(): Promise<string> {
        this._waitingForFeedback = true;
        this._feedbackPromise = new Promise<string>((resolve) => {
            this._feedbackResolver = resolve;
        });
        return this._feedbackPromise;
    }

    /** Wait for workflow completion signal (auto_loop_update) */
    private async waitForWorkflowCompletion(): Promise<void> {
        this._waitingForWorkflowCompletion = true;
        this._workflowCompletionPromise = new Promise<void>((resolve) => {
            this._workflowCompletionResolver = resolve;
        });
        return this._workflowCompletionPromise;
    }

    /** Signal that workflow completion was received */
    public signalWorkflowCompletion(): void {
        if (this._workflowCompletionResolver) {
            this._workflowCompletionResolver();
            this._workflowCompletionResolver = null;
            this._workflowCompletionPromise = null;
            this._waitingForWorkflowCompletion = false;
        }
    }

    public async runAutonomousLoop(initialMessage: string, initialContext: any): Promise<void> {
        /**
         * Unified autonomous execution loop with mid-iteration feedback support.
         *
         * Flow:
         * 1. Send message to Python (planning or execution)
         * 2. Python processes and may return LOOP_INCOMPLETE_REQUIRE_FEEDBACK
         * 3. If feedback needed, pause THIS iteration and wait for user input
         * 4. Continue same iteration with feedback (via continue statement)
         * 5. Only advance to next iteration when Python completes current one
         *
         * Key insight: Feedback continues the SAME iteration, not a new one.
         *
         * @see handleAutonomousPlanning() in agent-provider.ts
         * @see _handle_autonomous_unified() in workflow_orchestrator.py
         */
        try {
            this._autonomousMode = true;
            this._loopRunning = true;
            this.updateAutoModeButton();

            // Reset task tracking for new session to ensure fresh bubble
            this.chatCore.resetTaskTracking();

            // Show initial processing message
            this.chatCore.addMessage('assistant', 'Starting autonomous mode...', true, false);

            let currentMessage = initialMessage;
            let context = await this.chatCore.getContextForMessage(initialMessage);
            context.autonomousMode = true;
            context.autonomousModeContinue = false;
            context.lastExecutionFailed = this._lastExecutionFailed;
            context.errorCellIndex = this._errorCellIndex;
            context.executionResult = this._lastExecutionOutput;
            context.lastCellModifiedInAutoMode = this._lastCellModifiedInAutoMode;

            // Main autonomous loop - with feedback support
            while (this._autonomousMode) {
                if(this._currentWorkflowState === "LOOP_INCOMPLETE_REQUIRE_FEEDBACK") {
                    // Pause and wait for user feedback.
                    // This creates a synchronous pause point where the loop waits for
                    // user input via provideFeedback() called from ChatViewProvider.
                    this.chatCore.addMessage('assistant', 'Please provide feedback to proceed.', false, false);
                    currentMessage = await this.waitForUserFeedback();
                    context.autonomousModeContinue = false;
                } else if (this._currentWorkflowState === "LOOP_INCOMPLETE_FEEDBACK_INTERRUPT") {
                    // Handle feedback interrupt case where user provided feedback
                    // At this point, the interrupted iteration (last iteration) finished.
                    // Outputs were filtered after interruption, so we have to re-enable processing:
                    this.agentProvider.resumeToolOutputProcessing();
                    // And provide the feedback message for this iteration:
                    currentMessage = this._pendingInterruptFeedback || "";
                    this._pendingInterruptFeedback = null; // Clear after use
                    context.autonomousModeContinue = false;
                } else {
                    context.autonomousModeContinue = true;
                }

                // Start waiting for workflow completion signal before sending message
                const workflowCompletionPromise = this.waitForWorkflowCompletion();

                // This function returns once workflow has returned.
                // Tool outputs are come in through separate channel and are caught below.
                await this.agentProvider.handleAutonomousIteration(currentMessage, context);

                // Check if stop was requested during the agent call
                if (!this._autonomousMode) {
                    console.log('🛑 Autonomous mode stopped during iteration');
                    break;
                }

                // Workflow iteration completion checks
                // 1. Wait for workflow iteration completion signal (auto_loop_update)
                // This ensures that all messages have been queued, 
                // ie. that the following check tests all messages that belong to a worflow.
                await workflowCompletionPromise;
                // 2. Wait for all pending messages (execute_code, display, etc.) to complete
                // This ensures cell creation/execution finishes before advancing to next iteration.
                while (this.agentProvider.hasPendingMessages()) {
                    await new Promise(resolve => setTimeout(resolve, 100));
                }

                // Check if workflow completed
                if (this._currentWorkflowState === "LOOP_COMPLETE") {
                    break;
                }

                // Only advance to next iteration when Python completes the iteration
                if (this._autonomousMode) {
                    // Continue with empty message (execution mode)
                    currentMessage = ""; // Reset to empty for continuation
                    context = await this.chatCore.getContextForMessage(currentMessage);
                    context.autonomousMode = true;

                    // Add error context for autonomous execution
                    context.lastExecutionFailed = this._lastExecutionFailed;
                    context.errorCellIndex = this._errorCellIndex;
                    context.executionResult = this._lastExecutionOutput;
                    context.lastCellModifiedInAutoMode = this._lastCellModifiedInAutoMode;
                }
            }

            // Complete autonomous execution
            this.terminateAutonomousExecution();

        } catch (error: any) {
            console.error('Error in autonomous loop:', error);
            this.terminateAutonomousExecution();
        }
    }
    
    public async terminateAutonomousExecution(): Promise<void> {
        /**
         * Unified method to terminate autonomous execution cleanly.
         * Handles all termination scenarios with consistent state management.
         */
        if (!this._autonomousMode) {
            return; // Already terminated
        }
        this._autonomousMode = false;
        this._loopRunning = false;
        this._autoFollowEnabled = false;

        // Reset feedback state
        this._waitingForFeedback = false;
        this._currentWorkflowState = "";
        this._pendingInterruptFeedback = null;
        if (this._feedbackResolver) {
            this._feedbackResolver("");  // Resolve with empty string to break waiting
            this._feedbackResolver = null;
            this._feedbackPromise = null;
        }

        // Reset workflow completion state
        this._waitingForWorkflowCompletion = false;
        if (this._workflowCompletionResolver) {
            this._workflowCompletionResolver();  // Resolve to break waiting
            this._workflowCompletionResolver = null;
            this._workflowCompletionPromise = null;
        }
        
        // Update UI immediately
        this.updateAutoModeButton();
        
        // Reset task tracking to prevent overwriting completed task lists
        this.chatCore.resetTaskTracking();
        
        // Call agent stop (idempotent - safe to call multiple times)
        this.agentProvider.stopAutonomousExecution().catch(err => {
            console.error('Error calling agent stop:', err);
        });
    }

    public async handleFeedbackInterupt(feedbackMessage: string) {
        /**
         * Handle feedback-based interruption during execution.
         *
         * This handles the case where user provides feedback while the loop is executing
         * (not waiting for feedback). We need to:
         * 1. Pause incoming tool messages in VSCode
         * 2. Set state to require feedback so loop will pause at next iteration
         * 3. Store the feedback for when loop checks
         */
        try {
            if (this._waitingForFeedback) {
                // Already waiting for feedback, just provide it directly
                this.provideFeedback(feedbackMessage);
                return;
            }

            // Pause message processing to block incoming tool messages
            // This prevents any running tools from sending more output to the UI
            this.agentProvider.pauseToolOutputProcessing();

            // Set state to require feedback - this will cause the loop to pause
            // at the next iteration check
            this._currentWorkflowState = "LOOP_INCOMPLETE_FEEDBACK_INTERRUPT";

            // Store the feedback for when the loop creates a new promise
            // Don't call provideFeedback() yet - let the loop create the promise first
            this._pendingInterruptFeedback = feedbackMessage;

        } catch (error: any) {
            console.error('Error handling feedback interruption:', error);
            this.chatCore.addMessage('assistant', `❌ Error processing feedback: ${error.message}`, false, false);
        }
    }
    
    public async handleAutonomousCodeExecution(codeResponse: any): Promise<void> {
        try {
            // Check if this is a cell deletion response
            if (codeResponse.vscode_commands) {
                console.log('🗑️ Processing cell deletion commands');
                const commands = codeResponse.vscode_commands;
                // Execute deletion commands in reverse order
                for (const cmd of commands) {
                    if (cmd.command === 'deleteCell') {
                        console.log(`🗑️ Deleting cell ${cmd.cellIndex}`);
                        await this.notebookOps.deleteCell(cmd.cellIndex);

                        // Update _lastCellModifiedInAutoMode if the deleted cell affects it
                        if (this._lastCellModifiedInAutoMode >= cmd.cellIndex) {
                            // If we deleted the exact cell that was last modified, or a cell before it,
                            // adjust the index to account for the deletion
                            this._lastCellModifiedInAutoMode = Math.max(-1, this._lastCellModifiedInAutoMode - 1);
                            console.log(`📍 Updated lastCellModifiedInAutoMode to ${this._lastCellModifiedInAutoMode} after deleting cell ${cmd.cellIndex}`);
                        }
                    }
                }

                return;
            }

            const code = codeResponse.code?.trim();
            const cellNumber = codeResponse.positioning_info?.target_cell;
            const recoveryStrategy = codeResponse.error_recovery_strategy;
            const shouldReplace = codeResponse.should_replace_code === "true";

            console.log('Using cell position:', cellNumber, 'with should replace:', shouldReplace, ' and recovery strategy:', recoveryStrategy);

            let notebookCell: vscode.NotebookCell;
            const cellType = codeResponse.cell_type || "code";  // Default to code if not specified

            if (cellType === "markdown") {
                // Markdown cell handling
                if (shouldReplace) {
                    notebookCell = await this.notebookOps.replaceMarkdown(code, cellNumber);
                } else {
                    notebookCell = await this.notebookOps.addMarkdown(code, cellNumber);
                }
            } else {
                // Code cell handling (existing logic)
                if (shouldReplace) {
                    notebookCell = await this.notebookOps.replaceCode(code, cellNumber, true);

                    // Handle REPLACE_AND_RESTART strategy
                    if (recoveryStrategy === 'REPLACE_AND_RESTART') {
                        await this._handleRestartAndRunToCellStrategy(cellNumber);
                    }
                } else {
                    notebookCell = await this.notebookOps.addCode(code, cellNumber, true);
                }
            }

            // Extract execution results from object
            if (cellType === "markdown") {
                // Markdown cells don't execute, so always mark as success
                console.log('Markdown cell added:', cellNumber);
                this._errorCellIndex = -1;
                this._lastExecutionFailed = false;
                this._lastExecutionOutput = ""; // Markdown cells have no output
            } else {
                // Check if cell was terminated by execution monitor
                const wasTerminated = this.notebookOps.lastTerminatedCellIndex === notebookCell.index;

                if (wasTerminated) {
                    console.log(`Cell execution was TERMINATED by monitor: ${cellNumber}`);
                    console.log(`Termination reason: ${this.notebookOps.terminationReason}`);

                    // Treat termination as a failure requiring retry
                    this._errorCellIndex = notebookCell.index;
                    this._lastExecutionFailed = true;
                    this._lastExecutionOutput = `[EXECUTION TERMINATED BY MONITORING AGENT]\n${this.notebookOps.terminationReason}\n\nPartial outputs:\n${this.notebookOps.formatCellOutputToString(notebookCell)}`;

                    // Clear termination state now that we've handled it
                    this.notebookOps.clearTerminationState();
                } else {
                    // Normal execution success/failure handling
                    const executionSuccess = notebookCell.executionSummary?.success;
                    if (executionSuccess) {
                        console.log('Cell execution succeeded:', cellNumber);
                        this._errorCellIndex = -1;
                    } else {
                        console.log('Cell execution failed:', cellNumber);
                        this._errorCellIndex = notebookCell.index;
                    }
                    this._lastExecutionFailed = !executionSuccess;
                    this._lastExecutionOutput = this.notebookOps.formatCellOutputToString(notebookCell);
                }
            }
            this._lastCellModifiedInAutoMode = notebookCell.index;
        } catch (error) {
            console.error('Error in autonomous code execution:', error);
        }
    }
    
    private async _handleRestartAndRunToCellStrategy(errorCellIndex: number): Promise<void> {
        /**
         * Handle REPLACE_AND_RESTART error recovery strategy:
         * 1. Restart the kernel
         * 2. Run all cells up to and including the error cell
         */
        try {
            // Step 1: Restart the kernel
            await vscode.commands.executeCommand('jupyter.restartkernel');
            
            // Wait a bit for kernel to fully restart
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Step 2: Run all cells up to and including the error cell
            const editor = vscode.window.activeNotebookEditor;
            if (editor) {
                console.log(`🔄 Running cells 0 through ${errorCellIndex} after kernel restart`);
                
                // Execute cells from 0 to errorCellIndex (inclusive)
                for (let i = 0; i <= errorCellIndex && i < editor.notebook.cellCount; i++) {
                    const cell = editor.notebook.cellAt(i);
                    if (cell.kind === vscode.NotebookCellKind.Code) {
                        try {
                            console.log(`🔄 Executing cell ${i}`);
                            await vscode.commands.executeCommand('notebook.cell.execute', {
                                ranges: [{ start: i, end: i + 1 }],
                                document: editor.notebook.uri
                            });
                            
                            // Wait a bit between executions to avoid overwhelming
                            await new Promise(resolve => setTimeout(resolve, 500));
                        } catch (error: any) {
                            console.error(`Error executing cell ${i}:`, error);
                            // Continue with other cells even if one fails
                        }
                    }
                }
            } else {
                console.error('No active notebook editor for restart and run strategy');
                this.chatCore.addMessage('assistant', '❌ Failed to find notebook for restart and run', false, false);
            }
            
        } catch (error: any) {
            console.error('Error in restart and run to cell strategy:', error);
            this.chatCore.addMessage('assistant', `❌ Error during restart and run: ${error.message}`, false, false);
        }
    }
    
    public revealCell(cellIndex: number): void {
        /**
         * Reveals a specific cell in the notebook editor when auto-follow is enabled and autonomous mode is active.
         */
        if (!this._autoFollowEnabled || !this._autonomousMode) {
            return;
        }

        const editor = vscode.window.activeNotebookEditor;
        if (!editor) {
            return;
        }

        if (cellIndex < 0 || cellIndex >= editor.notebook.cellCount) {
            return;
        }

        try {
            const range = new vscode.NotebookRange(cellIndex, cellIndex + 1);
            editor.revealRange(range, vscode.NotebookEditorRevealType.InCenterIfOutsideViewport);
            editor.selection = range;
        } catch (error: any) {
            console.error(`📍 Failed to reveal cell ${cellIndex}:`, error.message);
        }
    }
}