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

    // Track whether code was actually executed in current iteration
    // Used to prevent learning explanation from showing after planning phase
    private _executionOccurredThisIteration: boolean = false;

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

    // Flag to prevent duplicate feedback messages in guided/learning mode
    private _checkpointShown: boolean = false;

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
     * States: LOOP_COMPLETE, LOOP_INCOMPLETE
     * @see ChatViewProvider._handleMessage() for state routing
     */
    setWorkflowState(state: string) {
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
            this.updateAutoModeButton(); // Update UI back to running state
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
         * Unified autonomous execution loop.
         *
         * Flow:
         * 1. Send message to Python (planning or execution)
         * 2. Python processes and returns LOOP_INCOMPLETE or LOOP_COMPLETE
         * 3. VSCode pauses in guided/learning mode after task completion
         * 4. User clicks Continue → next iteration starts
         *
         * @see handleAutonomousPlanning() in agent-provider.ts
         */
        try {
            this._autonomousMode = true;
            this._loopRunning = true;
            this.updateAutoModeButton();

            // Track notebook before user might click away during planning
            const editor = vscode.window.activeNotebookEditor;
            if (editor) {
                this.notebookOps.setTrackedNotebook(editor.notebook);
            } else {
                // Try to find any visible notebook editor
                const visibleEditors = vscode.window.visibleNotebookEditors;
                console.log(`[KAI] No active notebook editor at start. Visible editors: ${visibleEditors.length}`);
                if (visibleEditors.length > 0) {
                    this.notebookOps.setTrackedNotebook(visibleEditors[0].notebook);
                    console.log(`[KAI] Using first visible notebook editor as fallback`);
                } else {
                    console.warn('[KAI] WARNING: No notebook editor found at autonomous mode start!');
                }
            }

            // Reset task tracking for new session to ensure fresh bubble
            this.chatCore.resetTaskTracking();

            // Show activity indicator
            this.chatCore.showIndicator('thinking');

            let currentMessage = initialMessage;
            let context = await this.chatCore.getContextForMessage(initialMessage);
            context.autonomousMode = true;
            context.autonomousModeContinue = false;  // First iteration - triggers planning
            context.lastExecutionFailed = this._lastExecutionFailed;
            context.errorCellIndex = this._errorCellIndex;
            context.executionResult = this._lastExecutionOutput;
            context.lastCellModifiedInAutoMode = this._lastCellModifiedInAutoMode;

            let isFirstIteration = true;  // Track if this is the very first call

            // Main autonomous loop
            while (this._autonomousMode) {
                // Reset execution flag at start of each iteration
                // This ensures learning explanation only triggers after actual code execution
                this._executionOccurredThisIteration = false;

                // Handle feedback interrupt case (user typed during execution)
                if (this._currentWorkflowState === "LOOP_INCOMPLETE_FEEDBACK_INTERRUPT") {
                    this.agentProvider.resumeToolOutputProcessing();
                    currentMessage = this._pendingInterruptFeedback || "";
                    this._pendingInterruptFeedback = null;
                    context.autonomousModeContinue = false;
                } else if (!isFirstIteration) {
                    // Only set to true on subsequent iterations (after first planning call)
                    context.autonomousModeContinue = true;
                }
                // On first iteration, autonomousModeContinue stays false (set above)

                // Start waiting for workflow completion signal before sending message
                const workflowCompletionPromise = this.waitForWorkflowCompletion();

                // This function returns once workflow has returned.
                // Tool outputs come in through separate channel.
                await this.agentProvider.handleAutonomousIteration(currentMessage, context);

                // After first iteration completes, mark it so subsequent calls use autonomousModeContinue=true
                isFirstIteration = false;

                // Check if stop was requested during the agent call
                if (!this._autonomousMode) {
                    break;
                }

                // Wait for workflow completion and pending messages
                await workflowCompletionPromise;
                while (this.agentProvider.hasPendingMessages()) {
                    await new Promise(resolve => setTimeout(resolve, 100));
                }

                // Check if workflow completed
                if (this._currentWorkflowState === "LOOP_COMPLETE") {
                    break;
                }

                // Pause conditions for Tutorial (guided) mode:
                // - If code execution succeeded → show learning explanation and pause for continue
                // - If code execution failed → skip pause, continue to retry loop automatically
                // - If no code was executed (e.g., planning phase) → skip pause
                //
                // VSCode decides whether to pause based on:
                // 1. Learning mode is enabled (chatCore.learningMode)
                // 2. Code was actually executed this iteration (_executionOccurredThisIteration)
                // 3. Execution succeeded (!_lastExecutionFailed)
                const isLearningModeEnabled = this.chatCore.learningMode;
                const codeWasExecuted = this._executionOccurredThisIteration;
                const executionSucceeded = !this._lastExecutionFailed;

                // Pause and show learning explanation if:
                // - Learning mode is enabled AND
                // - Code was actually executed (not just planning) AND
                // - Execution succeeded
                const shouldPauseForLearning = isLearningModeEnabled && codeWasExecuted && executionSucceeded;

                if (isLearningModeEnabled && codeWasExecuted && !executionSucceeded) {
                    console.log('[KAI] Skipping learning pause - execution failed, continuing to retry');
                } else if (isLearningModeEnabled && !codeWasExecuted) {
                    console.log('[KAI] Skipping learning pause - no code executed this iteration (planning phase)');
                }

                if (shouldPauseForLearning) {
                    this.chatCore.taskJustCompleted = false;

                    // Request learning explanation from Python now that we know execution succeeded
                    // Keep indicator showing while waiting for the explanation
                    if (!this._checkpointShown) {
                        console.log('[KAI] Requesting learning explanation after successful execution');
                        // CRITICAL: Update context with the ACTUAL execution output before sending
                        // The context was built before execution, so executionResult would be stale
                        context.executionResult = this._lastExecutionOutput;
                        console.log('[KAI] Learning explanation executionResult length:', context.executionResult?.length || 0);
                        console.log('[KAI] Learning explanation executionResult preview:', context.executionResult?.substring(0, 200) || '(empty)');
                        // Request learning explanation from Python - this sends the explanation via display message
                        await this.agentProvider.requestLearningExplanation(context);
                        // Wait for any pending messages (the learning explanation)
                        while (this.agentProvider.hasPendingMessages()) {
                            await new Promise(resolve => setTimeout(resolve, 100));
                        }
                    }

                    // Hide activity indicator now that learning explanation is ready
                    this.chatCore.hideIndicator();

                    // Add checkpoint (continue button) after learning explanation
                    if (!this._checkpointShown) {
                        this.chatCore.addMessage('assistant', '', false, false, {isCheckpoint: true});
                        this._checkpointShown = true;
                    }

                    this._waitingForFeedback = true;
                    this.updateAutoModeButton();
                    currentMessage = await this.waitForUserFeedback();
                    this._checkpointShown = false;

                    // Show activity indicator when continuing
                    this.chatCore.showIndicator('thinking');

                    // Rebuild context with latest execution state before continuing
                    context = await this.chatCore.getContextForMessage(currentMessage);
                    context.autonomousMode = true;
                    context.lastExecutionFailed = this._lastExecutionFailed;
                    context.errorCellIndex = this._errorCellIndex;
                    context.executionResult = this._lastExecutionOutput;
                    context.lastCellModifiedInAutoMode = this._lastCellModifiedInAutoMode;
                    // autonomousModeContinue will be set at top of loop
                    continue; // Go back to loop start with feedback
                } else if (this.chatCore.taskJustCompleted) {
                    this.chatCore.taskJustCompleted = false;
                }

                // Continue to next iteration
                if (this._autonomousMode) {
                    currentMessage = "";
                    context = await this.chatCore.getContextForMessage(currentMessage);
                    context.autonomousMode = true;
                    context.lastExecutionFailed = this._lastExecutionFailed;
                    context.errorCellIndex = this._errorCellIndex;
                    context.executionResult = this._lastExecutionOutput;
                    context.lastCellModifiedInAutoMode = this._lastCellModifiedInAutoMode;
                }
            }

            // Complete autonomous execution
            this.terminateAutonomousExecution();

        } catch (error: any) {
            console.log('[KAI ERROR] Autonomous loop error:', error?.message || error);
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
        this._checkpointShown = false;
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

        // Clear tracked notebook reference
        this.notebookOps.setTrackedNotebook(undefined);

        // Hide activity indicator
        this.chatCore.hideIndicator();

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
        // Safety check: don't process if waiting for feedback
        if (this._waitingForFeedback) {
            console.log('🛑 Skipping code execution - waiting for user feedback');
            return;
        }

        try {
            // Switch indicator to show we're working in Jupyter
            this.chatCore.showIndicator('working in jupyter notebook');

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
            const shouldReplace = codeResponse.should_replace === true;
            const restartRequired = codeResponse.restart_required === true;  // For backtracking

            console.log('[KAI] Using cell position:', cellNumber, 'with should replace:', shouldReplace, ', recovery strategy:', recoveryStrategy, ', restart required:', restartRequired);

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
                    // For backtracking with restart: add cell first (don't execute yet),
                    // then restart and run all cells up to and including the new cell
                    if (restartRequired) {
                        // Add cell without executing
                        notebookCell = await this.notebookOps.addCode(code, cellNumber, false);
                        // Restart and run all cells up to the new cell
                        await this._handleRestartAndRunToCellStrategy(notebookCell.index);
                    } else {
                        notebookCell = await this.notebookOps.addCode(code, cellNumber, true);
                    }
                }
            }

            // Extract execution results from object
            if (cellType === "markdown") {
                // Markdown cells don't execute, so always mark as success
                console.log('Markdown cell added:', cellNumber);
                this._errorCellIndex = -1;
                this._lastExecutionFailed = false;
                this._lastExecutionOutput = ""; // Markdown cells have no output (content is in last_executed_cell)
                // Markdown cells (reasoning steps) should also trigger learning explanations
                this._executionOccurredThisIteration = true;
            } else {
                // Code cell was executed - mark this so learning explanation can trigger
                this._executionOccurredThisIteration = true;

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

            // Switch indicator back to thinking (Python will continue processing)
            // But not if we're waiting for user feedback
            if (!this._waitingForFeedback) {
                this.chatCore.showIndicator('thinking');
            }
        } catch (error) {
            console.error('Error in autonomous code execution:', error);
            // On error, show thinking indicator only if not waiting for feedback
            if (!this._waitingForFeedback) {
                this.chatCore.showIndicator('thinking');
            }
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
            // Use tracked notebook editor (not activeNotebookEditor) to handle cases where
            // another tab is focused but the notebook is still visible
            const editor = this.notebookOps.getNotebookEditor();
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
                console.error('No notebook editor found for restart and run strategy');
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

        // Use tracked notebook editor (not activeNotebookEditor) to handle cases where
        // another tab is focused but the notebook is still visible
        const editor = this.notebookOps.getNotebookEditor();
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