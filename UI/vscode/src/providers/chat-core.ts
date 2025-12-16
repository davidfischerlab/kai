import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

/**
 * ChatCore - Core chat message management and LLM context preparation
 * 
 * Primary Responsibilities:
 * - Message storage and conversation history management
 * - Context preparation for LLM interactions (notebook, execution, modification data)
 * - Task list formatting and display message handling
 * - Execution history token-based trimming for LLM efficiency
 * - UI state coordination (streaming, toggles)
 * 
 * Key Features:
 * - Manages conversation flow and message chronology
 * - Formats complex data structures for LLM consumption
 * - Provides smart context trimming based on token limits
 * - Handles autonomous mode state checking for UI updates
 * 
 * Architecture Position:
 * - Core component for all chat-related data processing
 * - Delegates storage to NotebookOperations, focuses on formatting
 * - Provides clean interface between raw data and LLM requirements
 */
export class ChatCore {
    private _messages: Array<{role: string, content: string, timestamp: string, isIndicator?: boolean, isButtonAction?: boolean, intent?: string, toolUsage?: Array<{name: string, query?: string, status: string, collections?: string[]}>, buttonsDisabled?: boolean, buttonsUsed?: boolean, metadata?: any }> = [];
    private taskMessageIndex: number = -1;
    private critiqueMessageIndex: number = -1;
    private agentNotificationMessageIndex: number = -1;
    
    // Context preparation limits for LLM efficiency
    private readonly MAX_CONTEXT_TOKENS = 8000; // Token limit for both execution and modification history in LLM context
    
    // Mode toggles  
    private _ragEnabled: boolean = true;
    private _turboEnabled: boolean = true;
    
    // Reference to notebook operations (set after construction)
    private notebookOps: any;

    // Reference workflow IDs storage (for current autonomous session)
    private _storedReferenceWorkflows: string | null = null;
    private _storedCritique: string | null = null;
    private _storedAgentNotification: string | null = null;
    
    constructor(
        private updateWebview: () => void,
        private _storeToolUsage: (tool: {name: string, query?: string, status: string, collections?: string[]}) => void, // Used in constructor binding
        private getAutonomousExecutionStatus?: () => boolean
    ) {}

    /**
     * Sets the NotebookOperations instance for accessing notebook history data.
     * Must be called after construction due to circular dependency resolution.
     * 
     * @param notebookOps - NotebookOperations instance for execution/modification history
     * @interaction Called by ChatViewProvider during initialization
     */
    setNotebookOperations(notebookOps: any) {
        this.notebookOps = notebookOps;
    }

    /**
     * Gets the complete conversation history including all messages.
     * 
     * @returns Array of all messages with metadata
     * @interaction Read by ChatViewProvider for webview updates
     */
    get messages() { return this._messages; }
    
    /**
     * Gets the current RAG (Retrieval Augmented Generation) enabled state.
     * @returns Boolean indicating if RAG is enabled
     * @interaction Used by agent context preparation
     */
    get ragEnabled() { return this._ragEnabled; }
    /**
     * Sets the RAG enabled state.
     * @param value - New RAG enabled state
     * @interaction Modified by UI toggles in ChatViewProvider
     */
    set ragEnabled(value: boolean) { this._ragEnabled = value; }
    
    /**
     * Gets the current Turbo mode (faster LLM) enabled state.
     * @returns Boolean indicating if Turbo mode is enabled
     * @interaction Used by agent to switch between LLM models
     */
    get turboEnabled() { return this._turboEnabled; }
    /**
     * Sets the Turbo mode enabled state.
     * @param value - New Turbo mode state
     * @interaction Modified by UI toggles and autonomous mode
     */
    set turboEnabled(value: boolean) { this._turboEnabled = value; }

    // MESSAGE HISTORY 

    /**
     * Adds a new message to the conversation history and triggers UI update.
     * Handles different message types including thinking indicators and button actions.
     * 
     * @param role - Message role ('user' or 'assistant')
     * @param content - Message content text
     * @param isThinking - Whether this is a temporary thinking indicator
     * @param isButtonAction - Whether this message is from a button click
     * @param metadata - Additional message metadata (e.g., task list flag)
     * @interaction Called by ChatViewProvider, AutonomousExecution, and agent responses
     */
    public addMessage(role: string, content: string, isThinking: boolean = false, isButtonAction: boolean = false, metadata?: any) {
        const now = new Date();
        this._messages.push({
            role,
            content,
            timestamp: now.toLocaleTimeString('en-US', { hour12: false }),
            isButtonAction,
            buttonsUsed: false,
            isIndicator: isThinking,
            metadata: metadata
        });
        this.updateWebview();
    }

    /**
     * Clears all messages and resets task tracking state.
     * Used when starting fresh conversations.
     * 
     * @interaction Called by ChatViewProvider on clear button click
     */
    public clearMessages(): void {
        this._messages.length = 0;
        this.taskMessageIndex = -1;
    }

    /**
     * Marks buttons in a specific message as used to prevent duplicate clicks.
     * 
     * @param messageIndex - Index of message containing buttons
     * @returns True if buttons were marked, false if message not found
     * @interaction Called when user clicks action buttons in messages
     */
    public markButtonsUsed(messageIndex: number): boolean {
        if (messageIndex >= 0 && messageIndex < this._messages.length) {
            this._messages[messageIndex].buttonsUsed = true;
            return true;
        }
        return false;
    }

    /**
     * Removes the thinking indicator message if it's the last message.
     * Used to clean up temporary status messages.
     * 
     * @returns True if indicator was removed, false if not found
     * @interaction Called when agent response completes or errors
     */
    public removeThinkingIndicator(): boolean {
        // Remove the last message if it's a thinking indicator
        const lastIndex = this._messages.length - 1;
        if (lastIndex >= 0) {
            const message = this._messages[lastIndex];
            if (message.isIndicator) {
                this._messages.splice(lastIndex, 1);
                return true;
            }
        }
        return false;
    }

    /**
     * Resets task list tracking for autonomous mode.
     * Clears the tracked task list message index and stored reference workflows.
     *
     * @interaction Called when autonomous mode ends or resets
     */
    public resetTaskTracking(): void {
        this.taskMessageIndex = -1;
        this.critiqueMessageIndex = -1;
        this.agentNotificationMessageIndex = -1;
        this._storedReferenceWorkflows = null;
        this._storedCritique = null;
        this._storedAgentNotification = null;
    }

    /**
     * Removes any existing critique message from the chat.
     * Called when a new task list arrives or when critique is replaced.
     */
    private _removeCritiqueMessage(): void {
        if (this.critiqueMessageIndex >= 0 && this.critiqueMessageIndex < this._messages.length) {
            this._messages.splice(this.critiqueMessageIndex, 1);

            // Adjust task message index if it was after the removed critique
            if (this.taskMessageIndex > this.critiqueMessageIndex) {
                this.taskMessageIndex--;
            }

            // Adjust agent notification index if it was after the removed critique
            if (this.agentNotificationMessageIndex > this.critiqueMessageIndex) {
                this.agentNotificationMessageIndex--;
            }

            this.critiqueMessageIndex = -1;
        }
    }

    /**
     * Removes any existing agent notification message from the chat.
     * Called when a new task list arrives or when agent notification is replaced.
     */
    private _removeAgentNotificationMessage(): void {
        if (this.agentNotificationMessageIndex >= 0 && this.agentNotificationMessageIndex < this._messages.length) {
            this._messages.splice(this.agentNotificationMessageIndex, 1);

            // Adjust task message index if it was after the removed agent notification
            if (this.taskMessageIndex > this.agentNotificationMessageIndex) {
                this.taskMessageIndex--;
            }

            // Adjust critique index if it was after the removed agent notification
            if (this.critiqueMessageIndex > this.agentNotificationMessageIndex) {
                this.critiqueMessageIndex--;
            }

            this.agentNotificationMessageIndex = -1;
        }
    }

    /**
     * Stores reference workflow IDs from ReferenceWorkflowSelectionTool output.
     * These IDs will be displayed above the Analysis Plan in subsequent task lists.
     *
     * @param data - Reference workflow data from tool output
     * @interaction Called by ChatViewProvider when reference_workflows message type is received
     */
    public storeReferenceWorkflows(data: any): void {
        if (data && data.text) {
            this._storedReferenceWorkflows = data.text;
        }
    }

    /**
     * Inserts a user feedback message at a specific position in conversation history.
     * Ensures feedback appears in correct chronological order after the original message.
     * Updates any tracked indices that might be affected by the insertion.
     * 
     * @param message - Feedback message content
     * @param afterIndex - Index to insert after (typically the original user message)
     * @interaction Called by AutonomousExecution when user provides feedback
     */
    public insertFeedbackMessage(message: string, afterIndex: number): void {
        const feedbackMessage = {
            role: 'user',
            content: message,
            timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
            isButtonAction: false,
            buttonsUsed: false,
            isIndicator: false,
            metadata: { feedbackMessage: true }
        };

        // Insert after the specified index (original user message)
        this._messages.splice(afterIndex + 1, 0, feedbackMessage);
        
        // Update all subsequent message indices that might be tracked
        if (this.taskMessageIndex > afterIndex) {
            this.taskMessageIndex++;
        }

        this.updateWebview();
    }

    // CONTEXT TO PASS TO PYTHON

    /**
     * Prepares complete context for LLM message processing.
     * Aggregates notebook state, execution history, and conversation context.
     * Intelligently selects relevant context based on message content.
     * 
     * @param message - The message being processed
     * @returns Complete context object for agent processing
     * @interaction Called by ChatViewProvider and AutonomousExecution before sending to agent
     */
    public async getContextForMessage(message: string): Promise<any> {
        const context: any = {};

        // Get notebook editor - use tracked notebook to handle cases where another tab is focused
        const editor = this.notebookOps?.getNotebookEditor() || vscode.window.activeNotebookEditor;
        if (editor) {
            const selection = editor.selections[0];
            if (selection) {
                const cell = editor.notebook.cellAt(selection.start);
                context.currentCell = cell.document.getText();
                context.currentCellIndex = cell.index;
            }

            context.notebookPath = editor.notebook.uri.fsPath;
            context.notebookUri = editor.notebook.uri.toString();
            context.totalCells = editor.notebook.cellCount;
        }
        
        context.ragEnabled = this._ragEnabled;
        context.turboEnabled = this._turboEnabled;
        
        const config = vscode.workspace.getConfiguration('kai_agent');
        const apiKey = config.get('ollamaApiKey', '');
        if (apiKey) {
            context.ollamaApiKey = apiKey;
        }
        
        // Delegate to NotebookOperations for structured context
        context.conversationHistory = this._getConversationHistoryContext();
        context.executionHistory = this.notebookOps.executionHistory;
        context.modificationHistory = this.notebookOps.modificationHistory;
        context.notebookStructure = this.notebookOps.getNotebookStructure();

        return context;
    }

    private _getConversationHistoryContext(): Array<{role: string, content: string, timestamp: string, metadata?: any}> {
        const estimateTokens = (text: string) => Math.ceil(text.length / 4); // ~4 chars per token
        
        // Get recent conversation messages, working backwards until we hit token limit
        const allMessages = this._messages.filter(msg => !msg.isIndicator && msg.content.trim().length > 0);
        let totalTokens = 0;
        let selectedMessages: Array<{role: string, content: string, timestamp: string, metadata?: any}> = [];
        
        // Start from most recent and work backwards
        for (let i = allMessages.length - 1; i >= 0; i--) {
            const msg = allMessages[i];
            const entryTokens = estimateTokens(msg.content);
            
            if (totalTokens + entryTokens > this.MAX_CONTEXT_TOKENS && selectedMessages.length > 0) {
                break; // Stop if adding this message would exceed token limit
            }
            
            selectedMessages.unshift(msg); // Add to beginning to maintain chronological order
            totalTokens += entryTokens;
        }
        
        return selectedMessages.map(msg => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp,
            metadata: msg.metadata
        }));
    }

    // DISPLAY

    /**
     * Handles task lists and status update display messages from the agent.
     * Replaces thinking indicators with actual content when appropriate.
     *
     * @param displayResponse - Display message from agent with text and intent
     * @interaction Called by AgentProvider when agent sends display messages
     */
    public async handleTaskListMessage(displayResponse: any): Promise<void> {
        try {
            // Skip autonomous-related display messages if autonomous mode is no longer active
            if (this.getAutonomousExecutionStatus && !this.getAutonomousExecutionStatus()) {
                return;
            }

            // Check if this is a critique-only message
            if (displayResponse.critique !== undefined && !displayResponse.text) {
                // This is a critique message - add it as a separate bubble
                if (displayResponse.critique && displayResponse.critique.trim()) {
                    // Remove any existing critique message
                    this._removeCritiqueMessage();

                    // Add new critique message right after the task list
                    const critiqueText = `**Reviewer agent:**\n\n${displayResponse.critique.trim()}`;
                    this.addMessage('assistant', critiqueText, false, false, {isCritique: true});
                    this.critiqueMessageIndex = this._messages.length - 1;
                }
                this.updateWebview();
                return;
            }

            // Check if this is an agent notification-only message
            if (displayResponse.agent_notification !== undefined && !displayResponse.text && !displayResponse.critique) {
                // This is an agent notification message - add it as a separate bubble
                if (displayResponse.agent_notification &&
                    typeof displayResponse.agent_notification === 'string' &&
                    displayResponse.agent_notification.trim()) {
                    // Remove any existing agent notification message
                    this._removeAgentNotificationMessage();

                    // Add new agent notification message right after the task list
                    const agentNotificationText = `**Analyst agent:**\n\n${displayResponse.agent_notification.trim()}`;
                    this.addMessage('assistant', agentNotificationText, false, false, {isAgentNotification: true});
                    this.agentNotificationMessageIndex = this._messages.length - 1;
                }
                this.updateWebview();
                return;
            }

            // Check if this is a combined critique + agent notification message (no task list)
            if (!displayResponse.text && (displayResponse.critique || displayResponse.agent_notification)) {
                // Handle critique if present
                if (displayResponse.critique && displayResponse.critique.trim()) {
                    this._removeCritiqueMessage();
                    const critiqueText = `**Reviewer agent:**\n\n${displayResponse.critique.trim()}`;
                    this.addMessage('assistant', critiqueText, false, false, {isCritique: true});
                    this.critiqueMessageIndex = this._messages.length - 1;
                }

                // Handle agent notification if present
                if (displayResponse.agent_notification &&
                    typeof displayResponse.agent_notification === 'string' &&
                    displayResponse.agent_notification.trim()) {
                    this._removeAgentNotificationMessage();
                    const agentNotificationText = `**Analyst agent:**\n\n${displayResponse.agent_notification.trim()}`;
                    this.addMessage('assistant', agentNotificationText, false, false, {isAgentNotification: true});
                    this.agentNotificationMessageIndex = this._messages.length - 1;
                }

                this.updateWebview();
                return;
            }

            // Regular task list message handling
            let messageText = displayResponse.text;

            // Format the task list from JSON to readable format
            messageText = this._formatTaskList(messageText);

            // Remove any existing critique and agent notification messages when a new task list arrives
            this._removeCritiqueMessage();
            this._removeAgentNotificationMessage();

            // Update existing task message if it exists, replace thinking message, or create new one
            if (this.taskMessageIndex >= 0 && this.taskMessageIndex < this._messages.length) {
                // Update existing task message in place
                this._messages[this.taskMessageIndex].content = messageText;
                this._messages[this.taskMessageIndex].timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
            } else {
                // Find and replace any thinking indicator with the task list
                const thinkingIndex = this._messages.findIndex(m => m.isIndicator);
                if (thinkingIndex >= 0) {
                    this._messages[thinkingIndex].content = messageText;
                    this._messages[thinkingIndex].isIndicator = false;
                    this._messages[thinkingIndex].metadata = {isTaskList: true};
                    this._messages[thinkingIndex].timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
                    this.taskMessageIndex = thinkingIndex;
                } else {
                    // Create new task message and track its index
                    this.addMessage('assistant', messageText, false, false, {isTaskList: true});
                    this.taskMessageIndex = this._messages.length - 1;
                }
            }

            // Handle agent notification if provided - display as green box below task list
            if (displayResponse.agent_notification &&
                typeof displayResponse.agent_notification === 'string' &&
                displayResponse.agent_notification.trim()) {
                // Store the agent notification
                this._storedAgentNotification = displayResponse.agent_notification.trim();

                // Add agent notification message right after the task list
                const agentNotificationText = `**Analyst agent:**\n\n${this._storedAgentNotification}`;
                this.addMessage('assistant', agentNotificationText, false, false, {isAgentNotification: true});
                this.agentNotificationMessageIndex = this._messages.length - 1;
            }

            // Handle critique if provided with task list - display as red box below agent notification
            if (displayResponse.critique && displayResponse.critique.trim()) {
                // Store the critique
                this._storedCritique = displayResponse.critique.trim();

                // Add critique message after task list and agent notification
                const critiqueText = `**Reviewer agent:**\n\n${this._storedCritique}`;
                this.addMessage('assistant', critiqueText, false, false, {isCritique: true});
                this.critiqueMessageIndex = this._messages.length - 1;
            }

            this.updateWebview();
        } catch (error) {
            console.error('Error in task list display message handling:', error);
        }
    }

    /**
     * Handles display messages from the agent, excluding task lists and status updates.
     * 
     * @param displayResponse - Display message from agent with text and intent
     * @interaction Called by AgentProvider when agent sends display messages
     */
    public async handleDisplayMessage(displayResponse: any): Promise<void> {
        try {
            let messageText = displayResponse.text;
            
            // Skip error messages from tool failures during autonomous mode
            if (messageText.trim().length > 0) {
                // Only add non-empty messages
                this.addMessage('assistant', messageText);
            }
        } catch (error) {
            console.error('Error in display message handling:', error);
        }
    }

    /**
     * Formats JSON task list into readable markdown format.
     * Converts task status to visual indicators (✅, 🏃, ⏳).
     * Handles malformed JSON gracefully.
     *
     * @param text - Raw text potentially containing JSON task list
     * @returns Formatted markdown task list or original text if parsing fails
     * @interaction Called by handleTaskListMessage for task list display
     */
    private _formatTaskList(text: string): string {
        try {
            const jsonMatch = text.match(/\{[\s\S]*\}/);
            if (!jsonMatch) return text;

            const jsonText = jsonMatch[0];
            let braceCount = 0;
            for (const char of jsonText) {
                if (char === '{') braceCount++;
                if (char === '}') braceCount--;
            }
            if (braceCount !== 0) return text;

            const taskData = JSON.parse(jsonText);
            if (!taskData.tasks) return text;

            let formatted = '';

            // Add reference workflow IDs if available (from stored data or task data)
            const referenceWorkflows = this._storedReferenceWorkflows || taskData.reference_workflow_ids;
            if (referenceWorkflows) {
                formatted += `**Retrieved reference workflows**  \n`;
                formatted += `${referenceWorkflows}`;
                formatted += `\n\n`;
            }

            formatted += `**Chain-of-thought:**\n`;

            taskData.tasks.forEach((task: any) => {
                // Check if task is a reasoning task
                const isReasoning = task.task.toLowerCase().includes('[reasoning]');

                // Different icons for reasoning vs code tasks
                let status: string;
                if (isReasoning) {
                    status = task.status === 'completed' ? '✅' :
                            task.status === 'active' ? '🧠' : '💭';
                } else {
                    status = task.status === 'completed' ? '✅' :
                            task.status === 'active' ? '🏃' : '⏳';
                }

                // Option: Add subtle visual distinction for reasoning tasks
                const taskText = isReasoning ? `*${task.task}*` : task.task;
                formatted += `${status} ${task.id}. ${taskText}\n`;
            });

            return formatted;
        } catch (e) {
            console.error('Failed to parse task list:', e);
            return text;
        }
    }

    /**
     * Loads and returns HTML template for webview chat interface.
     * Reads from template file with fallback for errors.
     * 
     * @param _webview - VSCode webview instance (unused but required by interface)
     * @returns HTML content for webview
     * @interaction Called by ChatViewProvider when creating webview
     */
    public getHtmlForWebview(_webview: vscode.Webview): string {
        try {
            // Get the path to the template file
            const templatePath = path.join(__dirname, '..', 'templates', 'chat-template.html');
            
            // Read the template file
            const templateContent = fs.readFileSync(templatePath, 'utf8');
            
            return templateContent;
        } catch (error) {
            console.error('Error loading chat template:', error);
            // Fallback to a simple template if file loading fails
            return `<!DOCTYPE html>
<html>
<head>
    <title>KAI: CHAT INTERFACE - Error</title>
</head>
<body>
    <p>Error loading chat template. Please check the installation.</p>
</body>
</html>`;
        }
    }

}