import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Message types for categorizing chat messages.
 * - user/assistant: Regular conversation messages
 * - task_list: Chain-of-thought display (singleton)
 * - agent_notif: Analyst agent notification (singleton)
 * - critique: Reviewer agent feedback (singleton)
 * - learning: Learning explanations (accumulating)
 * - checkpoint: Continue button (singleton)
 * - indicator: Activity spinner (singleton, always last)
 */
type MessageType = 'user' | 'assistant' | 'task_list' | 'agent_notif' | 'critique' | 'learning' | 'checkpoint' | 'indicator';

/**
 * Structured chat message with explicit type and ID.
 */
interface ChatMessage {
    id: string;
    type: MessageType;
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
    metadata?: Record<string, any>;
    isButtonAction?: boolean;
    buttonsUsed?: boolean;
    suggestions?: Array<{icon: string; label: string; action: string}>;
    intent?: string;
    toolUsage?: Array<{name: string; query?: string; status: string; collections?: string[]}>;
    // Legacy compatibility fields
    isIndicator?: boolean;
    indicatorText?: string;
}

/**
 * ChatCore - Core chat message management and LLM context preparation
 *
 * Architecture:
 * - Separate storage for different message types (conversation, singletons, accumulating)
 * - Computed `messages` getter assembles display order with guaranteed indicator-last ordering
 * - Type-routed message handling eliminates fragile index tracking
 *
 * Display Order (guaranteed):
 * 1. Conversation messages (user/assistant exchanges)
 * 2. Task list (if exists)
 * 3. Agent notification (if exists)
 * 4. Critique (if exists)
 * 5. Learning messages (if any)
 * 6. Checkpoint (if exists)
 * 7. Indicator (if visible) - ALWAYS LAST
 */
export class ChatCore {
    // ========== MESSAGE STORAGE ==========

    // Conversation messages (user/assistant exchanges)
    private _conversation: ChatMessage[] = [];

    // Singleton messages (only one of each can exist)
    private _taskList: ChatMessage | null = null;
    private _agentNotification: ChatMessage | null = null;
    private _critique: ChatMessage | null = null;
    private _checkpoint: ChatMessage | null = null;

    // Accumulating messages
    private _learningMessages: ChatMessage[] = [];

    // Indicator state (not stored as message, just state)
    private _indicator: { visible: boolean; text: string } = { visible: false, text: '' };

    // ========== CONFIGURATION ==========

    private readonly MAX_CONTEXT_TOKENS = 8000;

    // Mode toggles
    private _ragEnabled: boolean = true;
    private _turboEnabled: boolean = true;
    private _interactionMode: 'chat' | 'guided' | 'autonomous' = 'autonomous';
    private _learningMode: boolean = false;

    // Task completion tracking for guided mode checkpoints
    private _lastTaskStates: Record<string, string> = {};
    private _taskJustCompleted: boolean = false;

    // Reference to notebook operations (set after construction)
    private notebookOps: any;

    // Reference workflow storage
    private _storedReferenceWorkflows: string | null = null;
    private _storedReferenceNotebooks: Array<{
        id: string;
        title: string;
        source_path: string;
        percentage: number;
        cells: Array<{index: number; type: string; preview: string}>;
    }> | null = null;

    // Callback for sending targeted messages to webview
    private _sendToWebview: ((msg: any) => void) | null = null;

    constructor(
        private updateWebview: () => void,
        private _storeToolUsage: (tool: {name: string, query?: string, status: string, collections?: string[]}) => void,
        private getAutonomousExecutionStatus?: () => boolean
    ) {}

    // ========== COMPUTED MESSAGES GETTER ==========

    /**
     * Returns all messages in guaranteed display order.
     * Indicator is ALWAYS last when visible.
     */
    get messages(): ChatMessage[] {
        const result: ChatMessage[] = [...this._conversation];

        if (this._taskList) result.push(this._taskList);
        if (this._agentNotification) result.push(this._agentNotification);
        if (this._critique) result.push(this._critique);
        result.push(...this._learningMessages);
        if (this._checkpoint) result.push(this._checkpoint);

        // Indicator ALWAYS last
        if (this._indicator.visible) {
            result.push({
                id: 'indicator',
                type: 'indicator',
                role: 'assistant',
                content: '',
                timestamp: '',
                metadata: { isIndicator: true, indicatorText: this._indicator.text },
                isIndicator: true,
                indicatorText: this._indicator.text
            });
        }

        return result;
    }

    // ========== PROPERTY GETTERS/SETTERS ==========

    get ragEnabled() { return this._ragEnabled; }
    set ragEnabled(value: boolean) { this._ragEnabled = value; }

    get turboEnabled() { return this._turboEnabled; }
    set turboEnabled(value: boolean) { this._turboEnabled = value; }

    get interactionMode() { return this._interactionMode; }
    set interactionMode(value: 'chat' | 'guided' | 'autonomous') { this._interactionMode = value; }

    get learningMode() { return this._learningMode; }
    set learningMode(value: boolean) { this._learningMode = value; }

    get taskJustCompleted() { return this._taskJustCompleted; }
    set taskJustCompleted(value: boolean) { this._taskJustCompleted = value; }

    get storedReferenceNotebooks() { return this._storedReferenceNotebooks; }

    // ========== INITIALIZATION ==========

    setNotebookOperations(notebookOps: any) {
        this.notebookOps = notebookOps;
    }

    public setSendToWebview(callback: (msg: any) => void): void {
        this._sendToWebview = callback;
    }

    // ========== MESSAGE MANAGEMENT ==========

    /**
     * Adds a message, routing to appropriate storage based on type.
     * This maintains backwards compatibility with existing callers.
     */
    public addMessage(
        role: string,
        content: string,
        isThinking: boolean = false,
        isButtonAction: boolean = false,
        metadata?: any,
        suggestions?: Array<{icon: string, label: string, action: string}>,
        indicatorText?: string
    ) {
        // Indicator: delegate to showIndicator
        if (isThinking) {
            this.showIndicator(indicatorText || 'thinking');
            return;
        }

        // Checkpoint: singleton
        if (metadata?.isCheckpoint) {
            this._checkpoint = this._createMessage('checkpoint', role, content, metadata);
            this._notify();
            return;
        }

        // Learning explanation: accumulate
        if (metadata?.isLearningExplanation) {
            const msg = this._createMessage('learning', role, content, metadata);
            this._learningMessages.push(msg);
            this._notify();
            return;
        }

        // Critique: singleton
        if (metadata?.isCritique) {
            this._critique = this._createMessage('critique', role, content, metadata);
            this._notify();
            return;
        }

        // Agent notification: singleton
        if (metadata?.isAgentNotification) {
            this._agentNotification = this._createMessage('agent_notif', role, content, metadata);
            this._notify();
            return;
        }

        // Task list: singleton (usually handled by handleTaskListMessage)
        if (metadata?.isTaskList) {
            this._taskList = this._createMessage('task_list', role, content, metadata);
            this._notify();
            return;
        }

        // Default: conversation message
        const msgType: MessageType = role === 'user' ? 'user' : 'assistant';
        const msg = this._createMessage(msgType, role, content, metadata);
        msg.isButtonAction = isButtonAction;
        msg.suggestions = suggestions;
        this._conversation.push(msg);
        this._notify();
    }

    /**
     * Clears all messages and resets state.
     */
    public clearMessages(): void {
        this._conversation = [];
        this._taskList = null;
        this._agentNotification = null;
        this._critique = null;
        this._checkpoint = null;
        this._learningMessages = [];
        this._indicator = { visible: false, text: '' };
        this._storedReferenceWorkflows = null;
        this._storedReferenceNotebooks = null;
        this._lastTaskStates = {};
        this._taskJustCompleted = false;
    }

    /**
     * Marks buttons in a message as used.
     */
    public markButtonsUsed(messageIndex: number): boolean {
        const allMessages = this.messages;
        if (messageIndex >= 0 && messageIndex < allMessages.length) {
            const target = allMessages[messageIndex];
            // Find in conversation array and mark
            const convMsg = this._conversation.find(m => m.id === target.id);
            if (convMsg) {
                convMsg.buttonsUsed = true;
                return true;
            }
        }
        return false;
    }

    // ========== INDICATOR MANAGEMENT ==========

    /**
     * Shows the activity indicator with specified text.
     */
    public showIndicator(text: string): void {
        this._indicator = { visible: true, text };
        this._notify();
    }

    /**
     * Hides the activity indicator.
     */
    public hideIndicator(): void {
        this._indicator = { visible: false, text: '' };
        this._notify();
    }

    /**
     * Updates indicator text without full re-render.
     */
    public updateIndicatorText(text: string, sendToWebview: (msg: any) => void): void {
        this._indicator.text = text;
        sendToWebview({ type: 'updateIndicator', text: text });
    }

    /**
     * Legacy method - now just hides indicator.
     */
    public removeThinkingIndicator(): boolean {
        if (this._indicator.visible) {
            this._indicator = { visible: false, text: '' };
            return true;
        }
        return false;
    }

    // ========== TRACKING RESET ==========

    /**
     * Resets task-related tracking for new autonomous session.
     */
    public resetTaskTracking(): void {
        this._taskList = null;
        this._agentNotification = null;
        this._critique = null;
        this._learningMessages = [];
        this._storedReferenceWorkflows = null;
        this._storedReferenceNotebooks = null;
        this._lastTaskStates = {};
        this._taskJustCompleted = false;
        // Keep conversation and indicator state
    }

    /**
     * Removes checkpoint message.
     */
    public removeCheckpointMessages(): void {
        this._checkpoint = null;
        this._notify();
    }

    // ========== REFERENCE WORKFLOWS ==========

    public storeReferenceWorkflows(data: any): void {
        if (data && data.text) {
            this._storedReferenceWorkflows = data.text;
        }
        if (data && data.notebooks) {
            this._storedReferenceNotebooks = data.notebooks;
        }
    }

    // ========== FEEDBACK INSERTION ==========

    /**
     * Inserts a user feedback message at a specific position.
     */
    public insertFeedbackMessage(message: string, afterIndex: number): void {
        const feedbackMessage = this._createMessage('user', 'user', message, { feedbackMessage: true });

        // Find the conversation message at the given index and insert after it
        const allMessages = this.messages;
        if (afterIndex >= 0 && afterIndex < allMessages.length) {
            const targetId = allMessages[afterIndex].id;
            const convIndex = this._conversation.findIndex(m => m.id === targetId);
            if (convIndex >= 0) {
                this._conversation.splice(convIndex + 1, 0, feedbackMessage);
            } else {
                // Target wasn't in conversation, just append
                this._conversation.push(feedbackMessage);
            }
        } else {
            this._conversation.push(feedbackMessage);
        }

        this._notify();
    }

    // ========== DISPLAY MESSAGE HANDLERS ==========

    /**
     * Handles task list display messages from the agent.
     */
    public async handleTaskListMessage(displayResponse: any): Promise<void> {
        try {
            const autonomousActive = this.getAutonomousExecutionStatus && this.getAutonomousExecutionStatus();
            const hasTaskListText = displayResponse.text !== undefined;

            if (!autonomousActive && !hasTaskListText) {
                return;
            }

            // Handle critique-only messages
            if (displayResponse.critique !== undefined && !displayResponse.text) {
                if (displayResponse.critique && displayResponse.critique.trim()) {
                    const critiqueText = `**Reviewer agent:**\n\n${displayResponse.critique.trim()}`;
                    this._critique = this._createMessage('critique', 'assistant', critiqueText, { isCritique: true });
                }
                this._notify();
                return;
            }

            // Handle agent notification-only messages
            if (displayResponse.agent_notification !== undefined && !displayResponse.text && !displayResponse.critique) {
                if (displayResponse.agent_notification && typeof displayResponse.agent_notification === 'string' && displayResponse.agent_notification.trim()) {
                    const notifText = `**Analyst agent:**\n\n${displayResponse.agent_notification.trim()}`;
                    this._agentNotification = this._createMessage('agent_notif', 'assistant', notifText, { isAgentNotification: true });
                }
                this._notify();
                return;
            }

            // Handle combined critique + agent notification (no task list)
            if (!displayResponse.text && (displayResponse.critique || displayResponse.agent_notification)) {
                if (displayResponse.critique && displayResponse.critique.trim()) {
                    const critiqueText = `**Reviewer agent:**\n\n${displayResponse.critique.trim()}`;
                    this._critique = this._createMessage('critique', 'assistant', critiqueText, { isCritique: true });
                }
                if (displayResponse.agent_notification && typeof displayResponse.agent_notification === 'string' && displayResponse.agent_notification.trim()) {
                    const notifText = `**Analyst agent:**\n\n${displayResponse.agent_notification.trim()}`;
                    this._agentNotification = this._createMessage('agent_notif', 'assistant', notifText, { isAgentNotification: true });
                }
                this._notify();
                return;
            }

            // Regular task list message
            const rawText = displayResponse.text;
            const { text: messageText, tasks: structuredTasks } = this._formatTaskList(rawText);

            const taskMetadata: any = { isTaskList: true };
            if (structuredTasks) {
                taskMetadata.tasks = structuredTasks;
            }
            if (this._storedReferenceNotebooks && this._storedReferenceNotebooks.length > 0) {
                taskMetadata.referenceNotebooks = this._storedReferenceNotebooks;
            }

            // Update task list singleton - no index tracking needed
            this._taskList = this._createMessage('task_list', 'assistant', messageText, taskMetadata);

            // Clear previous iteration's agent notification and critique
            // These should only display for one iteration
            this._agentNotification = null;
            this._critique = null;

            // Handle agent notification if present with task list (for this iteration only)
            if (displayResponse.agent_notification && typeof displayResponse.agent_notification === 'string' && displayResponse.agent_notification.trim()) {
                const notifText = `**Analyst agent:**\n\n${displayResponse.agent_notification.trim()}`;
                this._agentNotification = this._createMessage('agent_notif', 'assistant', notifText, { isAgentNotification: true });
            }

            // Handle critique if present with task list (for this iteration only)
            if (displayResponse.critique && displayResponse.critique.trim()) {
                const critiqueText = `**Reviewer agent:**\n\n${displayResponse.critique.trim()}`;
                this._critique = this._createMessage('critique', 'assistant', critiqueText, { isCritique: true });
            }

            // Send targeted task list update if possible
            if (this._sendToWebview && structuredTasks) {
                const taskIndex = this.messages.findIndex(m => m.type === 'task_list');
                if (taskIndex >= 0) {
                    this._sendToWebview({
                        type: 'updateTaskCards',
                        taskIndex: taskIndex,
                        tasks: structuredTasks
                    });
                }
            }

            this._notify();
        } catch (error) {
            console.error('Error in task list display message handling:', error);
        }
    }

    /**
     * Handles non-task-list display messages from the agent.
     */
    public async handleDisplayMessage(displayResponse: any): Promise<void> {
        try {
            let messageText = displayResponse.text;

            if (messageText && messageText.trim().length > 0) {
                const suggestions = displayResponse.suggestions || undefined;

                let metadata: any = undefined;
                if (displayResponse.isLearningExplanation) {
                    metadata = { isLearningExplanation: true };
                    if (displayResponse.referenceNotebook) {
                        metadata.referenceNotebook = displayResponse.referenceNotebook;
                    }
                }

                this.addMessage('assistant', messageText, false, false, metadata, suggestions);
            }
        } catch (error) {
            console.error('Error in display message handling:', error);
        }
    }

    // ========== CONTEXT PREPARATION ==========

    public async getContextForMessage(message: string): Promise<any> {
        const context: any = {};

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
        context.interactionMode = this._interactionMode;
        context.learningMode = this._learningMode;

        const config = vscode.workspace.getConfiguration('kai_agent');
        const apiKey = config.get('ollamaApiKey', '');
        if (apiKey) {
            context.ollamaApiKey = apiKey;
        }

        context.conversationHistory = this._getConversationHistoryContext();
        context.executionHistory = this.notebookOps?.executionHistory || [];
        context.modificationHistory = this.notebookOps?.modificationHistory || [];
        context.notebookStructure = this.notebookOps?.getNotebookStructure() || [];

        return context;
    }

    private _getConversationHistoryContext(): Array<{role: string, content: string, timestamp: string, metadata?: any}> {
        const estimateTokens = (text: string) => Math.ceil(text.length / 4);

        // Filter to non-indicator, non-empty messages
        const allMessages = this.messages.filter(msg => msg.type !== 'indicator' && msg.content.trim().length > 0);
        let totalTokens = 0;
        let selectedMessages: typeof allMessages = [];

        // Start from most recent and work backwards
        for (let i = allMessages.length - 1; i >= 0; i--) {
            const msg = allMessages[i];
            const entryTokens = estimateTokens(msg.content);

            if (totalTokens + entryTokens > this.MAX_CONTEXT_TOKENS && selectedMessages.length > 0) {
                break;
            }

            selectedMessages.unshift(msg);
            totalTokens += entryTokens;
        }

        return selectedMessages.map(msg => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp,
            metadata: msg.metadata
        }));
    }

    // ========== TASK LIST FORMATTING ==========

    private _formatTaskList(text: string): { text: string; tasks: Array<{id: string; task: string; status: string; isReasoning: boolean}> | null } {
        try {
            const jsonMatch = text.match(/\{[\s\S]*\}/);
            if (!jsonMatch) return { text, tasks: null };

            const jsonText = jsonMatch[0];
            let braceCount = 0;
            for (const char of jsonText) {
                if (char === '{') braceCount++;
                if (char === '}') braceCount--;
            }
            if (braceCount !== 0) return { text, tasks: null };

            const taskData = JSON.parse(jsonText);
            if (!taskData.tasks) return { text, tasks: null };

            // Detect task completion for guided mode checkpoints
            const currentTaskStates: Record<string, string> = {};
            taskData.tasks.forEach((task: any) => {
                currentTaskStates[task.id] = task.status;
                if (this._lastTaskStates[task.id] === 'active' && task.status === 'completed') {
                    this._taskJustCompleted = true;
                }
            });
            this._lastTaskStates = currentTaskStates;

            let formatted = '**Chain-of-thought:**\n';
            const structuredTasks: Array<{id: string; task: string; status: string; isReasoning: boolean}> = [];

            taskData.tasks.forEach((task: any) => {
                const isReasoning = task.task.toLowerCase().includes('[reasoning]');

                structuredTasks.push({
                    id: task.id.toString(),
                    task: task.task,
                    status: task.status,
                    isReasoning
                });

                let status: string;
                if (isReasoning) {
                    status = task.status === 'completed' ? '✅' :
                            task.status === 'active' ? '🧠' : '💭';
                } else {
                    status = task.status === 'completed' ? '✅' :
                            task.status === 'active' ? '🏃' : '⏳';
                }

                const taskText = isReasoning ? `*${task.task}*` : task.task;
                formatted += `${status} ${task.id}. ${taskText}\n`;
            });

            return { text: formatted, tasks: structuredTasks };
        } catch (e) {
            console.error('Failed to parse task list:', e);
            return { text, tasks: null };
        }
    }

    // ========== HTML TEMPLATE ==========

    public getHtmlForWebview(_webview: vscode.Webview): string {
        try {
            const templatePath = path.join(__dirname, '..', 'templates', 'chat-template.html');
            const templateContent = fs.readFileSync(templatePath, 'utf8');
            return templateContent;
        } catch (error) {
            console.error('Error loading chat template:', error);
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

    // ========== HELPER METHODS ==========

    private _createMessage(type: MessageType, role: string, content: string, metadata?: any): ChatMessage {
        return {
            id: this._generateId(),
            type,
            role: role as 'user' | 'assistant',
            content,
            timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
            metadata
        };
    }

    private _generateId(): string {
        return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    private _notify(): void {
        this.updateWebview();
    }
}
