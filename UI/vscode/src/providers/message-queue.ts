/**
 * MessageQueue - Manages pending messages that require execution completion
 *
 * This class tracks messages from Python that will trigger notebook cell execution,
 * ensuring that autonomous workflows don't proceed until execution actually completes.
 *
 * **Async Processing Chain:**
 * Messages are processed through an async chain that waits for actual notebook completion:
 * 1. Message added to queue
 * 2. Async processing: handleMessage → handleAutonomousCodeExecution → executeCell
 * 3. executeCell waits for VSCode notebook execution (up to 30min timeout)
 * 4. Only after execution completes is message removed from queue
 * 5. Autonomous loop waits for empty queue before next iteration
 *
 * **Synchronous Behavior via Async Coordination:**
 * The autonomous loop achieves synchronous behavior by waiting for the central
 * completion register (this queue) to be empty, ensuring all async work is done.
 *
 * Key Concepts:
 * - Messages from Python VSCodeCommunicator contain execution-triggering content
 * - We queue these messages instead of processing them immediately
 * - Only mark them complete when the associated notebook execution finishes
 * - Autonomous loop waits for empty queue before next iteration
 */

export interface PendingMessage {
    id: string;
    type: 'tool_result' | 'workflow_result' | 'auto_loop_update';
    payload: any;
    completed: boolean;
}

export class MessageQueue {
    private pendingMessages = new Map<string, PendingMessage>();
    private messageIdCounter = 0;

    /**
     * Add a message to the pending queue
     */
    addMessage(message: any): string {
        const messageId = this.generateMessageId();

        const pendingMessage: PendingMessage = {
            id: messageId,
            type: message.type,
            payload: message,
            completed: false
        };

        this.pendingMessages.set(messageId, pendingMessage);
        return messageId;
    }

    /**
     * Remove and return a message when its execution finishes
     */
    removeMessage(messageId: string): PendingMessage | null {
        const message = this.pendingMessages.get(messageId);
        if (message) {
            this.pendingMessages.delete(messageId);
            return message;
        }
        return null;
    }

    /**
     * Check if there are any pending messages
     */
    hasPendingMessages(): boolean {
        return this.pendingMessages.size > 0;
    }

    /**
     * Generate unique message ID
     */
    private generateMessageId(): string {
        return `msg_${Date.now()}_${++this.messageIdCounter}`;
    }

    /**
     * Clear all messages (for session reset)
     */
    clear(): void {
        this.pendingMessages.clear();
    }
}