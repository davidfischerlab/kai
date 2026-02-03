/**
 * Unit tests for ChatCore message display logic.
 *
 * Tests the key architectural guarantees:
 * 1. Message ordering: conversation → task list → notifications → learning → checkpoint → indicator
 * 2. Indicator is ALWAYS last when visible
 * 3. Singleton messages (task list, checkpoint, etc.) don't duplicate
 * 4. Message type routing via addMessage()
 */

import * as assert from 'assert';

// ============================================================
// ISOLATED TEST IMPLEMENTATION
// ============================================================
// We test the core message management logic in isolation without
// importing the actual chat-core.ts (which has vscode dependencies).
// This tests the same architectural patterns and algorithms.

type MessageType = 'user' | 'assistant' | 'task_list' | 'agent_notif' | 'critique' | 'learning' | 'checkpoint' | 'indicator';

interface ChatMessage {
    id: string;
    type: MessageType;
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
    metadata?: Record<string, any>;
    isIndicator?: boolean;
    indicatorText?: string;
}

/**
 * Simplified ChatCore that mirrors the production implementation
 * but without vscode dependencies. Tests the core message logic.
 */
class TestChatCore {
    private _conversation: ChatMessage[] = [];
    private _taskList: ChatMessage | null = null;
    private _agentNotification: ChatMessage | null = null;
    private _critique: ChatMessage | null = null;
    private _checkpoint: ChatMessage | null = null;
    private _learningMessages: ChatMessage[] = [];
    private _indicator: { visible: boolean; text: string } = { visible: false, text: '' };
    private _storedReferenceNotebooks: Array<{id: string; title: string}> | null = null;

    private _idCounter = 0;

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

    addMessage(
        role: string,
        content: string,
        isThinking: boolean = false,
        _isButtonAction: boolean = false,
        metadata?: any
    ) {
        if (isThinking) {
            this.showIndicator('thinking');
            return;
        }

        if (metadata?.isCheckpoint) {
            this._checkpoint = this._createMessage('checkpoint', role, content, metadata);
            return;
        }

        if (metadata?.isLearningExplanation) {
            this._learningMessages.push(this._createMessage('learning', role, content, metadata));
            return;
        }

        if (metadata?.isCritique) {
            this._critique = this._createMessage('critique', role, content, metadata);
            return;
        }

        if (metadata?.isAgentNotification) {
            this._agentNotification = this._createMessage('agent_notif', role, content, metadata);
            return;
        }

        if (metadata?.isTaskList) {
            this._taskList = this._createMessage('task_list', role, content, metadata);
            return;
        }

        const msgType: MessageType = role === 'user' ? 'user' : 'assistant';
        this._conversation.push(this._createMessage(msgType, role, content, metadata));
    }

    showIndicator(text: string): void {
        this._indicator = { visible: true, text };
    }

    hideIndicator(): void {
        this._indicator = { visible: false, text: '' };
    }

    setTaskList(content: string, metadata?: any): void {
        const taskMetadata: any = { isTaskList: true, ...metadata };
        // Include reference notebooks if stored
        if (this._storedReferenceNotebooks && this._storedReferenceNotebooks.length > 0) {
            taskMetadata.referenceNotebooks = this._storedReferenceNotebooks;
        }
        this._taskList = this._createMessage('task_list', 'assistant', content, taskMetadata);
    }

    storeReferenceWorkflows(data: { notebooks?: Array<{id: string; title: string}> }): void {
        if (data && data.notebooks) {
            this._storedReferenceNotebooks = data.notebooks;
        }
    }

    setAgentNotification(content: string): void {
        this._agentNotification = this._createMessage('agent_notif', 'assistant', content, { isAgentNotification: true });
    }

    setCritique(content: string): void {
        this._critique = this._createMessage('critique', 'assistant', content, { isCritique: true });
    }

    setCheckpoint(): void {
        this._checkpoint = this._createMessage('checkpoint', 'assistant', '', { isCheckpoint: true });
    }

    addLearningMessage(content: string): void {
        this._learningMessages.push(this._createMessage('learning', 'assistant', content, { isLearningExplanation: true }));
    }

    clearMessages(): void {
        this._conversation = [];
        this._taskList = null;
        this._agentNotification = null;
        this._critique = null;
        this._checkpoint = null;
        this._learningMessages = [];
        this._indicator = { visible: false, text: '' };
    }

    resetTaskTracking(): void {
        this._taskList = null;
        this._agentNotification = null;
        this._critique = null;
        this._learningMessages = [];
        this._storedReferenceNotebooks = null;
    }

    removeCheckpointMessages(): void {
        this._checkpoint = null;
    }

    private _createMessage(type: MessageType, role: string, content: string, metadata?: any): ChatMessage {
        return {
            id: `msg-${++this._idCounter}`,
            type,
            role: role as 'user' | 'assistant',
            content,
            timestamp: new Date().toISOString(),
            metadata
        };
    }
}


// ============================================================
// TESTS
// ============================================================

describe('ChatCore Message Display Logic', () => {

    describe('Message Ordering', () => {

        it('should return empty array when no messages exist', () => {
            const core = new TestChatCore();
            assert.deepStrictEqual(core.messages, []);
        });

        it('should return conversation messages in order', () => {
            const core = new TestChatCore();
            core.addMessage('user', 'Hello');
            core.addMessage('assistant', 'Hi there');
            core.addMessage('user', 'How are you?');

            const messages = core.messages;
            assert.strictEqual(messages.length, 3);
            assert.strictEqual(messages[0].content, 'Hello');
            assert.strictEqual(messages[1].content, 'Hi there');
            assert.strictEqual(messages[2].content, 'How are you?');
        });

        it('should place task list after conversation messages', () => {
            const core = new TestChatCore();
            core.addMessage('user', 'Do a task');
            core.setTaskList('Task 1, Task 2');

            const messages = core.messages;
            assert.strictEqual(messages.length, 2);
            assert.strictEqual(messages[0].type, 'user');
            assert.strictEqual(messages[1].type, 'task_list');
        });

        it('should maintain order: conversation → task list → agent notif → critique', () => {
            const core = new TestChatCore();
            core.addMessage('user', 'Start');
            core.setTaskList('Tasks');
            core.setAgentNotification('Analysis complete');
            core.setCritique('Review feedback');

            const messages = core.messages;
            assert.strictEqual(messages.length, 4);
            assert.strictEqual(messages[0].type, 'user');
            assert.strictEqual(messages[1].type, 'task_list');
            assert.strictEqual(messages[2].type, 'agent_notif');
            assert.strictEqual(messages[3].type, 'critique');
        });

        it('should place learning messages after critique', () => {
            const core = new TestChatCore();
            core.addMessage('user', 'Start');
            core.setTaskList('Tasks');
            core.setCritique('Feedback');
            core.addLearningMessage('Explanation 1');
            core.addLearningMessage('Explanation 2');

            const messages = core.messages;
            assert.strictEqual(messages.length, 5);
            assert.strictEqual(messages[3].type, 'learning');
            assert.strictEqual(messages[4].type, 'learning');
        });

        it('should place checkpoint after learning messages', () => {
            const core = new TestChatCore();
            core.addMessage('user', 'Start');
            core.addLearningMessage('Explanation');
            core.setCheckpoint();

            const messages = core.messages;
            assert.strictEqual(messages.length, 3);
            assert.strictEqual(messages[1].type, 'learning');
            assert.strictEqual(messages[2].type, 'checkpoint');
        });
    });


    describe('Indicator Always Last', () => {

        it('should place indicator at the end when visible', () => {
            const core = new TestChatCore();
            core.addMessage('user', 'Hello');
            core.showIndicator('thinking');

            const messages = core.messages;
            assert.strictEqual(messages.length, 2);
            assert.strictEqual(messages[1].type, 'indicator');
            assert.strictEqual(messages[1].isIndicator, true);
        });

        it('should keep indicator last even with task list present', () => {
            const core = new TestChatCore();
            core.showIndicator('thinking');
            core.addMessage('user', 'Start');
            core.setTaskList('Tasks');

            const messages = core.messages;
            const lastMessage = messages[messages.length - 1];
            assert.strictEqual(lastMessage.type, 'indicator');
        });

        it('should keep indicator last with all message types present', () => {
            const core = new TestChatCore();
            core.addMessage('user', 'Start');
            core.setTaskList('Tasks');
            core.setAgentNotification('Analysis');
            core.setCritique('Review');
            core.addLearningMessage('Explanation');
            core.setCheckpoint();
            core.showIndicator('working');

            const messages = core.messages;
            assert.strictEqual(messages.length, 7);
            assert.strictEqual(messages[6].type, 'indicator');
            assert.strictEqual(messages[6].indicatorText, 'working');
        });

        it('should not include indicator when hidden', () => {
            const core = new TestChatCore();
            core.addMessage('user', 'Hello');
            core.showIndicator('thinking');
            core.hideIndicator();

            const messages = core.messages;
            assert.strictEqual(messages.length, 1);
            assert.strictEqual(messages[0].type, 'user');
        });

        it('should keep indicator last after adding learning message', () => {
            const core = new TestChatCore();
            core.showIndicator('thinking');
            core.addLearningMessage('Explanation');

            const messages = core.messages;
            assert.strictEqual(messages.length, 2);
            assert.strictEqual(messages[0].type, 'learning');
            assert.strictEqual(messages[1].type, 'indicator');
        });

        it('should keep indicator last after adding checkpoint', () => {
            const core = new TestChatCore();
            core.addLearningMessage('Explanation');
            core.showIndicator('thinking');
            core.setCheckpoint();

            const messages = core.messages;
            const lastMessage = messages[messages.length - 1];
            assert.strictEqual(lastMessage.type, 'indicator');
        });
    });


    describe('Singleton Messages', () => {

        it('should only have one task list even after multiple updates', () => {
            const core = new TestChatCore();
            core.setTaskList('Task list v1');
            core.setTaskList('Task list v2');
            core.setTaskList('Task list v3');

            const taskLists = core.messages.filter(m => m.type === 'task_list');
            assert.strictEqual(taskLists.length, 1);
            assert.strictEqual(taskLists[0].content, 'Task list v3');
        });

        it('should only have one critique even after multiple updates', () => {
            const core = new TestChatCore();
            core.setCritique('Critique v1');
            core.setCritique('Critique v2');

            const critiques = core.messages.filter(m => m.type === 'critique');
            assert.strictEqual(critiques.length, 1);
            assert.strictEqual(critiques[0].content, 'Critique v2');
        });

        it('should only have one agent notification even after multiple updates', () => {
            const core = new TestChatCore();
            core.setAgentNotification('Notif v1');
            core.setAgentNotification('Notif v2');

            const notifs = core.messages.filter(m => m.type === 'agent_notif');
            assert.strictEqual(notifs.length, 1);
            assert.strictEqual(notifs[0].content, 'Notif v2');
        });

        it('should only have one checkpoint', () => {
            const core = new TestChatCore();
            core.setCheckpoint();
            core.setCheckpoint();
            core.setCheckpoint();

            const checkpoints = core.messages.filter(m => m.type === 'checkpoint');
            assert.strictEqual(checkpoints.length, 1);
        });

        it('should accumulate learning messages (not singleton)', () => {
            const core = new TestChatCore();
            core.addLearningMessage('Explanation 1');
            core.addLearningMessage('Explanation 2');
            core.addLearningMessage('Explanation 3');

            const learnings = core.messages.filter(m => m.type === 'learning');
            assert.strictEqual(learnings.length, 3);
        });
    });


    describe('addMessage Type Routing', () => {

        it('should route isThinking to indicator', () => {
            const core = new TestChatCore();
            core.addMessage('assistant', '', true); // isThinking = true

            const messages = core.messages;
            assert.strictEqual(messages.length, 1);
            assert.strictEqual(messages[0].type, 'indicator');
        });

        it('should route isCheckpoint metadata to checkpoint', () => {
            const core = new TestChatCore();
            core.addMessage('assistant', 'Continue?', false, false, { isCheckpoint: true });

            const messages = core.messages;
            assert.strictEqual(messages.length, 1);
            assert.strictEqual(messages[0].type, 'checkpoint');
        });

        it('should route isLearningExplanation metadata to learning', () => {
            const core = new TestChatCore();
            core.addMessage('assistant', 'This explains...', false, false, { isLearningExplanation: true });

            const messages = core.messages;
            assert.strictEqual(messages.length, 1);
            assert.strictEqual(messages[0].type, 'learning');
        });

        it('should route isCritique metadata to critique', () => {
            const core = new TestChatCore();
            core.addMessage('assistant', 'Feedback...', false, false, { isCritique: true });

            const messages = core.messages;
            assert.strictEqual(messages.length, 1);
            assert.strictEqual(messages[0].type, 'critique');
        });

        it('should route isAgentNotification metadata to agent_notif', () => {
            const core = new TestChatCore();
            core.addMessage('assistant', 'Analysis...', false, false, { isAgentNotification: true });

            const messages = core.messages;
            assert.strictEqual(messages.length, 1);
            assert.strictEqual(messages[0].type, 'agent_notif');
        });

        it('should route isTaskList metadata to task_list', () => {
            const core = new TestChatCore();
            core.addMessage('assistant', 'Tasks...', false, false, { isTaskList: true });

            const messages = core.messages;
            assert.strictEqual(messages.length, 1);
            assert.strictEqual(messages[0].type, 'task_list');
        });

        it('should route user role to conversation', () => {
            const core = new TestChatCore();
            core.addMessage('user', 'Hello');

            const messages = core.messages;
            assert.strictEqual(messages.length, 1);
            assert.strictEqual(messages[0].type, 'user');
        });

        it('should route assistant without special metadata to conversation', () => {
            const core = new TestChatCore();
            core.addMessage('assistant', 'Hello back');

            const messages = core.messages;
            assert.strictEqual(messages.length, 1);
            assert.strictEqual(messages[0].type, 'assistant');
        });
    });


    describe('Reset Operations', () => {

        it('should clear all messages on clearMessages', () => {
            const core = new TestChatCore();
            core.addMessage('user', 'Hello');
            core.setTaskList('Tasks');
            core.addLearningMessage('Explanation');
            core.showIndicator('thinking');

            core.clearMessages();

            assert.strictEqual(core.messages.length, 0);
        });

        it('should reset task tracking but keep conversation', () => {
            const core = new TestChatCore();
            core.addMessage('user', 'Hello');
            core.setTaskList('Tasks');
            core.setAgentNotification('Analysis');
            core.setCritique('Review');
            core.addLearningMessage('Explanation');

            core.resetTaskTracking();

            const messages = core.messages;
            assert.strictEqual(messages.length, 1);
            assert.strictEqual(messages[0].type, 'user');
        });

        it('should remove checkpoint on removeCheckpointMessages', () => {
            const core = new TestChatCore();
            core.addMessage('user', 'Hello');
            core.setCheckpoint();

            assert.strictEqual(core.messages.length, 2);

            core.removeCheckpointMessages();

            assert.strictEqual(core.messages.length, 1);
            assert.strictEqual(core.messages[0].type, 'user');
        });
    });


    describe('Complex Scenarios', () => {

        it('should handle Tutorial mode flow: task → learning → checkpoint → indicator hidden', () => {
            const core = new TestChatCore();

            // User sends request
            core.addMessage('user', 'Explain cell types');
            core.showIndicator('thinking');

            // Task list arrives
            core.setTaskList('1. Import data\n2. Process\n3. Annotate');

            // Code executes, learning explanation arrives
            core.addLearningMessage('This code imports the scanpy library...');

            // Checkpoint appears, indicator hides
            core.setCheckpoint();
            core.hideIndicator();

            const messages = core.messages;
            assert.strictEqual(messages.length, 4);
            assert.strictEqual(messages[0].type, 'user');
            assert.strictEqual(messages[1].type, 'task_list');
            assert.strictEqual(messages[2].type, 'learning');
            assert.strictEqual(messages[3].type, 'checkpoint');

            // No indicator since hidden
            assert.ok(!messages.some(m => m.type === 'indicator'));
        });

        it('should handle task list update without duplicating', () => {
            const core = new TestChatCore();
            core.addMessage('user', 'Start analysis');

            // Initial task list
            core.setTaskList('1. [pending] Import\n2. [pending] Process');

            // Task 1 completes
            core.setTaskList('1. [done] Import\n2. [active] Process');

            // Task 2 completes
            core.setTaskList('1. [done] Import\n2. [done] Process');

            const taskLists = core.messages.filter(m => m.type === 'task_list');
            assert.strictEqual(taskLists.length, 1);
            assert.ok(taskLists[0].content.includes('[done] Process'));
        });

        it('should maintain indicator at bottom through multiple operations', () => {
            const core = new TestChatCore();

            core.showIndicator('thinking');
            assert.strictEqual(core.messages[core.messages.length - 1].type, 'indicator');

            core.addMessage('user', 'Hello');
            assert.strictEqual(core.messages[core.messages.length - 1].type, 'indicator');

            core.setTaskList('Tasks');
            assert.strictEqual(core.messages[core.messages.length - 1].type, 'indicator');

            core.addLearningMessage('Explanation 1');
            assert.strictEqual(core.messages[core.messages.length - 1].type, 'indicator');

            core.addLearningMessage('Explanation 2');
            assert.strictEqual(core.messages[core.messages.length - 1].type, 'indicator');

            core.setCheckpoint();
            assert.strictEqual(core.messages[core.messages.length - 1].type, 'indicator');
        });
    });


    describe('Reference Notebooks', () => {

        it('should include reference notebooks in task list metadata when stored before task list', () => {
            const core = new TestChatCore();

            // Store reference notebooks first (like Python sends them)
            core.storeReferenceWorkflows({
                notebooks: [
                    { id: 'nb1', title: 'Tutorial 1' },
                    { id: 'nb2', title: 'Tutorial 2' }
                ]
            });

            // Then task list arrives
            core.setTaskList('1. Import\n2. Process');

            const taskList = core.messages.find(m => m.type === 'task_list');
            assert.ok(taskList, 'Task list should exist');
            assert.ok(taskList!.metadata?.referenceNotebooks, 'Task list should have referenceNotebooks in metadata');
            assert.strictEqual(taskList!.metadata?.referenceNotebooks.length, 2);
        });

        it('should NOT include reference notebooks if none stored', () => {
            const core = new TestChatCore();

            // No reference notebooks stored
            core.setTaskList('1. Import\n2. Process');

            const taskList = core.messages.find(m => m.type === 'task_list');
            assert.ok(taskList, 'Task list should exist');
            assert.ok(!taskList!.metadata?.referenceNotebooks, 'Task list should NOT have referenceNotebooks');
        });

        it('should clear reference notebooks on resetTaskTracking', () => {
            const core = new TestChatCore();

            // Store notebooks
            core.storeReferenceWorkflows({
                notebooks: [{ id: 'nb1', title: 'Tutorial 1' }]
            });

            // Reset tracking
            core.resetTaskTracking();

            // New task list should NOT have notebooks
            core.setTaskList('New tasks');

            const taskList = core.messages.find(m => m.type === 'task_list');
            assert.ok(!taskList!.metadata?.referenceNotebooks, 'Task list should NOT have referenceNotebooks after reset');
        });

        it('should preserve reference notebooks across multiple task list updates', () => {
            const core = new TestChatCore();

            // Store notebooks
            core.storeReferenceWorkflows({
                notebooks: [{ id: 'nb1', title: 'Tutorial 1' }]
            });

            // First task list
            core.setTaskList('1. [pending] Task A');

            // Update task list
            core.setTaskList('1. [done] Task A');

            const taskList = core.messages.find(m => m.type === 'task_list');
            assert.ok(taskList!.metadata?.referenceNotebooks, 'Updated task list should still have referenceNotebooks');
            assert.strictEqual(taskList!.metadata?.referenceNotebooks.length, 1);
        });

        it('should handle empty notebooks array', () => {
            const core = new TestChatCore();

            // Store empty array
            core.storeReferenceWorkflows({ notebooks: [] });

            core.setTaskList('Tasks');

            const taskList = core.messages.find(m => m.type === 'task_list');
            // Empty array should not be included
            assert.ok(!taskList!.metadata?.referenceNotebooks, 'Empty notebooks should not be included');
        });
    });
});


// ============================================================
// PRODUCTION CODE TESTS
// ============================================================
// These tests import and test the actual ChatCore class to ensure
// the production code matches the algorithmic tests above.

import { ChatCore } from '../providers/chat-core';

describe('ChatCore Production Implementation', () => {
    let updateWebviewCalled: boolean;
    let storeToolUsageCalled: boolean;

    function createProductionChatCore(): ChatCore {
        updateWebviewCalled = false;
        storeToolUsageCalled = false;
        return new ChatCore(
            () => { updateWebviewCalled = true; },
            () => { storeToolUsageCalled = true; },
            () => false
        );
    }

    describe('Message Ordering (Production)', () => {

        it('should return empty array when no messages exist', () => {
            const core = createProductionChatCore();
            assert.deepStrictEqual(core.messages, []);
        });

        it('should place indicator last after conversation messages', () => {
            const core = createProductionChatCore();
            core.addMessage('user', 'Hello');
            core.showIndicator('thinking');

            const messages = core.messages;
            assert.strictEqual(messages.length, 2);
            assert.strictEqual(messages[0].type, 'user');
            assert.strictEqual(messages[1].type, 'indicator');
        });

        it('should place indicator last with all message types', () => {
            const core = createProductionChatCore();
            core.addMessage('user', 'Start');
            core.addMessage('assistant', 'Task list', false, false, { isTaskList: true });
            core.addMessage('assistant', 'Analysis', false, false, { isAgentNotification: true });
            core.addMessage('assistant', 'Review', false, false, { isCritique: true });
            core.addMessage('assistant', 'Learning', false, false, { isLearningExplanation: true });
            core.addMessage('assistant', 'Continue', false, false, { isCheckpoint: true });
            core.showIndicator('working');

            const messages = core.messages;
            assert.strictEqual(messages[messages.length - 1].type, 'indicator');
        });
    });

    describe('Singleton Messages (Production)', () => {

        it('should only have one task list after multiple updates', () => {
            const core = createProductionChatCore();
            core.addMessage('assistant', 'v1', false, false, { isTaskList: true });
            core.addMessage('assistant', 'v2', false, false, { isTaskList: true });
            core.addMessage('assistant', 'v3', false, false, { isTaskList: true });

            const taskLists = core.messages.filter(m => m.type === 'task_list');
            assert.strictEqual(taskLists.length, 1);
            assert.strictEqual(taskLists[0].content, 'v3');
        });

        it('should accumulate learning messages', () => {
            const core = createProductionChatCore();
            core.addMessage('assistant', 'Explanation 1', false, false, { isLearningExplanation: true });
            core.addMessage('assistant', 'Explanation 2', false, false, { isLearningExplanation: true });

            const learnings = core.messages.filter(m => m.type === 'learning');
            assert.strictEqual(learnings.length, 2);
        });
    });

    describe('Reference Notebooks (Production)', () => {

        it('should store reference notebooks from storeReferenceWorkflows', () => {
            const core = createProductionChatCore();
            core.storeReferenceWorkflows({
                text: 'Notebook info',
                notebooks: [
                    { id: 'nb1', title: 'Tutorial 1', source_path: '/path/1', percentage: 50, cells: [] },
                    { id: 'nb2', title: 'Tutorial 2', source_path: '/path/2', percentage: 75, cells: [] }
                ]
            });

            assert.ok(core.storedReferenceNotebooks);
            assert.strictEqual(core.storedReferenceNotebooks?.length, 2);
        });

        it('should include reference notebooks in task list via handleTaskListMessage', async () => {
            const core = createProductionChatCore();

            // Store notebooks first
            core.storeReferenceWorkflows({
                text: 'Notebook info',
                notebooks: [
                    { id: 'org/repo/nb1.ipynb', title: 'Tutorial 1', source_path: '/path/1', percentage: 50, cells: [] }
                ]
            });

            // Then handle task list message
            await core.handleTaskListMessage({
                text: '{"tasks": [{"id": 1, "task": "Import data", "status": "pending"}]}'
            });

            const taskList = core.messages.find(m => m.type === 'task_list');
            assert.ok(taskList, 'Task list should exist');
            assert.ok(taskList!.metadata?.referenceNotebooks, 'Task list should have referenceNotebooks');
            assert.strictEqual(taskList!.metadata?.referenceNotebooks.length, 1);
        });

        it('should clear reference notebooks on clearMessages', () => {
            const core = createProductionChatCore();
            core.storeReferenceWorkflows({
                notebooks: [{ id: 'nb1', title: 'Tutorial 1', source_path: '', percentage: 50, cells: [] }]
            });
            assert.ok(core.storedReferenceNotebooks);

            core.clearMessages();

            assert.strictEqual(core.storedReferenceNotebooks, null);
        });

        it('should clear reference notebooks on resetTaskTracking', () => {
            const core = createProductionChatCore();
            core.storeReferenceWorkflows({
                notebooks: [{ id: 'nb1', title: 'Tutorial 1', source_path: '', percentage: 50, cells: [] }]
            });
            assert.ok(core.storedReferenceNotebooks);

            core.resetTaskTracking();

            assert.strictEqual(core.storedReferenceNotebooks, null);
        });
    });

    describe('handleTaskListMessage (Production)', () => {

        it('should parse JSON task list and create structured tasks', async () => {
            const core = createProductionChatCore();

            await core.handleTaskListMessage({
                text: '{"tasks": [{"id": 1, "task": "Task A", "status": "pending"}, {"id": 2, "task": "Task B", "status": "active"}]}'
            });

            const taskList = core.messages.find(m => m.type === 'task_list');
            assert.ok(taskList, 'Task list should exist');
            assert.ok(taskList!.metadata?.tasks, 'Task list should have structured tasks');
            assert.strictEqual(taskList!.metadata?.tasks.length, 2);
        });

        it('should handle agent notification with task list', async () => {
            const core = createProductionChatCore();

            await core.handleTaskListMessage({
                text: '{"tasks": [{"id": 1, "task": "Task A", "status": "pending"}]}',
                agent_notification: 'Analysis complete'
            });

            const notif = core.messages.find(m => m.type === 'agent_notif');
            assert.ok(notif, 'Agent notification should exist');
            assert.ok(notif!.content.includes('Analysis complete'));
        });

        it('should handle critique with task list', async () => {
            const core = createProductionChatCore();

            await core.handleTaskListMessage({
                text: '{"tasks": [{"id": 1, "task": "Task A", "status": "pending"}]}',
                critique: 'Please improve'
            });

            const critique = core.messages.find(m => m.type === 'critique');
            assert.ok(critique, 'Critique should exist');
            assert.ok(critique!.content.includes('Please improve'));
        });

        it('should update singleton task list without duplicating', async () => {
            const core = createProductionChatCore();

            await core.handleTaskListMessage({
                text: '{"tasks": [{"id": 1, "task": "Task A", "status": "pending"}]}'
            });

            await core.handleTaskListMessage({
                text: '{"tasks": [{"id": 1, "task": "Task A", "status": "completed"}]}'
            });

            const taskLists = core.messages.filter(m => m.type === 'task_list');
            assert.strictEqual(taskLists.length, 1, 'Should only have one task list');
        });
    });

    describe('Indicator Position Guarantees (Production)', () => {

        it('should have indicator with empty timestamp (not a stored message)', () => {
            const core = createProductionChatCore();

            // Add some messages with real timestamps
            core.addMessage('user', 'Hello');
            core.showIndicator('thinking');

            const messages = core.messages;
            const indicator = messages.find(m => m.type === 'indicator');

            assert.ok(indicator, 'Indicator should exist');
            assert.strictEqual(indicator!.timestamp, '',
                'Indicator timestamp should be empty (computed, not stored)');
            assert.strictEqual(indicator!.id, 'indicator',
                'Indicator id should be "indicator" (fixed, not generated)');
        });

        it('should never have indicator in _conversation array', () => {
            const core = createProductionChatCore();

            core.addMessage('user', 'Hello');
            core.showIndicator('thinking');
            core.addMessage('assistant', 'Response', false, false, { isLearningExplanation: true });

            // Access messages getter
            const messages = core.messages;

            // The indicator should only appear once, as the last message
            const indicators = messages.filter(m => m.type === 'indicator' || m.isIndicator);
            assert.strictEqual(indicators.length, 1,
                `Should have exactly 1 indicator but found ${indicators.length}`);
            assert.strictEqual(messages[messages.length - 1].type, 'indicator',
                'The single indicator should be at the end');
        });

        it('should place indicator AFTER learning message when both present', () => {
            const core = createProductionChatCore();

            // Add learning message first
            core.addMessage('assistant', 'This explains the code...', false, false, { isLearningExplanation: true });

            // Show indicator
            core.showIndicator('thinking');

            const messages = core.messages;
            const learningIndex = messages.findIndex(m => m.type === 'learning');
            const indicatorIndex = messages.findIndex(m => m.type === 'indicator');

            assert.ok(learningIndex >= 0, 'Learning message should exist');
            assert.ok(indicatorIndex >= 0, 'Indicator should exist');
            assert.ok(indicatorIndex > learningIndex,
                `Indicator (index ${indicatorIndex}) should be AFTER learning (index ${learningIndex})`);
        });

        it('should place indicator AFTER checkpoint when both present', () => {
            const core = createProductionChatCore();

            // Add checkpoint
            core.addMessage('assistant', '', false, false, { isCheckpoint: true });

            // Show indicator
            core.showIndicator('working');

            const messages = core.messages;
            const checkpointIndex = messages.findIndex(m => m.type === 'checkpoint');
            const indicatorIndex = messages.findIndex(m => m.type === 'indicator');

            assert.ok(checkpointIndex >= 0, 'Checkpoint should exist');
            assert.ok(indicatorIndex >= 0, 'Indicator should exist');
            assert.ok(indicatorIndex > checkpointIndex,
                `Indicator (index ${indicatorIndex}) should be AFTER checkpoint (index ${checkpointIndex})`);
        });

        it('should place indicator last in full tutorial flow: task → learning → checkpoint → indicator', () => {
            const core = createProductionChatCore();

            // Simulate full tutorial mode flow
            core.addMessage('user', 'Analyze this data');
            core.addMessage('assistant', 'Task list', false, false, { isTaskList: true });
            core.addMessage('assistant', 'This code imports scanpy...', false, false, { isLearningExplanation: true });
            core.addMessage('assistant', '', false, false, { isCheckpoint: true });
            core.showIndicator('thinking');

            const messages = core.messages;
            const lastMessage = messages[messages.length - 1];

            assert.strictEqual(lastMessage.type, 'indicator',
                `Last message should be indicator but was ${lastMessage.type}`);
        });

        it('should keep indicator last even when learning messages are added after indicator is shown', () => {
            const core = createProductionChatCore();

            // Show indicator first
            core.showIndicator('processing');

            // Add learning message while indicator is showing
            core.addMessage('assistant', 'Explanation 1', false, false, { isLearningExplanation: true });
            core.addMessage('assistant', 'Explanation 2', false, false, { isLearningExplanation: true });

            const messages = core.messages;
            const lastMessage = messages[messages.length - 1];

            assert.strictEqual(lastMessage.type, 'indicator',
                `Last message should be indicator but was ${lastMessage.type}`);

            // Verify learning messages are before indicator
            const learningMessages = messages.filter(m => m.type === 'learning');
            assert.strictEqual(learningMessages.length, 2, 'Should have 2 learning messages');
        });
    });

    describe('Tutorial Mode Complete Flow (Production)', () => {

        it('should have correct message order after full tutorial flow', async () => {
            const core = createProductionChatCore();

            // 1. User sends message
            core.addMessage('user', 'Analyze this data');

            // 2. Indicator shows while processing
            core.showIndicator('thinking');

            // 3. Reference workflows stored
            core.storeReferenceWorkflows({
                notebooks: [{ id: 'nb1', title: 'Tutorial', source_path: '', percentage: 50, cells: [] }]
            });

            // 4. Task list arrives
            await core.handleTaskListMessage({
                text: '{"tasks": [{"id": 1, "task": "Step 1", "status": "active"}]}'
            });

            // Verify indicator is still last at this point
            let messages = core.messages;
            assert.strictEqual(messages[messages.length - 1].type, 'indicator',
                'Indicator should be last after task list');

            // 5. Learning explanation arrives
            core.addMessage('assistant', 'This code does...', false, false, { isLearningExplanation: true });

            // Verify indicator is still last
            messages = core.messages;
            assert.strictEqual(messages[messages.length - 1].type, 'indicator',
                'Indicator should be last after learning message');

            // 6. Hide indicator and show checkpoint (feedback waiting state)
            core.hideIndicator();
            core.addMessage('assistant', '', false, false, { isCheckpoint: true });

            // Verify final order: user → task_list → learning → checkpoint (no indicator)
            messages = core.messages;
            assert.strictEqual(messages[0].type, 'user', 'First should be user');
            assert.strictEqual(messages[1].type, 'task_list', 'Second should be task_list');
            assert.strictEqual(messages[2].type, 'learning', 'Third should be learning');
            assert.strictEqual(messages[3].type, 'checkpoint', 'Fourth should be checkpoint');
            assert.strictEqual(messages.length, 4, 'Should have exactly 4 messages (no indicator)');
        });

        it('should never have indicator between task_list and learning messages', () => {
            const core = createProductionChatCore();

            core.addMessage('user', 'Test');
            core.showIndicator('thinking');
            core.addMessage('assistant', 'Tasks', false, false, { isTaskList: true });
            core.addMessage('assistant', 'Explanation', false, false, { isLearningExplanation: true });

            const messages = core.messages;

            // Find indices
            const taskListIndex = messages.findIndex(m => m.type === 'task_list');
            const learningIndex = messages.findIndex(m => m.type === 'learning');
            const indicatorIndex = messages.findIndex(m => m.type === 'indicator');

            assert.ok(taskListIndex >= 0, 'Task list should exist');
            assert.ok(learningIndex >= 0, 'Learning should exist');
            assert.ok(indicatorIndex >= 0, 'Indicator should exist');

            // Indicator must be after BOTH task_list and learning
            assert.ok(indicatorIndex > taskListIndex,
                `Indicator (${indicatorIndex}) must be after task_list (${taskListIndex})`);
            assert.ok(indicatorIndex > learningIndex,
                `Indicator (${indicatorIndex}) must be after learning (${learningIndex})`);

            // Learning must be after task_list
            assert.ok(learningIndex > taskListIndex,
                `Learning (${learningIndex}) must be after task_list (${taskListIndex})`);
        });
    });

    describe('Indicator Visibility During Feedback State (Production)', () => {

        it('should not show indicator when checkpoint is present and waiting for feedback', () => {
            const core = createProductionChatCore();

            // Simulate learning mode pause state:
            // 1. Task list exists
            // 2. Learning explanation shown
            // 3. Checkpoint shown (waiting for user to click continue)
            // 4. Indicator should be HIDDEN
            core.addMessage('assistant', 'Tasks...', false, false, { isTaskList: true });
            core.addMessage('assistant', 'This explains...', false, false, { isLearningExplanation: true });
            core.addMessage('assistant', '', false, false, { isCheckpoint: true });

            // In proper flow, indicator should be hidden when waiting for feedback
            // If someone accidentally shows indicator here, it's a bug
            core.hideIndicator();

            const messages = core.messages;
            const hasIndicator = messages.some(m => m.type === 'indicator');

            assert.strictEqual(hasIndicator, false,
                'Indicator should NOT be visible when checkpoint is shown (waiting for feedback)');
        });

        it('should show indicator only after checkpoint is removed (user clicked continue)', () => {
            const core = createProductionChatCore();

            // Setup feedback waiting state
            core.addMessage('assistant', 'Tasks...', false, false, { isTaskList: true });
            core.addMessage('assistant', 'This explains...', false, false, { isLearningExplanation: true });
            core.addMessage('assistant', '', false, false, { isCheckpoint: true });
            core.hideIndicator();

            // Verify no indicator while waiting
            assert.strictEqual(core.messages.some(m => m.type === 'indicator'), false,
                'Indicator should be hidden while waiting for feedback');

            // User clicks continue - checkpoint removed, indicator shown
            core.removeCheckpointMessages();
            core.showIndicator('thinking');

            const messages = core.messages;
            const hasCheckpoint = messages.some(m => m.type === 'checkpoint');
            const hasIndicator = messages.some(m => m.type === 'indicator');

            assert.strictEqual(hasCheckpoint, false, 'Checkpoint should be removed');
            assert.strictEqual(hasIndicator, true, 'Indicator should be visible after continue');
        });

        it('should have indicator as last message after continue is clicked', () => {
            const core = createProductionChatCore();

            // Full flow: task → learning → checkpoint → (user continues) → indicator
            core.addMessage('assistant', 'Tasks...', false, false, { isTaskList: true });
            core.addMessage('assistant', 'Explanation', false, false, { isLearningExplanation: true });
            core.addMessage('assistant', '', false, false, { isCheckpoint: true });
            core.hideIndicator();

            // User continues
            core.removeCheckpointMessages();
            core.showIndicator('thinking');

            const messages = core.messages;
            const lastMessage = messages[messages.length - 1];

            assert.strictEqual(lastMessage.type, 'indicator',
                `After continue, last message should be indicator but was ${lastMessage.type}`);
        });
    });
});
