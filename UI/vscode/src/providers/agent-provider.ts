import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import { MessageQueue } from './message-queue';

/**
 * KaiAgentProvider - Python subprocess interface with unified promise-based communication
 *
 * **Communication Architecture:**
 * 1. Promise-based requests: All Python requests return promises that resolve when processing completes
 * 2. Real-time messages: Workflow progress and results stream via VSCodeCommunicator messages
 *
 * **Message Flow:**
 * ```
 * VSCode Request → Python Processing → VSCode Promise Resolution
 *              ↗ (during processing) → Real-time VSCodeCommunicator messages → VSCode UI
 * ```
 *
 * **Public API Methods:**
 * - `sendRegularRequest(message, context)` - Send regular chat requests
 * - `handleAutonomousPlanning(message, context)` - Handle autonomous planning and feedback (unified)
 * - `handleAutonomousExecution(context)` - Continue autonomous execution
 * - `pauseAutonomousExecution()` - Pause autonomous execution for feedback
 * - `stopAutonomousExecution()` - Stop autonomous execution
 * - `setMessageCallback(callback)` - Set callback for real-time messages
 *
 * **Message Types Sent to messageCallback:**
 * - `display`: Standard display messages → ChatCore.handleDisplayMessage()
 * - `task_list_display`: Task list updates → ChatCore.handleTaskListMessage()
 * - `execute_code`: Autonomous code execution → AutonomousExecution.handleAutonomousCodeExecution()
 * - `ui_control`: Interface state control → ChatViewProvider._handleMessage()
 * - `auto_loop_update`: Autonomous workflow signals → ChatViewProvider._handleMessage()
 * - `regular_chat_complete`: Regular chat completion → ChatViewProvider._handleMessage()
 */
export class KaiAgentProvider {
    private pythonPath: string;
    private pythonProcess?: ChildProcess;
    private currentSession?: string;
    private isInitialized = false;
    private isFullyReady = false; // Track if LLM is loaded and ready
    
    public get initialized(): boolean {
        return this.isInitialized;
    }
    
    public get fullyReady(): boolean {
        return this.isFullyReady;
    }
    
    private lastErrorMessage?: string; // Track last error to avoid duplicate logging
    private initializationPromise?: Promise<void>;
    private requestId = 0;
    private pendingRequests = new Map<string, { resolve: (value: any) => void; reject: (error: Error) => void; timeout: NodeJS.Timeout }>();
    private messageQueue: MessageQueue;
    private messageFilteringPaused = false; // Filter incoming tool messages when paused
    
    constructor(private context: vscode.ExtensionContext) {
        const config = vscode.workspace.getConfiguration('kai_agent');
        this.pythonPath = config.get('pythonPath', 'python');

        // Initialize message queue
        this.messageQueue = new MessageQueue();

        // Initialize agent with .initializeAgent() later when environment is defined.
    }
    
    async initializePythonAgent(turboEnabled: boolean): Promise<void> {
        if (this.initializationPromise) {
            return this.initializationPromise;
        }
        
        this.initializationPromise = new Promise<void>((resolve, reject) => {
            
            // Load Python script from external file
            const scriptPath = path.join(__dirname, 'python-subprocess.py');
            console.log('Loading Python script from:', scriptPath);

            // Get API key from VSCode settings
            const config = vscode.workspace.getConfiguration('kai_agent');
            const apiKey = config.get('ollamaApiKey', '');

            this.pythonProcess = spawn(this.pythonPath, [scriptPath, turboEnabled.toString(), apiKey], {
                // No need to set cwd or PYTHONPATH - kai is installed as a package
                stdio: ['pipe', 'pipe', 'pipe'], // Explicit stdio configuration
                env: {
                    ...process.env,
                    // Disable telemetry and analytics
                    DISABLE_TELEMETRY: '1',
                    POSTHOG_DISABLED: '1',
                    DO_NOT_TRACK: '1',
                    DISABLE_ANALYTICS: '1',
                    // Force unbuffered output
                    PYTHONUNBUFFERED: '1'
                }
            });
            
            // Add error handler for spawn failures
            this.pythonProcess.on('error', (error) => {
                console.error('[ERROR] Failed to spawn Python process:', error);
                console.error('[ERROR] Python path:', this.pythonPath);
                reject(new Error(`Failed to start Python process: ${error.message}`));
            });
            
            let initBuffer = '';
            
            const onData = (data: Buffer) => {
                initBuffer += data.toString();
                const lines = initBuffer.split('\n');
                initBuffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const response = JSON.parse(line);
                            if (response.type === 'initialized') {
                                this.isInitialized = true;
                                this.pythonProcess!.stdout!.off('data', onData);
                                this.pythonProcess!.stdout!.on('data', this.handleResponse.bind(this));
                                resolve();
                            } else if (response.type === 'error') {
                                reject(new Error(response.message));
                            }
                        } catch (e) {
                            // Ignore non-JSON init output
                        }
                    }
                }
            };
            
            this.pythonProcess.stdout!.on('data', onData);
            
            this.pythonProcess.stderr!.on('data', (data) => {
                // Filter out all stderr messages except critical errors
                const stderrText = data.toString();
                
                // Skip debug messages that aren't actual errors
                if (stderrText.includes('PARSED REQUEST:') || 
                    stderrText.includes('PARSING JSON:') ||
                    stderrText.includes('HANDLING CHAT REQUEST:') ||
                    stderrText.includes('RECEIVED STDIN LINE:') ||
                    stderrText.includes('WAITING FOR STDIN LINE') ||
                    stderrText.includes('Creating KaiAgent') ||
                    stderrText.includes('KaiAgent created') ||
                    stderrText.includes('Agent ready!') ||
                    stderrText.includes('[STREAM]') ||
                    stderrText.includes('Stream started')) {
                    return; // Skip these debug messages entirely
                }
                
                // Only log critical errors with stack traces
                if (stderrText.includes('Traceback') ||
                    stderrText.includes('ModuleNotFoundError') ||
                    stderrText.includes('ImportError') ||
                    stderrText.includes('SyntaxError')) {
                    // Truncate long messages to first 5000 characters for debugging
                    const truncated = stderrText.length > 5000 ? stderrText.substring(0, 5000) + '...[truncated]' : stderrText;
                    // Only log once per unique error to avoid spam
                    if (!this.lastErrorMessage || this.lastErrorMessage !== truncated) {
                        console.error('Python error:', truncated);
                        this.lastErrorMessage = truncated;
                    }
                }
                // All other stderr messages are silently ignored
            });
            
            this.pythonProcess.on('exit', (code) => {
                this.isInitialized = false;
                this.isFullyReady = false;
                this.pythonProcess = undefined;
                // Reject any pending requests
                for (const [id, { reject, timeout }] of this.pendingRequests) {
                    clearTimeout(timeout);
                    reject(new Error('Python process exited'));
                }
                this.pendingRequests.clear();
            });
            
            // Timeout for initialization
            setTimeout(() => {
                if (!this.isInitialized) {
                    reject(new Error('Agent initialization timed out'));
                }
            }, 120000); // 120 second timeout for initialization (2 minutes)
        });
        
        return this.initializationPromise;
    }
    
    private responseBuffer: string = '';
    
    /**
     * Handle responses from Python subprocess.
     *
     * Processes two types of messages:
     * 1. **Promise Resolution**: Messages with request_id resolve pending promises
     * 2. **Real-time Communication**: VSCodeCommunicator messages sent via messageCallback
     *
     * **Why Async Message Processing is Required:**
     * Some messages (execute_code, auto_loop_update) trigger notebook cell execution,
     * which is inherently async in VSCode. To ensure the autonomous loop doesn't proceed
     * before execution completes, these messages are processed through an async chain:
     *
     * execute_code → queueAndProcessMessage → _handleMessage → handleAutonomousCodeExecution → executeCell
     *
     * Each step awaits the previous, ensuring messages are only marked complete after
     * actual notebook execution finishes (not just when processing starts).
     *
     * Complete Message Type Routing:
     * - `response`: Resolves promise for standard chat → Promise resolution
     * - `console_log`: Development console output → console.log()
     * - `status`: Agent status updates → VSCode status bar
     * - `initialized`: Agent initialization complete → isInitialized = true
     * - `display`: Standard display messages → messageCallback('display')
     * - `task_list_display`: Task list updates → messageCallback('task_list_display')
     * - `execute_code`: Autonomous code execution → messageCallback('execute_code') [ASYNC CHAIN]
     * - `no_output`: Only queue message to track completion, do not process → messageCallback('no_output')
     * - `workflow_result`: Workflow completion signals → messageCallback('auto_loop_update'|'regular_chat_complete') [ASYNC CHAIN]
     * - `error`: Error responses → Promise rejection
     */
    private handleResponse(data: Buffer) {
        this.responseBuffer += data.toString();
        const lines = this.responseBuffer.split('\n');
        // Keep the last incomplete line in buffer
        this.responseBuffer = lines.pop() || '';
        
        for (const line of lines) {
            if (line.trim()) {
                // Skip lines that don't start with { (not JSON)
                if (!line.trim().startsWith('{')) {
                    console.log('Skipping non-JSON line:', line);
                    continue;
                }
                
                try {
                    const response = JSON.parse(line);
                    
                    // Handle status and debug messages first (they don't have request_id)
                    if (response.type === 'status') {
                        // Handle status messages (don't resolve/reject, just log)
                        vscode.window.setStatusBarMessage(`KaiAgent: ${response.message}`, 3000);
                        
                        // Track when agent is fully ready (LLM loaded)
                        if (response.message === 'Agent ready!') {
                            console.log('Agent ready!');
                            this.isFullyReady = true;
                        }
                        continue; // Status messages don't need further processing
                    } else if (response.type === 'initialized') {
                        // Handle initialization complete signal
                        this.isInitialized = true;
                        continue; // Initialization messages don't need further processing
                    } else if (response.type === 'console_log') {
                        // Handle timing and debug console messages
                        console.log(response.message);
                        continue;
                    } else if (response.type === 'response') {
                        // Handle standard responses, e.g. in chat mode
                        const requestId = response.request_id;
                        if (this.pendingRequests.has(requestId)) {
                            const { resolve, timeout } = this.pendingRequests.get(requestId)!;
                            clearTimeout(timeout);
                            this.pendingRequests.delete(requestId);
                            this.currentSession = response.session_id;
                            resolve({
                                response: response.response,
                                session_id: response.session_id,
                                request_id: requestId
                            });
                        }
                        continue;
                    } else if (response.type === 'workflow_result') {
                        // Handle status response from workflow completion (asynchronous)
                        // Queue and process workflow completion signals
                        if (response.auto_loop_update) {
                            // Autonomous workflow completion
                            this.queueAndProcessMessage('auto_loop_update', {
                                status: response.auto_loop_update
                            });
                        } else if (response.regular_chat_update) {
                            // Regular chat completion - process immediately (no queuing needed)
                            if (this.messageCallback) {
                                this.messageCallback('regular_chat_complete', {
                                    message: response.regular_chat_update
                                });
                            }
                        }
                        continue;
                    } else if (response.type === 'execute_code') {
                        // Filter out tool messages when paused for feedback interrupt
                        if (this.messageFilteringPaused) {
                            continue;
                        }

                        // Queue and process autonomous code execution
                        const enrichedResponse = { ...response.response };
                        if (response.code) {
                            enrichedResponse.code = response.code;
                        }
                        this.queueAndProcessMessage('execute_code', enrichedResponse);
                        continue;
                    } else if (response.type === 'display') {
                        // Filter out tool messages when paused for feedback interrupt
                        if (this.messageFilteringPaused) {
                            continue;
                        }
                        this.queueAndProcessMessage('display', response.response);
                        continue;
                    } else if (response.type === 'task_list_display') {
                        // Filter out tool messages when paused for feedback interrupt
                        if (this.messageFilteringPaused) {
                            continue;
                        }
                        this.queueAndProcessMessage('task_list_display', response.response);
                        continue;
                    } else if (response.type === 'no_output') {
                        // Filter out tool messages when paused for feedback interrupt
                        if (this.messageFilteringPaused) {
                            continue;
                        }
                        this.queueAndProcessMessage('no_output', response.response);
                        continue;
                    } else if (response.type === 'reference_workflows') {
                        // Filter out tool messages when paused for feedback interrupt
                        if (this.messageFilteringPaused) {
                            continue;
                        }
                        this.queueAndProcessMessage('reference_workflows', response.response);
                        continue;
                    } else if (response.type === 'execution_progress_check_response') {
                        // Handle execution progress check response
                        const requestId = response.request_id;
                        if (this.pendingRequests.has(requestId)) {
                            const { resolve, timeout } = this.pendingRequests.get(requestId)!;
                            clearTimeout(timeout);
                            this.pendingRequests.delete(requestId);
                            resolve({
                                action: response.action,
                                feedback: response.feedback
                            });
                        }
                        continue;
                    } else if (response.type === 'error') {
                        // Handle error responses
                        const requestId = response.request_id;
                        if (this.pendingRequests.has(requestId)) {
                            const { reject, timeout } = this.pendingRequests.get(requestId)!;
                            clearTimeout(timeout);
                            this.pendingRequests.delete(requestId);
                            reject(new Error(response.message));
                        }
                    }
                } catch (e) {
                    console.log('Response parse error:', e, 'Line length:', line.length, 'First 100 chars:', line.substring(0, 100));
                }
            }
        }
    }
    
    private stopAgent() {
        if (this.pythonProcess) {
            this.pythonProcess.kill();
            this.pythonProcess = undefined;
        }
        this.isInitialized = false;
        this.isFullyReady = false;
        this.initializationPromise = undefined;
    }

    // Message callback for non-streaming messages - returns true if processing completed
    private messageCallback: ((type: string, data: any) => Promise<boolean>) | undefined;

    public setMessageCallback(callback: (type: string, data: any) => Promise<boolean>) {
        this.messageCallback = callback;
    }

    /**
     * Queue and process a message asynchronously
     */
    private async queueAndProcessMessage(type: string, data: any): Promise<void> {
        if (!this.messageCallback) {
            return;
        }

        // Queue the message
        const messageId = this.messageQueue.addMessage({ type, data });

        try {
            // Process the message asynchronously and get completion status
            const isCompleted = await this.messageCallback(type, data);

            // If processing completed, remove it from queue
            if (isCompleted) {
                this.messageQueue.removeMessage(messageId);
            }
        } catch (error) {
            console.error('Error processing message:', error);
            // Remove from queue on error to prevent hanging
            this.messageQueue.removeMessage(messageId);
        }
    }

    /**
     * Check if there are pending messages in the queue
     */
    public hasPendingMessages(): boolean {
        return this.messageQueue.hasPendingMessages();
    }

    dispose() {
        this.stopAgent();
    }
    
    private async sendRequest<T = any>(
        type: string,
        payload: any = {},
        timeout: number = 300000
    ): Promise<T> {
        await this.ensureAgentReady();
        if (!this.pythonProcess) {
            throw new Error('Python process not available');
        }

        const requestId = (++this.requestId).toString();
        const request = { type, request_id: requestId, ...payload };

        return new Promise((resolve, reject) => {
            const timeoutHandle = setTimeout(() => {
                this.pendingRequests.delete(requestId);
                reject(new Error(`Request ${type} timed out after ${timeout/60000} minutes`));
            }, timeout);

            this.pendingRequests.set(requestId, { resolve, reject, timeout: timeoutHandle });
            this.pythonProcess!.stdin!.write(JSON.stringify(request) + '\n');
        });
    }

    private async ensureAgentReady(): Promise<void> {
        // Wait for agent to be ready (initialization should be triggered elsewhere)
        while (!this.isInitialized) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    async sendRegularRequest(message: string, context?: any): Promise<{ response: any, session_id: string, request_id: string }> {
        const timeout = 300000; // 5 minutes for regular requests
        return this.sendRequest('chat', { message, context: context || {} }, timeout);
    }
    
    async handleAutonomousIteration(message: string, context: any): Promise<{ status: string }> {
        // Unified autonomous handler: Handles both initial planning and user feedback
        // Python side uses intent classification to route to appropriate workflow
        const autonomousContext = { ...context, autonomousMode: true };
        return this.sendRequest('chat', { message, context: autonomousContext }, 1800000); // 30 minutes
    }

    async pauseAutonomousExecution(): Promise<string> {
        await this.ensureAgentReady();
        // Pause autonomous execution for feedback
        if (!this.pythonProcess || !this.pythonProcess.stdin) {
            throw new Error('Python process not available');
        }

        const requestId = `pause_${Date.now()}`;
        const request = {
            type: 'pause_autonomous',
            request_id: requestId
        };

        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                this.pendingRequests.delete(requestId);
                reject(new Error('Pause request timed out after 5 seconds'));
            }, 5000);

            this.pendingRequests.set(requestId, { 
                resolve: (result) => {
                    resolve(result.result || result.response || 'Paused successfully');
                }, 
                reject, 
                timeout 
            });

            // Send request to Python process
            this.pythonProcess!.stdin!.write(JSON.stringify(request) + '\n');
        });
    }

    async stopAutonomousExecution(): Promise<string> {
        const result = await this.sendRequest<{result?: string, response?: string}>('stop_autonomous', {}, 15000);
        return result.result || result.response || 'Stopped successfully';
    }

    /**
     * Pause message filtering to block incoming tool messages during feedback interrupts
     */
    pauseToolOutputProcessing(): void {
        this.messageFilteringPaused = true;
    }

    /**
     * Resume message processing after feedback interrupt is handled
     */
    resumeToolOutputProcessing(): void {
        this.messageFilteringPaused = false;
    }

    /**
     * Check if a long-running cell execution should continue or be terminated
     * @param cellCode - The code currently being executed
     * @param elapsedSeconds - Time elapsed since execution started (in seconds)
     * @param partialOutputs - Outputs captured so far
     * @returns Promise<{shouldContinue: boolean, feedback: string}> - decision and feedback
     */
    async checkExecutionProgress(cellCode: string, elapsedSeconds: number, partialOutputs: string): Promise<{shouldContinue: boolean, feedback: string}> {
        await this.ensureAgentReady();

        // Get active task from current notebook context (if available)
        const activeTask = "Unknown task"; // TODO: Extract from current workflow state if needed

        const context = {
            current_cell: cellCode,
            elapsed_time: elapsedSeconds,
            partial_outputs: partialOutputs,
            active_task: activeTask
        };

        try {
            const result = await this.sendRequest<{action: string, feedback: string}>(
                'execution_progress_check',
                { context },
                30000 // 30 second timeout for progress check
            );

            console.log(`Execution monitor: ${result.action} - ${result.feedback}`);
            return {
                shouldContinue: result.action === 'continue',
                feedback: result.feedback
            };
        } catch (error: any) {
            // If monitoring fails, default to continuing (conservative approach)
            console.error('Execution progress check failed, defaulting to continue:', error.message);
            return {
                shouldContinue: true,
                feedback: "Monitoring failed, defaulting to continue"
            };
        }
    }

}

    