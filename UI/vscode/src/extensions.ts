import * as vscode from 'vscode';
import { KaiAgentProvider } from './providers/agent-provider';
import { ChatViewProvider } from './providers/chat-view-provider';
import { registerCommands } from './commands';
import { KernelFix } from './kernel-fix';

let agentProvider: KaiAgentProvider;
let chatProvider: ChatViewProvider;

export function activate(context: vscode.ExtensionContext) {
    console.log('Kai Agent extension is activating...');

    // Initialize the agent provider
    agentProvider = new KaiAgentProvider(context);

    // Initialize and register the chat view provider
    chatProvider = new ChatViewProvider(context.extensionUri, agentProvider);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            ChatViewProvider.viewType,
            chatProvider,
            {
                webviewOptions: {
                    retainContextWhenHidden: true
                }
            }
        )
    );

    // Register commands
    registerCommands(context, agentProvider);

    // Status bar item
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.text = "$(sparkle) Kai Ready";
    statusBarItem.tooltip = "Kai Agent is ready to help";
    statusBarItem.command = 'kai_agent.showChat';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Watch for notebook changes
    context.subscriptions.push(
        vscode.window.onDidChangeActiveNotebookEditor(async (editor) => {
            if (editor) {
                updateStatusBar(statusBarItem, editor);
                // Proactively add kernel fix when notebook is opened
                try {
                    const fixAdded = await KernelFix.ensureFixCellExists();
                    if (fixAdded) {
                        vscode.window.showInformationMessage('🔧 Added kernel stability fix cell to your notebook for macOS compatibility');
                    }
                } catch (error) {
                    // Silent fail - don't block notebook functionality
                    console.log('Kernel fix failed:', error);
                }
            }
        })
    );

    console.log('Kai Agent extension activated!');
}

function updateStatusBar(statusBar: vscode.StatusBarItem, editor: vscode.NotebookEditor) {
    // Update status bar based on notebook state
    statusBar.text = "$(sparkle) Kai Active";
    statusBar.tooltip = `Analyzing: ${editor.notebook.uri.fsPath.split('/').pop()}`;
}

export function deactivate() {
    console.log('Kai Agent extension deactivated');
    if (agentProvider) {
        agentProvider.dispose();
    }
}