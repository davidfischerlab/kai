import * as vscode from 'vscode';
import { KaiAgentProvider as KaiAgentProvider } from './providers/agent-provider';

/**
 * Register essential commands - only those used by buttons and chat interface
 */
export function registerCommands(
    context: vscode.ExtensionContext,
    agentProvider: KaiAgentProvider
): void {

    // Essential UI command - used by status bar
    context.subscriptions.push(
        vscode.commands.registerCommand('kai_agent.showChat', async () => {
            // Focus the chat view
            await vscode.commands.executeCommand('kai_agent_Chat.focus');
        })
    );

    // Cell context menu commands - these appear as buttons on notebook cells
    context.subscriptions.push(
        vscode.commands.registerCommand('kai_agent.fixCell', async () => {
            await processCellCommand(agentProvider, 'fix');
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('kai_agent.deleteCell', async () => {
            await processCellCommand(agentProvider, 'remove');
        })
    );

}

/**
 * Process cell-based commands using the unified chat interface
 */
async function processCellCommand(agentProvider: KaiAgentProvider, action: string): Promise<void> {
    const editor = vscode.window.activeNotebookEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active notebook found');
        return;
    }
    
    try {
        const cell = editor.notebook.cellAt(editor.selection.start);
        const code = cell.document.getText();
        
        // Build prompt based on action
        let prompt = '';
        switch (action) {
            case 'fix':
                prompt = `Fix this code that has an error:\n\n\`\`\`python\n${code}\n\`\`\``;
                break;
            case 'remove':
                // For remove, we could delete the cell directly
                await vscode.commands.executeCommand('notebook.cell.delete');
                return;
            default:
                prompt = `Help with this code:\n\n\`\`\`python\n${code}\n\`\`\``;
        }
        
        const context = {
            action: action,
            cell_type: cell.kind === vscode.NotebookCellKind.Code ? 'code' : 'markdown',
            cell_content: code
        };
        
        const result = await agentProvider.sendRegularRequest(prompt, context);
        
        // Show result in webview
        showResultInWebview(`Kai Agent: ${action}`, typeof result.response === 'string' ? result.response : JSON.stringify(result.response));
        
    } catch (error: any) {
        vscode.window.showErrorMessage(`Error processing cell: ${error.message}`);
    }
}

/**
 * Show result in a webview panel
 */
function showResultInWebview(title: string, content: string): void {
    const panel = vscode.window.createWebviewPanel(
        'kaiAgentResult',
        title,
        vscode.ViewColumn.Beside,
        {}
    );
    
    panel.webview.html = `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>${title}</title>
            <style>
                body { 
                    font-family: var(--vscode-font-family);
                    padding: 20px;
                    background-color: var(--vscode-editor-background);
                    color: var(--vscode-editor-foreground);
                }
                pre { 
                    background-color: var(--vscode-textBlockQuote-background);
                    padding: 10px; 
                    border-radius: 5px;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }
            </style>
        </head>
        <body>
            <pre>${content}</pre>
        </body>
        </html>
    `;
}