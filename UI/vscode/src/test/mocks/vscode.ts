/**
 * Mock VSCode API for unit testing.
 * Provides minimal stubs for the vscode module APIs used by chat-core.ts
 */

export const window = {
    activeNotebookEditor: undefined as any,
};

export const workspace = {
    getConfiguration: (_section?: string) => ({
        get: (_key: string, defaultValue?: any) => defaultValue,
    }),
    notebookDocuments: [] as any[],
    onDidOpenNotebookDocument: () => ({ dispose: () => {} }),
    onDidCloseNotebookDocument: () => ({ dispose: () => {} }),
};

export class Uri {
    static file(path: string) {
        return { fsPath: path, toString: () => `file://${path}` };
    }
}

export type Webview = {
    html: string;
};
