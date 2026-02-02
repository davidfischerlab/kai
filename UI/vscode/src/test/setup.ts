/**
 * Test setup - registers module mocks before tests run.
 * This file is loaded first by mocha to set up the test environment.
 */

import Module from 'module';
import * as originalFs from 'fs';

// Mock vscode module
const vscodeMock = {
    window: {
        activeNotebookEditor: undefined as any,
    },
    workspace: {
        getConfiguration: (_section?: string) => ({
            get: (_key: string, defaultValue?: any) => defaultValue,
        }),
        notebookDocuments: [] as any[],
    },
    Uri: {
        file: (path: string) => ({ fsPath: path, toString: () => `file://${path}` }),
    },
};

// Mock fs module for template loading
const fsMock = {
    ...originalFs,
    readFileSync: (filePath: string | number | Buffer | URL, options?: any) => {
        if (typeof filePath === 'string' && filePath.includes('chat-template.html')) {
            return '<html><body>Mock Template</body></html>';
        }
        return originalFs.readFileSync(filePath, options);
    },
};

// Use Module._load to intercept module loading (works in modern Node.js)
const originalLoad = (Module as any)._load;
(Module as any)._load = function(request: string, parent: any, isMain: boolean) {
    if (request === 'vscode') {
        return vscodeMock;
    }
    if (request === 'fs') {
        return fsMock;
    }
    return originalLoad.apply(this, [request, parent, isMain]);
};

// Export for use in tests
export { vscodeMock };
