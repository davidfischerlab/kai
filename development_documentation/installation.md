# VSCode extension installation during development cycles
Follow these exact steps each time you want to make changes to the VSCode extension.

1. **Compile TypeScript to JavaScript:**
   ```bash
   npm run compile
   ```
2. **Package the extension:**
   ```bash
   vsce package
   ```
3. **Uninstall old version:**
   ```bash
   /Applications/Visual\ Studio\ Code.app/Contents/Resources/app/bin/code --uninstall-extension local.kai-agent-vscode
   ```
4. **Install updated version:**
   ```bash
   /Applications/Visual\ Studio\ Code.app/Contents/Resources/app/bin/code --install-extension kai-agent-vscode-0.1.0.vsix --force
   ```
