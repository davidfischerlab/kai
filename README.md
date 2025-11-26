# kai

An agentic AI for building single-cell omics analyses in jupyter notebooks, interfaced via a VS Code extension. Read our [preprint](https://www.biorxiv.org/content/10.1101/2025.11.24.689256v1).

## Usage notices

**Security:**
- kai executes LLM-generated code with full Python permissions, including downloading to, accessing, creating, and deleting files in your file system.

**Costs:**
- kai uses remote services to execute LLMs (e.g., via ollama). Make sure you understand API key usage and cost implications.
- At the time of writing, ollama offers a flatrate monthly subscription (Pro) for US$20 that has allowed us to navigate costs.

**Kernel stability:**
- We have observed some issues with Jupyter kernels in VS Code that lead to kernel crashes upon execution of certain pieces of Python code. By extension, this crashes kai.
- To prevent this, we are temporarily auto-adding a Jupyter notebook cell with conservative kernel settings at the top of any notebook that is loaded while kai is enabled.
- If you want to avoid having this cell added to open notebooks in other projects, disable the kai extension in the VS Code extension menu after usage (see below).

**LLM provider requirements:**
- kai depends on a provider for LLM execution. We have implemented ollama-cloud in this current version. ollama-cloud provides many of the current open-source LLMs.
- To use kai, you will need an [ollama-cloud](https://ollama.com/cloud) subscription and API key.
- Similarly, kai agents could be relayed to OpenAI or Claude via respective API keys and minimal adaptors in the code base.
- You can also use kai with LLMs interfaced via a local ollama instance (this requires a ollama installation on the device and pulling the relevant models). However, overall performance will likely suffer if you use smaller LLMs because of local hardware restrictions. You may be able to run sufficiently large models this way if you locally have access to GPUs.

**Notebook usage:**
- We have run kai with one Jupyter notebook open at a time. We will work on improving this.
- For now, do not close the active notebook and do not open other notebooks while using kai.
- We typically mitigate this by using kai in a separate VS Code window and doing other work in other windows.

**Debugging:**
Kai does not post detailed logs and errors to the chat interface. To monitor kai's actions and troubleshoot issues:
1. Open VS Code Developer Tools: **Help** → **Toggle Developer Tools**
2. Click the **Console** tab in the developer tools panel
3. Monitor this console if kai hangs or behaves unexpectedly

Note that there are two potential sources of errors: the python package and the VS Code extension. kai is set up so that logs and errors from both of these elements land in this console.

## Project state
This is a reproducibility release for the attached preprint. We will fix issues and features in the near future, feel free to add any observations as issues on this repository.

## Repository structure
This repository contains:

- **development_documents:** Curated descriptions for key topics that you can read or provide to software development co-pilots to accelerate orientation in the code base. Note that co-pilots frequently get confused about the python-VS Code extension interface unless guided by prompts or context from such development documents.
- **kai:** The python package.
- **scripts:** Scripts for building the retrieval database from scratch.
- **UI:** User interface with the VS Code extension.

Note that both the python package and the VS Code extension are maintained in this same repository for now so that it's easier to iterate on their interface. They may be split into separate repositories in the future. We may also split the retrieval database (currently hosted in the python package with the agentic system) into a separate python package in the future.

## Quick start

Once installed, the Kai extension appears in the VSCode sidebar (left as a square with "kai" written in the middle). When you click on it, a chat interface opens as a tab in your VS Code window. Before interacting with the interface, load the jupyter notebook you want to work on, it will appear in VS Code next to the chat interface. You can send messages here and use buttons:

### Button toolbar

- **🔧 (fix):** Analyzes and fixes errors in the current cell
- **➡️ (continue):** Generates code to continue your analysis in a new cell
- **💬 (review):** Explains the current cell in the context of your notebook
- **📚 RAG (on/off):** Toggle retrieval-augmented generation (RAG)
- **🤖 Auto (on/off):** Toggle Autonomous mode (agent works through tasks independently)
- **👀 Auto-follow (on/off):** Toggle auto-scroll to cells during autonomous execution (the notebook view follows the actions of kai)

### Chat Interface
The first message in the chat will initialize the agentic system, which takes a bit longer than the subsequent messages, you will see an `initializing` indicator in the chat.

If `🤖 Auto` is off, sending a message will result in a response to which you can then respond again - a standard chat setting.

If `🤖 Auto` is on, kai will make an analysis plan based on your next message and will then address it until it assesses completion. Unless you interrupt kai, it only expects a message from you once it has completed (i.e. when the task list is finished and the blue send button appears again). You will often see green and red messages appear and disappear that help understanding what kai is doing but that will not be persisted in the chat.

### Code Buttons (In chat responses)

When Kai suggests code, you'll see action buttons:

- **+** Insert code below the current cell
- **↻** Replace the current cell with new code
- **▶** Insert and execute code immediately
- **▶** (on replace) Replace and execute code immediately

### Modes

**Regular mode (default):** Ask questions, get suggestions, manually accept code changes

**Autonomous mode (🤖):** Kai works through complex tasks independently:
1. Enable autonomous mode (🤖 button)
2. Describe your analysis goal
3. Kai will iteratively write, execute, and debug code
4. Provide feedback anytime by typing in the chat
5. Click "Stop" to exit autonomous mode

## Installation
kai consists of a python package (the agentic system) and a VS Code extension that need to be installed separately. 

### Prepare python environments
You need a python environment for the agentic system and one for the jupyter notebooks that you will work on - note that they do not have to be the same, we recommend using different ones because the agentic system does not need access to any single-cell analysis specific analysis packages. For example, if you are using mamba:

```bash
mamba create -n kai_agent python=3.11
mamba create -n kai_jupyter python=3.11
```

### Clone the kai repository for installation

```bash
git clone https://github.com/davidfischerlab/kai.git
```

### Install kai python package

Find the kai clone from the previous step:
```bash
# git clone https://github.com/davidfischerlab/kai.git
mamba activate kai_agent
cd kai
pip install -e .
```

### Download retrieval database (recommended)

kai uses a retrieval-augmented generation (RAG) system to access knowledge about the scverse single-cell analysis ecosystem. Download the pre-built retrieval database:

```bash
# Make sure you're still in the kai directory and kai_agent environment is active
python scripts/download_retrieval_data.py 251121
```

**Different version:** To download a different version (e.g., future updates):
```bash
python scripts/download_retrieval_data.py YYMMDD
```

**Verification:** To verify the download and extraction:
```bash
python scripts/download_retrieval_data.py 251121 --verify
```

**Manual download:** To manually download and use a version, obtain an uncompressed directory of the retrieval database (named `retrieval` with sub-directories `chromadb`, `notebook_summaries`, etc.) and copy it into your path for local kai files (per default this is `~/.kai_agent`). You should have a directory structure such as this one: `~/.kai_agent/retrieval/chromadb`.

**Skip this step** if you plan to build the retrieval database from scratch (see scripts in `scripts/` directory) or if you want to use kai without RAG capabilities (note: performance will be significantly reduced).

### Install kai VS Code extension

1. **Navigate to the extension code:**
```bash
# git clone https://github.com/davidfischerlab/kai.git
# cd kai
cd UI/vscode
```

2. **Install Node.js dependencies:**
```bash
npm install
```

3. **Compile TypeScript to JavaScript:**
```bash
npm run compile
```

4. **Package the extension:**
```bash
vsce package
```

You can now install the extension, either from your shell, e.g. here in MacOS, or directly in VS Code. If you are updating the extension, you can force uninstall it first as shown below.

5. **Uninstall old version:**
```bash
/Applications/Visual\ Studio\ Code.app/Contents/Resources/app/bin/code --uninstall-extension local.kai-agent-vscode
```

6. **Install updated version:**
```bash
/Applications/Visual\ Studio\ Code.app/Contents/Resources/app/bin/code --install-extension kai-agent-vscode-0.1.0.vsix --force
```

**Note:** Node.js and npm are required to build the VSCode extension. Install from [nodejs.org](https://nodejs.org/).

**Note on reinstallation:** After reinstallation, reload the VS Code window (toolbar: ">Developer: Reload Window") or restart VS Code. In doubt, uninstall always before reinstalling. You do not need make changes to the python package and environment when updating the extension.

### Configure VS Code extension

After installing the extension, configure it in VSCode settings:

1. **Set Python path** (required):
   - Open VSCode Settings (`Cmd+,` or `Ctrl+,`)
   - Search for "kai agent python"
   - Set `kai_agent.pythonPath` to your kai_agent environment's Python interpreter
   - Example: `/path/to/mambaforge/envs/kai_agent/bin/python`
   - To find the path, run in your terminal:
     ```bash
     mamba activate kai_agent
     which python
     ```

2. **Set Ollama API key** (required):
   - **Option A - VSCode Settings** (recommended):
     - Search for "ollama api key" in VSCode settings
     - Set `kai_agent.ollamaApiKey` to your API key
   - **Option B - Environment Variable**:
     ```bash
     export OLLAMA_API_KEY="your-api-key-here"
     ```
     Add to your `~/.zshrc` or `~/.bashrc` to persist

3. **Set Jupyter kernel** (required for notebooks):
   - Open a Jupyter notebook in VSCode
   - Click the kernel selector in the top-right corner
   - Select the `kai_jupyter` environment you created earlier
   - This ensures kai can execute code in your notebook with the correct dependencies

4. **Reload VSCode** to activate the extension

### Enable and disable kai VS Code extension
Once installed, you can enable and disable the extension without affecting the installation in VS Code's extension menu.

**Note:** Extension installation paths shown above are for MacOS. On Windows, use the path to your VSCode installation (typically `C:\Program Files\Microsoft VS Code\bin\code.cmd`). On Linux, use `code` if it's in your PATH.