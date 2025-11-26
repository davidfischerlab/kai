# Ollama Turbo API Keys Setup

This document explains how to configure API keys for Ollama Turbo mode, which enables using Ollama's remote processing capability for more powerful models.

## What is Ollama Turbo?

Ollama Turbo allows you to use Ollama's remote servers to run larger, more powerful models (like `gpt-oss:120b`) instead of running models locally. This provides:

- Access to more powerful models (120B parameters vs 20B locally)
- Faster processing on high-end hardware
- No local resource consumption
- Better performance for complex tasks

## Configuration Options

There are two ways to provide your Ollama API key:

### Option 1: VSCode Settings (Recommended)

1. Open VSCode Settings (`Ctrl+,` or `Cmd+,`)
2. Search for "ollama api key"
3. Set `kai_agent.ollamaApiKey` to your API key
4. This keeps the key in VSCode's local settings (not synced or tracked by git)

### Option 2: Environment Variable

Set the environment variable before starting VSCode:

```bash
export OLLAMA_API_KEY="your-api-key-here"
code
```

Or add it to your shell profile (`.bashrc`, `.zshrc`, etc.):
```bash
echo 'export OLLAMA_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

## Using Turbo Mode

Once configured, you can enable Turbo mode in the VSCode chat interface:

1. Open the BioAgent chat panel
2. Click the 💰 (money) button to toggle Turbo mode
3. When enabled, requests will use Ollama's remote servers
4. You'll see console logs indicating when Turbo mode is active:
   ```
   🚀 TURBO MODE ENABLED: Switched to Ollama Turbo (gpt-oss:120b) for this request
   ```

## Priority Order

The system checks for API keys in this order:

1. **VSCode Settings** (`kai_agent.ollamaApiKey`)
2. **Environment Variable** (`OLLAMA_API_KEY`)
3. **No Authentication** (may work if your local Ollama client is authenticated)

## Security Notes

- ✅ **VSCode Settings**: Stored locally, not synced or committed to git
- ✅ **Environment Variables**: Local to your machine, not committed to git  
- ❌ **Never commit API keys to git repositories**
- ❌ **Never share API keys in plain text**

## Cost Considerations

- Turbo mode uses Ollama's remote servers and may incur costs
- Check Ollama's pricing at [ollama.com/pricing](https://ollama.com/pricing)
- The 💰 emoji indicates when you're using paid resources
- You can toggle Turbo on/off per request to control costs
