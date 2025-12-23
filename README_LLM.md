# LLM Integration for Intelligent Chatbot

## Quick Start

### Option 1: DeepSeek (Recommended - Most Affordable)

1. **Get API Key**: Sign up at https://platform.deepseek.com
2. **Install dependency**:
   ```bash
   pip install openai
   ```
3. **Create config file** (`data/models/llm_config.json`):
   ```json
   {
     "enabled": true,
     "provider": "deepseek",
     "api_key": "sk-your-key-here",
     "model": "deepseek-chat"
   }
   ```
4. **Restart the web app** - LLM will automatically enhance responses!

### Option 2: OpenAI GPT-3.5-turbo

1. **Get API Key**: https://platform.openai.com
2. **Install dependency**:
   ```bash
   pip install openai
   ```
3. **Create config file** (`data/models/llm_config.json`):
   ```json
   {
     "enabled": true,
     "provider": "openai",
     "api_key": "sk-your-key-here",
     "model": "gpt-3.5-turbo"
   }
   ```

### Option 3: Ollama (Free, Local)

1. **Install Ollama**: https://ollama.ai
2. **Pull a model**:
   ```bash
   ollama pull llama2
   # or
   ollama pull mistral
   ```
3. **Create config file** (`data/models/llm_config.json`):
   ```json
   {
     "enabled": true,
     "provider": "ollama",
     "model": "llama2",
     "base_url": "http://localhost:11434"
   }
   ```

## What It Does

- **Enhanced Query Understanding**: Better intent detection and entity extraction
- **Natural Responses**: More conversational and helpful responses
- **Intelligent Summaries**: AI-generated summaries of analysis results
- **Context Awareness**: Better understanding of follow-up questions

## Cost Comparison

- **DeepSeek**: Very affordable (~$0.14 per 1M tokens)
- **OpenAI GPT-3.5**: Moderate (~$0.50 per 1M tokens)
- **Ollama**: Free (runs locally)

## Disable LLM

Set `"enabled": false` in `llm_config.json` or delete the file. The chatbot will use rule-based responses (still works great!).

## More Details

See `docs/LLM_INTEGRATION_GUIDE.md` for complete documentation.

