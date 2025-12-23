# LLM Integration Guide

## Overview

The Log Analyzer chatbot can be enhanced with Language Models (LLMs) to provide more intelligent, natural responses similar to ChatGPT or DeepSeek. The system supports multiple LLM providers and falls back to rule-based responses if LLM is not available.

## Supported Providers

### 1. DeepSeek API
- **Provider**: `deepseek`
- **Model**: `deepseek-chat`
- **Cost**: Affordable pricing
- **Setup**: Requires API key

### 2. OpenAI API
- **Provider**: `openai`
- **Model**: `gpt-3.5-turbo` or `gpt-4`
- **Cost**: Pay-per-use
- **Setup**: Requires API key

### 3. Ollama (Local)
- **Provider**: `ollama`
- **Model**: `llama2`, `mistral`, `codellama`, etc.
- **Cost**: Free (runs locally)
- **Setup**: Requires Ollama installed locally

### 4. Hugging Face Transformers
- **Provider**: `huggingface`
- **Model**: `distilgpt2`, `gpt2`, etc.
- **Cost**: Free (runs locally)
- **Setup**: Requires transformers library

## Configuration

### Option 1: Configuration File (Recommended)

Create `data/models/llm_config.json`:

```json
{
  "enabled": true,
  "provider": "deepseek",
  "api_key": "your-api-key-here",
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com"
}
```

### Option 2: Environment Variables

Set environment variables:

```bash
# For DeepSeek
export DEEPSEEK_API_KEY="your-api-key"

# For OpenAI
export OPENAI_API_KEY="your-api-key"
```

## Setup Instructions

### DeepSeek API

1. **Get API Key**: Sign up at https://platform.deepseek.com
2. **Install Dependencies**:
   ```bash
   pip install openai
   ```
3. **Configure**:
   ```json
   {
     "enabled": true,
     "provider": "deepseek",
     "api_key": "sk-...",
     "model": "deepseek-chat"
   }
   ```

### OpenAI API

1. **Get API Key**: Sign up at https://platform.openai.com
2. **Install Dependencies**:
   ```bash
   pip install openai
   ```
3. **Configure**:
   ```json
   {
     "enabled": true,
     "provider": "openai",
     "api_key": "sk-...",
     "model": "gpt-3.5-turbo"
   }
   ```

### Ollama (Local)

1. **Install Ollama**: https://ollama.ai
2. **Pull Model**:
   ```bash
   ollama pull llama2
   # or
   ollama pull mistral
   ```
3. **Start Ollama**: Usually runs automatically
4. **Install Dependencies**:
   ```bash
   pip install requests
   ```
5. **Configure**:
   ```json
   {
     "enabled": true,
     "provider": "ollama",
     "model": "llama2",
     "base_url": "http://localhost:11434"
   }
   ```

### Hugging Face (Local)

1. **Install Dependencies**:
   ```bash
   pip install transformers torch
   ```
2. **Configure**:
   ```json
   {
     "enabled": true,
     "provider": "huggingface",
     "model": "distilgpt2"
   }
   ```

## How It Works

### 1. Query Understanding Enhancement

When a user asks a question, the LLM helps understand:
- Intent (question type, follow-up, comparison)
- Entities (components, endpoints, error codes)
- Context references ("this", "that", "the error")

### 2. Response Enhancement

The LLM enhances rule-based responses to be:
- More natural and conversational
- Better structured
- More actionable
- Context-aware

### 3. Intelligent Summaries

LLM can generate concise summaries of analysis results.

## Features

### Enhanced Query Understanding
- Better intent detection
- Entity extraction
- Reference resolution
- Context awareness

### Natural Responses
- Conversational tone
- Better explanations
- Actionable recommendations
- Context-aware answers

### Fallback Behavior
- If LLM is unavailable, uses rule-based responses
- Graceful degradation
- No breaking changes

## Example Usage

### Without LLM (Rule-Based)
```
User: "Why did this fail?"
Bot: "Why this failure occurred: Based on root cause analysis..."
```

### With LLM (Enhanced)
```
User: "Why did this fail?"
Bot: "Based on the analysis, the failure occurred because the endpoint 
/mtna-ability/EaiEnvelopeSoapQSService returned a 404 error, indicating 
the service is not deployed or not accessible. This affected 5 operations 
in the transaction flow..."
```

## Cost Considerations

### DeepSeek
- Very affordable
- Good for production use
- Recommended for most users

### OpenAI GPT-3.5-turbo
- Moderate cost
- High quality
- Good balance

### Ollama
- Free (local)
- No API costs
- Requires local resources
- Best for privacy-sensitive environments

### Hugging Face
- Free (local)
- Smaller models
- Fast inference
- Good for testing

## Troubleshooting

### LLM Not Working
1. Check `llm_config.json` exists and is valid
2. Verify API key is set correctly
3. Check network connectivity (for API providers)
4. Review logs for error messages

### Slow Responses
- Use smaller models (GPT-3.5-turbo vs GPT-4)
- Use local models (Ollama) for faster responses
- Adjust `max_tokens` in configuration

### High Costs
- Use DeepSeek (most affordable)
- Use local models (Ollama/HuggingFace)
- Disable LLM for non-critical queries

## Best Practices

1. **Start with DeepSeek**: Most affordable and effective
2. **Use Local Models for Privacy**: Ollama for sensitive data
3. **Monitor Usage**: Track API calls and costs
4. **Fallback Always Works**: System degrades gracefully
5. **Test First**: Try with small queries before production

## Configuration Examples

### Minimal Config (DeepSeek)
```json
{
  "enabled": true,
  "provider": "deepseek",
  "api_key": "sk-..."
}
```

### Full Config (OpenAI)
```json
{
  "enabled": true,
  "provider": "openai",
  "api_key": "sk-...",
  "model": "gpt-3.5-turbo"
}
```

### Local Config (Ollama)
```json
{
  "enabled": true,
  "provider": "ollama",
  "model": "llama2",
  "base_url": "http://localhost:11434"
}
```

## Next Steps

1. Choose a provider based on your needs
2. Create `data/models/llm_config.json`
3. Set API key (if using API provider)
4. Restart the web application
5. Test with sample queries

The chatbot will automatically use LLM enhancement when available!

