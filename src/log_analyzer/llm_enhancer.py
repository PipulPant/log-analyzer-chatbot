#!/usr/bin/env python3
"""
LLM Enhancer for Intelligent Chat
=================================
Integrates language models (DeepSeek, OpenAI, Ollama, etc.) to enhance chatbot intelligence.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMEnhancer:
    """Enhances chatbot responses using language models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM enhancer.
        
        Args:
            config: Configuration dict with provider settings
                {
                    'provider': 'deepseek' | 'openai' | 'ollama' | 'huggingface' | None,
                    'api_key': str (for API providers),
                    'model': str (model name),
                    'base_url': str (for custom endpoints),
                    'enabled': bool
                }
        """
        self.config = config or {}
        self.provider = self.config.get('provider', 'none').lower()
        self.enabled = self.config.get('enabled', False)
        self.model = None
        self.client = None
        
        if self.enabled and self.provider != 'none':
            self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the LLM provider."""
        try:
            if self.provider == 'deepseek':
                self._init_deepseek()
            elif self.provider == 'openai':
                self._init_openai()
            elif self.provider == 'ollama':
                self._init_ollama()
            elif self.provider == 'huggingface':
                self._init_huggingface()
            else:
                logger.warning(f"Unknown LLM provider: {self.provider}")
                self.enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize LLM provider {self.provider}: {e}")
            self.enabled = False
    
    def _init_deepseek(self):
        """Initialize DeepSeek API."""
        try:
            from openai import OpenAI
            
            api_key = self.config.get('api_key') or os.getenv('DEEPSEEK_API_KEY')
            base_url = self.config.get('base_url', 'https://api.deepseek.com')
            model = self.config.get('model', 'deepseek-chat')
            
            if not api_key:
                logger.warning("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")
                self.enabled = False
                return
            
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = model
            logger.info(f"✓ DeepSeek LLM initialized with model: {model}")
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek: {e}")
            self.enabled = False
    
    def _init_openai(self):
        """Initialize OpenAI API."""
        try:
            from openai import OpenAI
            
            api_key = self.config.get('api_key') or os.getenv('OPENAI_API_KEY')
            model = self.config.get('model', 'gpt-3.5-turbo')
            
            if not api_key:
                logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
                self.enabled = False
                return
            
            self.client = OpenAI(api_key=api_key)
            self.model = model
            logger.info(f"✓ OpenAI LLM initialized with model: {model}")
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            self.enabled = False
    
    def _init_ollama(self):
        """Initialize Ollama (local models)."""
        try:
            import requests
            
            base_url = self.config.get('base_url', 'http://localhost:11434')
            model = self.config.get('model', 'llama2')
            
            # Test connection
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.warning(f"Ollama not available at {base_url}")
                self.enabled = False
                return
            
            self.base_url = base_url
            self.model = model
            logger.info(f"✓ Ollama LLM initialized with model: {model}")
        except ImportError:
            logger.warning("requests package not installed. Install with: pip install requests")
            self.enabled = False
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self.enabled = False
    
    def _init_huggingface(self):
        """Initialize Hugging Face transformers."""
        try:
            from transformers import pipeline
            
            model_name = self.config.get('model', 'distilgpt2')
            self.pipeline = pipeline('text-generation', model=model_name)
            self.model = model_name
            logger.info(f"✓ Hugging Face LLM initialized with model: {model_name}")
        except ImportError:
            logger.warning("transformers package not installed. Install with: pip install transformers torch")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face: {e}")
            self.enabled = False
    
    def enhance_query_understanding(
        self, 
        query: str, 
        conversation_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to better understand user query.
        
        Returns enhanced query classification with better intent understanding.
        """
        if not self.enabled:
            return None
        
        try:
            # Build context prompt
            context_info = ""
            if conversation_context:
                prev_failures = conversation_context.get('previous_failures', [])
                if prev_failures:
                    pf = prev_failures[0]
                    context_info = f"""
Previous analysis context:
- Root cause: {pf.get('root_cause_type', 'Unknown')}
- Component: {pf.get('component', 'Unknown')}
- Error: {str(pf.get('message', ''))[:200]}
"""
            
            prompt = f"""You are a log analysis assistant. Analyze this user query and provide structured understanding.

User Query: "{query}"
{context_info}

Provide a JSON response with:
{{
    "intent": "question|follow_up|comparison|analysis",
    "question_type": "what|why|how|explain|other",
    "entities": {{"component": "...", "endpoint": "...", "error_code": "..."}},
    "needs_context": true/false,
    "is_reference": true/false
}}"""
            
            if self.provider in ['deepseek', 'openai']:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful log analysis assistant. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                result = json.loads(response.choices[0].message.content)
                return result
            elif self.provider == 'ollama':
                import requests
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt + "\n\nRespond with JSON only:",
                        "stream": False
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    result = json.loads(response.json()['response'])
                    return result
        except Exception as e:
            logger.warning(f"LLM query understanding failed: {e}")
        
        return None
    
    def enhance_response(
        self,
        base_response: str,
        query: str,
        conversation_context: Optional[Dict] = None,
        analysis_results: Optional[Dict] = None
    ) -> str:
        """
        Use LLM to enhance and make response more natural and intelligent.
        
        Args:
            base_response: The rule-based response
            query: Original user query
            conversation_context: Conversation context
            analysis_results: Analysis results if available
        
        Returns:
            Enhanced, more natural response
        """
        if not self.enabled:
            return base_response
        
        try:
            # Build context
            context_info = ""
            if conversation_context:
                prev_failures = conversation_context.get('previous_failures', [])
                if prev_failures:
                    pf = prev_failures[0]
                    context_info = f"""
Previous Analysis:
- Root Cause: {pf.get('root_cause_type', 'Unknown')}
- Component: {pf.get('component', 'Unknown')}
- Impact: {pf.get('impact_count', 0)} operations
- Error: {str(pf.get('message', ''))[:300]}
"""
            
            if analysis_results:
                failures = analysis_results.get('failures', [])
                primary_failures = analysis_results.get('root_cause_analysis', {}).get('primary_failures', [])
                context_info += f"""
Current Analysis:
- Total Failures: {len(failures)}
- Root Causes: {len(primary_failures)}
"""
            
            prompt = f"""You are an intelligent log analysis assistant. Enhance this response to be more natural, helpful, and conversational while keeping all technical details accurate.

User Query: "{query}"

{context_info}

Current Response:
{base_response}

Provide an enhanced, more natural response that:
1. Is conversational and helpful
2. Maintains all technical accuracy
3. Provides actionable insights
4. References previous context when relevant
5. Uses clear, professional language

Enhanced Response:"""
            
            if self.provider in ['deepseek', 'openai']:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful log analysis assistant. Provide clear, accurate, and actionable responses."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                enhanced = response.choices[0].message.content.strip()
                return enhanced
            elif self.provider == 'ollama':
                import requests
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60
                )
                if response.status_code == 200:
                    enhanced = response.json()['response'].strip()
                    return enhanced
        except Exception as e:
            logger.warning(f"LLM response enhancement failed: {e}, using base response")
        
        return base_response
    
    def generate_intelligent_summary(
        self,
        analysis_results: Dict[str, Any],
        conversation_context: Optional[Dict] = None
    ) -> str:
        """
        Use LLM to generate an intelligent summary of analysis results.
        
        Returns:
            Natural language summary
        """
        if not self.enabled:
            return None
        
        try:
            failures = analysis_results.get('failures', [])
            primary_failures = analysis_results.get('root_cause_analysis', {}).get('primary_failures', [])
            stats = analysis_results.get('stats', {})
            
            # Build summary data
            summary_data = {
                "total_failures": len(failures),
                "root_causes": len(primary_failures),
                "critical_errors": stats.get('critical_errors', 0),
                "http_errors": stats.get('http_errors', 0),
            }
            
            if primary_failures:
                pf = primary_failures[0]
                summary_data["primary_failure"] = {
                    "type": pf.get('root_cause_type', pf.get('type', 'Unknown')),
                    "component": pf.get('component', 'Unknown'),
                    "impact": pf.get('impact_count', 0),
                    "message": str(pf.get('message', ''))[:300]
                }
            
            prompt = f"""Generate a concise, intelligent summary of this log analysis:

{json.dumps(summary_data, indent=2)}

Provide a clear, actionable summary (2-3 sentences) highlighting:
1. The main issue
2. Impact
3. Key recommendation

Summary:"""
            
            if self.provider in ['deepseek', 'openai']:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a log analysis expert. Provide concise, actionable summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=200
                )
                return response.choices[0].message.content.strip()
            elif self.provider == 'ollama':
                import requests
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    return response.json()['response'].strip()
        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}")
        
        return None

