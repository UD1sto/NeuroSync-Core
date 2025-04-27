import os
import json
import requests
from enum import Enum
import openai
from typing import List, Dict, Any, Union, Optional, Callable

class LLMProvider(Enum):
    LLAMA_3_1 = "llama3_1"
    LLAMA_3_2 = "llama3_2"
    OPENAI = "openai"

class LLMService:
    """Unified interface for LLM services that can switch between local Llama models and OpenAI"""
    
    def __init__(self, provider: Union[str, LLMProvider] = None):
        """
        Initialize the LLM service
        
        Args:
            provider: The LLM provider to use (defaults to value from environment variable LLM_PROVIDER)
        """
        # Get provider from environment if not specified
        if provider is None:
            provider = os.getenv("LLM_PROVIDER", LLMProvider.OPENAI.value)
        
        # Convert string to enum if needed
        if isinstance(provider, str):
            provider = LLMProvider(provider)
            
        self.provider = provider
        
        # Load provider-specific configuration
        if self.provider == LLMProvider.OPENAI:
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            # Initialize the OpenAI client (new style for v1.0.0+)
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == LLMProvider.LLAMA_3_1:
            self.endpoint = os.getenv("LLAMA_3_1_ENDPOINT", "http://127.0.0.1:5050/generate_llama")
            self.stream_endpoint = os.getenv("LLAMA_3_1_STREAM_ENDPOINT", "http://127.0.0.1:5050/generate_stream")
        elif self.provider == LLMProvider.LLAMA_3_2:
            self.endpoint = os.getenv("LLAMA_3_2_ENDPOINT", "http://127.0.0.1:5050/generate_llama")
            self.stream_endpoint = os.getenv("LLAMA_3_2_STREAM_ENDPOINT", "http://127.0.0.1:5050/generate_stream")
    
    def generate(self, 
                messages: List[Dict[str, str]], 
                max_tokens: int = 1000,
                temperature: float = 0.7,
                top_p: float = 0.9) -> str:
        """
        Generate a response from the LLM
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            The generated text
        """
        if self.provider == LLMProvider.OPENAI:
            return self._generate_openai(messages, max_tokens, temperature, top_p)
        elif self.provider in [LLMProvider.LLAMA_3_1, LLMProvider.LLAMA_3_2]:
            return self._generate_llama(messages, max_tokens, temperature, top_p)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_stream(self, 
                       messages: List[Dict[str, str]], 
                       max_tokens: int = 1000,
                       temperature: float = 0.7,
                       top_p: float = 0.9,
                       callback: Optional[Callable[[str], None]] = None):
        """
        Generate a streaming response from the LLM
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            callback: Optional callback function that receives each token
            
        Returns:
            Iterator that yields tokens for LLAMA or entire response for OpenAI
        """
        if self.provider == LLMProvider.OPENAI:
            return self._generate_stream_openai(messages, max_tokens, temperature, top_p, callback)
        elif self.provider in [LLMProvider.LLAMA_3_1, LLMProvider.LLAMA_3_2]:
            return self._generate_stream_llama(messages, max_tokens, temperature, top_p, callback)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_openai(self, messages, max_tokens, temperature, top_p):
        """Generate text using OpenAI API"""
        try:
            # Use new OpenAI client style for v1.0.0+
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_stream_openai(self, messages, max_tokens, temperature, top_p, callback):
        """Generate streaming text using OpenAI API"""
        try:
            # Use new OpenAI client style for v1.0.0+
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True
            )
            
            collected_content = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_content += content
                    if callback:
                        callback(content)
                    yield content
            
            return collected_content
        except Exception as e:
            print(f"OpenAI API streaming error: {e}")
            error_msg = f"Error generating response: {str(e)}"
            if callback:
                callback(error_msg)
            yield error_msg
    
    def _generate_llama(self, messages, max_tokens, temperature, top_p):
        """Generate text using local Llama API"""
        try:
            payload = {
                "messages": messages,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
            response = requests.post(self.endpoint, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            return response_data.get("assistant", {}).get("content", "")
        except Exception as e:
            print(f"Llama API error: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_stream_llama(self, messages, max_tokens, temperature, top_p, callback):
        """Generate streaming text using local Llama API"""
        try:
            payload = {
                "messages": messages,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
            with requests.post(self.stream_endpoint, json=payload, stream=True) as response:
                response.raise_for_status()
                
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        if callback:
                            callback(chunk)
                        yield chunk
        except Exception as e:
            print(f"Llama API streaming error: {e}")
            error_msg = f"Error generating response: {str(e)}"
            if callback:
                callback(error_msg)
            yield error_msg 