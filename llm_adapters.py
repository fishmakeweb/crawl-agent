"""
LLM Provider Adapters
Abstraction layer for different LLM providers (Gemini, OpenAI, custom APIs)
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator
import asyncio
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    text: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingResponse:
    """Standardized embedding response"""
    embedding: List[float]
    model: str
    dimensions: int
    tokens_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMAdapter(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = config.get("provider", "unknown")
    
    @abstractmethod
    async def generate_text_async(
        self,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    async def generate_embedding_async(
        self,
        text: str,
        task_type: str = "retrieval_document"
    ) -> EmbeddingResponse:
        """Generate embedding vector"""
        pass
    
    @property
    @abstractmethod
    def supports_json_mode(self) -> bool:
        """Whether provider supports JSON mode"""
        pass
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether provider supports streaming"""
        pass
    
    @property
    @abstractmethod
    def default_embedding_dimension(self) -> int:
        """Default embedding dimension for this provider"""
        pass


class GeminiAdapter(LLMAdapter):
    """Adapter for Google Gemini API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError("google-generativeai package required for Gemini provider")
        
        # Configure Gemini
        api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not provided")
        
        self.genai.configure(api_key=api_key)
        
        # Initialize model
        self.model_name = config.get("model_name", "gemini-2.0-flash-exp")
        self.embedding_model = config.get("embedding_model", "models/embedding-001")
        self.model = self.genai.GenerativeModel(self.model_name)
        
        logger.info(f"✅ GeminiAdapter initialized with model {self.model_name}")
    
    async def generate_text_async(
        self,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Generate text using Gemini"""
        try:
            # Build generation config with max_output_tokens
            # Gemini 1.5 Pro supports up to 65536 output tokens
            generation_config_params = {
                "max_output_tokens": kwargs.pop("max_tokens", 65536)  # Max for Gemini 1.5 Pro
            }
            
            # Handle JSON mode
            if kwargs.get("response_mime_type") or kwargs.get("json_mode"):
                generation_config_params["response_mime_type"] = "application/json"
                kwargs.pop("json_mode", None)
                kwargs.pop("response_mime_type", None)
            
            generation_config = self.genai.GenerationConfig(**generation_config_params)
            
            # Generate content with config
            response = await self.model.generate_content_async(
                prompt,
                generation_config=generation_config,
                **kwargs
            )
            
            return LLMResponse(
                text=response.text,
                model=self.model_name,
                tokens_used=None,  # Gemini doesn't always return token count
                finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
                metadata={"provider": "gemini"}
            )
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
    
    async def generate_embedding_async(
        self,
        text: str,
        task_type: str = "retrieval_document"
    ) -> EmbeddingResponse:
        """Generate embedding using Gemini"""
        try:
            result = await self.genai.embed_content_async(
                model=self.embedding_model,
                content=text,
                task_type=task_type
            )
            
            embedding = result.get("embedding", [])
            
            return EmbeddingResponse(
                embedding=embedding,
                model=self.embedding_model,
                dimensions=len(embedding),
                tokens_used=None,
                metadata={"provider": "gemini", "task_type": task_type}
            )
            
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise
    
    @property
    def supports_json_mode(self) -> bool:
        return True
    
    @property
    def supports_streaming(self) -> bool:
        return True
    
    @property
    def default_embedding_dimension(self) -> int:
        return 768


class OpenAICompatibleAdapter(LLMAdapter):
    """Adapter for OpenAI-compatible APIs (OpenAI, Together, Groq, custom)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        try:
            from openai import AsyncOpenAI
            self.AsyncOpenAI = AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required for OpenAI-compatible providers")
        
        # Get configuration
        base_url = config.get("base_url")
        api_key = config.get("api_key")
        
        if not base_url:
            raise ValueError("base_url required for external LLM provider")
        if not api_key:
            raise ValueError("api_key required for external LLM provider")
        
        # Initialize OpenAI client
        self.client = self.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=config.get("timeout", 120.0)
        )
        
        self.model_name = config.get("model_name", "gpt-4o-mini")
        self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 8000)
        
        # Embedding dimension (varies by provider)
        self._embedding_dimension = config.get("embedding_dimension", 1536)
        
        logger.info(f"✅ OpenAICompatibleAdapter initialized with base_url={base_url}, model={self.model_name}")
    
    async def generate_text_async(
        self,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Generate text using OpenAI-compatible API"""
        try:
            # Extract kwargs
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            json_mode = kwargs.get("json_mode", False)
            
            # Build messages
            messages = [
                {"role": "system", "content": "You are a helpful web scraping and data extraction assistant."},
                {"role": "user", "content": prompt}
            ]
            
            # Build request params
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add JSON mode if supported and requested
            if json_mode:
                request_params["response_format"] = {"type": "json_object"}
            
            # Generate completion
            response = await self.client.chat.completions.create(**request_params)
            
            return LLMResponse(
                text=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "provider": "openai_compatible",
                    "base_url": self.client.base_url
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI-compatible generation failed: {e}")
            raise
    
    async def generate_embedding_async(
        self,
        text: str,
        task_type: str = "retrieval_document"
    ) -> EmbeddingResponse:
        """Generate embedding using OpenAI-compatible API"""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            return EmbeddingResponse(
                embedding=embedding,
                model=response.model,
                dimensions=len(embedding),
                tokens_used=response.usage.total_tokens if response.usage else None,
                metadata={
                    "provider": "openai_compatible",
                    "task_type": task_type
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI-compatible embedding failed: {e}")
            # Some providers don't support embeddings - return zero vector
            logger.warning(f"Falling back to zero vector for embeddings")
            return EmbeddingResponse(
                embedding=[0.0] * self._embedding_dimension,
                model=self.embedding_model,
                dimensions=self._embedding_dimension,
                tokens_used=0,
                metadata={"provider": "openai_compatible", "fallback": True}
            )
    
    @property
    def supports_json_mode(self) -> bool:
        # Most OpenAI-compatible APIs support JSON mode
        return True
    
    @property
    def supports_streaming(self) -> bool:
        return True
    
    @property
    def default_embedding_dimension(self) -> int:
        return self._embedding_dimension


def create_adapter(config: Dict[str, Any]) -> LLMAdapter:
    """Factory function to create appropriate adapter"""
    provider = config.get("provider", "gemini").lower()
    
    if provider == "gemini":
        return GeminiAdapter(config)
    elif provider in ["openai", "azure", "together", "groq", "custom", "external"]:
        return OpenAICompatibleAdapter(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
