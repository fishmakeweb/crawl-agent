"""
Configuration for Self-Learning Agent System
"""
import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class AgentMode(Enum):
    """Agent operation modes"""
    TRAINING = "training"  # Active learning with feedback
    PRODUCTION = "production"  # Fixed resources, optimized


class LLMProvider(Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"  # Google Gemini (default)
    OPENAI = "openai"  # OpenAI GPT models
    AZURE = "azure"  # Azure OpenAI
    TOGETHER = "together"  # Together AI
    GROQ = "groq"  # Groq (fast inference)
    CUSTOM = "custom"  # Custom OpenAI-compatible API


@dataclass
class TrainingConfig:
    """Training-specific configuration"""
    UPDATE_FREQUENCY: int = 5  # Update resources every N rollouts
    MAX_ROLLOUTS_PER_SESSION: int = 100
    PARALLEL_RUNNERS: int = 4

    # Initial resource limits (will be auto-adjusted by RL controller)
    INITIAL_KNOWLEDGE_STORE_MAX_SIZE_GB: float = 2.0
    INITIAL_PATTERN_RETENTION_DAYS: int = 30
    INITIAL_MIN_PATTERN_FREQUENCY: int = 3  # Keep patterns seen 3+ times

    # Learning rates
    RL_LEARNING_RATE: float = 0.1
    META_LEARNING_RATE: float = 0.01

    # Consolidation
    CONSOLIDATION_INTERVAL_HOURS: int = 24
    SIMILARITY_THRESHOLD: float = 0.85
    MIN_CLUSTER_SIZE: int = 2


@dataclass
class LLMConfig:
    """LLM provider configuration (Gemini or external)"""
    # Provider selection
    PROVIDER: LLMProvider = LLMProvider.GEMINI
    
    # Gemini-specific config
    API_KEY: str = ""  # Gemini API key
    MODEL: str = "gemini-2.0-flash-exp"  # Gemini model name
    EMBEDDING_MODEL: str = "models/embedding-001"  # Gemini embedding model

    # External provider config (OpenAI-compatible APIs)
    EXTERNAL_BASE_URL: Optional[str] = None  # e.g., https://one.keyai.shop/v1
    EXTERNAL_API_KEY: Optional[str] = None  # External API key
    EXTERNAL_MODEL_NAME: Optional[str] = None  # e.g., gpt-4o-mini
    EXTERNAL_EMBEDDING_MODEL: Optional[str] = None  # e.g., text-embedding-3-small
    EXTERNAL_EMBEDDING_DIMENSION: int = 1536  # OpenAI default: 1536, Gemini: 768
    
    # General LLM settings
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 8000
    TIMEOUT: float = 120.0

    # Multi-model routing (Gemini only)
    ROUTING_ENABLED: bool = True
    
    # Rate limits (free tier defaults, upgrade for paid)
    MAX_RPM: int = 15  # Requests per minute
    MAX_TPM: int = 1000000  # Tokens per minute (1M for stable models)
    TPM_WARNING_THRESHOLD: float = 0.8  # Warn at 80%
    
    # Complexity thresholds
    COMPLEX_TASK_TOKEN_THRESHOLD: int = 5000
    LARGE_BATCH_THRESHOLD: int = 10

    # Cost optimization
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 1000

    BATCHING_ENABLED: bool = True
    MAX_BATCH_SIZE: int = 8
    BATCH_TIMEOUT_SECONDS: float = 1.0

    # Fallback to local LLM for cost savings
    LOCAL_LLM_ENABLED: bool = False
    LOCAL_LLM_ENDPOINT: Optional[str] = None
    LOCAL_LLM_THRESHOLD_CHARS: int = 5000
    
    # Model definitions with costs and limits
    MODELS: dict = None
    
    def get_adapter_config(self) -> dict:
        """Get configuration for LLM adapter"""
        if self.PROVIDER == LLMProvider.GEMINI:
            return {
                "provider": "gemini",
                "api_key": self.API_KEY,
                "model_name": self.MODEL,
                "embedding_model": self.EMBEDDING_MODEL,
                "temperature": self.TEMPERATURE,
                "max_tokens": self.MAX_TOKENS
            }
        else:
            # External OpenAI-compatible provider
            return {
                "provider": self.PROVIDER.value,
                "base_url": self.EXTERNAL_BASE_URL,
                "api_key": self.EXTERNAL_API_KEY,
                "model_name": self.EXTERNAL_MODEL_NAME or "gpt-4o-mini",
                "embedding_model": self.EXTERNAL_EMBEDDING_MODEL or "text-embedding-3-small",
                "embedding_dimension": self.EXTERNAL_EMBEDDING_DIMENSION,
                "temperature": self.TEMPERATURE,
                "max_tokens": self.MAX_TOKENS,
                "timeout": self.TIMEOUT
            }
    
    def __post_init__(self):
        """Initialize model definitions if not provided"""
        if self.MODELS is None:
            self.MODELS = {
                "gemini-2.0-flash": {
                    "cost_per_1m_input": 0.15,
                    "cost_per_1m_output": 0.60,
                    "latency_avg_ms": 500,
                    "rpm_limit": 15,
                    "tpm_limit": 1000000,
                },
                "gemini-2.0-flash-lite": {
                    "cost_per_1m_input": 0.075,
                    "cost_per_1m_output": 0.30,
                    "latency_avg_ms": 300,
                    "rpm_limit": 30,
                    "tpm_limit": 1500000,
                },
                "gemini-2.5-flash": {
                    "cost_per_1m_input": 0.30,
                    "cost_per_1m_output": 2.50,
                    "latency_avg_ms": 800,
                    "rpm_limit": 10,
                    "tpm_limit": 1000000,
                },
                "gemini-2.5-pro": {
                    "cost_per_1m_input": 1.25,
                    "cost_per_1m_output": 5.00,
                    "latency_avg_ms": 2000,
                    "rpm_limit": 5,
                    "tpm_limit": 500000,
                },
                "learnlm-2.0-flash": {
                    "cost_per_1m_input": 0.25,
                    "cost_per_1m_output": 1.00,
                    "latency_avg_ms": 600,
                    "rpm_limit": 15,
                    "tpm_limit": 500000,
                },
            }


@dataclass
class KnowledgeStoreConfig:
    """Hybrid knowledge store configuration"""
    # Vector store (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    VECTOR_DIMENSION: int = 3072  # External embedding dimension (text-embedding-3-large)
    COLLECTION_NAME: str = "crawler_patterns"

    # Graph store (Neo4j)
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # Cache layer (Redis)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CACHE_TTL_SECONDS: int = 3600

    # Retrieval
    TOP_K_PATTERNS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7


@dataclass
class ServiceConfig:
    """Service endpoints and ports"""
    PRODUCTION_SERVICE_PORT: int = 5014
    TRAINING_SERVICE_PORT: int = 5020

    PRODUCTION_AGENT_ENDPOINT: str = "http://localhost:8004"
    TRAINING_AGENT_ENDPOINT: str = "http://localhost:8091"

    TRAINING_UI_PORT: int = 3001


class Config:
    """Main configuration class"""

    def __init__(self):
        # Load from environment
        self.MODE = AgentMode(os.getenv("MODE", "training"))

        # Initialize sub-configs
        self.training = TrainingConfig()
        
        # Determine LLM provider
        provider_str = os.getenv("LLM_PROVIDER", "gemini").lower()
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            provider = LLMProvider.GEMINI
        
        # Initialize LLM config
        self.llm = LLMConfig(
            PROVIDER=provider,
            API_KEY=os.getenv("GEMINI_API_KEY", ""),
            MODEL=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
            EMBEDDING_MODEL=os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001"),
            EXTERNAL_BASE_URL=os.getenv("EXTERNAL_LLM_BASE_URL"),
            EXTERNAL_API_KEY=os.getenv("EXTERNAL_LLM_API_KEY"),
            EXTERNAL_MODEL_NAME=os.getenv("EXTERNAL_LLM_MODEL_NAME"),
            EXTERNAL_EMBEDDING_MODEL=os.getenv("EXTERNAL_EMBEDDING_MODEL"),
            EXTERNAL_EMBEDDING_DIMENSION=int(os.getenv("EXTERNAL_EMBEDDING_DIMENSION", "1536")),
            TEMPERATURE=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            MAX_TOKENS=int(os.getenv("LLM_MAX_TOKENS", "8000")),
            LOCAL_LLM_ENABLED=os.getenv("LOCAL_LLM_ENABLED", "false").lower() == "true",
            LOCAL_LLM_ENDPOINT=os.getenv("LOCAL_LLM_ENDPOINT")
        )
        
        # Keep backward compatibility alias
        self.gemini = self.llm
        self.knowledge_store = KnowledgeStoreConfig(
            QDRANT_HOST=os.getenv("QDRANT_HOST", "localhost"),
            NEO4J_URI=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            NEO4J_USER=os.getenv("NEO4J_USER", "neo4j"),
            NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD", "password"),
            REDIS_HOST=os.getenv("REDIS_HOST", "localhost")
        )
        self.service = ServiceConfig()

        # Lightning Store (PostgreSQL)
        self.lightning_store_url = os.getenv(
            "LIGHTNING_STORE_URL",
            "postgresql://postgres:password@localhost:5432/lightning"
        )

        # Frozen resources path (production mode)
        self.frozen_resources_path = os.getenv(
            "FROZEN_RESOURCES_PATH",
            "./frozen_resources/latest.json"
        )

    def is_training_mode(self) -> bool:
        return self.MODE == AgentMode.TRAINING

    def is_production_mode(self) -> bool:
        return self.MODE == AgentMode.PRODUCTION

    def validate(self):
        """Validate configuration"""
        # Validate LLM config based on provider
        if self.llm.PROVIDER == LLMProvider.GEMINI:
            if not self.llm.API_KEY:
                raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        else:
            # External provider validation
            if not self.llm.EXTERNAL_BASE_URL:
                raise ValueError(f"EXTERNAL_LLM_BASE_URL is required when LLM_PROVIDER={self.llm.PROVIDER.value}")
            if not self.llm.EXTERNAL_API_KEY:
                raise ValueError(f"EXTERNAL_LLM_API_KEY is required when LLM_PROVIDER={self.llm.PROVIDER.value}")

        if self.is_production_mode() and not os.path.exists(self.frozen_resources_path):
            raise ValueError(f"Frozen resources not found: {self.frozen_resources_path}")

        return True


# Global config instance
config = Config()
