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
class GeminiConfig:
    """Gemini API configuration and optimization"""
    API_KEY: str
    MODEL: str = "gemini-2.0-flash"  # Stable model for agentic tasks
    EMBEDDING_MODEL: str = "models/text-embedding-004"

    # Multi-model routing
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
    VECTOR_DIMENSION: int = 768  # Gemini embedding size
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
        self.gemini = GeminiConfig(
            API_KEY=os.getenv("GEMINI_API_KEY", ""),
            LOCAL_LLM_ENABLED=os.getenv("LOCAL_LLM_ENABLED", "false").lower() == "true",
            LOCAL_LLM_ENDPOINT=os.getenv("LOCAL_LLM_ENDPOINT")
        )
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
            "./frozen_resources_v10.json"
        )

    def is_training_mode(self) -> bool:
        return self.MODE == AgentMode.TRAINING

    def is_production_mode(self) -> bool:
        return self.MODE == AgentMode.PRODUCTION

    def validate(self):
        """Validate configuration"""
        if not self.gemini.API_KEY:
            raise ValueError("GEMINI_API_KEY is required")

        if self.is_production_mode() and not os.path.exists(self.frozen_resources_path):
            raise ValueError(f"Frozen resources not found: {self.frozen_resources_path}")

        return True


# Global config instance
config = Config()
