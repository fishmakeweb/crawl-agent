"""
Production Agent Entry Point
Serves frozen resources with optimized performance
"""
import asyncio
import sys
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import os
import logging

from config import Config, LLMProvider
from gemini_client import GeminiClient  # Now supports multiple providers
from agents.shared_crawler_agent import SharedCrawlerAgent

# RAG Services
from services.conversation_rag_service import ConversationRAGService

# Add crawl4ai-agent to path for Kafka publisher
sys.path.append(os.path.join(os.path.dirname(__file__), '../crawl4ai-agent'))
from kafka_publisher import Crawl4AIKafkaPublisher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize
app = FastAPI(title="Production Crawler Agent", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config
config = Config()
config.MODE = "production"
config.validate()

# Initialize LLM client (supports Gemini and external providers)
llm_client = GeminiClient(config.llm)
gemini_client = llm_client  # Keep backward compatibility alias

# Initialize Kafka publisher for real-time progress events
kafka_publisher = Crawl4AIKafkaPublisher()
logger.info(f"Kafka publisher initialized (enabled={kafka_publisher.enabled})")

# Load frozen resources
frozen_resources = {}
if os.path.exists(config.frozen_resources_path):
    with open(config.frozen_resources_path, 'r') as f:
        frozen_resources = json.load(f)
    print(f"✅ Loaded frozen resources version {frozen_resources.get('version', 'unknown')}")
else:
    print(f"⚠️  No frozen resources found at {config.frozen_resources_path}")
    frozen_resources = {
        "version": 0,
        "extraction_prompt": "",
        "crawl_config": {},
        "domain_patterns": {}
    }

# Initialize agent in production mode
agent = SharedCrawlerAgent(gemini_client, mode="production")

# Initialize RAG service for conversation memory
# Detect actual embedding dimension by generating a test embedding
logger.info("Detecting embedding dimension...")
test_embedding = asyncio.run(llm_client.embed("test"))
vector_dim = len(test_embedding)
logger.info(f"✅ Detected {vector_dim}D embeddings for {config.llm.PROVIDER.value}")

# Try to reuse existing Qdrant connection from HybridKnowledgeStore
qdrant_client = None
try:
    from knowledge.hybrid_knowledge_store import HybridKnowledgeStore
    from training_queue_manager import TrainingQueueManager
    
    # Create minimal RL controller for knowledge store (not used in production)
    class MinimalRLController:
        def observe(self, *args, **kwargs): pass
        def get_adjusted_params(self): return {}
    
    knowledge_store = HybridKnowledgeStore(llm_client, MinimalRLController(), config.knowledge_store)
    qdrant_client = knowledge_store.vector_store
    logger.info("✅ Reusing Qdrant connection from HybridKnowledgeStore")
except Exception as e:
    logger.info(f"HybridKnowledgeStore not available ({e}), creating new Qdrant connection")
    from qdrant_client import QdrantClient
    qdrant_client = QdrantClient(
        host=config.knowledge_store.QDRANT_HOST,
        port=config.knowledge_store.QDRANT_PORT
    )
    logger.info(f"✅ Created new Qdrant connection: {config.knowledge_store.QDRANT_HOST}:{config.knowledge_store.QDRANT_PORT}")

rag_service = ConversationRAGService(qdrant_client, llm_client, vector_dimension=vector_dim)
logger.info("✅ RAG service initialized for conversation memory")

print(f"🚀 Production Agent started on port 8000")


# Request models
class CrawlRequest(BaseModel):
    url: str
    prompt: str  # Renamed from user_description for consistency with agent_server
    user_description: Optional[str] = None  # Kept for backward compatibility
    extraction_schema: Optional[Dict[str, Any]] = None
    extract_schema: Optional[Dict[str, Any]] = None  # Alias
    job_id: Optional[str] = None
    user_id: Optional[str] = None
    navigation_steps: Optional[List[Dict[str, Any]]] = None
    max_pages: int = 50


class EmbeddingData(BaseModel):
    """Pre-generated embedding data to eliminate separate .NET API call"""
    embedding_text: str
    embedding_vector: List[float]
    schema_type: str  # "product_list", "article", "generic_data"
    quality_score: float  # 0.0 - 1.0


class NavigationResult(BaseModel):
    """Result of navigation execution"""
    final_url: str
    executed_steps: List[Dict[str, Any]]
    pages_collected: int


class CrawlResponse(BaseModel):
    """Response from crawl operation - OPTIMIZED with pre-generated embeddings"""
    success: bool
    data: List[Dict[str, Any]]
    navigation_result: Optional[NavigationResult] = None
    execution_time_ms: float
    conversation_name: str  # NEW: Pre-generated conversation name
    embedding_data: Optional[EmbeddingData] = None  # NEW: Pre-generated embedding
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ========== RAG Chat Models ==========

class RAGChatRequest(BaseModel):
    """Request for RAG-powered chat"""
    conversation_id: str
    message_id: str
    message: str
    assignment_context: Optional[str] = None
    user_id: str
    top_k: int = 5
    time_window_hours: int = 24


class RAGChatResponse(BaseModel):
    """Response from RAG chat"""
    success: bool
    response: str
    response_id: str
    retrieved_context_count: int
    relevance_scores: List[float]
    error: Optional[str] = None


class IndexCrawlResultRequest(BaseModel):
    """Request to index crawl results for future retrieval"""
    conversation_id: str
    message_id: str
    crawl_job_id: str
    result_summary: str


# ========== Existing Models ==========

class QueryRequest(BaseModel):
    """Request for RAG query"""
    context: str
    query: str


class SummaryRequest(BaseModel):
    job_id: Optional[str] = None
    data: Any
    source: str = "manual"
    prompt: Optional[str] = None


class EmbeddingRequest(BaseModel):
    """Request for generating vector embeddings"""
    text: str
    model: Optional[str] = "models/embedding-001"


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": "production",
        "llm_provider": config.llm.PROVIDER.value,
        "embedding_dimension": llm_client.adapter.default_embedding_dimension,
        "resources_version": frozen_resources.get("version", 0),
        "llm_stats": gemini_client.get_stats()
    }


@app.post("/crawl", response_model=CrawlResponse)
async def crawl(request: CrawlRequest):
    """Execute production crawl with frozen resources and Kafka events"""
    try:
        # Support both 'prompt' and 'user_description' fields
        prompt = request.prompt or request.user_description or ""
        extract_schema = request.extract_schema or request.extraction_schema or {}
        
        task = {
            "url": request.url,
            "prompt": prompt,
            "user_description": prompt,  # For backward compatibility
            "extraction_schema": extract_schema,
            "job_id": request.job_id,
            "user_id": request.user_id,
            "navigation_steps": request.navigation_steps,
            "max_pages": request.max_pages
        }

        # Execute with frozen resources and Kafka publisher
        result = await agent.execute_crawl(
            task, 
            resources=frozen_resources,
            kafka_publisher=kafka_publisher
        )

        # Extract embedding data from result (pre-generated by intelligent_crawl)
        embedding_data = result.get("embedding_data")
        embedding_data_model = None
        if embedding_data:
            embedding_data_model = EmbeddingData(
                embedding_text=embedding_data.get("embedding_text", ""),
                embedding_vector=embedding_data.get("embedding_vector", []),
                schema_type=embedding_data.get("schema_type", "generic_data"),
                quality_score=embedding_data.get("quality_score", 0.5)
            )

        # Extract navigation result
        nav_result = result.get("navigation_result")
        nav_result_model = None
        if nav_result:
            nav_result_model = NavigationResult(
                final_url=nav_result.get("final_url", request.url),
                executed_steps=nav_result.get("executed_steps", []),
                pages_collected=nav_result.get("pages_collected", 1)
            )

        response = CrawlResponse(
            success=result["success"],
            data=result.get("data", []),
            navigation_result=nav_result_model,
            execution_time_ms=result.get("execution_time_ms", 0.0),
            conversation_name=result.get("conversation_name", "Data Collection"),
            embedding_data=embedding_data_model,
            metadata=result.get("metadata", {}),
            error=result.get("error")
        )
        
        # Debug: Log the ACTUAL JSON that will be sent to C#
        response_dict = response.dict()
        logger.info("=" * 80)
        logger.info("📤 JSON BEING SENT TO C#:")
        logger.info(f"   conversation_name: '{response_dict.get('conversation_name')}'")
        logger.info(f"   success: {response_dict.get('success')}")
        logger.info(f"   data: {len(response_dict.get('data', []))} items")
        logger.info(f"   First 200 chars of JSON: {json.dumps(response_dict)[:200]}")
        logger.info("=" * 80)
        
        return response

    except Exception as e:
        logger.error(f"Crawl failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get production agent statistics"""
    return {
        "mode": "production",
        "resources_version": frozen_resources.get("version", 0),
        "frozen_at": frozen_resources.get("frozen_at"),
        "gemini_stats": gemini_client.get_stats(),
        "domain_patterns_count": len(frozen_resources.get("domain_patterns", {}))
    }


@app.post("/query")
async def answer_query(request: QueryRequest):
    """Answer a question based on provided context (RAG)"""
    try:
        # Use agent's base crawler for query answering
        answer = await agent.base_crawler.answer_query(request.context, request.query)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summary")
async def generate_summary_endpoint(request: SummaryRequest):
    """Generate summary and chart recommendations based on data"""
    try:
        result = await agent.base_crawler.generate_summary(request.data, request.prompt)
        return result
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embedding")
async def generate_embedding(request: EmbeddingRequest):
    """
    Generate vector embedding for text using configured LLM provider.
    Returns embedding vector (dimension varies by provider: Gemini=768, OpenAI=1536)
    """
    try:
        # Truncate text if too long
        text = request.text[:10000] if len(request.text) > 10000 else request.text
        
        # Use LLM client's embed method (provider-agnostic)
        embedding = await llm_client.embed(text)
        
        logger.info(f"Generated embedding with {len(embedding)} dimensions using {config.llm.PROVIDER.value}")
        
        return {
            "success": True,
            "embedding": embedding,
            "dimensions": len(embedding),
            "model": request.model,
            "provider": config.llm.PROVIDER.value
        }
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/chat", response_model=RAGChatResponse)
async def rag_chat(request: RAGChatRequest):
    """
    Handle RAG-powered chat with conversation memory.
    Retrieves relevant context from previous messages and generates contextual response.
    """
    try:
        # Generate unique response ID as valid GUID
        response_id = str(uuid.uuid4())
        
        # Step 1: Create conversation collection if not exists
        await rag_service.create_conversation_collection(request.conversation_id)
        
        # Step 2: Retrieve relevant context from past messages
        retrieved_contexts = await rag_service.retrieve_context(
            request.conversation_id,
            request.message,
            top_k=request.top_k,
            time_window_hours=request.time_window_hours
        )
        
        relevance_scores = [ctx['score'] for ctx in retrieved_contexts]
        logger.info(f"Retrieved {len(retrieved_contexts)} relevant messages for conversation {request.conversation_id}")
        if relevance_scores:
            logger.info(f"Relevance scores: {relevance_scores}")
        
        # Step 3: Build enriched prompt with context
        enriched_prompt = rag_service.build_rag_prompt(
            retrieved_context=retrieved_contexts,
            current_query=request.message,
            assignment_context=request.assignment_context
        )
        
        # Step 4: Generate response using LLM
        logger.info(f"Generating RAG response for conversation {request.conversation_id}")
        ai_response = await llm_client.generate(enriched_prompt)
        
        if not ai_response:
            raise ValueError("LLM returned empty response")
        
        # Step 5: Index user message for future retrieval
        await rag_service.index_message(
            conversation_id=request.conversation_id,
            message_id=request.message_id,
            role="user",
            content=request.message,
            message_type="text",
            metadata={"user_id": request.user_id}
        )
        
        # Step 6: Index AI response for future retrieval
        await rag_service.index_message(
            conversation_id=request.conversation_id,
            message_id=response_id,
            role="assistant",
            content=ai_response,
            message_type="text",
            metadata={"user_id": request.user_id}
        )
        
        logger.info(f"Successfully handled RAG chat for conversation {request.conversation_id}")
        
        return RAGChatResponse(
            success=True,
            response=ai_response,
            response_id=response_id,
            retrieved_context_count=len(retrieved_contexts),
            relevance_scores=relevance_scores
        )
        
    except Exception as e:
        logger.error(f"RAG chat failed: {str(e)}", exc_info=True)
        return RAGChatResponse(
            success=False,
            response="",
            response_id="",
            retrieved_context_count=0,
            relevance_scores=[],
            error=str(e)
        )


@app.post("/rag/index-crawl-result")
async def index_crawl_result(request: IndexCrawlResultRequest):
    """
    Index a crawl result for future RAG retrieval.
    Marks message with has_crawl_results=true and crawl_job_id metadata.
    """
    try:
        # Create conversation collection if not exists
        await rag_service.create_conversation_collection(request.conversation_id)
        
        # Index crawl result with special metadata
        await rag_service.index_message(
            conversation_id=request.conversation_id,
            message_id=request.message_id,
            role="assistant",
            content=request.result_summary,
            message_type="crawl_result",
            has_crawl_results=True,
            crawl_job_id=request.crawl_job_id,
            metadata={"crawl_job_id": request.crawl_job_id}
        )
        
        logger.info(f"Indexed crawl result {request.crawl_job_id} for conversation {request.conversation_id}")
        
        return {
            "success": True,
            "message": "Crawl result indexed successfully",
            "conversation_id": request.conversation_id,
            "message_id": request.message_id
        }
        
    except Exception as e:
        logger.error(f"Failed to index crawl result: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/rag/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation's collection from Qdrant.
    Use when conversation is no longer needed to free resources.
    """
    try:
        await rag_service.delete_conversation(conversation_id)
        
        logger.info(f"Deleted conversation collection: {conversation_id}")
        
        return {
            "success": True,
            "message": f"Conversation {conversation_id} deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Production Crawler Agent",
        "version": "1.0.0",
        "mode": "production",
        "status": "running",
        "resources_version": frozen_resources.get("version", 0),
        "endpoints": {
            "health": "/health",
            "crawl": "/crawl (POST)",
            "query": "/query (POST)",
            "summary": "/summary (POST)",
            "embedding": "/embedding (POST)",
            "rag_chat": "/rag/chat (POST)",
            "rag_index_crawl": "/rag/index-crawl-result (POST)",
            "rag_delete": "/rag/conversation/{id} (DELETE)",
            "stats": "/stats (GET)"
        }
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully close Kafka publisher on shutdown"""
    logger.info("Shutting down production agent...")
    if kafka_publisher:
        kafka_publisher.close()
        logger.info("Kafka publisher closed")


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging to suppress health check spam
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=log_config,
        access_log=True
    )
