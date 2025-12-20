"""
Conversation RAG Service - Separate from HybridKnowledgeStore
Manages conversation-specific embeddings and retrieval
"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue, Range
)
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ConversationRAGService:
    """
    Manages conversation history in Qdrant for RAG-based chat
    Separate from pattern storage (crawler_patterns collection)
    """
    
    def __init__(self, qdrant_client: QdrantClient, llm_client, vector_dimension: int = 768):
        """
        Args:
            qdrant_client: Existing Qdrant connection
            llm_client: GeminiClient for embeddings (supports both Gemini and OpenAI)
            vector_dimension: 768 for Gemini, 1536 for OpenAI
        """
        self.qdrant = qdrant_client
        self.llm_client = llm_client
        self.vector_dimension = vector_dimension
        logger.info(f"ConversationRAGService initialized with {vector_dimension}D vectors")
    
    def _get_collection_name(self, conversation_id: str) -> str:
        """Get collection name for conversation"""
        return f"conv_{conversation_id.replace('-', '_')}"
    
    async def create_conversation_collection(self, conversation_id: str):
        """Create Qdrant collection for a new conversation"""
        collection_name = self._get_collection_name(conversation_id)
        
        try:
            # Check if exists
            collections = self.qdrant.get_collections().collections
            if collection_name in [c.name for c in collections]:
                logger.info(f"Collection {collection_name} already exists")
                return collection_name
            
            # Create new collection
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✅ Created collection: {collection_name}")
            return collection_name
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise
    
    async def index_message(
        self,
        conversation_id: str,
        message_id: str,
        content: str,
        role: str,
        message_type: str = "user_message",
        metadata: Optional[Dict] = None
    ):
        """
        Store a message with its embedding
        
        Args:
            conversation_id: Conversation GUID
            message_id: Message GUID
            content: Message text
            role: "user" | "assistant" | "system"
            message_type: "user_message" | "crawl_request" | "agent_response"
            metadata: Additional data (crawl_job_id, urls, etc.)
        """
        try:
            # Ensure collection exists
            collection_name = await self.create_conversation_collection(conversation_id)
            
            # Generate embedding using LLM client (auto-detects provider)
            embedding = await self.llm_client.embed(content)
            
            # Prepare payload
            payload = {
                "message_id": message_id,
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "message_type": message_type,
                "timestamp": int(datetime.utcnow().timestamp()),
                "char_count": len(content),
                "has_crawl_results": metadata.get("has_crawl_results", False) if metadata else False,
                "crawl_job_id": metadata.get("crawl_job_id") if metadata else None,
                "metadata": metadata or {}
            }
            
            # Upsert to Qdrant
            from uuid import uuid4
            # Generate numeric ID from message_id hash for Qdrant
            point_id = abs(hash(message_id)) % (10 ** 10)
            
            self.qdrant.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            logger.info(f"✅ Indexed message {message_id[:8]}... in {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to index message: {e}")
            raise
    
    async def retrieve_context(
        self,
        conversation_id: str,
        query: str,
        top_k: int = 5,
        time_window_hours: Optional[int] = 24,
        min_score: float = 0.4,  # Lowered from 0.65 for better recall
        filter_role: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant messages for RAG
        
        Args:
            conversation_id: Conversation ID
            query: User's new message
            top_k: Number of results
            time_window_hours: Only search recent messages
            min_score: Minimum cosine similarity
            filter_role: Filter by role (e.g., only "assistant")
        
        Returns:
            List of {id, score, payload} dicts
        """
        try:
            collection_name = self._get_collection_name(conversation_id)
            
            # Check if collection exists
            collections = self.qdrant.get_collections().collections
            if collection_name not in [c.name for c in collections]:
                logger.info(f"Collection {collection_name} does not exist yet")
                return []
            
            # Generate query embedding
            query_vector = await self.llm_client.embed(query)
            
            # Build filter conditions
            filter_conditions = []
            
            if time_window_hours:
                cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
                logger.info(f"Time filter: messages after {cutoff} (timestamp >= {int(cutoff.timestamp())})")
                filter_conditions.append(
                    FieldCondition(
                        key="timestamp",
                        range=Range(gte=int(cutoff.timestamp()))
                    )
                )
            
            if filter_role:
                filter_conditions.append(
                    FieldCondition(
                        key="role",
                        match=MatchValue(value=filter_role)
                    )
                )
            
            query_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Search Qdrant - using scroll + manual scoring for compatibility
            from qdrant_client.models import PointStruct, ScoredPoint
            import numpy as np
            
            # Scroll to get all points (or limit to reasonable number)
            logger.info(f"Scrolling collection {collection_name} with filter: {query_filter}")
            scroll_result = self.qdrant.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=min(top_k * 5, 100),  # Get more than needed for scoring
                with_payload=True,
                with_vectors=True
            )
            
            points = scroll_result[0] if scroll_result else []
            logger.info(f"Scroll returned {len(points)} points")
            
            # Manually calculate cosine similarity scores
            results = []
            query_norm = np.linalg.norm(query_vector)
            
            for point in points:
                point_vector = point.vector
                point_norm = np.linalg.norm(point_vector)
                
                # Cosine similarity
                if query_norm > 0 and point_norm > 0:
                    similarity = np.dot(query_vector, point_vector) / (query_norm * point_norm)
                    logger.info(f"Point {point.id}: similarity={similarity:.3f}, threshold={min_score}, role={point.payload.get('role')}")
                    if similarity >= min_score:
                        results.append({
                            "id": point.id,
                            "score": float(similarity),
                            "payload": point.payload
                        })
            
            # Sort by score and limit
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
            
            scores = [f"{r['score']:.3f}" for r in results]
            logger.info(f"Retrieved {len(results)} relevant messages (scores: {scores})")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []
    
    def build_rag_prompt(
        self,
        retrieved_context: List[Dict],
        current_query: str,
        assignment_context: Optional[str] = None,
        system_prompt: str = "You are a helpful AI assistant for web crawling and data analysis."
    ) -> str:
        """Build enriched prompt with retrieved conversation history"""
        
        parts = []
        
        # System prompt
        parts.append(f"# SYSTEM\n{system_prompt}\n")
        
        # Assignment context (if provided)
        if assignment_context:
            parts.append(f"# ASSIGNMENT CONTEXT\n{assignment_context}\n")
        
        # Retrieved conversation history (sorted by timestamp)
        if retrieved_context:
            parts.append("# RELEVANT CONVERSATION HISTORY\n")
            sorted_context = sorted(retrieved_context, key=lambda x: x["payload"]["timestamp"])
            
            for item in sorted_context:
                role = item["payload"]["role"]
                content = item["payload"]["content"]
                score = item["score"]
                
                role_emoji = "👤" if role == "user" else "🤖"
                parts.append(f"{role_emoji} {role.upper()} (relevance: {score:.2f}):\n{content}\n")
        
        # Current query
        parts.append(f"---\n# CURRENT QUESTION\n{current_query}\n")
        
        # Instruction
        parts.append(
            "\n**INSTRUCTION:** Use the conversation history above to provide a contextually "
            "accurate response. Reference specific information from previous crawls when relevant."
        )
        
        return "\n".join(parts)
    
    async def delete_conversation(self, conversation_id: str):
        """Delete entire conversation collection"""
        try:
            collection_name = self._get_collection_name(conversation_id)
            self.qdrant.delete_collection(collection_name)
            logger.info(f"🗑️ Deleted conversation: {collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete conversation {conversation_id}: {e}")
