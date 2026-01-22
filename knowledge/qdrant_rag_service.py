"""
Qdrant RAG Service for Query Answering
Provides semantic search using Qdrant vector database for product queries.

Flow:
.NET → gửi data + question → Python → embed data vào Qdrant → query Qdrant → LLM format → trả về

Updated Flow (with Code Generation):
1. Classify query (computational vs non-computational)
2. Computational → CodeGeneratorService → accurate numerical results
3. Non-computational → traditional RAG path
"""
import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

import google.generativeai as genai

# Import simplified unified SmartRAG module
try:
    from smart_rag import SmartRAG, get_smart_rag
    SMART_RAG_AVAILABLE = True
except ImportError as e:
    SMART_RAG_AVAILABLE = False
    logger.warning(f"SmartRAG not available: {e}")

logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "models/text-embedding-004"  # Google's embedding model
EMBEDDING_DIMENSION = 768
COLLECTION_PREFIX = "crawl_rag_"
TOP_K_RESULTS = 500  # Max number of results to retrieve (increased for large datasets)


class QdrantRAGService:
    """
    RAG service using Qdrant for semantic search on product data.
    
    Flow:
    1. Receive context data (JSON products) from .NET
    2. Parse products and create text representations
    3. Classify query (computational vs non-computational)
    4a. Computational → CodeGeneratorService (accurate calculations)
    4b. Non-computational → Embed + semantic search
    5. Return relevant products for LLM to answer
    """
    
    def __init__(self, qdrant_host: str = None, qdrant_port: int = None):
        """Initialize Qdrant client and embedding model."""
        self.qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = qdrant_port or int(os.getenv("QDRANT_PORT", "6333"))
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            host=self.qdrant_host,
            port=self.qdrant_port,
            timeout=30
        )
        
        # Configure Gemini for embeddings
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        
        # Simplified SmartRAG component (lazy initialization)
        self._smart_rag: Optional[SmartRAG] = None
        
        logger.info(f"QdrantRAGService initialized: {self.qdrant_host}:{self.qdrant_port}")
        logger.info(f"SmartRAG available: {SMART_RAG_AVAILABLE}")
    
    def _generate_collection_name(self, session_id: str = None) -> str:
        """Generate unique collection name for this session."""
        if session_id:
            # Hash session_id to ensure valid collection name
            hash_suffix = hashlib.md5(session_id.encode()).hexdigest()[:12]
            return f"{COLLECTION_PREFIX}{hash_suffix}"
        # Default: use timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{COLLECTION_PREFIX}{timestamp}"
    
    def _parse_products(self, context: str) -> List[Dict[str, Any]]:
        """Parse context string to extract products."""
        import re
        
        try:
            # Try parsing as JSON
            data = json.loads(context)
            
            # Handle different data formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Look for common keys that contain product arrays
                for key in ["products", "items", "data", "results"]:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                # Single product
                return [data]
            
            return []
        except json.JSONDecodeError:
            # Not JSON, try to extract from text
            logger.warning("Context is not pure JSON, attempting text extraction")
            
            products = []
            
            # Try to find JSON objects in text first - use more sophisticated extraction
            # This regex handles nested objects better
            def extract_json_objects(text):
                """Extract all valid JSON objects from text, handling nested braces."""
                objects = []
                i = 0
                while i < len(text):
                    if text[i] == '{':
                        # Found potential JSON start
                        depth = 0
                        start = i
                        in_string = False
                        escape_next = False
                        
                        while i < len(text):
                            char = text[i]
                            
                            if escape_next:
                                escape_next = False
                            elif char == '\\':
                                escape_next = True
                            elif char == '"':
                                in_string = not in_string
                            elif not in_string:
                                if char == '{':
                                    depth += 1
                                elif char == '}':
                                    depth -= 1
                                    if depth == 0:
                                        # Found complete JSON object
                                        json_str = text[start:i+1]
                                        try:
                                            obj = json.loads(json_str)
                                            objects.append(obj)
                                        except:
                                            pass
                                        break
                            i += 1
                    i += 1
                return objects
            
            json_objects = extract_json_objects(context)
            for obj in json_objects:
                if isinstance(obj, dict) and any(k in obj for k in ['name', 'title', 'product_name', 'productName', 'price', 'brand']):
                    products.append(obj)
            
            if products:
                return products
            
            # If no JSON found, treat as plain text - split by sentences or patterns
            logger.info("No JSON found, treating context as plain text chunks")
            
            # Split by common delimiters: sentences ending with period, numbered items, newlines
            # Pattern: "Sản phẩm 1: ...", "Product 1: ...", or sentences
            lines = re.split(r'(?:\.|\n|\|)', context)
            lines = [l.strip() for l in lines if l.strip() and len(l.strip()) > 5]
            
            # If we have line splits, create pseudo-products from each chunk
            if lines:
                for idx, line in enumerate(lines):
                    products.append({
                        "raw_text": line,
                        "chunk_index": idx + 1,
                        "_is_text_chunk": True
                    })
                logger.info(f"Extracted {len(products)} text chunks from context")
                return products
            
            # Last resort: treat entire context as single item
            if context.strip():
                products.append({
                    "raw_text": context.strip(),
                    "chunk_index": 1,
                    "_is_text_chunk": True
                })
            
            return products
    
    def _product_to_text(self, product: Dict[str, Any], index: int) -> str:
        """Convert a product dict to searchable text."""
        # Handle raw text chunks from non-JSON context
        if product.get("_is_text_chunk"):
            raw = product.get("raw_text", "")
            return f"Nội dung: {raw} | Số thứ tự: {index + 1}"
        
        parts = []
        
        # Common product fields with Vietnamese labels for better search
        name = product.get("name") or product.get("title") or product.get("product_name") or product.get("productName", "")
        brand = product.get("brand") or product.get("manufacturer", "")
        price = product.get("price") or product.get("price_usd") or product.get("price_vnd") or product.get("salePrice", "")
        category = product.get("category", "")
        description = product.get("description", "")
        
        if name:
            parts.append(f"Tên sản phẩm: {name}")
        if brand:
            parts.append(f"Thương hiệu: {brand}")
        if price:
            parts.append(f"Giá: {price}")
        if category:
            parts.append(f"Danh mục: {category}")
        if description:
            parts.append(f"Mô tả: {description[:200]}")  # Limit description length
        
        # Add numeric index for counting
        parts.append(f"Số thứ tự: {index + 1}")
        
        return " | ".join(parts) if parts else json.dumps(product, ensure_ascii=False)
    
    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using Google's embedding model."""
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise
    
    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for query (uses different task_type)."""
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise
    
    def _create_collection(self, collection_name: str) -> bool:
        """Create Qdrant collection if not exists."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if exists:
                # Delete existing collection for fresh data
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def _delete_collection(self, collection_name: str) -> bool:
        """Delete Qdrant collection."""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    async def index_context(self, context: str, session_id: str = None) -> tuple[str, int]:
        """
        Index context data into Qdrant.
        
        Args:
            context: JSON string with products/data
            session_id: Optional session ID for collection naming
            
        Returns:
            Tuple of (collection_name, product_count)
        """
        collection_name = self._generate_collection_name(session_id)
        
        # Create collection
        if not self._create_collection(collection_name):
            raise RuntimeError(f"Failed to create collection {collection_name}")
        
        # Parse products
        products = self._parse_products(context)
        
        if not products:
            logger.warning("No products found in context")
            return collection_name, 0
        
        # Index each product as a point
        logger.info(f"Indexing {len(products)} products into Qdrant")
        points = []
        
        for i, product in enumerate(products):
            text = self._product_to_text(product, i)
            embedding = self._embed_text(text)
            
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": text,
                    "product_data": product,
                    "product_index": i
                }
            )
            points.append(point)
            
            # Log progress every 50 products
            if (i + 1) % 50 == 0:
                logger.info(f"Embedded {i + 1}/{len(products)} products")
        
        # Upsert points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )
        
        logger.info(f"Indexed {len(points)} products to {collection_name}")
        return collection_name, len(points)
    
    async def search(self, collection_name: str, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """
        Search Qdrant collection for relevant products.
        
        Args:
            collection_name: Qdrant collection name
            query: User's question
            top_k: Number of results to return
            
        Returns:
            List of relevant products with scores
        """
        import requests as http_requests
        
        try:
            query_embedding = self._embed_query(query)
            
            # Use REST API directly for compatibility with Qdrant 1.7.x
            # POST /collections/{collection_name}/points/search
            search_url = f"http://{self.qdrant_host}:{self.qdrant_port}/collections/{collection_name}/points/search"
            
            search_body = {
                "vector": query_embedding,
                "limit": top_k,
                "with_payload": True
            }
            
            response = http_requests.post(search_url, json=search_body)
            
            if response.status_code != 200:
                logger.error(f"Search API error: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            results_list = data.get("result", [])
            
            products = []
            for hit in results_list:
                payload = hit.get("payload", {})
                products.append({
                    "text": payload.get("text", ""),
                    "score": hit.get("score", 0),
                    "product_data": payload.get("product_data"),
                    "product_index": payload.get("product_index", -1)
                })
            
            logger.info(f"Found {len(products)} relevant products for query")
            return products
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def answer_with_rag(
        self, 
        context: str, 
        query: str, 
        session_id: str = None,
        gemini_model = None,
        gemini_client = None
    ) -> str:
        """
        Simplified RAG pipeline using unified SmartRAG module.
        
        Simplified Flow (3 lines of code):
        1. Parse products from context
        2. Call SmartRAG.process_query() → handles everything
        3. Return formatted result
        
        NO keyword matching, NO branching, NO complexity.
        Single unified SmartRAG handles: analysis + code generation + execution + formatting.
        
        Args:
            context: JSON string with products/data
            query: User's question
            session_id: Optional session ID
            gemini_model: Gemini model (legacy, deprecated)
            gemini_client: GeminiClient instance (required)
            
        Returns:
            Natural language answer or structured data (JSON for charts)
        """
        try:
            # Step 1: Parse products from context
            products = self._parse_products(context)
            
            if not products:
                return "Không tìm thấy sản phẩm nào trong dữ liệu đã crawl."
            
            logger.info(f"📦 Parsed {len(products)} products from context")
            
            # Ensure LLM client is available
            if not gemini_client:
                logger.error("❌ No LLM client provided")
                return "Lỗi: Không có LLM client để xử lý câu hỏi."
            
            # Check if SmartRAG is available
            if not SMART_RAG_AVAILABLE:
                logger.error("❌ SmartRAG module not available")
                return "Lỗi: Hệ thống SmartRAG chưa được cài đặt."
            
            # Step 2: Initialize SmartRAG (lazy)
            if self._smart_rag is None:
                self._smart_rag = get_smart_rag(llm_client=gemini_client)
                logger.info("✅ SmartRAG initialized")
            
            # Step 3: Process query with SmartRAG (handles everything)
            logger.info(f"🚀 Processing query with SmartRAG: {query[:50]}...")
            
            code_result = await self._smart_rag.process_query(
                query=query,
                products=products
            )
            
            if not code_result.success:
                logger.error(f"❌ SmartRAG failed: {code_result.error}")
                return code_result.display_text  # Error message
            
            logger.info(f"✅ SmartRAG success: {code_result.format_type}, "
                       f"time={code_result.execution_time_ms:.0f}ms")
            
            # Step 4: Return formatted result
            return code_result.display_text
            
        except Exception as e:
            logger.error(f"❌ Simplified RAG pipeline failed: {e}", exc_info=True)
            return f"Lỗi khi xử lý câu hỏi: {str(e)}"
    
    def cleanup_old_collections(self, max_age_hours: int = 24) -> int:
        """
        Cleanup old collections to free up space.
        
        Args:
            max_age_hours: Delete collections older than this
            
        Returns:
            Number of collections deleted
        """
        try:
            collections = self.client.get_collections().collections
            deleted = 0
            
            for collection in collections:
                if collection.name.startswith(COLLECTION_PREFIX):
                    # Extract timestamp if present
                    try:
                        parts = collection.name.replace(COLLECTION_PREFIX, "").split("_")
                        if len(parts) >= 3:
                            date_str = f"{parts[0]}_{parts[1]}"
                            created_at = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                            age = datetime.now() - created_at
                            
                            if age > timedelta(hours=max_age_hours):
                                self._delete_collection(collection.name)
                                deleted += 1
                    except (ValueError, IndexError):
                        # Can't parse date, skip
                        pass
            
            logger.info(f"Cleanup: deleted {deleted} old collections")
            return deleted
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0


# Singleton instance
_rag_service: Optional[QdrantRAGService] = None

def get_rag_service() -> Optional[QdrantRAGService]:
    """Get or create RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        try:
            _rag_service = QdrantRAGService()
        except Exception as e:
            logger.warning(f"Failed to initialize Qdrant RAG service: {e}")
            return None
    return _rag_service
