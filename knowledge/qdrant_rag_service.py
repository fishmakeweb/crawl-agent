"""
Qdrant RAG Service for Query Answering
Provides semantic search using Qdrant vector database for product queries.

Flow:
.NET → gửi data + question → Python → embed data vào Qdrant → query Qdrant → LLM format → trả về
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
    3. Embed each product using Google's embedding model
    4. Store in Qdrant collection (temporary, per session)
    5. Query using semantic search
    6. Return relevant products for LLM to answer
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
        
        logger.info(f"QdrantRAGService initialized: {self.qdrant_host}:{self.qdrant_port}")
    
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
        Full RAG pipeline: index → search → generate answer.
        
        Args:
            context: JSON string with products/data
            query: User's question
            session_id: Optional session ID
            gemini_model: Gemini model for text generation (legacy, deprecated)
            gemini_client: GeminiClient instance (preferred - supports multiple providers)
            
        Returns:
            Natural language answer
        """
        try:
            # Step 1: Index context data
            collection_name, product_count = await self.index_context(context, session_id)
            
            if product_count == 0:
                return "Không tìm thấy sản phẩm nào trong dữ liệu đã crawl."
            
            # Step 2: Search for relevant products (get ALL indexed products for listing queries)
            # Use product_count as top_k to ensure we retrieve all items
            search_limit = min(product_count, TOP_K_RESULTS)  # Use actual count or max limit
            relevant_products = await self.search(collection_name, query, top_k=search_limit)
            
            if not relevant_products:
                return "Không tìm thấy thông tin liên quan trong dữ liệu đã crawl."
            
            # Step 3: Build context from relevant products
            rag_context_parts = []
            rag_context_parts.append(f"TỔNG SỐ SẢN PHẨM ĐÃ CRAWL: {product_count}")
            rag_context_parts.append("")
            rag_context_parts.append("SẢN PHẨM LIÊN QUAN (từ semantic search):")
            
            # Pre-count brands for accurate visualization
            import re
            brand_counts = {}
            products_without_brand = []
            
            for i, product in enumerate(relevant_products):
                rag_context_parts.append(f"\n[Sản phẩm {i+1}, Độ liên quan: {product['score']:.2f}]")
                rag_context_parts.append(product['text'])
                
                brand_found = False
                brand = None
                
                # Try to get brand from product_data first
                if product['product_data']:
                    rag_context_parts.append(f"Raw data: {json.dumps(product['product_data'], ensure_ascii=False)}")
                    
                    pdata = product['product_data']
                    if isinstance(pdata, dict):
                        brand = pdata.get('brand') or pdata.get('Brand') or pdata.get('thuong_hieu') or pdata.get('Thương hiệu')
                    
                    if brand and isinstance(brand, str) and brand.strip():
                        brand = brand.strip()
                        brand_counts[brand] = brand_counts.get(brand, 0) + 1
                        brand_found = True
                
                # If no brand from product_data, try regex on text
                if not brand_found and product['text']:
                    text = product['text']
                    # Try multiple patterns to extract brand (both with and without Vietnamese diacritics)
                    patterns = [
                        r'Thương hiệu[:\s]+([^\n\|,]+)',  # "Thương hiệu: XXX" (with diacritics)
                        r'Thuong hieu[:\s]+([^\n\|,]+)',  # "Thuong hieu: XXX" (without diacritics)
                        r'\|\s*Thương hiệu[:\s]*([^\n\|]+)\s*\|',  # "| Thương hiệu: XXX |"
                        r'\|\s*Thuong hieu[:\s]*([^\n\|]+)\s*\|',  # "| Thuong hieu: XXX |"
                        r'Brand[:\s]+([^\n\|,]+)',  # "Brand: XXX"
                        r'Nhãn hiệu[:\s]+([^\n\|,]+)',  # "Nhãn hiệu: XXX"
                        r'Nhan hieu[:\s]+([^\n\|,]+)',  # "Nhan hieu: XXX"
                    ]
                    
                    for pattern in patterns:
                        brand_match = re.search(pattern, text, re.IGNORECASE)
                        if brand_match:
                            brand = brand_match.group(1).strip()
                            if brand:
                                brand_counts[brand] = brand_counts.get(brand, 0) + 1
                                brand_found = True
                                break
                
                if not brand_found:
                    products_without_brand.append(i + 1)
            
            # Log for debugging
            logger.info(f"📊 Brand counting: {len(relevant_products)} products processed")
            logger.info(f"📊 Brands found: {len(brand_counts)} unique brands")
            logger.info(f"📊 Brand counts: {brand_counts}")
            logger.info(f"📊 Total counted: {sum(brand_counts.values())}")
            if products_without_brand:
                logger.warning(f"📊 Products without brand: {products_without_brand[:10]}...")  # Show first 10
            
            rag_context = "\n".join(rag_context_parts)
            
            # Step 4: Generate answer with Gemini
            
            # Detect visualization request
            viz_keywords = ['vẽ biểu đồ', 've bieu do', 'chart', 'graph', 'visualization', 'visualize', 'tạo biểu đồ', 'tao bieu do']
            is_viz_request = any(keyword in query.lower() for keyword in viz_keywords)
            
            if is_viz_request:
                # Detect chart type from query
                query_lower = query.lower()
                if any(kw in query_lower for kw in ['tròn', 'pie', 'donut', 'doughnut']):
                    suggested_chart_type = "pie"
                elif any(kw in query_lower for kw in ['đường', 'line', 'trend']):
                    suggested_chart_type = "line"
                else:
                    suggested_chart_type = "bar"
                
                # Check if query is about brands/thương hiệu and we have pre-counted data
                is_brand_query = any(kw in query_lower for kw in ['thương hiệu', 'thuong hieu', 'brand', 'nhãn hiệu', 'nhan hieu'])
                
                if is_brand_query and brand_counts:
                    # Sort by count descending
                    sorted_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
                    labels = [b[0] for b in sorted_brands]
                    data = [b[1] for b in sorted_brands]
                    total_counted = sum(data)
                    
                    # Generate chart JSON directly from pre-counted data
                    chart_json = {
                        "chart_type": suggested_chart_type,
                        "data": data,
                        "labels": labels
                    }
                    
                    logger.info(f"📊 Pre-counted brand data: {len(brand_counts)} brands, {total_counted} products")
                    logger.info(f"📊 Brand counts: {brand_counts}")
                    
                    # Build response with pre-counted accurate data
                    prompt = f"""Bạn là trợ lý AI. Dữ liệu đã được đếm chính xác bằng code:

**TỔNG SỐ SẢN PHẨM:** {product_count}
**SỐ THƯƠNG HIỆU:** {len(brand_counts)}

**THỐNG KÊ CHÍNH XÁC THEO THƯƠNG HIỆU (đã đếm bằng code):**
{json.dumps(brand_counts, ensure_ascii=False, indent=2)}

CÂU HỎI: {query}

**BẮT BUỘC:** Trả về summary ngắn gọn + JSON chính xác sau:

{json.dumps(chart_json, ensure_ascii=False)}

Ví dụ output:
Dựa trên {total_counted} sản phẩm từ {len(brand_counts)} thương hiệu, phân bố như sau:

{json.dumps(chart_json, ensure_ascii=False)}"""
                else:
                    # Fallback: Let LLM analyze (less accurate)
                    prompt = f"""Bạn là trợ lý AI chuyên phân tích và visualize dữ liệu sản phẩm.

{rag_context}

CÂU HỎI: {query}

**BẮT BUỘC: Trả về JSON với định dạng:**

```json
{{
  "chart_type": "{suggested_chart_type}",
  "data": [số1, số2, số3, ...],
  "labels": ["label1", "label2", "label3", ...]
}}
```

**HƯỚNG DẪN:**
1. Đếm CHÍNH XÁC từng sản phẩm theo tiêu chí được yêu cầu
2. KHÔNG được ước lượng - phải đếm từng item một
3. Kiểm tra lại tổng = {product_count}
4. **BẮT BUỘC** return JSON với chart_type, data, labels

**BẮT BUỘC: JSON phải có đủ 3 fields: chart_type, data, labels. KHÔNG có comments trong JSON.**"""
            else:
                # Normal prompt for non-visualization queries
                # Detect if this is a listing request
                list_keywords = ['liệt kê', 'liet ke', 'list', 'danh sách', 'danh sach', 'tất cả', 'tat ca', 'all', 'toàn bộ', 'toan bo', 'đầy đủ', 'day du', 'full']
                is_listing_request = any(keyword in query.lower() for keyword in list_keywords)
                
                if is_listing_request:
                    prompt = f"""Bạn là trợ lý AI chuyên phân tích dữ liệu sản phẩm đã được crawl.

**DỮ LIỆU SẢN PHẨM:**
{rag_context}

**CÂU HỎI:** {query}

**CHỈ THỊ BẮT BUỘC:**
1. LIỆT KÊ TẤT CẢ {product_count} SẢN PHẨM NGAY LẬP TỨC
2. KHÔNG ĐƯỢC hỏi xác nhận, không được viết "Hãy xác nhận", "có muốn xem tiếp không"
3. KHÔNG ĐƯỢC viết "Do độ dài...", "Quá dài...", "Tiếp tục nếu..."
4. KHÔNG ĐƯỢC dừng giữa chừng - PHẢI liệt kê từ sản phẩm #1 đến #{product_count}
5. Format: **[Số thứ tự]. [Tên sản phẩm]** | Thương hiệu: [Brand] | Giá: [Price]₫
6. BẮT ĐẦU NGAY với sản phẩm #1, KẾT THÚC với sản phẩm #{product_count}

**BẮT ĐẦU LIỆT KÊ:**
"""
                else:
                    prompt = f"""Bạn là trợ lý AI chuyên phân tích dữ liệu sản phẩm đã được crawl.
Hãy trả lời câu hỏi dựa trên dữ liệu được cung cấp.

{rag_context}

CÂU HỎI: {query}

HƯỚNG DẪN:
1. Phân tích kỹ dữ liệu để tìm câu trả lời chính xác
2. Nếu câu hỏi yêu cầu TÍNH TOÁN (đếm số lượng, tính trung bình, tổng, max, min):
   - SỬ DỤNG "TỔNG SỐ SẢN PHẨM ĐÃ CRAWL" cho câu hỏi đếm tổng
   - Thực hiện phép tính dựa trên dữ liệu sản phẩm
   - Đưa ra kết quả số cụ thể
3. Nếu không tìm thấy thông tin, hãy nói rõ
4. Trả lời bằng tiếng Việt, chính xác và chi tiết

CÂU TRẢ LỜI:"""

            # Step 4: Generate answer using available LLM
            # Priority: gemini_client (multi-provider) > gemini_model (legacy) > direct Gemini
            if gemini_client:
                # Use GeminiClient which supports multiple providers via adapter
                logger.info("Generating answer via GeminiClient (multi-provider)")
                answer = await gemini_client.generate(prompt)
                
                # DEBUG: Log answer length and preview for visualization queries
                if is_viz_request:
                    logger.info(f"📊 Visualization answer length: {len(answer)} chars")
                    logger.info(f"📊 Answer preview (first 500 chars): {answer[:500]}")
                    logger.info(f"📊 Answer preview (last 200 chars): {answer[-200:]}")
                    
            elif gemini_model:
                import asyncio
                response = await asyncio.to_thread(
                    gemini_model.generate_content, prompt
                )
                answer = response.text if hasattr(response, 'text') else str(response)
            else:
                # Fallback to direct Gemini call
                model = genai.GenerativeModel("models/gemini-2.0-flash")
                response = model.generate_content(prompt)
                answer = response.text if hasattr(response, 'text') else str(response)
            
            # Step 5: Cleanup collection
            self._delete_collection(collection_name)
            
            return answer
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}", exc_info=True)
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
