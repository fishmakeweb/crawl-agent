"""
Query Classification for Computational vs Non-Computational Queries

Robust classification using:
1. Keyword-based detection (fast, high precision)
2. Semantic LLM check (slower, handles edge cases)

Examples:
- Computational (trigger code-gen path):
  * "tính trung bình giá"
  * "bao nhiêu sản phẩm dưới 500k"
  * "tổng doanh thu nếu bán hết"
  * "top 5 thương hiệu phổ biến nhất"
  
- Non-computational (normal RAG path):
  * "liệt kê tất cả sản phẩm"
  * "mô tả sản phẩm X"
  * "so sánh hai sản phẩm"
"""
import logging
import re
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query classification types"""
    COMPUTATIONAL = "computational"  # Needs code generation for calculation
    LISTING = "listing"  # Simple listing (no calculation)
    DESCRIPTIVE = "descriptive"  # Description/comparison (no calculation)
    SEARCH = "search"  # Search/filter (no calculation)
    AMBIGUOUS = "ambiguous"  # Unclear, needs semantic check


@dataclass
class ClassificationResult:
    """Result of query classification"""
    query_type: QueryType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    requires_computation: bool
    
    def __str__(self):
        return f"{self.query_type.value} (confidence={self.confidence:.2f}): {self.reasoning}"


class QueryClassifier:
    """
    Classify queries into computational vs non-computational.
    
    Two-stage approach:
    1. Fast keyword-based classification (covers 90% of cases)
    2. Semantic LLM check for ambiguous queries (10% of cases)
    
    Usage:
        classifier = QueryClassifier(llm_client=gemini_client)
        result = await classifier.classify("tính trung bình giá")
        if result.requires_computation:
            # Use code generation path
        else:
            # Use normal RAG path
    """
    
    # Keywords for computational queries (Vietnamese + English)
    COMPUTATIONAL_KEYWORDS = {
        # Aggregation
        'tính', 'tinh', 'calculate', 'compute', 'tổng', 'tong', 'sum', 'total',
        'trung bình', 'trung binh', 'average', 'avg', 'mean',
        
        # Counting
        'bao nhiêu', 'bao nhieu', 'how many', 'count', 'số lượng', 'so luong',
        'đếm', 'dem',
        
        # Statistics
        'cao nhất', 'cao nhat', 'thấp nhất', 'thap nhat', 'highest', 'lowest',
        'max', 'min', 'maximum', 'minimum', 'lớn nhất', 'lon nhat', 'nhỏ nhất', 'nho nhat',
        
        # Ranking
        'top', 'xếp hạng', 'xep hang', 'rank', 'sort', 'sắp xếp', 'sap xep',
        'phổ biến', 'pho bien', 'popular', 'nhiều nhất', 'nhieu nhat', 'most',
        
        # Comparison with numbers
        'dưới', 'duoi', 'below', 'under', 'trên', 'tren', 'above', 'over',
        'từ', 'tu', 'from', 'đến', 'den', 'to', 'giữa', 'giua', 'between',
        
        # Revenue/money calculations
        'doanh thu', 'doanh số', 'revenue', 'sales', 'lợi nhuận', 'loi nhuan', 'profit',
        'giá trị', 'gia tri', 'value', 'chi phí', 'chi phi', 'cost',
        
        # Distribution/visualization (often needs counting)
        'phân bố', 'phan bo', 'distribution', 'tỷ lệ', 'ty le', 'ratio', 'percentage',
        'biểu đồ', 'bieu do', 'chart', 'graph',
    }
    
    # Keywords for listing queries
    LISTING_KEYWORDS = {
        'liệt kê', 'liet ke', 'list', 'danh sách', 'danh sach',
        'tất cả', 'tat ca', 'all', 'toàn bộ', 'toan bo', 'full',
        'show', 'hiển thị', 'hien thi', 'display',
    }
    
    # Keywords for descriptive queries
    DESCRIPTIVE_KEYWORDS = {
        'mô tả', 'mo ta', 'describe', 'description',
        'so sánh', 'so sanh', 'compare', 'comparison',
        'thông tin', 'thong tin', 'information', 'info',
        'chi tiết', 'chi tiet', 'detail', 'details',
        'là gì', 'la gi', 'what is', 'giải thích', 'giai thich', 'explain',
    }
    
    # Keywords for search/filter queries
    SEARCH_KEYWORDS = {
        'tìm', 'tim', 'find', 'search', 'tìm kiếm', 'tim kiem',
        'có', 'co', 'has', 'với', 'voi', 'with',
        'sản phẩm nào', 'san pham nao', 'which product',
    }
    
    def __init__(self, llm_client=None):
        """
        Initialize classifier.
        
        Args:
            llm_client: Optional LLM client for semantic classification (GeminiClient)
        """
        self.llm_client = llm_client
    
    def _keyword_classify(self, query: str) -> Tuple[QueryType, float, str]:
        """
        Fast keyword-based classification.
        
        Returns:
            (query_type, confidence, reasoning)
        """
        query_lower = query.lower()
        
        # Count matches for each category
        computational_score = sum(
            1 for kw in self.COMPUTATIONAL_KEYWORDS 
            if kw in query_lower
        )
        listing_score = sum(
            1 for kw in self.LISTING_KEYWORDS 
            if kw in query_lower
        )
        descriptive_score = sum(
            1 for kw in self.DESCRIPTIVE_KEYWORDS 
            if kw in query_lower
        )
        search_score = sum(
            1 for kw in self.SEARCH_KEYWORDS 
            if kw in query_lower
        )
        
        # Check for numeric patterns (strong signal for computational)
        has_numbers = bool(re.search(r'\d+[k|m|tr|triệu|nghìn|nghin]?', query_lower))
        has_comparison = bool(re.search(r'[><]=?|dưới|duoi|trên|tren|từ|tu|đến|den', query_lower))
        
        if has_numbers or has_comparison:
            computational_score += 2
        
        # Determine category based on scores
        max_score = max(computational_score, listing_score, descriptive_score, search_score)
        
        if max_score == 0:
            return QueryType.AMBIGUOUS, 0.0, "No keywords matched any category"
        
        # Computational has priority if score > 0
        if computational_score > 0:
            confidence = min(0.9, 0.5 + (computational_score * 0.15))
            reasoning = f"Found {computational_score} computational keywords"
            if has_numbers:
                reasoning += " + numeric patterns"
            return QueryType.COMPUTATIONAL, confidence, reasoning
        
        # Otherwise, check other categories
        if listing_score == max_score:
            # Check if it's pure listing (no computational intent)
            if listing_score > computational_score:
                confidence = min(0.85, 0.5 + (listing_score * 0.15))
                return QueryType.LISTING, confidence, f"Found {listing_score} listing keywords"
            else:
                return QueryType.AMBIGUOUS, 0.3, "Listing keywords but ambiguous intent"
        
        if descriptive_score == max_score:
            confidence = min(0.8, 0.5 + (descriptive_score * 0.15))
            return QueryType.DESCRIPTIVE, confidence, f"Found {descriptive_score} descriptive keywords"
        
        if search_score == max_score:
            confidence = min(0.75, 0.5 + (search_score * 0.15))
            return QueryType.SEARCH, confidence, f"Found {search_score} search keywords"
        
        return QueryType.AMBIGUOUS, 0.0, "Could not determine category"
    
    async def _semantic_classify(self, query: str) -> Tuple[QueryType, float, str]:
        """
        LLM-based semantic classification for ambiguous queries.
        
        Returns:
            (query_type, confidence, reasoning)
        """
        if not self.llm_client:
            logger.warning("No LLM client provided for semantic classification")
            return QueryType.AMBIGUOUS, 0.5, "No LLM available for semantic check"
        
        prompt = f"""Analyze this user query and determine if it requires NUMERICAL COMPUTATION/AGGREGATION.

QUERY: "{query}"

COMPUTATIONAL queries need:
- Counting (how many, số lượng)
- Math operations (sum, average, min, max, total)
- Rankings (top N, xếp hạng)
- Statistical analysis (mean, median, distribution)
- Comparisons with numbers (> 500k, dưới 1tr, top 5)
- Revenue/profit calculations

NON-COMPUTATIONAL queries:
- Listing all items (liệt kê tất cả)
- Describing a product (mô tả sản phẩm X)
- Comparing two specific items (so sánh A và B)
- Searching by name/keyword (tìm sản phẩm có từ X)

ANSWER IN THIS EXACT FORMAT:
CLASSIFICATION: [COMPUTATIONAL or NON_COMPUTATIONAL]
CONFIDENCE: [0-100]%
REASONING: [one sentence explanation]

Example:
CLASSIFICATION: COMPUTATIONAL
CONFIDENCE: 95%
REASONING: Query asks to count products under a price threshold (500k).
"""
        
        try:
            response = await self.llm_client.generate(prompt)
            
            # Parse LLM response
            classification_match = re.search(r'CLASSIFICATION:\s*(COMPUTATIONAL|NON_COMPUTATIONAL)', response, re.IGNORECASE)
            confidence_match = re.search(r'CONFIDENCE:\s*(\d+)%?', response)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response)
            
            if not classification_match:
                logger.warning(f"Could not parse LLM classification response: {response}")
                return QueryType.AMBIGUOUS, 0.5, "LLM response unparseable"
            
            is_computational = classification_match.group(1).upper() == "COMPUTATIONAL"
            confidence = int(confidence_match.group(1)) / 100.0 if confidence_match else 0.7
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "LLM semantic analysis"
            
            query_type = QueryType.COMPUTATIONAL if is_computational else QueryType.DESCRIPTIVE
            
            logger.info(f"Semantic classification: {query_type.value} (confidence={confidence:.2f})")
            return query_type, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Semantic classification failed: {e}")
            return QueryType.AMBIGUOUS, 0.5, f"LLM error: {str(e)}"
    
    async def classify(self, query: str) -> ClassificationResult:
        """
        Classify query into computational or non-computational.
        
        Strategy:
        1. Fast keyword-based classification
        2. If confidence >= 0.7, return immediately
        3. Otherwise, use semantic LLM check
        
        Args:
            query: User's question
            
        Returns:
            ClassificationResult with type, confidence, reasoning
        """
        if not query or not query.strip():
            return ClassificationResult(
                query_type=QueryType.AMBIGUOUS,
                confidence=0.0,
                reasoning="Empty query",
                requires_computation=False
            )
        
        # Step 1: Keyword-based classification
        query_type, confidence, reasoning = self._keyword_classify(query)
        
        logger.info(f"Keyword classification: {query_type.value} (confidence={confidence:.2f}) - {reasoning}")
        
        # Step 2: If high confidence or clearly non-computational, return immediately
        if confidence >= 0.7 or query_type in [QueryType.LISTING, QueryType.DESCRIPTIVE, QueryType.SEARCH]:
            requires_computation = (query_type == QueryType.COMPUTATIONAL)
            return ClassificationResult(
                query_type=query_type,
                confidence=confidence,
                reasoning=reasoning,
                requires_computation=requires_computation
            )
        
        # Step 3: Low confidence or ambiguous → use semantic check
        if query_type == QueryType.AMBIGUOUS or confidence < 0.7:
            logger.info("Low confidence, using semantic LLM check...")
            semantic_type, semantic_conf, semantic_reasoning = await self._semantic_classify(query)
            
            # Combine keyword + semantic results (semantic has higher weight)
            if semantic_conf > confidence:
                query_type = semantic_type
                confidence = semantic_conf
                reasoning = f"Semantic: {semantic_reasoning}"
            else:
                # Keep keyword result but note semantic disagreement
                reasoning = f"Keyword: {reasoning} | Semantic: {semantic_reasoning}"
        
        requires_computation = (query_type == QueryType.COMPUTATIONAL)
        
        return ClassificationResult(
            query_type=query_type,
            confidence=confidence,
            reasoning=reasoning,
            requires_computation=requires_computation
        )


# Singleton instance
_classifier: Optional[QueryClassifier] = None

def get_classifier(llm_client=None) -> QueryClassifier:
    """Get or create singleton classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier(llm_client=llm_client)
    return _classifier
