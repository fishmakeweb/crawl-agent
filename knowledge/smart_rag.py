"""
Unified Smart RAG Module (Simplified)
======================================

This module replaces the previous 4-module architecture:
- query_analyzer.py (720 LOC) 
- dynamic_prompt_builder.py (670 LOC)
- unified_code_generator.py (550 LOC)
- result_formatter.py (480 LOC)
Total: 2,420 LOC → ~800 LOC (67% reduction)

Key simplifications:
1. Single LLM call for analysis + code generation (was 2-5 calls)
2. 2 dataclasses instead of 6 (QueryIntent, CodeResult)
3. No Enum classes - use simple strings
4. Prompts externalized to prompts.py
5. Direct code execution without layers of abstraction

Flow: Query → LLM (analyze + generate code) → Execute → Format → Return
"""

import os
import re
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

logger = logging.getLogger(__name__)

# Import prompts from external configuration
try:
    from prompts import PROMPTS
except ImportError:
    logger.warning("prompts.py not found, using minimal defaults")
    PROMPTS = {}


# =============================================================================
# DATA MODELS (Simplified from 6 to 2 dataclasses)
# =============================================================================

@dataclass
class QueryIntent:
    """
    Simplified query analysis result.
    
    Replaces: QueryAnalysis (9 fields) + OperationType (Enum) + OutputFormat (Enum)
    """
    # Core fields
    intent_type: str  # "computational", "listing", "comparison", "visualization", "descriptive"
    operation: str    # "sum", "count", "average", "filter", "sort", "group", "find"
    output_format: str  # "number", "list", "chart", "table", "text"
    
    # Optional context
    target_fields: List[str] = field(default_factory=list)  # e.g., ["price", "brand"]
    filters: Dict[str, Any] = field(default_factory=dict)   # e.g., {"brand": "Apple"}
    reasoning: str = ""
    confidence: float = 0.0
    
    # Metadata
    requires_visualization: bool = False
    chart_type: Optional[str] = None  # "bar", "pie", "line", etc.


@dataclass
class CodeResult:
    """
    Code execution result with formatted output.
    
    Replaces: GenerationResult (14 fields) + FormattedResult (6 fields)
    """
    # Execution status
    success: bool
    result: Any  # The actual computed result
    
    # Display
    display_text: str  # Final formatted answer for user
    format_type: str   # "number", "list", "chart", "table", "text"
    
    # Debug info
    code: str = ""
    error: str = ""
    execution_time_ms: float = 0.0
    
    # Optional chart data
    chart_data: Optional[Dict[str, Any]] = None


# =============================================================================
# SAFE CODE EXECUTION (From unified_code_generator.py)
# =============================================================================

class SafeExecutionSandbox:
    """
    Secure Python code execution sandbox using RestrictedPython.
    Simplified from original 150 LOC to ~50 LOC.
    """
    
    def __init__(self):
        try:
            from RestrictedPython import compile_restricted, safe_globals
            from RestrictedPython.Guards import (
                guarded_iter_unpack_sequence,
                safe_iter,
                safe_builtins
            )
            self.compile_restricted = compile_restricted
            self.safe_globals = safe_globals
            self.guarded_iter_unpack_sequence = guarded_iter_unpack_sequence
            self.safe_iter = safe_iter
            self.safe_builtins = safe_builtins
            self.available = True
        except ImportError:
            logger.warning("RestrictedPython not available, using eval() - NOT SAFE FOR PRODUCTION")
            self.available = False
    
    def execute(self, code: str, products: List[Dict[str, Any]], timeout: int = 5) -> Tuple[bool, Any, str]:
        """
        Execute code safely with timeout.
        
        Returns:
            (success, result, error_message)
        """
        try:
            import signal
            from collections import Counter, defaultdict
            
            # Timeout handler
            def timeout_handler(signum, frame):
                raise TimeoutError("Code execution timeout")
            
            # Prepare safe execution context with all needed utilities
            safe_context = {
                'products': products,
                # Built-in functions
                'len': len,
                'sum': sum,
                'min': min,
                'max': max,
                'sorted': sorted,
                'set': set,
                'list': list,
                'dict': dict,
                'str': str,
                'int': int,
                'float': float,
                'round': round,
                'abs': abs,
                'enumerate': enumerate,
                'zip': zip,
                'range': range,
                'any': any,
                'all': all,
                # Collections utilities (no import needed in generated code)
                'Counter': Counter,
                'defaultdict': defaultdict,
                # RestrictedPython guards
                '__builtins__': self.safe_builtins if self.available else {},
                '_getiter_': self.safe_iter if self.available else iter,
                '_iter_unpack_sequence_': self.guarded_iter_unpack_sequence if self.available else iter,
            }
            
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                if self.available:
                    # Use RestrictedPython
                    byte_code = self.compile_restricted(code, '<string>', 'exec')
                    exec(byte_code, safe_context)
                else:
                    # Fallback to regular exec (NOT SAFE)
                    exec(code, safe_context)
                
                # Get result
                result = safe_context.get('result', None)
                
                # Cancel timeout
                signal.alarm(0)
                
                return True, result, ""
                
            except TimeoutError as e:
                signal.alarm(0)
                return False, None, f"Timeout: {str(e)}"
            except Exception as e:
                signal.alarm(0)
                return False, None, f"Execution error: {str(e)}"
                
        except Exception as e:
            return False, None, f"Sandbox error: {str(e)}"


# =============================================================================
# UNIFIED SMART RAG (Main Class)
# =============================================================================

class SmartRAG:
    """
    Unified RAG with dynamic code generation.
    
    Replaces:
    - QueryAnalyzer (200 LOC)
    - DynamicPromptBuilder (250 LOC)
    - UnifiedCodeGenerator (300 LOC)
    - ResultFormatter (200 LOC)
    
    Single LLM call approach:
    1. Analyze query + Generate code in ONE prompt
    2. Execute code safely
    3. Format result simply
    """
    
    def __init__(self, llm_client):
        """
        Initialize SmartRAG.
        
        Args:
            llm_client: GeminiClient or compatible LLM client
        """
        self.llm = llm_client
        self.sandbox = SafeExecutionSandbox()
        self.prompts = PROMPTS
        
        # Simple LRU cache for query analysis (replaces QueryAnalysisCache)
        self._analyze_cache = {}
        self.cache_size = 1000
        
        logger.info("SmartRAG initialized (unified module)")
    
    @lru_cache(maxsize=100)
    def _get_data_schema(self, products_hash: str, products_sample: str) -> Dict[str, str]:
        """
        Auto-detect data schema from products.
        Cached to avoid recomputation.
        
        Args:
            products_hash: Hash of products for cache key
            products_sample: JSON string of first product
            
        Returns:
            Dict mapping field names to types
        """
        try:
            sample = json.loads(products_sample)
            schema = {}
            
            for key, value in sample.items():
                if isinstance(value, (int, float)):
                    schema[key] = "number"
                elif isinstance(value, bool):
                    schema[key] = "boolean"
                elif isinstance(value, str):
                    schema[key] = "string"
                elif isinstance(value, list):
                    schema[key] = "array"
                elif isinstance(value, dict):
                    schema[key] = "object"
                else:
                    schema[key] = "unknown"
            
            return schema
        except:
            return {}
    
    async def process_query(
        self, 
        query: str, 
        products: List[Dict[str, Any]],
        use_cache: bool = True
    ) -> CodeResult:
        """
        Main entry point: Process query end-to-end.
        
        Flow:
        1. Check cache
        2. Single LLM call: analyze + generate code
        3. Execute code
        4. Format result
        5. Return CodeResult
        
        Args:
            query: User's question
            products: List of product dicts
            use_cache: Whether to use analysis cache
            
        Returns:
            CodeResult with formatted answer
        """
        start_time = datetime.now()
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(query, products)
            if cache_key in self._analyze_cache:
                logger.info(f"Cache hit for query: {query[:50]}...")
                cached_result = self._analyze_cache[cache_key]
                return cached_result
        
        try:
            # Step 1: Get data schema
            schema = self._get_data_schema(
                products_hash=str(hash(json.dumps(products[0], sort_keys=True))),
                products_sample=json.dumps(products[0], ensure_ascii=False)
            )
            
            # Step 2: Single LLM call - analyze + generate code
            logger.info("🤖 Calling LLM for analysis + code generation...")
            
            intent, code = await self._analyze_and_generate(query, products, schema)
            
            logger.info(f"📊 Intent: {intent.intent_type}, Operation: {intent.operation}")
            logger.info(f"💻 Generated code ({len(code)} chars)")
            
            # Step 3: Execute code
            logger.info("⚙️ Executing code...")
            
            success, result, error = self.sandbox.execute(code, products)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if not success:
                logger.error(f"❌ Execution failed: {error}")
                return CodeResult(
                    success=False,
                    result=None,
                    display_text=f"Xin lỗi, không thể tính toán kết quả. Lỗi: {error}",
                    format_type="text",
                    code=code,
                    error=error,
                    execution_time_ms=execution_time
                )
            
            logger.info(f"✅ Execution success: {type(result).__name__}")
            
            # Step 4: Format result
            formatted_result = await self._format_result(result, intent, query)
            
            # Create final CodeResult
            code_result = CodeResult(
                success=True,
                result=result,
                display_text=formatted_result["text"],
                format_type=formatted_result["format"],
                code=code,
                execution_time_ms=execution_time,
                chart_data=formatted_result.get("chart_data")
            )
            
            # Cache result
            if use_cache:
                self._update_cache(cache_key, code_result)
            
            logger.info(f"🎉 Query processed in {execution_time:.0f}ms")
            
            return code_result
            
        except Exception as e:
            logger.error(f"❌ SmartRAG error: {e}", exc_info=True)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return CodeResult(
                success=False,
                result=None,
                display_text=f"Lỗi xử lý câu hỏi: {str(e)}",
                format_type="text",
                error=str(e),
                execution_time_ms=execution_time
            )
    
    async def _pre_classify_visualization(self, query: str) -> bool:
        """
        Quick LLM call to determine if query needs chart/visualization.
        
        This is more accurate than keyword matching because the LLM understands
        semantic intent like "thống kê phân bổ" = statistics = needs chart.
        
        Args:
            query: User's query string
            
        Returns:
            True if query needs chart, False for text/list
        """
        prompt = f"""You are a query classifier. Determine if this query needs a CHART/VISUALIZATION or just TEXT/LIST.

Query: "{query}"

**CRITICAL RULES:**
1. If query contains ANY statistics/distribution keywords → ANSWER: CHART
2. If query just asks for list/count/total → ANSWER: TEXT

**CHART Keywords (Statistics/Distribution/Grouping):**
Vietnamese: "thống kê", "phân bổ", "tỷ lệ", "tỷ trọng", "phân phối", "phân tích", "theo nhóm", "theo loại", "theo thương hiệu", "biểu đồ", "so sánh theo"
English: "statistics", "distribution", "breakdown", "ratio", "proportion", "analyze by", "group by", "chart", "graph", "visualization"

**TEXT Keywords (Simple List/Count):**
Vietnamese: "danh sách", "liệt kê", "tìm", "có bao nhiêu", "tổng số", "chi tiết"
English: "list all", "show me", "find", "how many", "total number", "details"

**Pattern Detection:**
- "thống kê" + ANY field → CHART (statistics implies visualization)
- "phân bổ" + ANY field → CHART (distribution implies visualization)
- "tỷ lệ" / "tỷ trọng" → CHART (ratios need visualization)
- "theo" + field name → CHART if combined with stats keywords, else TEXT
- Just "danh sách" / "list" → TEXT

**Examples:**
✅ CHART: "Thống kê phân bổ sản phẩm theo thương hiệu" (statistics + distribution)
✅ CHART: "Phân tích số lượng theo nhóm" (analyze + grouping)
✅ CHART: "Tỷ lệ sản phẩm theo giá" (ratio/proportion)
✅ CHART: "Vẽ biểu đồ tròn theo brand" (explicit chart)
❌ TEXT: "Danh sách tất cả sản phẩm" (just list)
❌ TEXT: "Có bao nhiêu sản phẩm?" (simple count)
❌ TEXT: "Tìm sản phẩm rẻ nhất" (search/find)

**Answer with ONLY one word: CHART or TEXT**

Answer:"""
        
        try:
            logger.info(f"🔍 Classifying query: '{query}'")
            response = await self.llm.generate(prompt)
            answer = response.strip().upper()
            is_chart = "CHART" in answer
            logger.info(f"📈 Pre-classification result: {'CHART' if is_chart else 'TEXT'} (response: {answer[:50]}...)")
            return is_chart
        except Exception as e:
            logger.warning(f"Pre-classification failed: {e}, defaulting to TEXT")
            return False
    
    async def _analyze_and_generate(
        self,
        query: str,
        products: List[Dict[str, Any]],
        schema: Dict[str, str]
    ) -> Tuple[QueryIntent, str]:
        """
        Single LLM call: Analyze query intent + Generate Python code.
        
        This replaces separate calls to QueryAnalyzer + DynamicPromptBuilder + UnifiedCodeGenerator.
        
        Returns:
            (QueryIntent, generated_code)
        """
        # Step 1: Pre-classify if visualization is needed (quick focused LLM call)
        needs_chart = await self._pre_classify_visualization(query)
        logger.info(f"📊 Pre-classification: needs_chart={needs_chart}")
        
        # Step 2: Build unified prompt with visualization hint
        prompt = self._build_unified_prompt(query, products, schema, force_visualization=needs_chart)
        
        # Call LLM (use generate method from GeminiClient)
        response = await self.llm.generate(prompt)
        
        # Parse response (expected format: JSON + code block)
        intent, code = self._parse_llm_response(response)
        
        return intent, code
    
    def _build_unified_prompt(
        self,
        query: str,
        products: List[Dict[str, Any]],
        schema: Dict[str, str],
        force_visualization: bool = False
    ) -> str:
        """
        Build unified prompt for analysis + code generation.
        
        Uses external prompt template from prompts.py.
        
        Args:
            query: User's query
            products: Product data
            schema: Data schema
            force_visualization: If True, adds hint to generate chart
        """
        # Get base prompt template
        template = self.prompts.get("unified_analysis_and_code", self._get_default_prompt())
        
        # Prepare context
        schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
        sample_products = json.dumps(products[:3], indent=2, ensure_ascii=False)
        
        # Format prompt
        prompt = template.format(
            query=query,
            schema=schema_str,
            sample_products=sample_products,
            total_products=len(products)
        )
        
        # Add visualization hint if needed
        if force_visualization:
            prompt += """\n\n**🎯 CRITICAL INSTRUCTION - VISUALIZATION REQUIRED:**
This query has been pre-classified as needing a CHART/VISUALIZATION.
You MUST:
1. Set `"intent_type": "visualization"`
2. Set `"output_format": "chart"`
3. Set `"requires_visualization": true`
4. Set `"chart_type"` to "pie" (for grouping/distribution) or "bar" (for comparisons)
5. Generate code that returns: `result = {{"labels": [...], "values": [...]}}`

Do NOT return a list or text format. The user expects a visual chart.
"""
        
        return prompt
    
    def _get_default_prompt(self) -> str:
        """Fallback prompt if prompts.py not available."""
        return """Analyze the user's query and generate Python code to answer it.

User Query: {query}

Available Data:
- Total products: {total_products}
- Data schema: {schema}
- Sample products: {sample_products}

Your response must be in this EXACT format:

```json
{{
  "intent_type": "computational|listing|comparison|visualization|descriptive",
  "operation": "sum|count|average|filter|sort|group|find",
  "output_format": "number|list|chart|table|text",
  "target_fields": ["field1", "field2"],
  "filters": {{}},
  "reasoning": "Brief explanation",
  "confidence": 0.95,
  "requires_visualization": false,
  "chart_type": null
}}
```

```python
# Python code to compute the answer
# Available: products (list of dicts)
# Must set: result (the computed value)

result = ...  # Your code here
```

CRITICAL:
1. Code must be executable Python
2. Use only: products, len, sum, min, max, sorted, set, list, dict, str, int, float, round, abs
3. Must assign final result to variable named 'result'
4. Keep code simple and efficient
"""
    
    def _parse_llm_response(self, response: str) -> Tuple[QueryIntent, str]:
        """
        Parse LLM response to extract intent JSON and code.
        
        Expected format:
        ```json
        { "intent_type": "...", ... }
        ```
        
        ```python
        result = ...
        ```
        """
        try:
            # Extract JSON block
            json_match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON block found in LLM response")
            
            intent_data = json.loads(json_match.group(1))
            
            # Extract Python code block
            code_match = re.search(r'```python\s*\n(.*?)\n```', response, re.DOTALL)
            if not code_match:
                raise ValueError("No Python code block found in LLM response")
            
            code = code_match.group(1).strip()
            
            # Build QueryIntent
            intent = QueryIntent(
                intent_type=intent_data.get("intent_type", "descriptive"),
                operation=intent_data.get("operation", "find"),
                output_format=intent_data.get("output_format", "text"),
                target_fields=intent_data.get("target_fields", []),
                filters=intent_data.get("filters", {}),
                reasoning=intent_data.get("reasoning", ""),
                confidence=intent_data.get("confidence", 0.0),
                requires_visualization=intent_data.get("requires_visualization", False),
                chart_type=intent_data.get("chart_type")
            )
            
            return intent, code
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response was: {response[:500]}...")
            
            # Fallback: simple intent + error code
            fallback_intent = QueryIntent(
                intent_type="descriptive",
                operation="find",
                output_format="text",
                reasoning=f"Parse error: {str(e)}",
                confidence=0.0
            )
            
            fallback_code = "result = 'Error: Could not parse LLM response'"
            
            return fallback_intent, fallback_code
    
    def _format_special_result(self, result: Dict[str, Any], intent: QueryIntent, query: str) -> Dict[str, Any]:
        """Format special result types like max_product, min_product."""
        result_type = result.get("type")
        name = result.get("name", "Không rõ")
        price = result.get("price", 0)
        
        # Format price
        formatted_price = f"{price:,.0f}₫" if isinstance(price, (int, float)) else str(price)
        
        if result_type == "max_product":
            text = f"🏆 Sản phẩm có giá cao nhất:\n• {name}\n• Giá: {formatted_price}"
        elif result_type == "min_product":
            text = f"💰 Sản phẩm có giá thấp nhất:\n• {name}\n• Giá: {formatted_price}"
        else:
            text = f"📦 {name} - {formatted_price}"
        
        return {
            "text": text,
            "format": "text"
        }
    
    async def _format_result(
        self,
        result: Any,
        intent: QueryIntent,
        query: str
    ) -> Dict[str, Any]:
        """
        Format execution result for display.
        
        Returns:
            {
                "text": "Natural language answer",
                "format": "number|list|chart|table|text",
                "chart_data": {...} (optional)
            }
        """
        # Check if result is a dict with metadata (like max_product, min_product)
        if isinstance(result, dict) and result.get("type") in ["max_product", "min_product"]:
            return self._format_special_result(result, intent, query)
        
        # Check if it's a max/min operation with product dict (even without "type" field)
        if isinstance(result, dict) and intent.operation in ["max", "min"]:
            return await self._format_product_with_llm(result, intent, query)
        
        format_type = intent.output_format
        
        # Format based on type
        if format_type == "number":
            return await self._format_number(result, intent, query)
        
        elif format_type == "list":
            return self._format_list(result, intent, query)
        
        elif format_type == "chart":
            return await self._format_chart(result, intent, query)
        
        elif format_type == "table":
            return self._format_table(result, intent, query)
        
        else:  # text
            return self._format_text(result, intent, query)
    
    async def _format_product_with_llm(
        self,
        product: Dict[str, Any],
        intent: QueryIntent,
        query: str
    ) -> Dict[str, Any]:
        """Use LLM to format product information naturally based on query context."""
        try:
            # Extract product details
            name = product.get("product_name") or product.get("name") or product.get("title", "Unknown")
            price = product.get("price") or product.get("salePrice", 0)
            brand = product.get("brand", "")
            description = product.get("description", "")
            
            # Format price
            if isinstance(price, (int, float)):
                formatted_price = f"{price:,.0f}₫"
            else:
                formatted_price = str(price)
            
            # Build product info string
            product_info = f"Product: {name}\nPrice: {formatted_price}"
            if brand:
                product_info += f"\nBrand: {brand}"
            
            # Map operation to context
            op_context = {
                "max": "highest price / most expensive",
                "min": "lowest price / cheapest",
                "best": "best rated / top quality"
            }.get(intent.operation, "matching")
            
            prompt = f"""Generate a natural, conversational response for this product query.

User Query: "{query}"
Operation: {op_context}

Product Information:
{product_info}

Generate a brief, natural Vietnamese response that:
1. Starts with an appropriate emoji (🏆 for best/max, 💰 for cheapest, ⭐ for ratings)
2. Directly answers the user's question
3. Presents the product information conversationally
4. Keeps it concise (2-3 sentences max)

Examples:
- "🏆 Sản phẩm có giá cao nhất là [name] với giá [price]. Đây là dòng sản phẩm cao cấp từ [brand]."
- "💰 Sản phẩm giá rẻ nhất bạn có thể tìm thấy là [name], chỉ [price]. Lựa chọn tiết kiệm nhưng vẫn đảm bảo chất lượng."

Your response (Vietnamese, 2-3 sentences):""" 
            
            response = await self.llm.generate(prompt, max_tokens=200)
            response_text = response.strip()
            
            # Validate response
            if len(response_text) > 500 or '{' in response_text:
                # Fallback to simple format if LLM returns code or too long
                logger.warning("LLM product response invalid, using fallback")
                if intent.operation == "max":
                    response_text = f"🏆 Sản phẩm có giá cao nhất: {name}\n💵 Giá: {formatted_price}"
                else:
                    response_text = f"💰 Sản phẩm có giá thấp nhất: {name}\n💵 Giá: {formatted_price}"
            
            logger.info(f"📝 Generated product response: {response_text[:80]}...")
            return {
                "text": response_text,
                "format": "text"
            }
            
        except Exception as e:
            logger.error(f"LLM product formatting failed: {e}, using fallback")
            # Fallback formatting
            name = product.get("product_name") or product.get("name", "Unknown")
            price = product.get("price", 0)
            formatted_price = f"{price:,.0f}₫" if isinstance(price, (int, float)) else str(price)
            
            if intent.operation == "max":
                text = f"🏆 Sản phẩm có giá cao nhất: {name}\n💵 Giá: {formatted_price}"
            else:
                text = f"💰 Sản phẩm có giá thấp nhất: {name}\n💵 Giá: {formatted_price}"
            
            return {
                "text": text,
                "format": "text"
            }
    
    async def _generate_number_insight(
        self,
        query: str,
        num: float,
        formatted_value: str,
        operation: str,
        is_price: bool
    ) -> str:
        """Use LLM to generate contextual insight for numerical results."""
        try:
            # Get additional context if available
            context_info = ""
            if hasattr(self, '_last_products_data') and self._last_products_data:
                prices = [p.get('price', 0) or p.get('salePrice', 0) for p in self._last_products_data if p.get('price') or p.get('salePrice')]
                if prices and is_price:
                    min_price = min(prices)
                    max_price = max(prices)
                    count = len(prices)
                    context_info = f"\nTotal Products: {count}\nPrice Range: {min_price:,.0f}₫ - {max_price:,.0f}₫"
            
            # Map operation to Vietnamese
            op_map = {
                "average": "trung bình",
                "sum": "tổng",
                "count": "số lượng",
                "max": "cao nhất",
                "min": "thấp nhất"
            }
            op_vn = op_map.get(operation, operation)
            
            prompt = f"""Generate a natural, insightful response for this statistical query.

User Query: "{query}"
Operation: {op_vn}
Result: {formatted_value}
{context_info}

Generate a 1-2 sentence response that:
1. Starts with an appropriate emoji (📊/💰/📦/🔝/💵)
2. States the result clearly
3. Adds contextual insight (e.g., comparison to range, what this means)
4. Uses natural Vietnamese language

Examples:
- "📊 Giá trung bình là 502,900₫. Giá dao động trong khoảng 200,000₫ - 1,500,000₫, cho thấy sự đa dạng về phân khúc giá."
- "📦 Tổng số 19 sản phẩm được tìm thấy, phù hợp với nhu cầu tìm kiếm sản phẩm chăm sóc da."
- "🔝 Sản phẩm đắt nhất có giá 1,500,000₫, thuộc phân khúc cao cấp."

Your response (1-2 sentences only):"""
            
            response = await self.llm.generate(prompt, max_tokens=150)
            insight = response.strip()
            
            # Validation
            if len(insight) > 400 or '```' in insight:
                logger.warning("LLM number insight too long or contains code, using simple format")
                return self._simple_number_format(formatted_value, operation)
            
            logger.info(f"📝 Generated number insight: {insight[:100]}...")
            return insight
            
        except Exception as e:
            logger.error(f"LLM number insight generation failed: {e}, using fallback")
            return self._simple_number_format(formatted_value, operation)
    
    def _simple_number_format(self, formatted_value: str, operation: str) -> str:
        """Simple fallback formatting for numbers."""
        emoji_map = {
            "average": "📊",
            "sum": "💰",
            "count": "📦",
            "max": "🔝",
            "min": "💵"
        }
        text_map = {
            "average": "Giá trung bình",
            "sum": "Tổng giá trị",
            "count": "Tổng số",
            "max": "Giá trị cao nhất",
            "min": "Giá trị thấp nhất"
        }
        emoji = emoji_map.get(operation, "📊")
        text = text_map.get(operation, "Kết quả")
        return f"{emoji} {text}: {formatted_value}"
    
    async def _format_number(self, result: Any, intent: QueryIntent, query: str) -> Dict[str, Any]:
        """Format numerical result with LLM-generated contextual insights."""
        try:
            num = float(result)
            
            # Format with thousands separator
            if num >= 1000:
                formatted = f"{num:,.0f}"
            elif num != int(num):
                formatted = f"{num:,.2f}"
            else:
                formatted = f"{int(num)}"
            
            # Detect if it's a price query
            price_keywords = ["giá", "price", "tiền", "đồng", "vnd", "usd", "cost"]
            is_price = any(keyword in query.lower() for keyword in price_keywords)
            
            # Add currency for prices
            if is_price and "count" not in intent.operation and "số lượng" not in query.lower():
                formatted_value = f"{formatted}₫"
            else:
                formatted_value = formatted
            
            # Generate LLM-based contextual response
            text = await self._generate_number_insight(query, num, formatted_value, intent.operation, is_price)
            
            return {
                "text": text,
                "format": "number"
            }
        except Exception as e:
            logger.error(f"Number formatting error: {e}")
            return {
                "text": str(result),
                "format": "text"
            }
    
    def _format_list(self, result: Any, intent: QueryIntent, query: str) -> Dict[str, Any]:
        """Format list result with descriptive header and product details."""
        if not isinstance(result, (list, tuple)):
            result = [result]
        
        # Check if query asks for prices
        price_query = any(keyword in query.lower() for keyword in ["giá", "price", "tiền"])
        
        # Detect query intent for header
        is_sorted_by_price = any(keyword in query.lower() for keyword in ["đắt", "rẻ", "cao", "thấp", "expensive", "cheap", "highest", "lowest"])
        
        # Format as numbered list with details
        lines = []
        prices_in_list = []
        
        for i, item in enumerate(result, 1):
            if isinstance(item, dict):
                # Format dict with name and price
                name = item.get("name") or item.get("title") or "Sản phẩm"
                price = item.get("price") or item.get("salePrice")
                
                if price:
                    prices_in_list.append(price if isinstance(price, (int, float)) else 0)
                
                if price and name:
                    lines.append(f"{i}. {name} - {price:,.0f}₫" if isinstance(price, (int, float)) else f"{i}. {name} - {price}")
                elif name:
                    lines.append(f"{i}. {name}")
                else:
                    lines.append(f"{i}. {str(item)[:100]}")
            else:
                # Simple value - check if it's a price
                if price_query and isinstance(item, (int, float)):
                    lines.append(f"{i}. {item:,.0f}₫")
                    prices_in_list.append(item)
                else:
                    lines.append(f"{i}. {item}")
        
        # Create contextual header based on query intent
        header = ""
        if is_sorted_by_price and prices_in_list:
            if "đắt" in query.lower() or "cao" in query.lower() or "highest" in query.lower() or "expensive" in query.lower():
                min_p, max_p = min(prices_in_list), max(prices_in_list)
                header = f"📋 Top {len(result)} sản phẩm đắt nhất (giá từ {min_p:,.0f}₫ - {max_p:,.0f}₫):\n"
            elif "rẻ" in query.lower() or "thấp" in query.lower() or "lowest" in query.lower() or "cheap" in query.lower():
                min_p, max_p = min(prices_in_list), max(prices_in_list)
                header = f"📋 Top {len(result)} sản phẩm rẻ nhất (giá từ {min_p:,.0f}₫ - {max_p:,.0f}₫):\n"
            else:
                header = f"📋 Danh sách {len(result)} sản phẩm:\n"
        else:
            header = f"📋 Danh sách {len(result)} kết quả:\n"
        
        text = header + "\n".join(lines)
        
        return {
            "text": text,
            "format": "list"
        }
    
    async def _generate_chart_summary_structured(
        self,
        query: str,
        labels: List[str],
        values: List[float],
        chart_type: str
    ) -> Dict[str, Any]:
        """
        Generate structured chart response with title, insights, and content using LLM.
        
        Args:
            query: Original user query
            labels: Chart category labels
            values: Chart data values
            chart_type: Type of chart (pie, bar, etc.)
            
        Returns:
            Dict with keys: title, insightHighlights, content
        """
        try:
            # Calculate statistics
            total = sum(values)
            sorted_items = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
            
            # Build statistics context
            stats = []
            for i, (label, value) in enumerate(sorted_items[:5], 1):
                pct = (value / total * 100) if total > 0 else 0
                stats.append(f"{i}. {label}: {value} ({pct:.1f}%)")
            
            # Create prompt for LLM to generate structured JSON
            prompt = f"""You are a data analyst. Analyze this chart data and provide structured insights.

User Query: "{query}"
Chart Type: {chart_type}
Total Items: {total}

Top Categories:
{chr(10).join(stats)}

Generate a JSON response with this EXACT structure:
{{
  "title": "A concise descriptive title (Vietnamese, 10-15 words max)",
  "insightHighlights": [
    "First key insight (e.g., top performer, trend, comparison)",
    "Second key insight (e.g., bottom performer, gap analysis)",
    "Third key insight (e.g., distribution pattern, recommendation)"
  ],
  "content": "A 2-3 sentence natural summary starting with emoji (📊/🥧/📈)"
}}

Insight Guidelines:
- Generate 3-5 specific, actionable insights
- Focus on: rankings (top/bottom), trends, comparisons, patterns
- Use Vietnamese language, be specific with numbers
- Avoid generic statements, make insights data-driven

Example Response:
{{
  "title": "Thống kê phân bổ sản phẩm theo thương hiệu",
  "insightHighlights": [
    "Cetaphil dẫn đầu với 20 sản phẩm (42.5% thị phần)",
    "Top 3 thương hiệu chiếm 65% tổng sản phẩm",
    "8/12 thương hiệu có dưới 5 sản phẩm (cơ hội mở rộng)"
  ],
  "content": "📊 Phân bổ sản phẩm theo thương hiệu cho thấy sự tập trung cao vào Cetaphil với 42.5% thị phần. Tổng cộng 47 sản phẩm được phân bổ trong 12 thương hiệu."
}}

Your JSON response (valid JSON only, no markdown):"""
            
            response = await self.llm.generate(prompt, max_tokens=400)
            response = response.strip()
            
            # Try to parse JSON response
            try:
                # Remove markdown code blocks if present
                if '```' in response:
                    import re
                    match = re.search(r'```(?:json)?\s*({[^`]+})\s*```', response, re.DOTALL)
                    if match:
                        response = match.group(1)
                
                structured = json.loads(response)
                
                # Validate structure
                if isinstance(structured, dict) and "title" in structured and "insightHighlights" in structured and "content" in structured:
                    # Ensure insights is a list
                    if not isinstance(structured["insightHighlights"], list):
                        structured["insightHighlights"] = [str(structured["insightHighlights"])]
                    
                    logger.info(f"📝 Generated structured insights: {len(structured['insightHighlights'])} items")
                    return structured
                else:
                    logger.warning("LLM response missing required fields, using fallback")
                    return self._generate_simple_chart_summary_structured(query, labels, values, total)
                    
            except json.JSONDecodeError as je:
                logger.warning(f"LLM response not valid JSON: {je}, using fallback")
                return self._generate_simple_chart_summary_structured(query, labels, values, total)
            
        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}, using fallback")
            return self._generate_simple_chart_summary_structured(query, labels, values, sum(values))
    
    def _generate_simple_chart_summary_structured(
        self,
        query: str,
        labels: List[str],
        values: List[float],
        total: float
    ) -> Dict[str, Any]:
        """
        Generate simple structured summary without LLM (fallback).
        Returns: {"title": str, "insightHighlights": [str], "content": str}
        """
        sorted_items = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
        
        # Generate title
        if "phân bổ" in query.lower() or "distribution" in query.lower():
            title = f"Phân bổ {len(labels)} nhóm"
        elif "thống kê" in query.lower() or "statistics" in query.lower():
            title = f"Thống kê {len(labels)} nhóm"
        else:
            title = f"Dữ liệu {len(labels)} nhóm"
        
        # Generate insights
        insights = []
        
        # Top performer
        if sorted_items:
            top_label, top_value = sorted_items[0]
            top_pct = (top_value / total * 100) if total > 0 else 0
            insights.append(f"Dẫn đầu: {top_label} với {top_value:.0f} ({top_pct:.1f}%)")
        
        # Top 3 concentration
        if len(sorted_items) >= 3:
            top3_total = sum(v for _, v in sorted_items[:3])
            top3_pct = (top3_total / total * 100) if total > 0 else 0
            insights.append(f"Top 3 chiếm {top3_pct:.1f}% tổng số")
        
        # Bottom performer or distribution pattern
        if len(sorted_items) > 5:
            small_groups = [v for _, v in sorted_items if v < (total / len(sorted_items))]  # Below average
            if small_groups:
                insights.append(f"{len(small_groups)} nhóm dưới mức trung bình")
        
        # Total summary
        insights.append(f"Tổng cộng: {total:.0f} items trong {len(labels)} nhóm")
        
        # Generate content summary
        content = f"📊 {title}. "
        if sorted_items:
            top_label, top_value = sorted_items[0]
            top_pct = (top_value / total * 100) if total > 0 else 0
            content += f"{top_label} chiếm tỷ trọng lớn nhất với {top_value:.0f} ({top_pct:.1f}%). "
        content += f"Tổng cộng {total:.0f} items được phân bổ trong {len(labels)} nhóm."
        
        return {
            "title": title,
            "insightHighlights": insights[:5],  # Limit to 5 insights
            "content": content
        }
    
    async def _format_chart(self, result: Any, intent: QueryIntent, query: str) -> Dict[str, Any]:
        """Format chart result for C# parser with LLM-generated summary.
        
        C# expects format: {"chart_type":"pie","data":[...],"labels":[...]}
        This matches TryExtractPythonChartJson() in CrawlerChatHub.cs
        """
        try:
            # Result should be dict with labels + values
            if isinstance(result, dict) and "labels" in result and "values" in result:
                # Transform to C# expected format
                chart_type = intent.chart_type or "pie"
                labels = result["labels"]
                values = result["values"]
                
                chart_data = {
                    "chart_type": chart_type,  # C# expects "chart_type" not "type"
                    "data": values,   # C# expects "data" not "series"
                    "labels": labels
                }
                
                # Generate structured summary with insights using LLM
                structured_summary = await self._generate_chart_summary_structured(query, labels, values, chart_type)
                
                # Build new structured format for C#
                # Format: {"title": "...", "insightHighlights": [...], "content": "...", "chart": {...}}
                structured_response = {
                    "title": structured_summary.get("title", query),
                    "insightHighlights": structured_summary.get("insightHighlights", []),
                    "content": structured_summary.get("content", ""),
                    "chart": chart_data
                }
                
                # Embed structured JSON in text response for C# to extract
                # C# will detect and parse this new format
                display_text = f"{structured_summary.get('content', '')}\n\n```json\n{json.dumps(structured_response, ensure_ascii=False)}\n```"
                
                return {
                    "text": display_text,
                    "format": "chart",
                    "chart_data": structured_response
                }
            else:
                # Fallback to text
                return {
                    "text": json.dumps(result, ensure_ascii=False, indent=2),
                    "format": "text"
                }
        except Exception as e:
            logger.error(f"Chart formatting error: {e}")
            return {
                "text": str(result),
                "format": "text"
            }
    
    def _format_table(self, result: Any, intent: QueryIntent, query: str) -> Dict[str, Any]:
        """Format table result."""
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            # Format as markdown table
            keys = list(result[0].keys())
            
            # Header
            header = "| " + " | ".join(keys) + " |"
            separator = "|" + "|".join(["---"] * len(keys)) + "|"
            
            # Rows
            rows = []
            for item in result:
                row = "| " + " | ".join(str(item.get(k, "")) for k in keys) + " |"
                rows.append(row)
            
            text = "\n".join([header, separator] + rows)
            
            return {
                "text": text,
                "format": "table"
            }
        else:
            return {
                "text": str(result),
                "format": "text"
            }
    
    def _format_text(self, result: Any, intent: QueryIntent, query: str) -> Dict[str, Any]:
        """Format generic text result."""
        if isinstance(result, (list, dict)):
            text = json.dumps(result, ensure_ascii=False, indent=2)
        else:
            text = str(result)
        
        return {
            "text": text,
            "format": "text"
        }
    
    def _get_cache_key(self, query: str, products: List[Dict[str, Any]]) -> str:
        """Generate cache key from query + products."""
        # Hash products to create stable key
        products_hash = hashlib.md5(
            json.dumps(products, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        # Combine with query
        key = f"{query.lower().strip()}_{products_hash}"
        
        return key
    
    def _update_cache(self, key: str, result: CodeResult):
        """Update LRU cache."""
        # Simple LRU: if cache full, remove oldest
        if len(self._analyze_cache) >= self.cache_size:
            # Remove first item (oldest)
            first_key = next(iter(self._analyze_cache))
            del self._analyze_cache[first_key]
        
        self._analyze_cache[key] = result


# =============================================================================
# SINGLETON FACTORY
# =============================================================================

_smart_rag_instance: Optional[SmartRAG] = None

def get_smart_rag(llm_client) -> SmartRAG:
    """Get or create SmartRAG singleton."""
    global _smart_rag_instance
    
    if _smart_rag_instance is None:
        _smart_rag_instance = SmartRAG(llm_client)
        logger.info("Created new SmartRAG instance")
    
    return _smart_rag_instance
