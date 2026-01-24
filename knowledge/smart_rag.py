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
        
        # Source metadata tracking for comparison queries
        self._sources_metadata = []
        self._last_products_data = []
        
        logger.info("SmartRAG initialized (unified module)")
    
    def _normalize_filename(self, filename: str) -> str:
        """
        Normalize filename by removing underscores, hyphens, spaces, and converting to lowercase.
        This enables matching "haircarecrawlnonpicarefixed.csv" with "haircare_crawl_non_picare_fixed.csv"
        
        Args:
            filename: Original filename
            
        Returns:
            Normalized filename (lowercase, no separators)
        """
        return filename.replace('_', '').replace('-', '').replace(' ', '').lower()
    
    def _extract_sources_metadata(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract and group source metadata from products.
        Enables comparison queries like "compare products from link A vs link B".
        
        Args:
            products: List of product dicts with _source, _crawl_job_id, _source_url fields
            
        Returns:
            List of unique sources with metadata:
            [
                {
                    "type": "crawl_job",
                    "job_id": "abc-123",
                    "url": "https://...",
                    "normalized_url": "haircarecrawlnonpicarefixed.csv",
                    "job_prompt": "crawl shopee",
                    "count": 10
                },
                ...
            ]
        """
        sources = {}
        
        for product in products:
            source_type = product.get("_source", "unknown")
            job_id = product.get("_crawl_job_id", "unknown")
            source_url = product.get("_source_url", "N/A")
            
            key = f"{source_type}_{job_id}"
            
            if key not in sources:
                # Add both original and normalized filename for fuzzy matching
                normalized_url = self._normalize_filename(source_url) if source_url != "N/A" else "N/A"
                
                sources[key] = {
                    "type": source_type,
                    "job_id": job_id,
                    "url": source_url,
                    "url_normalized": normalized_url,
                    "job_prompt": product.get("_job_prompt", "N/A"),
                    "count": 0
                }
            
            sources[key]["count"] += 1
        
        return list(sources.values())
    
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
        2. Extract source metadata from products
        3. Single LLM call: analyze + generate code
        4. Execute code
        5. Format result with source context
        6. Return CodeResult
        
        Args:
            query: User's question
            products: List of product dicts (may contain _source, _crawl_job_id, _source_url metadata)
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
            # Extract source metadata for comparison queries
            sources_metadata = self._extract_sources_metadata(products)
            if sources_metadata:
                logger.info(f"📊 Found {len(sources_metadata)} unique data sources")
                for source in sources_metadata:
                    logger.info(f"  - {source['type']}: {source.get('job_prompt', 'N/A')[:50]}... ({source['count']} products)")
            
            # Store for use in formatters
            self._sources_metadata = sources_metadata
            self._last_products_data = products  # Keep for context in number insights
            
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
            logger.info(f"🔍 DEBUG - Generated code:\n{code}")
            logger.info(f"🔍 DEBUG - First product structure: {json.dumps(products[0], indent=2, ensure_ascii=False)}")
            
            # Step 3: Execute code with retry mechanism
            logger.info("⚙️ Executing code...")
            
            success, result, error = self.sandbox.execute(code, products)
            
            # Retry up to 2 times if execution fails (skip for timeout/sandbox errors)
            max_retries = 2
            retry_count = 0
            
            while not success and retry_count < max_retries:
                # Skip retry for non-fixable errors
                if "Timeout" in error or "Sandbox error" in error:
                    logger.warning(f"⏭️ Skipping retry for non-fixable error: {error}")
                    break
                
                retry_count += 1
                logger.warning(f"🔄 Retry attempt {retry_count}/{max_retries} - Error: {error}")
                
                # Ask LLM to fix the code
                fixed_code = await self._retry_with_error_fix(query, code, error, intent, schema)
                
                if fixed_code and fixed_code != code:
                    logger.info(f"🔧 LLM generated fixed code ({len(fixed_code)} chars)")
                    code = fixed_code
                    success, result, error = self.sandbox.execute(code, products)
                    
                    if success:
                        logger.info(f"✅ Retry {retry_count} succeeded!")
                        break
                else:
                    logger.warning("❌ LLM failed to generate fix, stopping retries")
                    break
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if not success:
                logger.error(f"❌ Execution failed after {retry_count} retries: {error}")
                
                # Generate user-friendly error message instead of raw error
                friendly_error = await self._generate_user_friendly_error(query, error, intent)
                
                return CodeResult(
                    success=False,
                    result=None,
                    display_text=friendly_error,
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
    
    async def _retry_with_error_fix(
        self,
        query: str,
        failed_code: str,
        error: str,
        intent: QueryIntent,
        schema: Dict[str, str]
    ) -> Optional[str]:
        """
        Ask LLM to fix failed code based on error message.
        
        Args:
            query: Original user query
            failed_code: Code that failed execution
            error: Error message from execution
            intent: Query intent
            schema: Data schema
            
        Returns:
            Fixed code or None if LLM failed
        """
        try:
            # Get error recovery prompt from templates
            template = self.prompts.get("error_recovery", "")
            
            if not template:
                # Fallback inline prompt
                template = """The previous code failed with an error. Please fix it.

Original Query: {query}
Original Code:
```python
{code}
```

Error:
{error}

Common fixes:
1. Price ranges like "349000 ~ 698000" → Use parse_price() helper
2. None values on string methods → Check if value exists before calling .lower()
3. Missing imports → Counter/defaultdict are already available, don't import
4. Field not found → Use .get() with defaults

Generate ONLY the fixed Python code in ```python``` block (no JSON, no explanation):"""
            
            prompt = template.format(
                query=query,
                code=failed_code,
                error=error
            )
            
            response = await self.llm.generate(prompt, max_tokens=600)
            
            # Extract code from response
            code_match = re.search(r'```python\s*\n(.*?)\n```', response, re.DOTALL)
            if code_match:
                fixed_code = code_match.group(1).strip()
                logger.info(f"🔧 LLM provided fixed code")
                return fixed_code
            else:
                logger.warning("❌ LLM response missing code block")
                return None
                
        except Exception as e:
            logger.error(f"Error in retry_with_error_fix: {e}")
            return None
    
    async def _generate_user_friendly_error(
        self,
        query: str,
        error: str,
        intent: QueryIntent
    ) -> str:
        """
        Generate user-friendly Vietnamese error message instead of raw error log.
        
        Args:
            query: User's query
            error: Raw error message
            intent: Query intent
            
        Returns:
            User-friendly error message in Vietnamese
        """
        try:
            prompt = f"""Generate a helpful, user-friendly error message in Vietnamese for this failed query.

User Query: "{query}"
Intent: {intent.intent_type}
Operation: {intent.operation}
Technical Error: {error}

Generate a 1-2 sentence Vietnamese message that:
1. Starts with apologetic emoji (😔/❌/⚠️)
2. Explains what went wrong in simple terms (avoid technical jargon)
3. Suggests what the user might try instead or why it failed
4. Is polite and helpful
5. Does NOT include raw error messages or code

Common error patterns:
- "could not convert string to float" → Data has unexpected format (price ranges, text instead of numbers)
- "NoneType object has no attribute" → Some products missing required information
- "KeyError" → Field doesn't exist in data
- "division by zero" → No matching products found

Examples:
- "😔 Xin lỗi, không thể tính toán vì một số sản phẩm có định dạng giá không chuẩn (ví dụ: khoảng giá). Vui lòng thử với tập dữ liệu khác."
- "⚠️ Không tìm thấy sản phẩm nào phù hợp với điều kiện tìm kiếm. Vui lòng kiểm tra lại tên file hoặc URL."
- "❌ Một số sản phẩm thiếu thông tin cần thiết để thực hiện phép tính này. Vui lòng thử câu hỏi khác."

Your message (Vietnamese, 1-2 sentences, NO technical details):"""
            
            response = await self.llm.generate(prompt, max_tokens=150)
            friendly_msg = response.strip()
            
            # Validate
            if len(friendly_msg) > 300 or '{' in friendly_msg or 'Error:' in friendly_msg:
                logger.warning("LLM error message too technical, using fallback")
                return self._simple_error_message(error)
            
            logger.info(f"📝 Generated friendly error: {friendly_msg[:80]}...")
            return friendly_msg
            
        except Exception as e:
            logger.error(f"Failed to generate friendly error: {e}")
            return self._simple_error_message(error)
    
    def _simple_error_message(self, error: str) -> str:
        """Simple fallback error message."""
        if "could not convert" in error.lower() and "float" in error.lower():
            return "😔 Xin lỗi, không thể tính toán vì dữ liệu có định dạng không phù hợp. Một số sản phẩm có thể có giá dạng khoảng hoặc định dạng đặc biệt."
        elif "nonetype" in error.lower() and "attribute" in error.lower():
            return "⚠️ Một số sản phẩm thiếu thông tin cần thiết. Vui lòng thử với bộ lọc khác hoặc câu hỏi đơn giản hơn."
        elif "keyerror" in error.lower() or "not found" in error.lower():
            return "❌ Không tìm thấy trường dữ liệu cần thiết trong sản phẩm. Vui lòng kiểm tra lại câu hỏi."
        elif "division by zero" in error.lower() or "no matching" in error.lower():
            return "📭 Không tìm thấy sản phẩm nào phù hợp với điều kiện tìm kiếm. Vui lòng thử với điều kiện khác."
        else:
            return "😔 Xin lỗi, không thể xử lý câu hỏi này. Vui lòng thử diễn đạt lại hoặc đơn giản hóa câu hỏi."
    
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
        
        # Extract sources metadata for filtering/comparison queries
        sources_metadata = self._extract_sources_metadata(products)
        
        # Build sources_info with normalized filenames for better matching
        sources_list = []
        for s in sources_metadata:
            url = s['url']
            # If it's a CSV filename, add normalized version (remove underscores/spaces)
            if s['type'] == 'csv_file' and url != 'N/A':
                normalized = url.replace('_', '').replace('-', '').replace(' ', '').lower()
                sources_list.append(
                    f"- {s['type']} ({s['count']} products): {url} (also matches: {normalized}) | Prompt: {s['job_prompt']}"
                )
            else:
                sources_list.append(
                    f"- {s['type']} ({s['count']} products): {url} | Prompt: {s['job_prompt']}"
                )
        
        sources_info = "\n".join(sources_list) if sources_list else "No source metadata available"
        
        # Format prompt with sources_info included
        prompt = template.format(
            query=query,
            schema=schema_str,
            sample_products=sample_products,
            total_products=len(products),
            sources_info=sources_info
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
        return """Analyze the user's query and generate Python code that DIRECTLY returns a formatted Vietnamese answer string.

User Query: {query}

Available Data:
- Total products: {total_products}
- Data schema: {schema}
- Sample products: {sample_products}

⚠️ IMPORTANT - SOURCE FILTERING:
When user asks about "sản phẩm trong file X.csv" or "từ link Y", you MUST filter products first!

Each product has metadata fields:
- _source: "crawl_job" or "csv_file" 
- _source_url: URL (for crawl) or "haircare_crawl_non_picare_fixed.csv" (for CSV)
- _crawl_job_id: unique ID
- _job_prompt: original prompt

Available Data Sources:
{sources_info}

FILTERING RULES:
1. "sản phẩm trong file X.csv" → filter by: `if 'X.csv' in p.get('_source_url', '')`
2. "sản phẩm từ link/URL Y" → filter by: `if 'Y' in p.get('_source_url', '')`  
3. "sản phẩm từ crawl" → filter by: `if p.get('_source') == 'crawl_job'`
4. "sản phẩm từ file/CSV" → filter by: `if p.get('_source') == 'csv_file'`
5. No mention of source → use ALL products (no filter)

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
# Python code that returns a FORMATTED VIETNAMESE STRING with emoji
# Available: products (list of dicts with metadata: _source, _source_url, _crawl_job_id, _job_prompt)
# Must set: result (formatted string answer)

# ⚠️ CRITICAL: If user mentions "file X" or "link Y", FILTER FIRST!
# Pattern: filtered = [p for p in products if 'keyword' in p.get('_source_url', '')]

# Example 1: "Có bao nhiêu sản phẩm trong file haircare.csv" ⚠️ FILTER REQUIRED
csv_products = [p for p in products if 'haircare.csv' in p.get('_source_url', '')]
result = f"📄 Có {{len(csv_products)}} sản phẩm trong file haircare.csv"

# Example 2: "Sản phẩm đắt nhất và rẻ nhất"
# Example 2: "Sản phẩm đắt nhất và rẻ nhất"
max_p = max(products, key=lambda p: p.get('price', 0) or 0)
min_p = min(products, key=lambda p: p.get('price', 0) or 0)
result = f"🏆 Sản phẩm đắt nhất là {{max_p.get('name', 'N/A')}} ({{max_p.get('price', 0):,}}₫), 💰 sản phẩm rẻ nhất là {{min_p.get('name', 'N/A')}} ({{min_p.get('price', 0):,}}₫)"

# Example 3: "Có bao nhiêu sản phẩm" (no filter - all products)
# Example 3: "Có bao nhiêu sản phẩm" (no filter - all products)
result = f"📦 Tổng cộng {{len(products)}} sản phẩm được tìm thấy"

# Example 4: "Top 5 sản phẩm đắt nhất"
# Example 4: "Top 5 sản phẩm đắt nhất"
top_5 = sorted(products, key=lambda p: p.get('price', 0) or 0, reverse=True)[:5]
lines = [f"{{i+1}}. {{p.get('name', 'N/A')}} - {{p.get('price', 0):,}}₫" for i, p in enumerate(top_5)]
result = "🔝 Top 5 sản phẩm đắt nhất:\n" + "\n".join(lines)

# Example 5: "Giá trung bình"
avg_price = sum(p.get('price', 0) or 0 for p in products) / len(products) if products else 0
result = f"📊 Giá trung bình là {{avg_price:,.0f}}₫"

# Example 6: "Có bao nhiêu sản phẩm từ link https://picare.vn/..." ⚠️ FILTER BY URL
crawl_products = [p for p in products if 'picare.vn' in p.get('_source_url', '')]
result = f"🔗 Có {{len(crawl_products)}} sản phẩm được crawl từ picare.vn"

# Example 7: "So sánh sản phẩm từ link A vs file B" ⚠️ COMPARISON REQUIRES FILTERING
link_a = [p for p in products if 'example.com' in p.get('_source_url', '')]
link_b = [p for p in products if 'shopee.vn' in p.get('_source_url', '')]
avg_a = sum(p.get('price', 0) or 0 for p in link_a) / len(link_a) if link_a else 0
avg_b = sum(p.get('price', 0) or 0 for p in link_b) / len(link_b) if link_b else 0
result = f"📊 So sánh:\n- Link A: {{len(link_a)}} sản phẩm, giá TB {{avg_a:,.0f}}₫\n- Link B: {{len(link_b)}} sản phẩm, giá TB {{avg_b:,.0f}}₫"

# Example 8: "Sản phẩm từ file CSV" ⚠️ FILTER BY SOURCE TYPE
csv_only = [p for p in products if p.get('_source') == 'csv_file']
result = f"📁 Có {{len(csv_only)}} sản phẩm từ file CSV"
```

CRITICAL INSTRUCTIONS:
1. Code MUST return a formatted Vietnamese string starting with emoji
2. Use emojis: 🏆(max), 💰(min), 📊(avg), 📦(count), 🔝(top), 💵(price), 📋(list), 📄(file), 🕷️(crawl)
3. For numbers: Format with :, (e.g., {{price:,.0f}}₫)
4. For SOURCE FILTERING: Use _source_url, _source, _crawl_job_id fields
5. For COMPARISONS: Filter products by source, then compare stats
4. For lists: Include header + numbered items
5. For comparisons: "A là [value], B là [value]"
6. Always use .get() with defaults to avoid KeyError
7. Only for CHARTS output_format="chart": return {{"labels": [...], "values": [...]}}
8. Keep natural conversational Vietnamese
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
        
        NEW APPROACH: If code returns a formatted string (with emoji), use it directly.
        This is more reliable than LLM formatting afterward.
        
        Returns:
            {
                "text": "Natural language answer",
                "format": "number|list|chart|table|text",
                "chart_data": {...} (optional)
            }
        """
        # PRIORITY 1: If code already returned a formatted string with emoji, use it directly
        if isinstance(result, str) and len(result) > 0:
            # Check if it starts with common Vietnamese answer emoji
            emoji_prefixes = ['🏆', '💰', '📊', '📦', '🔝', '💵', '📋', '💡', '🔍', '⭐', '🥧', '📈']
            if any(result.startswith(e) for e in emoji_prefixes):
                logger.info("✅ Using formatted string from code execution directly")
                return {
                    "text": result,
                    "format": "text"
                }
        
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
            return await self._format_list(result, intent, query)
        
        elif format_type == "chart":
            return await self._format_chart(result, intent, query)
        
        elif format_type == "table":
            return await self._format_table(result, intent, query)
        
        else:  # text
            return await self._format_text(result, intent, query)
    
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
        result_type: str  # "percentage", "price", or "number"
    ) -> str:
        """Use LLM to generate contextual insight for numerical results."""
        try:
            # Get additional context if available
            context_info = ""
            if hasattr(self, '_last_products_data') and self._last_products_data:
                # Extract and parse prices (handle string format like "117,000₫")
                prices = []
                for p in self._last_products_data:
                    price_str = p.get('price') or p.get('salePrice')
                    if price_str:
                        try:
                            # Convert string price to float
                            price_val = float(str(price_str).replace('₫', '').replace(',', '').strip())
                            prices.append(price_val)
                        except (ValueError, AttributeError):
                            pass
                
                if prices and result_type == "price":
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
            
            # Customize prompt based on result type
            if result_type == "percentage":
                prompt = f"""Generate a natural, insightful response for this percentage/ratio result.

User Query: "{query}"
Operation: {op_vn}
Result: {formatted_value}

Generate a 1-2 sentence response that:
1. Starts with an appropriate emoji (📊/💹/📉/📈)
2. States the percentage result clearly
3. Provides context about what this percentage means (e.g., "discount rate of 13.12% means products are moderately discounted")
4. Uses natural Vietnamese language
5. NEVER mention price ranges - this is a percentage, not a price!

Examples for discount rates:
- "📊 Tỉ lệ giảm giá trung bình là 13.12%. Đây là mức giảm giá vừa phải, cho thấy các sản phẩm có chương trình khuyến mãi nhẹ."
- "💹 Tỉ lệ giảm giá trung bình đạt 25.5%, thể hiện chương trình khuyến mãi hấp dẫn cho khách hàng."
- "📉 Tỉ lệ giảm giá trung bình chỉ 5.2%, cho thấy các sản phẩm ít được giảm giá."

Your response (1-2 sentences only, about PERCENTAGE not price):"""
            else:
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
            
            # Detect query type
            percentage_keywords = ["tỉ lệ", "phần trăm", "percent", "%", "giảm giá", "discount", "rate"]
            is_percentage = any(keyword in query.lower() for keyword in percentage_keywords)
            
            price_keywords = ["giá", "price", "tiền", "đồng", "vnd", "usd", "cost"]
            is_price = any(keyword in query.lower() for keyword in price_keywords)
            
            # Add appropriate suffix
            if is_percentage:
                # It's a percentage/ratio result
                formatted_value = f"{formatted}%"
                result_type = "percentage"
            elif is_price and "count" not in intent.operation and "số lượng" not in query.lower():
                # It's a price result
                formatted_value = f"{formatted}₫"
                result_type = "price"
            else:
                # Just a number
                formatted_value = formatted
                result_type = "number"
            
            # Generate LLM-based contextual response
            text = await self._generate_number_insight(query, num, formatted_value, intent.operation, result_type)
            
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
    
    async def _format_list_with_llm(self, result: Any, intent: QueryIntent, query: str) -> Dict[str, Any]:
        """Format list result using LLM for contextual header and natural language."""
        if not isinstance(result, (list, tuple)):
            result = [result]
        
        try:
            # Format items as numbered list
            lines = []
            prices_in_list = []
            
            for i, item in enumerate(result, 1):
                if isinstance(item, dict):
                    # Format dict with name and price
                    name = item.get("name") or item.get("product_name") or item.get("title") or "Sản phẩm"
                    price = item.get("price") or item.get("salePrice")
                    
                    # Convert price to number for statistics
                    if price:
                        try:
                            # Handle both numeric and string prices
                            numeric_price = float(str(price).replace(",", "").replace("₫", "").strip()) if isinstance(price, str) else float(price)
                            prices_in_list.append(numeric_price)
                        except (ValueError, AttributeError):
                            # Can't convert to number, skip for stats but still display
                            pass
                    
                    if price and name:
                        lines.append(f"{i}. {name} - {price:,.0f}₫" if isinstance(price, (int, float)) else f"{i}. {name} - {price}")
                    elif name:
                        lines.append(f"{i}. {name}")
                    else:
                        lines.append(f"{i}. {str(item)[:100]}")
                else:
                    # Simple value
                    if isinstance(item, (int, float)):
                        lines.append(f"{i}. {item:,.0f}")
                        prices_in_list.append(item)
                    else:
                        lines.append(f"{i}. {item}")
            
            # Calculate statistics for LLM context
            stats_context = f"Total items: {len(result)}"
            if prices_in_list:
                min_p, max_p = min(prices_in_list), max(prices_in_list)
                avg_p = sum(prices_in_list) / len(prices_in_list)
                stats_context += f"\nPrice range: {min_p:,.0f}₫ - {max_p:,.0f}₫\nAverage price: {avg_p:,.0f}₫"
            
            # Use LLM to generate contextual header
            prompt = f"""Generate a brief, contextual header for this list result.

User Query: "{query}"
Operation: {intent.operation}
{stats_context}

Generate a 1-line Vietnamese header that:
1. Starts with emoji (📋 for general lists, 🔝 for top/best, 💰 for cheap, 🏆 for expensive)
2. Describes what the list shows based on the query
3. Includes key statistics (count, price range if relevant)
4. Is concise (under 80 characters)

Examples:
- "📋 Danh sách 19 sản phẩm (giá từ 200,000₫ - 1,500,000₫)"
- "🔝 Top 5 sản phẩm đắt nhất (1,200,000₫ - 1,500,000₫)"
- "💰 5 sản phẩm rẻ nhất (200,000₫ - 350,000₫)"
- "📋 10 kết quả tìm kiếm phù hợp"

Your header (1 line only, Vietnamese):"""
            
            response = await self.llm.generate(prompt, max_tokens=100)
            header = response.strip()
            
            # Validate header
            if len(header) > 150 or '\n' in header or '{' in header:
                logger.warning("LLM header invalid, using simple format")
                header = f"📋 Danh sách {len(result)} kết quả:"
            
            # Ensure header ends with colon
            if not header.endswith(':'):
                header += ':'
            
            text = header + "\n" + "\n".join(lines)
            
            logger.info(f"📝 Generated list header: {header}")
            return {
                "text": text,
                "format": "list"
            }
            
        except Exception as e:
            logger.error(f"LLM list formatting failed: {e}, using fallback")
            # Fallback to simple header
            header = f"📋 Danh sách {len(result)} kết quả:"
            text = header + "\n" + "\n".join(lines)
            return {
                "text": text,
                "format": "list"
            }
    
    async def _format_list(self, result: Any, intent: QueryIntent, query: str) -> Dict[str, Any]:
        """Format list result - now delegates to LLM-based formatter."""
        return await self._format_list_with_llm(result, intent, query)
    
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
            
            # Detect query type to customize insight generation
            query_lower = query.lower()
            is_comparison = any(keyword in query_lower for keyword in ["so sánh", "compare", "vs", "khác nhau", "difference"])
            is_percentage = any(keyword in query_lower for keyword in ["tỉ lệ", "phần trăm", "percent", "%"])
            is_distribution = any(keyword in query_lower for keyword in ["phân bổ", "distribution", "phân phối"])
            
            # Customize prompt based on query type
            if is_comparison:
                specific_instructions = """
IMPORTANT - This is a COMPARISON query. Focus your insights on:
- Which category has higher/lower values and by how much
- The percentage difference or gap between categories
- What the comparison reveals (e.g., "crawl data is 11.8% higher")
- Avoid generic statistics like "total value" unless directly relevant
- Generate ONLY 2-3 insights that directly answer the comparison question
"""
            elif is_percentage:
                specific_instructions = """
IMPORTANT - This is a PERCENTAGE/RATIO query. Focus your insights on:
- The actual percentage values and what they mean
- Whether percentages are high, low, or moderate
- Comparisons between different percentage groups
- Avoid mentioning absolute values unless they add context
- Generate ONLY 2-3 insights about the percentages
"""
            elif is_distribution:
                specific_instructions = """
IMPORTANT - This is a DISTRIBUTION query. Focus your insights on:
- How items are spread across categories
- Top performers and concentration patterns
- Gaps or imbalances in the distribution
- Generate 3-4 insights about the distribution pattern
"""
            else:
                specific_instructions = """
IMPORTANT - Analyze the query intent and generate insights that:
- Directly answer what the user asked
- Focus on relevant patterns in the data
- Avoid generic statements not related to the query
- Generate 2-4 insights based on query complexity
"""
            
            # Create prompt for LLM to generate structured JSON
            prompt = f"""You are a data analyst. Analyze this chart data and provide structured insights.

User Query: "{query}"
Chart Type: {chart_type}
Total Items: {total}

Top Categories:
{chr(10).join(stats)}

{specific_instructions}

Generate a JSON response with this EXACT structure:
{{
  "title": "A concise descriptive title that directly relates to the user query (Vietnamese, 10-15 words max)",
  "insightHighlights": [
    "First key insight that DIRECTLY answers the query",
    "Second insight providing additional relevant context"
  ],
  "content": "A 1-2 sentence natural summary starting with emoji (📊/🥧/📈) that directly answers the user's question"
}}

Guidelines:
- Be concise and specific - quality over quantity
- Every insight must be relevant to the user's actual question
- Use Vietnamese language naturally
- Include specific numbers from the data
- DO NOT generate generic insights unrelated to the query

Example for Comparison Query "So sánh giá crawl vs file":
{{
  "title": "So sánh giá sản phẩm: Crawl vs File CSV",
  "insightHighlights": [
    "Giá sản phẩm crawl chiếm ưu thế với 55.9% tổng giá trị",
    "Sản phẩm trong file CSV chỉ chiếm 44.1%, thấp hơn 11.8%"
  ],
  "content": "📊 So sánh giá sản phẩm cho thấy dữ liệu crawl có giá trị cao hơn, chiếm 55.9% so với 44.1% từ file CSV."
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
    
    async def _format_table(self, result: Any, intent: QueryIntent, query: str) -> Dict[str, Any]:
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
    
    async def _format_text_with_llm(self, result: Any, intent: QueryIntent, query: str) -> Dict[str, Any]:
        """Format generic text result using LLM for natural language response."""
        try:
            # Prepare result summary for LLM
            if isinstance(result, dict):
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
                result_type = "dictionary"
            elif isinstance(result, list):
                result_str = json.dumps(result[:10], ensure_ascii=False, indent=2)  # Limit to first 10 items
                if len(result) > 10:
                    result_str += f"\n... and {len(result) - 10} more items"
                result_type = "list"
            else:
                result_str = str(result)
                result_type = "simple value"
            
            prompt = f"""Generate a natural, conversational response for this query result.

User Query: "{query}"
Intent: {intent.intent_type}
Operation: {intent.operation}

Result Data ({result_type}):
{result_str}

Generate a natural Vietnamese response that:
1. Starts with an appropriate emoji (📊/📋/💡/🔍)
2. Directly answers the user's question in conversational language
3. Presents key information from the result (don't just repeat raw data)
4. Keeps it concise (2-4 sentences max)
5. NEVER output raw JSON or code blocks

Examples:
- "📊 Dữ liệu cho thấy có 15 sản phẩm thỏa mãn điều kiện tìm kiếm. Các sản phẩm này đều thuộc danh mục chăm sóc da với giá dao động từ 200,000₫ đến 1,500,000₫."
- "💡 Kết quả phân tích cho thấy sản phẩm có đặc điểm nổi bật về thành phần tự nhiên và phù hợp cho da nhạy cảm."
- "🔍 Tìm thấy thông tin chi tiết về sản phẩm bao gồm tên thương hiệu, giá cả và mô tả đầy đủ."

Your response (Vietnamese, conversational, NO JSON):"""
            
            response = await self.llm.generate(prompt, max_tokens=250)
            response_text = response.strip()
            
            # Validate response - ensure it's not JSON or code
            if '{' in response_text and '}' in response_text and '"' in response_text:
                logger.warning("LLM returned JSON-like content, using simple format")
                return self._simple_text_format(result, result_type)
            
            if len(response_text) > 600:
                logger.warning("LLM response too long, truncating")
                response_text = response_text[:597] + "..."
            
            logger.info(f"📝 Generated text response: {response_text[:100]}...")
            return {
                "text": response_text,
                "format": "text"
            }
            
        except Exception as e:
            logger.error(f"LLM text formatting failed: {e}, using fallback")
            return self._simple_text_format(result, "unknown")
    
    def _simple_text_format(self, result: Any, result_type: str) -> Dict[str, Any]:
        """Simple fallback formatting for text results."""
        if isinstance(result, (list, dict)):
            text = f"📋 Kết quả ({result_type}): " + json.dumps(result, ensure_ascii=False, indent=2)[:500]
        else:
            text = f"💡 Kết quả: {str(result)}"
        
        return {
            "text": text,
            "format": "text"
        }
    
    async def _format_text(self, result: Any, intent: QueryIntent, query: str) -> Dict[str, Any]:
        """Format generic text result - now delegates to LLM-based formatter."""
        return await self._format_text_with_llm(result, intent, query)
    
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
