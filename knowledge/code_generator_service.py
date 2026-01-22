"""
Code Generator Service - Main Orchestrator for LLM-Generated Analysis Functions

Workflow:
1. Classify query (computational vs non-computational)
2. Profile data to detect patterns
3. Generate Python analysis function via LLM
4. Execute function in sandbox
5. On error: classify error → retry (max 3) or fallback
6. Return numerical results + natural language answer

Features:
- Versioned prompt templates (A/B testing ready)
- Error classification (FIXABLE vs UNFIXABLE)
- Graceful degradation (fallback calculator)
- Structured logging for observability
- Flexible function signatures

Usage:
    service = CodeGeneratorService(llm_client=gemini_client)
    result = await service.answer_computational_query(
        query="tính trung bình giá",
        products=product_list
    )
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time

from code_executor import SafeExecutionSandbox, ExecutionResult, ExecutionStatus, get_sandbox
from query_classifier import QueryClassifier, QueryType, ClassificationResult, get_classifier
from data_profiler import DataPatternDetector, DataProfile, get_detector
from fallback_calculator import FallbackCalculator, FallbackResult, get_calculator
from logging_config import get_structured_logger

logger = get_structured_logger(__name__)


class ErrorCategory(Enum):
    """Error categories for retry decision"""
    FIXABLE = "fixable"  # LLM can rewrite function to fix
    UNFIXABLE = "unfixable"  # Data too inconsistent, stop retrying
    UNKNOWN = "unknown"


class PromptVersion(Enum):
    """Prompt template versions for A/B testing"""
    V1_BASIC = "v1_basic"
    V2_FLEXIBLE_SIGNATURE = "v2_flexible_signature"


@dataclass
class GenerationAttempt:
    """Record of a single code generation + execution attempt"""
    attempt_number: int
    generated_code: str
    execution_result: ExecutionResult
    error_category: Optional[ErrorCategory] = None
    error_feedback: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging"""
        return {
            "attempt": self.attempt_number,
            "timestamp": self.timestamp,
            "code_length": len(self.generated_code),
            "execution_status": self.execution_result.status.value,
            "execution_time_ms": self.execution_result.execution_time_ms,
            "error_category": self.error_category.value if self.error_category else None,
            "error": self.execution_result.error if not self.execution_result.is_success() else None
        }


@dataclass
class ComputationalResult:
    """Final result from computational query"""
    success: bool
    numerical_result: Optional[Dict[str, Any]] = None
    natural_language_answer: Optional[str] = None
    method_used: str = "unknown"  # "code_generation", "fallback", "error"
    attempts: List[GenerationAttempt] = field(default_factory=list)
    data_profile: Optional[DataProfile] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging"""
        return {
            "success": self.success,
            "method_used": self.method_used,
            "total_attempts": len(self.attempts),
            "numerical_result": self.numerical_result,
            "error_message": self.error_message,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "data_profile": self.data_profile.to_dict() if self.data_profile else None
        }


class CodeGeneratorService:
    """
    Main service for handling computational queries with code generation.
    
    Features:
    - Data profiling before generation
    - Retry logic with error classification
    - Graceful fallback to regex calculator
    - Versioned prompts
    - Structured logging
    """
    
    MAX_RETRIES = 3
    DEFAULT_PROMPT_VERSION = PromptVersion.V2_FLEXIBLE_SIGNATURE
    
    def __init__(
        self,
        llm_client,
        sandbox: Optional[SafeExecutionSandbox] = None,
        classifier: Optional[QueryClassifier] = None,
        detector: Optional[DataPatternDetector] = None,
        calculator: Optional[FallbackCalculator] = None,
        prompt_version: PromptVersion = DEFAULT_PROMPT_VERSION
    ):
        """
        Initialize code generator service.
        
        Args:
            llm_client: GeminiClient for LLM calls
            sandbox: Code execution sandbox (default: get_sandbox())
            classifier: Query classifier (default: get_classifier())
            detector: Data pattern detector (default: get_detector())
            calculator: Fallback calculator (default: get_calculator())
            prompt_version: Prompt template version
        """
        self.llm_client = llm_client
        self.sandbox = sandbox or get_sandbox()
        self.classifier = classifier or get_classifier(llm_client=llm_client)
        self.detector = detector or get_detector()
        self.calculator = calculator or get_calculator()
        self.prompt_version = prompt_version
    
    def _classify_error(
        self,
        execution_result: ExecutionResult,
        data_profile: DataProfile
    ) -> Tuple[ErrorCategory, str]:
        """
        Classify execution error to decide if retry is worthwhile.
        
        Args:
            execution_result: Result from code execution
            data_profile: Data profile for consistency check
            
        Returns:
            (error_category, feedback_message)
        """
        if execution_result.is_success():
            return ErrorCategory.FIXABLE, ""
        
        error_msg = execution_result.error or ""
        error_type = execution_result.error_type or ""
        
        # UNFIXABLE: Data too inconsistent
        if data_profile.consistency_score < 0.4:
            return (
                ErrorCategory.UNFIXABLE,
                f"Data consistency too low ({data_profile.consistency_score:.1%}). Cannot generate reliable code."
            )
        
        # UNFIXABLE: No valid price data for price-based queries
        if not data_profile.has_consistent_prices():
            if any(kw in error_msg.lower() for kw in ['price', 'giá', 'gia']):
                return (
                    ErrorCategory.UNFIXABLE,
                    "More than 50% of products lack valid price data. Cannot calculate."
                )
        
        # FIXABLE: Parsing errors (can instruct LLM to handle)
        if execution_result.status == ExecutionStatus.RUNTIME_ERROR:
            # KeyError, AttributeError, ValueError often fixable
            if error_type in ['KeyError', 'AttributeError', 'ValueError', 'TypeError']:
                feedback = f"Runtime error: {error_type}: {error_msg}. "
                
                # Give specific hints
                if 'KeyError' in error_type:
                    feedback += "Field not found. Use .get() with defaults. Check field names in data profile."
                elif 'ValueError' in error_type and 'invalid literal' in error_msg:
                    feedback += "Cannot parse value. Add try/except for type conversion. Handle currency symbols."
                elif 'TypeError' in error_type:
                    feedback += "Type mismatch. Check types in data profile and add type conversion."
                else:
                    feedback += "Add error handling for missing/invalid data."
                
                return ErrorCategory.FIXABLE, feedback
        
        # FIXABLE: Syntax errors (LLM can fix)
        if execution_result.status == ExecutionStatus.SYNTAX_ERROR:
            return (
                ErrorCategory.FIXABLE,
                f"Syntax error: {error_msg}. Fix Python syntax."
            )
        
        # UNFIXABLE: Timeout (infinite loop)
        if execution_result.status == ExecutionStatus.TIMEOUT:
            return (
                ErrorCategory.UNFIXABLE,
                "Function timed out (>5s). Likely infinite loop or too slow. Cannot retry."
            )
        
        # UNFIXABLE: Memory limit
        if execution_result.status == ExecutionStatus.MEMORY_LIMIT:
            return (
                ErrorCategory.UNFIXABLE,
                "Memory limit exceeded. Data too large or memory leak. Cannot retry."
            )
        
        # UNKNOWN: Other errors
        return (
            ErrorCategory.UNKNOWN,
            f"Unknown error: {error_type}: {error_msg}"
        )
    
    def _build_generation_prompt(
        self,
        query: str,
        data_profile: DataProfile,
        previous_attempt: Optional[GenerationAttempt] = None
    ) -> str:
        """
        Build prompt for code generation.
        
        Args:
            query: User's question
            data_profile: Detected data patterns
            previous_attempt: Previous failed attempt (for retry)
            
        Returns:
            Prompt string
        """
        if self.prompt_version == PromptVersion.V1_BASIC:
            return self._build_prompt_v1(query, data_profile, previous_attempt)
        elif self.prompt_version == PromptVersion.V2_FLEXIBLE_SIGNATURE:
            return self._build_prompt_v2(query, data_profile, previous_attempt)
        else:
            return self._build_prompt_v2(query, data_profile, previous_attempt)
    
    def _build_prompt_v1(
        self,
        query: str,
        data_profile: DataProfile,
        previous_attempt: Optional[GenerationAttempt] = None
    ) -> str:
        """Prompt v1: Basic fixed signature"""
        prompt_parts = []
        
        prompt_parts.append("You are a Python code generator for data analysis.")
        prompt_parts.append("")
        prompt_parts.append("TASK: Generate a Python function to answer the user's question.")
        prompt_parts.append("")
        prompt_parts.append(f"USER QUESTION: {query}")
        prompt_parts.append("")
        prompt_parts.append("DATA STRUCTURE:")
        prompt_parts.append(data_profile.to_llm_context())
        prompt_parts.append("")
        
        if previous_attempt:
            prompt_parts.append("PREVIOUS ATTEMPT FAILED:")
            prompt_parts.append(f"Error: {previous_attempt.error_feedback}")
            prompt_parts.append("Code:")
            prompt_parts.append("```python")
            prompt_parts.append(previous_attempt.generated_code)
            prompt_parts.append("```")
            prompt_parts.append("")
            prompt_parts.append("FIX THE ERROR and generate improved code.")
            prompt_parts.append("")
        
        prompt_parts.append("REQUIREMENTS:")
        prompt_parts.append("1. Function signature: `def analyze(products: List[Dict[str, Any]]) -> Dict[str, Any]:`")
        prompt_parts.append("2. Return a dict with numerical results (JSON-serializable)")
        prompt_parts.append("3. Handle missing/invalid data gracefully (use .get(), try/except)")
        prompt_parts.append("4. Remove currency symbols ($, ₫, đ) and separators (,.) before parsing prices")
        prompt_parts.append("5. Use only whitelisted modules: re, json, statistics")
        prompt_parts.append("6. NO imports inside function (re, json, statistics are already available)")
        prompt_parts.append("")
        prompt_parts.append("OUTPUT FORMAT:")
        prompt_parts.append("```python")
        prompt_parts.append("def analyze(products: List[Dict[str, Any]]) -> Dict[str, Any]:")
        prompt_parts.append("    # Your code here")
        prompt_parts.append("    return {\"result_key\": result_value}")
        prompt_parts.append("```")
        
        return "\n".join(prompt_parts)
    
    def _build_prompt_v2(
        self,
        query: str,
        data_profile: DataProfile,
        previous_attempt: Optional[GenerationAttempt] = None
    ) -> str:
        """Prompt v2: Flexible signature with more guidance"""
        prompt_parts = []
        
        prompt_parts.append("You are an expert Python code generator specializing in data analysis.")
        prompt_parts.append("")
        prompt_parts.append(f"**USER QUESTION:** {query}")
        prompt_parts.append("")
        prompt_parts.append("**DATA STRUCTURE:**")
        prompt_parts.append(data_profile.to_llm_context())
        prompt_parts.append("")
        
        if previous_attempt:
            prompt_parts.append("**⚠️ PREVIOUS ATTEMPT FAILED:**")
            prompt_parts.append(f"**Error:** {previous_attempt.error_feedback}")
            prompt_parts.append("")
            prompt_parts.append("**Previous Code:**")
            prompt_parts.append("```python")
            prompt_parts.append(previous_attempt.generated_code)
            prompt_parts.append("```")
            prompt_parts.append("")
            prompt_parts.append("**YOU MUST FIX THE ERROR.** Analyze the error carefully and generate corrected code.")
            prompt_parts.append("")
        
        prompt_parts.append("**TASK:**")
        prompt_parts.append("Generate a Python function that:")
        prompt_parts.append("1. Analyzes the product data to answer the question")
        prompt_parts.append("2. Returns accurate numerical results as a dict")
        prompt_parts.append("3. Handles all edge cases (missing data, parsing errors, etc.)")
        prompt_parts.append("")
        
        prompt_parts.append("**FUNCTION SIGNATURE OPTIONS:**")
        prompt_parts.append("Choose the most appropriate signature:")
        prompt_parts.append("")
        prompt_parts.append("```python")
        prompt_parts.append("# Option 1: Standard (recommended for structured data)")
        prompt_parts.append("def analyze(products: List[Dict[str, Any]]) -> Dict[str, Any]:")
        prompt_parts.append("    pass")
        prompt_parts.append("")
        prompt_parts.append("# Option 2: With query context")
        prompt_parts.append("def analyze_all(products: List[Dict[str, Any]], query: str) -> Dict[str, Any]:")
        prompt_parts.append("    pass")
        prompt_parts.append("```")
        prompt_parts.append("")
        
        prompt_parts.append("**CRITICAL RULES:**")
        prompt_parts.append("1. ✅ Use `.get(key, default)` for all dict access")
        prompt_parts.append("2. ✅ Wrap parsing in try/except blocks")
        prompt_parts.append("3. ✅ Clean prices: remove `$`, `₫`, `đ`, `,`, `.` before parsing")
        prompt_parts.append("4. ✅ Check for None/empty values before operations")
        prompt_parts.append("5. ✅ Return JSON-serializable dict (no custom objects)")
        prompt_parts.append("6. ❌ NO imports (re, json, statistics already available)")
        prompt_parts.append("7. ❌ NO infinite loops or recursive calls")
        prompt_parts.append("8. ❌ NO file I/O or network requests")
        prompt_parts.append("")
        
        prompt_parts.append("**EXAMPLE - Price cleaning:**")
        prompt_parts.append("```python")
        prompt_parts.append("def clean_price(price_value):")
        prompt_parts.append("    if isinstance(price_value, (int, float)):")
        prompt_parts.append("        return float(price_value)")
        prompt_parts.append("    if isinstance(price_value, str):")
        prompt_parts.append("        import re")
        prompt_parts.append("        cleaned = re.sub(r'[₫$đ,.\\s]', '', price_value)")
        prompt_parts.append("        try:")
        prompt_parts.append("            return float(cleaned)")
        prompt_parts.append("        except ValueError:")
        prompt_parts.append("            return None")
        prompt_parts.append("    return None")
        prompt_parts.append("```")
        prompt_parts.append("")
        
        prompt_parts.append("**OUTPUT FORMAT:**")
        prompt_parts.append("Return ONLY the Python function, no explanations:")
        prompt_parts.append("```python")
        prompt_parts.append("def analyze(products):")
        prompt_parts.append("    # Implementation")
        prompt_parts.append("    return {\"key\": value}")
        prompt_parts.append("```")
        
        return "\n".join(prompt_parts)
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """
        Extract Python code from LLM response.
        
        Tries multiple patterns:
        1. ```python ... ```
        2. ``` ... ```
        3. def ... (if raw code)
        
        Returns:
            Extracted code or None
        """
        # Pattern 1: ```python code block
        match = re.search(r'```python\s*\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Pattern 2: Generic ``` code block
        match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Pattern 3: Raw code starting with def
        if response.strip().startswith('def '):
            return response.strip()
        
        logger.warning("Could not extract code from LLM response")
        return None
    
    async def _generate_and_execute(
        self,
        query: str,
        products: List[Dict[str, Any]],
        data_profile: DataProfile,
        attempt_number: int,
        previous_attempt: Optional[GenerationAttempt] = None
    ) -> GenerationAttempt:
        """
        Single attempt: generate code + execute.
        
        Returns:
            GenerationAttempt with results
        """
        logger.info(f"📝 Attempt {attempt_number}: Generating code...")
        
        # Build prompt
        prompt = self._build_generation_prompt(query, data_profile, previous_attempt)
        
        # Call LLM
        response = await self.llm_client.generate(prompt)
        
        # Extract code
        code = self._extract_code_from_response(response)
        
        if not code:
            # Failed to extract
            execution_result = ExecutionResult(
                status=ExecutionStatus.SYNTAX_ERROR,
                error="Could not extract Python code from LLM response",
                error_type="ExtractionError"
            )
            
            attempt = GenerationAttempt(
                attempt_number=attempt_number,
                generated_code=response[:500],  # Store partial response
                execution_result=execution_result,
                error_category=ErrorCategory.FIXABLE,
                error_feedback="No code found in response. Return ONLY Python function in ```python block."
            )
            
            logger.error(f"❌ Attempt {attempt_number}: Code extraction failed")
            return attempt
        
        logger.info(f"Generated {len(code)} chars of code")
        
        # Determine function name from code
        func_name_match = re.search(r'def\s+(\w+)\s*\(', code)
        func_name = func_name_match.group(1) if func_name_match else "analyze"
        
        # Execute code
        logger.info(f"⚙️ Executing function '{func_name}'...")
        execution_result = self.sandbox.execute_function(code, func_name, products)
        
        # Classify error if failed
        error_category = None
        error_feedback = None
        
        if not execution_result.is_success():
            error_category, error_feedback = self._classify_error(execution_result, data_profile)
            logger.error(f"❌ Attempt {attempt_number}: {error_category.value} - {error_feedback}")
        else:
            logger.info(f"✅ Attempt {attempt_number}: Success in {execution_result.execution_time_ms:.2f}ms")
        
        attempt = GenerationAttempt(
            attempt_number=attempt_number,
            generated_code=code,
            execution_result=execution_result,
            error_category=error_category,
            error_feedback=error_feedback
        )
        
        return attempt
    
    async def answer_computational_query(
        self,
        query: str,
        products: List[Dict[str, Any]]
    ) -> ComputationalResult:
        """
        Answer computational query using code generation.
        
        Workflow:
        1. Profile data
        2. Generate + execute (retry up to MAX_RETRIES on FIXABLE errors)
        3. If all retries fail → try fallback calculator
        4. Format natural language answer
        
        Args:
            query: User's question
            products: List of product dicts
            
        Returns:
            ComputationalResult with numerical result + NL answer
        """
        start_time = time.time()
        logger.info(f"🔬 Starting computational query: {query}")
        
        # Step 1: Profile data
        logger.info("📊 Profiling data...")
        data_profile = self.detector.analyze(products)
        logger.log_data_profile(query, data_profile, len(products))
        logger.info(f"Data profile: {data_profile.consistency_score:.2%} consistency, {len(data_profile.field_schemas)} fields")
        
        # Step 2: Generate and execute with retries
        attempts: List[GenerationAttempt] = []
        previous_attempt = None
        
        for attempt_num in range(1, self.MAX_RETRIES + 1):
            attempt = await self._generate_and_execute(
                query=query,
                products=products,
                data_profile=data_profile,
                attempt_number=attempt_num,
                previous_attempt=previous_attempt
            )
            
            attempts.append(attempt)
            
            # Log generation attempt
            logger.log_generation_attempt(
                attempt_number=attempt_num,
                query=query,
                generated_code=attempt.generated_code,
                execution_result=attempt.execution_result,
                error_category=attempt.error_category.value if attempt.error_category else None
            )
            
            # Success!
            if attempt.execution_result.is_success():
                logger.info(f"✅ Code generation succeeded on attempt {attempt_num}")
                
                # Format natural language answer
                nl_answer = await self._format_answer(
                    query=query,
                    numerical_result=attempt.execution_result.output,
                    method="code_generation"
                )
                
                total_duration_ms = (time.time() - start_time) * 1000
                logger.log_final_result(
                    query=query,
                    success=True,
                    method_used="code_generation",
                    total_attempts=attempt_num,
                    total_duration_ms=total_duration_ms
                )
                
                return ComputationalResult(
                    success=True,
                    numerical_result=attempt.execution_result.output,
                    natural_language_answer=nl_answer,
                    method_used="code_generation",
                    attempts=attempts,
                    data_profile=data_profile
                )
            
            # Check if error is fixable
            if attempt.error_category == ErrorCategory.UNFIXABLE:
                logger.warning(f"⚠️ UNFIXABLE error on attempt {attempt_num}, stopping retries")
                logger.log_error_classification(
                    error_type=attempt.execution_result.error_type or "unknown",
                    error_category=attempt.error_category.value,
                    feedback=attempt.error_feedback or "",
                    is_retryable=False
                )
                break
            
            # Continue retrying if FIXABLE
            if attempt.error_category == ErrorCategory.FIXABLE:
                logger.info(f"🔄 Error is FIXABLE, will retry (attempt {attempt_num}/{self.MAX_RETRIES})")
                logger.log_error_classification(
                    error_type=attempt.execution_result.error_type or "unknown",
                    error_category=attempt.error_category.value,
                    feedback=attempt.error_feedback or "",
                    is_retryable=True
                )
                previous_attempt = attempt
            else:
                # UNKNOWN error - try once more but don't keep retrying
                logger.warning(f"⚠️ UNKNOWN error on attempt {attempt_num}")
                if attempt_num >= 2:
                    break
                previous_attempt = attempt
        
        # Step 3: All retries failed → try fallback
        logger.info("🔧 Code generation failed, trying fallback calculator...")
        fallback_result = self.calculator.calculate(query, products)
        logger.log_fallback_attempt(query, fallback_result, reason="code_generation_failed")
        
        if fallback_result.success:
            logger.info(f"✅ Fallback calculator succeeded: {fallback_result.method}")
            
            nl_answer = await self._format_answer(
                query=query,
                numerical_result=fallback_result.result,
                method="fallback"
            )
            
            total_duration_ms = (time.time() - start_time) * 1000
            logger.log_final_result(
                query=query,
                success=True,
                method_used="fallback",
                total_attempts=len(attempts),
                total_duration_ms=total_duration_ms
            )
            
            return ComputationalResult(
                success=True,
                numerical_result=fallback_result.result,
                natural_language_answer=nl_answer,
                method_used="fallback",
                attempts=attempts,
                data_profile=data_profile
            )
        
        # Step 4: Even fallback failed → return error
        logger.error("❌ All methods failed (code generation + fallback)")
        
        error_msg = self._build_error_message(attempts, fallback_result, data_profile)
        
        total_duration_ms = (time.time() - start_time) * 1000
        logger.log_final_result(
            query=query,
            success=False,
            method_used="error",
            total_attempts=len(attempts),
            total_duration_ms=total_duration_ms,
            error_message=error_msg
        )
        
        return ComputationalResult(
            success=False,
            method_used="error",
            attempts=attempts,
            data_profile=data_profile,
            error_message=error_msg
        )
    
    def _build_error_message(
        self,
        attempts: List[GenerationAttempt],
        fallback_result: FallbackResult,
        data_profile: DataProfile
    ) -> str:
        """Build user-friendly error message"""
        parts = []
        parts.append("Không thể tính toán kết quả chính xác do:")
        parts.append("")
        
        # Check data quality
        if data_profile.consistency_score < 0.5:
            parts.append(f"- Dữ liệu không đồng nhất (consistency: {data_profile.consistency_score:.1%})")
        
        if not data_profile.has_consistent_prices():
            parts.append("- Hơn 50% sản phẩm thiếu thông tin giá hợp lệ")
        
        # Show last attempt error
        if attempts:
            last_attempt = attempts[-1]
            if last_attempt.error_category == ErrorCategory.UNFIXABLE:
                parts.append(f"- Lỗi không thể khắc phục: {last_attempt.error_feedback}")
        
        # Show fallback error
        if not fallback_result.success:
            parts.append(f"- Phương pháp dự phòng cũng thất bại: {fallback_result.error}")
        
        parts.append("")
        parts.append("Đề xuất: Kiểm tra lại dữ liệu crawl hoặc đặt câu hỏi đơn giản hơn.")
        
        return "\n".join(parts)
    
    async def _format_answer(
        self,
        query: str,
        numerical_result: Dict[str, Any],
        method: str
    ) -> str:
        """
        Format numerical result into natural language answer.
        
        Args:
            query: User's question
            numerical_result: Dict with numerical results
            method: "code_generation" or "fallback"
            
        Returns:
            Natural language answer string
        """
        prompt_parts = []
        
        prompt_parts.append("Bạn là trợ lý AI chuyên trả lời câu hỏi về dữ liệu sản phẩm.")
        prompt_parts.append("")
        prompt_parts.append(f"CÂU HỎI: {query}")
        prompt_parts.append("")
        prompt_parts.append("KẾT QUẢ TÍNH TOÁN (đã được tính chính xác bằng code):")
        prompt_parts.append(json.dumps(numerical_result, ensure_ascii=False, indent=2))
        prompt_parts.append("")
        prompt_parts.append("NHIỆM VỤ:")
        prompt_parts.append("Trả lời câu hỏi bằng ngôn ngữ tự nhiên dựa trên kết quả tính toán trên.")
        prompt_parts.append("- KHÔNG ĐƯỢC tính toán lại")
        prompt_parts.append("- CHỈ diễn đạt kết quả đã có một cách rõ ràng")
        prompt_parts.append("- Trả lời ngắn gọn, súc tích (2-3 câu)")
        prompt_parts.append("")
        prompt_parts.append("CÂU TRẢ LỜI:")
        
        prompt = "\n".join(prompt_parts)
        
        try:
            answer = await self.llm_client.generate(prompt)
            return answer.strip()
        except Exception as e:
            logger.error(f"Failed to format answer: {e}")
            # Fallback: return JSON
            return f"Kết quả: {json.dumps(numerical_result, ensure_ascii=False)}"


# Singleton instance
_service: Optional[CodeGeneratorService] = None

def get_service(llm_client) -> CodeGeneratorService:
    """Get or create singleton service instance"""
    global _service
    if _service is None:
        _service = CodeGeneratorService(llm_client=llm_client)
    return _service
