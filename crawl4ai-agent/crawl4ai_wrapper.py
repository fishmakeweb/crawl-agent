"""
crawl4ai Wrapper with Gemini Integration
Provides intelligent navigation and data extraction
"""
import os
import time
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

from crawl4ai import AsyncWebCrawler, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.chunking_strategy import RegexChunking
import google.generativeai as genai

logger = logging.getLogger(__name__)


class Crawl4AIWrapper:
    """Wrapper around crawl4ai with Gemini LLM integration"""

    def __init__(
        self,
        kafka_publisher=None,
        gemini_client=None,
        model_name: Optional[str] = None
    ):
        """Initialize with Gemini API, optional shared client, and Kafka publisher."""
        self.kafka_publisher = kafka_publisher
        self.gemini_client = gemini_client
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")
        self.model = None
        
        # Pagination detection model (can be different/cheaper model)
        self.pagination_model_name = os.getenv("PAGINATION_DETECTION_MODEL", "models/gemini-2.0-flash-exp")
        self.pagination_model = None
        self.use_external_pagination_model = False  # Flag to track if using external API (not Gemini)
        self.min_pagination_confidence = float(os.getenv("MIN_PAGINATION_CONFIDENCE", "0.7"))
        
        # Re-detection strategy for adaptive pagination
        self.redetection_strategy = os.getenv("PAGINATION_REDETECTION_STRATEGY", "hybrid").lower()
        milestones_str = os.getenv("PAGINATION_REDETECTION_MILESTONES", "5,10,15,20,30")
        self.redetection_milestones = [int(m.strip()) for m in milestones_str.split(",") if m.strip().isdigit()]
        self.confidence_margin = float(os.getenv("PAGINATION_CONFIDENCE_MARGIN", "0.1"))
        
        # Concurrency control for parallel extraction
        self.max_concurrent_tabs = int(os.getenv("MAX_CONCURRENT_CRAWLER_TABS", "5"))
        self.kafka_verbose_events = os.getenv("KAFKA_VERBOSE_EVENTS", "false").lower() == "true"
        logger.info(f"Initialized with max_concurrent_tabs={self.max_concurrent_tabs}, kafka_verbose={self.kafka_verbose_events}, pagination_confidence_threshold={self.min_pagination_confidence}, redetection_strategy={self.redetection_strategy}, milestones={self.redetection_milestones}")

        if gemini_client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            
            # Check if pagination model is a Gemini model or external (OpenAI-compatible)
            if self.pagination_model_name.startswith("models/") or self.pagination_model_name.startswith("gemini"):
                # It's a Gemini model
                if self.pagination_model_name == self.model_name:
                    self.pagination_model = self.model
                    logger.info(f"Crawl4AIWrapper initialized with dedicated Gemini model (pagination uses same model)")
                else:
                    self.pagination_model = genai.GenerativeModel(self.pagination_model_name)
                    logger.info(f"Crawl4AIWrapper initialized with extraction model: {self.model_name}, pagination model: {self.pagination_model_name}")
            else:
                # It's an external model (OpenAI-compatible like gpt-4o-mini)
                self.use_external_pagination_model = True
                logger.info(f"Pagination uses external OpenAI-compatible model: {self.pagination_model_name}")
        else:
            # Shared Gemini client handles its own configuration/caching
            self.model = getattr(gemini_client, "model", None)
            
            # Check if pagination should use external model instead of shared Gemini client
            if self.pagination_model_name.startswith("models/") or self.pagination_model_name.startswith("gemini"):
                self.pagination_model = self.model  # Reuse shared client's model for pagination
                logger.info("Crawl4AIWrapper using shared Gemini client instance (for both extraction and pagination)")
            else:
                self.use_external_pagination_model = True
                logger.info(f"Pagination uses external OpenAI-compatible model via shared client: {self.pagination_model_name}")

    async def _generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using either the shared Gemini client or the local model."""
        if self.gemini_client is not None:
            return await self.gemini_client.generate(prompt, **kwargs)

        if self.model is None:
            raise RuntimeError("Gemini model is not configured")

        return await asyncio.to_thread(self._sync_generate, prompt, kwargs)

    def _sync_generate(self, prompt: str, extra_kwargs: Dict[str, Any]) -> str:
        """Synchronous generation helper for asyncio.to_thread."""
        response = self.model.generate_content(prompt, **extra_kwargs)
        if hasattr(response, "text") and response.text:
            return response.text
        return str(response)

    async def _generate_text_with_model(self, prompt: str, model_override: Optional[str] = None, **kwargs) -> str:
        """
        Generate text using specified model (supports external LLM override).
        
        Args:
            prompt: The prompt to send
            model_override: Optional model name to use instead of default
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        # PRIORITY 1: If external pagination model is configured, always use direct API call
        # This handles the case where pagination uses external LLM (gpt-4o-mini) but main model uses shared client (gpt-4o)
        if self.use_external_pagination_model:
            import httpx
            
            base_url = os.getenv("EXTERNAL_LLM_BASE_URL", "https://v98store.com/v1")
            api_key = os.getenv("EXTERNAL_LLM_API_KEY")
            model = model_override or os.getenv("EXTERNAL_LLM_MODEL_NAME", "gpt-4o")
            
            if not api_key:
                raise ValueError("EXTERNAL_LLM_API_KEY not configured for external model calls")
            
            logger.info(f"🔧 Using external pagination model: {model} (via {base_url})")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": kwargs.get("temperature", 0.3),
                        "max_tokens": kwargs.get("max_tokens", 2000)
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        
        # PRIORITY 2: If using shared Gemini client and it supports model override
        if self.gemini_client is not None and hasattr(self.gemini_client, 'generate_with_model'):
            return await self.gemini_client.generate_with_model(prompt, model=model_override, **kwargs)
        
        # PRIORITY 3: If using shared Gemini client without model override support (fallback to default)
        if self.gemini_client is not None:
            logger.warning(f"Shared client doesn't support model override, using default model instead of {model_override}")
            return await self.gemini_client.generate(prompt, **kwargs)

    async def answer_query(self, context: str, query: str) -> str:
        """
        Answer a question based on provided context using Gemini (RAG).
        
        Args:
            context: JSON or text context from previous crawls
            query: User's question
            
        Returns:
            Natural language answer
        """
        try:
            # Truncate context if too large (Gemini 2.0 Flash has 1M context, but let's be safe/fast)
            safe_context = context[:100000] 
            
            prompt = f"""
You are a helpful data assistant. Answer the user's question based ONLY on the provided context data.

CONTEXT DATA (JSON/Text):
{safe_context}

USER QUESTION: "{query}"

INSTRUCTIONS:
1. Analyze the context data to find the answer.
2. If the answer is found, provide a clear, concise summary.
3. If the answer is NOT in the context, say "I cannot find that information in the crawled data."
4. Do not hallucinate information not present in the context.

Answer:
"""
            return await self._generate_text(prompt)
        except Exception as e:
            logger.error(f"Query answer failed: {str(e)}", exc_info=True)
            return "Sorry, I encountered an error analyzing the data."

    async def intelligent_crawl(
        self,
        url: str,
        prompt: str,
        job_id: Optional[str] = None,
        user_id: Optional[str] = None,
        navigation_steps: Optional[List[Dict]] = None,
        extract_schema: Optional[Dict] = None,
        max_pages: int = 50
    ) -> Dict[str, Any]:
        """
        Execute intelligent crawl with navigation and extraction

        Args:
            url: Target URL
            prompt: User's natural language request
            job_id: Optional job ID from C# (for Kafka progress events)
            user_id: Optional user ID from C# (for Kafka progress events)
            navigation_steps: Optional pre-defined navigation steps
            extract_schema: Optional extraction schema
            max_pages: Maximum pages to crawl (pagination limit)

        Returns:
            Dictionary with success, data, navigation_result, execution_time_ms, 
            conversation_name, embedding_data (embedding_text, embedding_vector, schema_type, quality_score)
        """
        start_time = time.time()
        conversation_name = None
        analysis_result = None

        try:
            async with AsyncWebCrawler(verbose=True, headless=True) as crawler:
                # OPTIMIZED STEP 1: Combined analysis + navigation planning (single AI call)
                if not navigation_steps:
                    logger.info("OPTIMIZED: Analyzing page and planning navigation in single AI call...")

                    # PUBLISH EVENT 1: Navigation Planning Started
                    if self.kafka_publisher and job_id:
                        self.kafka_publisher.publish_progress(
                            "NavigationPlanningStarted",
                            job_id,
                            user_id or "unknown",
                            {"url": url, "prompt": prompt}
                        )

                    # Combined AI call: conversation name + navigation plan + extraction fields + page limit
                    analysis_result = await self._analyze_and_plan_with_name(crawler, url, prompt)
                    
                    conversation_name = analysis_result["conversation_name"]
                    navigation_steps = analysis_result["navigation_plan"]
                    extraction_fields = analysis_result["data_extraction_fields"]
                    pagination_override_info = analysis_result.get("pagination_override_info")  # Extract for Kafka event
                    
                    # NEW: Extract pagination control fields
                    skip_pagination = analysis_result.get("skip_pagination", False)
                    specific_pages = analysis_result.get("specific_pages")
                    start_page = analysis_result.get("start_page")
                    
                    # Override max_pages if user specified a limit in prompt (e.g., "3 trang đầu")
                    extracted_max_pages = analysis_result.get("extracted_max_pages")
                    if extracted_max_pages is not None and isinstance(extracted_max_pages, (int, float)):
                        max_pages = int(extracted_max_pages)
                        logger.info(f"📊 User specified page limit detected: {max_pages} pages (overriding default)")
                    
                    # Handle specific pages request
                    if specific_pages and len(specific_pages) > 0:
                        logger.info(f"🎯 SPECIFIC PAGES MODE: Will crawl only pages {specific_pages}")
                        # Generate URLs for specific pages
                        page_urls = await self._generate_specific_page_urls(url, specific_pages, start_page or 1)
                        logger.info(f"📋 Generated {len(page_urls)} URLs for specific pages")
                        
                        # Extract data from specific pages in parallel
                        extracted_data = await self._extract_data_parallel(
                            page_urls,
                            prompt,
                            extract_schema,
                            self.kafka_publisher,
                            job_id,
                            user_id
                        )
                        
                        execution_time = (time.time() - start_time) * 1000
                        
                        return {
                            "success": True,
                            "data": extracted_data,
                            "navigation_result": {
                                "final_url": url,
                                "executed_steps": [{"action": "extract_specific_pages", "pages": specific_pages, "success": True}],
                                "pages_collected": len(page_urls)
                            },
                            "execution_time_ms": execution_time,
                            "conversation_name": conversation_name
                        }
                    
                    logger.info(f"✅ OPTIMIZED: Generated name '{conversation_name}' + {len(navigation_steps)} steps in single call (max_pages={max_pages}, skip_pagination={skip_pagination})")

                    # PUBLISH EVENT 2: Navigation Planning Completed
                    if self.kafka_publisher and job_id:
                        event_data = {
                            "url": url,
                            "conversation_name": conversation_name,
                            "steps_count": len(navigation_steps),
                            "steps": navigation_steps,
                            "extraction_fields": extraction_fields
                        }
                        # Add pagination override info if available
                        if pagination_override_info:
                            event_data.update(pagination_override_info)
                        
                        self.kafka_publisher.publish_progress(
                            "NavigationPlanningCompleted",
                            job_id,
                            user_id or "unknown",
                            event_data
                        )
                else:
                    # Fallback: navigation steps provided externally
                    logger.info("Using provided navigation steps")
                    if conversation_name is None:
                        # Simple fallback name generation (no AI call)
                        words = prompt.split()
                        conversation_name = ' '.join(words[:6]) + ('...' if len(words) > 6 else '')

                # Step 2: Execute navigation steps (with dynamic pagination re-detection)
                # PUBLISH EVENT 3: Navigation Execution Started
                if self.kafka_publisher and job_id:
                    self.kafka_publisher.publish_progress(
                        "NavigationExecutionStarted",
                        job_id,
                        user_id or "unknown",
                        {
                            "url": url,
                            "total_steps": len(navigation_steps)
                        }
                    )

                navigation_result = await self._execute_navigation(
                    crawler, url, navigation_steps, max_pages,
                    original_prompt=prompt,
                    kafka_publisher=self.kafka_publisher,
                    job_id=job_id,
                    user_id=user_id
                )

                # Step 3: Extract data from final page(s)
                # Use parallel extraction when multiple pages collected
                page_urls = navigation_result.get("page_urls", [])
                collected_pages = navigation_result.get("collected_pages", [])
                
                if len(page_urls) > 0:
                    # Parallel extraction: spawn one crawler per page URL
                    logger.info(f"Using parallel extraction for {len(page_urls)} paginated pages")
                    extracted_data = await self._extract_data_parallel(
                        page_urls,
                        prompt,
                        extract_schema,
                        kafka_publisher=self.kafka_publisher,
                        job_id=job_id,
                        user_id=user_id
                    )
                else:
                    # Sequential extraction: single page or no pagination
                    logger.info("Using sequential extraction (single page)")
                    extracted_data = await self._extract_data(
                        crawler,
                        navigation_result["final_url"],
                        collected_pages,
                        prompt,
                        extract_schema,
                        kafka_publisher=self.kafka_publisher,
                        job_id=job_id,
                        user_id=user_id
                    )

                # OPTIMIZED STEP 4: Generate embedding + analyze data quality (single operation)
                logger.info("OPTIMIZED: Generating embedding + analyzing data quality...")
                
                # PUBLISH EVENT: Embedding Generation Started
                if self.kafka_publisher and job_id:
                    self.kafka_publisher.publish_progress(
                        "EmbeddingGenerationStarted",
                        job_id,
                        user_id or "unknown",
                        {"data_items": len(extracted_data) if isinstance(extracted_data, list) else 1}
                    )
                
                embedding_data = await self._generate_embedding_with_stats(
                    extracted_data if isinstance(extracted_data, list) else [extracted_data],
                    conversation_name,
                    prompt
                )
                
                # PUBLISH EVENT: Embedding Generation Completed
                if self.kafka_publisher and job_id:
                    self.kafka_publisher.publish_progress(
                        "EmbeddingGenerationCompleted",
                        job_id,
                        user_id or "unknown",
                        {
                            "schema_type": embedding_data["schema_type"],
                            "quality_score": embedding_data["quality_score"],
                            "embedding_dimensions": len(embedding_data["embedding_vector"])
                        }
                    )

                execution_time = (time.time() - start_time) * 1000

                return {
                    "success": True,
                    "data": extracted_data,
                    "navigation_result": navigation_result,
                    "execution_time_ms": execution_time,
                    "conversation_name": conversation_name,
                    "embedding_data": embedding_data,  # NEW: Pre-generated embedding
                    "error": None
                }

        except Exception as e:
            logger.error(f"Crawl failed: {str(e)}", exc_info=True)
            execution_time = (time.time() - start_time) * 1000

            # Generate conversation name even on failure if not already done
            if conversation_name is None:
                try:
                    words = prompt.split()
                    conversation_name = ' '.join(words[:6]) + ('...' if len(words) > 6 else '')
                except:
                    conversation_name = "Failed Crawl"

            return {
                "success": False,
                "data": [],
                "navigation_result": {
                    "final_url": url,
                    "executed_steps": [],
                    "pages_collected": 0
                },
                "execution_time_ms": execution_time,
                "conversation_name": conversation_name,
                "error": str(e)
            }

    async def analyze_and_plan(self, url: str, prompt: str) -> List[Dict[str, Any]]:
        """
        Analyze page and generate navigation plan (without executing)

        Args:
            url: Target URL
            prompt: User's intent

        Returns:
            List of navigation steps
        """
        async with AsyncWebCrawler(verbose=True, headless=True) as crawler:
            return await self._plan_navigation(crawler, url, prompt)

    async def _analyze_and_plan_with_name(
        self, crawler, url: str, prompt: str
    ) -> Dict[str, Any]:
        """
        OPTIMIZED: Combine conversation name generation + navigation planning in single AI call.
        This saves ~330 tokens compared to making 2 separate calls.

        Args:
            crawler: AsyncWebCrawler instance
            url: Target URL
            prompt: User's natural language request

        Returns:
            Dict with conversation_name, navigation_plan, and data_extraction_fields
        """
        try:
            # Crawl the initial page
            result = await crawler.arun(url=url)

            # Deep clean HTML first to remove JS/CSS (50-70% reduction)
            raw_html = result.html if result.html else result.cleaned_html
            deep_cleaned = self._deep_clean_html(raw_html)

            # STEP 0: Dedicated pagination detection (before main planning to save tokens)
            logger.info("Running dedicated pagination detection on full page...")
            pagination_detection = await self._detect_pagination_with_model(deep_cleaned, url, prompt)
            
            pagination_step_injected = None
            if pagination_detection.get("has_pagination"):
                confidence = pagination_detection.get("confidence", 0)
                logger.info(f"Pagination detected with confidence {confidence:.2f}: {pagination_detection.get('selector')}")
                
                # Pre-build pagination step to inject into plan
                pagination_step_injected = {
                    "action": "paginate",
                    "selector": pagination_detection.get("selector", "a[href*='page=']"),
                    "description": f"Navigate through all pages (pattern: {pagination_detection.get('href_pattern', 'detected')})"
                }
            else:
                logger.info(f"No pagination detected: {pagination_detection.get('reasoning')}")

            # Get first 30000 chars for comprehensive analysis
            page_html = deep_cleaned[:30000]
            logger.info(f"Analysis HTML: {len(raw_html)} -> {len(deep_cleaned)} chars ({100 - int(len(deep_cleaned)/len(raw_html)*100)}% reduction), using first {len(page_html)} chars")
            logger.info(f"📝 USER PROMPT: '{prompt}'")

            # COMBINED prompt: conversation name + navigation planning + page limit extraction
            analysis_prompt = f"""You are analyzing a webpage to help with data extraction.

USER REQUEST: "{prompt}"
WEBPAGE URL: {url}

PAGE HTML (simplified):
{page_html}

Your tasks:
1. Create a SHORT conversation name (max 6 words) that captures what the user wants
2. Plan the navigation strategy to fulfill the user's request
3. Identify the key data fields to extract
4. **IMPORTANT**: Analyze user INTENT regarding pagination

**CRITICAL INTENT ANALYSIS**:
- If user does NOT explicitly request multiple pages ("all pages", "next pages", "crawl X pages"), assume SINGLE-PAGE intent
- Phrases like "on this page", "from here", "this page", "current page" indicate single-page focus
- When in doubt, prefer single-page extraction (skip_pagination: true)
- Only enable pagination if user clearly requests: "all pages", "multiple pages", "crawl pages X to Y", "tất cả trang", or specific page counts

Respond in JSON format:
{{
    "conversation_name": "Short descriptive name (max 6 words)",
    "max_pages": <number or null>,
    "skip_pagination": <true or false>,
    "specific_pages": [<page numbers>] or null,
    "start_page": <number or null>,
    "navigation_plan": [
        {{"action": "click|select|input|scroll|paginate|extract|wait", "selector": "...", "value": "...", "description": "..."}}
    ],
    "data_extraction_fields": ["field1", "field2", "field3"]
}}

PAGINATION CONTROL DETECTION (Vietnamese & English):

**SKIP PAGINATION** (crawl only current page):
Vietnamese phrases:
- "trang này thôi", "chỉ trang này", "ở trang này", "từ trang này", "trên trang này", "lấy trang này", "chỉ lấy trang này", "lấy trang hiện tại" → skip_pagination: true, max_pages: 1
- "không crawl thêm", "không lấy thêm", "không cần trang khác" → skip_pagination: true, max_pages: 1

English phrases:
- "only this page", "this page only", "on this page", "from this page", "at this page", "current page only" → skip_pagination: true, max_pages: 1
- "just this page", "this single page", "single page", "no pagination", "don't crawl more" → skip_pagination: true, max_pages: 1
- Contextual: "product on this page", "data on this page", "info from here", "extract here" → skip_pagination: true, max_pages: 1

**SPECIFIC PAGES** (crawl exact page numbers):
- "crawl page 3 and 4", "trang 3 và 4", "page 3, 4, 5" → specific_pages: [3, 4] or [3, 4, 5], skip_pagination: true
- "only page 2", "chỉ trang 2" → specific_pages: [2], skip_pagination: true

**START PAGE** (detect from URL pattern):
- URL contains "/page/3" or "?page=3" or "?p=3" → start_page: 3
- Combined with specific_pages: URL has /page/3 + "crawl page 3 and 4" → start_page: 3, specific_pages: [3, 4]

**PAGE LIMIT** (crawl multiple pages from start):
- "3 trang đầu", "3 trang đầu tiên", "first 3 pages" → max_pages: 3, skip_pagination: false
- "5 trang", "5 pages" → max_pages: 5, skip_pagination: false
- "10 trang", "first 10 pages" → max_pages: 10, skip_pagination: false

**ALL PAGES** (unlimited pagination):
- "tất cả", "all pages", "toàn bộ", no limit mentioned → max_pages: null, skip_pagination: false

**DEFAULT BEHAVIOR** (Conservative approach - prefer single-page when ambiguous):
- If user prompt is vague/short AND contains location indicators ("this", "here", "current") → skip_pagination: true, max_pages: 1
- If user explicitly requests multi-page ("all", "multiple", "next pages", "X pages") → skip_pagination: false
- If URL has page number (/page/N) but NO multi-page request → skip_pagination: true, start_page: N, max_pages: 1
- If completely ambiguous AND page has pagination → skip_pagination: false, max_pages: null (only in this case)

NAVIGATION ACTIONS:
- "click": Navigate to new page via link/button (provide CSS selector)
- "select": Choose option from dropdown (provide selector + value)
- "input": Enter text into form field (provide selector + value)
- "scroll": Scroll to load lazy content
- "paginate": Navigate through multiple pages (provide next button selector)
- "extract": Extract data from current page (always include at end)
- "wait": Wait for dynamic content to load

CRITICAL - PAGINATION DETECTION:
E-commerce and listing pages often have pagination at BOTTOM of page.
- Look for: .pagination, .next-page, .load-more, a[rel='next'], button.page-next
- Text patterns: "Next", "Next Page", "Load More", page numbers (1 2 3...)
- If you see product/item listings, ALWAYS check for pagination
- Pagination comes BEFORE final "extract" action

Example responses:

1. Normal pagination (all pages):
{{
    "conversation_name": "iPhone Price Comparison",
    "max_pages": null,
    "skip_pagination": false,
    "specific_pages": null,
    "start_page": null,
    "navigation_plan": [
        {{"action": "click", "selector": "a[href*='electronics']", "description": "Navigate to electronics"}},
        {{"action": "paginate", "selector": ".pagination .next", "description": "Collect all product pages"}},
        {{"action": "extract", "description": "Extract product data"}}
    ],
    "data_extraction_fields": ["product_name", "price", "rating", "reviews_count"]
}}

2. Single page only ("trang này thôi"):
{{
    "conversation_name": "Current Page Products",
    "max_pages": 1,
    "skip_pagination": true,
    "specific_pages": null,
    "start_page": null,
    "navigation_plan": [
        {{"action": "extract", "description": "Extract product data from current page only"}}
    ],
    "data_extraction_fields": ["product_name", "price"]
}}

3. Specific pages ("crawl page 3 and 4" with URL /page/3):
{{
    "conversation_name": "Specific Pages Products",
    "max_pages": 2,
    "skip_pagination": true,
    "specific_pages": [3, 4],
    "start_page": 3,
    "navigation_plan": [
        {{"action": "extract", "description": "Extract data from pages 3 and 4"}}
    ],
    "data_extraction_fields": ["product_name", "price"]
}}

4. Vague prompt with page number in URL (URL: /page/8, prompt: "product info on this page"):
{{
    "conversation_name": "Current Page Product Info",
    "max_pages": 1,
    "skip_pagination": true,
    "specific_pages": null,
    "start_page": 8,
    "navigation_plan": [
        {{"action": "extract", "description": "Extract product info from current page (page 8) only"}}
    ],
    "data_extraction_fields": ["product_name", "brand", "price"]
}}
REASONING: User said "on this page" without mentioning other pages, URL shows /page/8 → Extract from page 8 only, no pagination

Return ONLY the JSON object, no other text.
"""

            response_text = (await self._generate_text(analysis_prompt)).strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)
            
            # Validate structure
            conversation_name = result.get("conversation_name", "Data Collection")
            navigation_plan = result.get("navigation_plan", [{"action": "extract", "description": "Extract data"}])
            extraction_fields = result.get("data_extraction_fields", [])
            extracted_max_pages = result.get("max_pages")  # Extract page limit from AI response
            
            # NEW: Extract pagination control fields
            skip_pagination = result.get("skip_pagination", False)
            specific_pages = result.get("specific_pages")  # List of page numbers or None
            start_page = result.get("start_page")  # Starting page number or None
            
            # SMART PAGINATION LOGIC: Remove pagination steps if skip_pagination=true
            if skip_pagination:
                # Remove all 'paginate' actions from navigation_plan
                original_plan_length = len(navigation_plan)
                navigation_plan = [step for step in navigation_plan if step.get("action") != "paginate"]
                removed_count = original_plan_length - len(navigation_plan)
                
                if removed_count > 0:
                    logger.info(f"🚫 Skip pagination enabled: Removed {removed_count} pagination step(s) from plan")
                
                # Override max_pages to 1 if not already set for specific pages
                if not specific_pages and extracted_max_pages != 1:
                    extracted_max_pages = 1
                    logger.info(f"🚫 Skip pagination: max_pages set to 1")
            
            # INJECT/OVERRIDE pagination step if detected by dedicated model (only if NOT skipping)
            pagination_override_info = None  # Track override for Kafka event
            if pagination_step_injected and not skip_pagination:
                confidence = pagination_detection.get("confidence", 0)
                is_fallback = "BeautifulSoup" in pagination_detection.get("reasoning", "")
                should_override = confidence >= 0.5 or is_fallback
                
                if should_override:
                    # Find if paginate already exists in plan
                    paginate_step = next((s for s in navigation_plan if s.get("action") == "paginate"), None)
                    
                    if paginate_step:
                        # Override existing pagination step with better detection
                        old_selector = paginate_step.get("selector", "none")
                        paginate_step["selector"] = pagination_step_injected["selector"]
                        paginate_step["description"] = f"Updated: {pagination_step_injected['description']} (overrode '{old_selector}')"
                        
                        # Track override for Kafka event
                        pagination_override_info = {
                            "pagination_selector_overridden": True,
                            "old_selector": old_selector,
                            "new_selector": pagination_step_injected["selector"],
                            "confidence": confidence,
                            "detection_method": "BeautifulSoup fallback" if is_fallback else "AI detection"
                        }
                        
                        logger.info(f"Overrode pagination selector: '{old_selector}' → '{paginate_step['selector']}' (confidence: {confidence:.2f}, method: {'BeautifulSoup' if is_fallback else 'AI'})")
                    else:
                        # No paginate yet → insert before extract
                        extract_index = next((i for i, s in enumerate(navigation_plan) if s.get("action") == "extract"), None)
                        if extract_index is not None:
                            navigation_plan.insert(extract_index, pagination_step_injected)
                            logger.info(f"Injected new pagination step at position {extract_index} (confidence: {confidence:.2f})")
                        else:
                            navigation_plan.append(pagination_step_injected)
                            logger.info(f"Injected new pagination step at end (confidence: {confidence:.2f})")
                else:
                    logger.info(f"Skipped pagination injection/override: confidence {confidence:.2f} below threshold (0.5)")
            
            logger.info(f"Analysis complete: '{conversation_name}' with {len(navigation_plan)} steps, {len(extraction_fields)} fields")
            if skip_pagination:
                logger.info(f"📄 Pagination mode: DISABLED (single page or specific pages)")
            if specific_pages:
                logger.info(f"📋 Specific pages requested: {specific_pages}")
            if start_page:
                logger.info(f"🎯 Starting from page: {start_page}")

            return {
                "conversation_name": conversation_name,
                "navigation_plan": navigation_plan,
                "data_extraction_fields": extraction_fields,
                "pagination_override_info": pagination_override_info,  # Pass to caller for Kafka event
                "extracted_max_pages": extracted_max_pages,  # Pass extracted page limit to caller
                "skip_pagination": skip_pagination,  # NEW: Pass pagination control flag
                "specific_pages": specific_pages,  # NEW: Pass specific page numbers
                "start_page": start_page  # NEW: Pass starting page number
            }

        except Exception as e:
            logger.error(f"Analysis and planning failed: {str(e)}", exc_info=True)
            # Return safe fallback
            return {
                "conversation_name": "Data Collection",
                "navigation_plan": [{"action": "extract", "description": "Extract data from current page"}],
                "data_extraction_fields": []
            }

    async def _plan_navigation(
        self, crawler, url: str, prompt: str
    ) -> List[Dict[str, Any]]:
        """
        DEPRECATED: Use _analyze_and_plan_with_name() instead for token efficiency.
        Kept for backward compatibility only.

        Use Gemini to analyze page structure and create navigation plan

        Args:
            crawler: AsyncWebCrawler instance
            url: Target URL
            prompt: User's natural language request

        Returns:
            List of navigation steps with actions, selectors, descriptions
        """
        try:
            # Crawl the initial page
            result = await crawler.arun(url=url)

            # Deep clean HTML first to remove JS/CSS (50-70% reduction)
            # This allows us to send MORE actual content to Gemini within token limits
            raw_html = result.html if result.html else result.cleaned_html
            deep_cleaned = self._deep_clean_html(raw_html)

            # Get first 30000 chars (increased from 8000 since JS/CSS removed)
            # This is critical for detecting pagination controls at bottom of page
            page_html = deep_cleaned[:30000]
            logger.info(f"Navigation planning HTML: {len(raw_html)} -> {len(deep_cleaned)} chars ({100 - int(len(deep_cleaned)/len(raw_html)*100)}% reduction), using first {len(page_html)} chars")

            # Prompt Gemini to analyze page and create navigation plan
            planning_prompt = f"""
You are an expert web navigation planner. Analyze this webpage and create a navigation plan.

USER REQUEST: "{prompt}"
TARGET URL: {url}

PAGE HTML (simplified):
{page_html}

Create a JSON array of navigation steps to fulfill the user's request. Each step should have:
- "action": one of ["click", "select", "input", "scroll", "paginate", "extract", "wait"]
- "selector": CSS selector for the target element (if applicable)
- "value": value to input/select (if applicable)
- "description": human-readable description of what this step does

IMPORTANT:
- For "click" actions, use specific CSS selectors (e.g., "a[href*='electronics']", "button.category")
- For "select" actions, provide the dropdown selector and value to select
- For "input" actions, provide the input field selector and value
- For "paginate" actions, provide the next/more button selector
- Include "extract" action at the end to extract data from the final page

CRITICAL - PAGINATION DETECTION:
E-commerce and listing pages often have pagination controls at the BOTTOM of the page.
- Look for pagination elements like: .pagination, .next-page, .load-more, a[rel='next'], button.page-next
- Common text patterns: "Next", "Next Page", "Load More", page numbers (1 2 3...)
- If you see product/item listings, ALWAYS check for pagination and include a "paginate" action
- Pagination selectors can be: "a:contains('Next')", ".pagination .next", "button[aria-label*='next']"
- The "paginate" action should come BEFORE the final "extract" action
- Example: If you see a product grid, include {{"action": "paginate", "selector": ".pagination .next", "description": "Navigate through all product pages"}}

Example response:
[
  {{"action": "click", "selector": "a[href*='category']", "description": "Click category link"}},
  {{"action": "select", "selector": "#brand-filter", "value": "iPhone", "description": "Select iPhone brand"}},
  {{"action": "paginate", "selector": ".next-page", "description": "Collect data from all pages"}},
  {{"action": "extract", "description": "Extract product prices"}}
]

Return ONLY the JSON array, no other text.
"""

            response_text = (await self._generate_text(planning_prompt)).strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            steps = json.loads(response_text)
            logger.info(f"Planned {len(steps)} navigation steps")

            return steps

        except Exception as e:
            logger.error(f"Navigation planning failed: {str(e)}", exc_info=True)
            # Return basic extraction as fallback
            return [{"action": "extract", "description": "Extract data from current page"}]

    def _should_reanalyze_after_step(self, step: Dict[str, Any]) -> bool:
        """
        Determine if we should re-analyze the page after this navigation step.

        Args:
            step: The executed navigation step

        Returns:
            True if re-analysis recommended, False otherwise
        """
        action = step.get("action")
        # Re-analyze after clicks that navigate to new pages
        if action == "click":
            return True
        # Maybe after selects that filter content
        if action == "select":
            return True
        # Not after pagination, extraction, scroll, or wait
        return False

    async def _reanalyze_for_pagination(
        self,
        crawler,
        current_url: str,
        html_content: str,
        original_prompt: str,
        visited_urls: set
    ) -> Optional[Dict[str, Any]]:
        """
        Re-analyze page after navigation to detect pagination controls.

        Args:
            crawler: AsyncWebCrawler instance
            current_url: Current page URL
            html_content: HTML of current page
            original_prompt: User's original request
            visited_urls: Set of URLs already analyzed (prevent loops)

        Returns:
            Pagination step dict if found, None otherwise
        """
        from bs4 import BeautifulSoup

        # Prevent re-analyzing same URL twice
        if current_url in visited_urls:
            logger.debug(f"Already analyzed {current_url}, skipping")
            return None

        visited_urls.add(current_url)

        try:
            # Deep clean HTML (same as planning)
            deep_cleaned = self._deep_clean_html(html_content)
            page_html = deep_cleaned[:30000]

            # Focused prompt for pagination detection only
            reanalysis_prompt = f"""
You are analyzing a webpage AFTER navigation to detect pagination controls.

ORIGINAL USER REQUEST: "{original_prompt}"
CURRENT PAGE URL: {current_url}

PAGE HTML (simplified):
{page_html}

QUESTION: Does this page have pagination controls for navigating through multiple pages of items/products/listings?

Look for:
- Pagination elements: .pagination, .next-page, .load-more, a[rel='next'], button.page-next
- Text patterns: "Next", "Next Page", "Load More", page numbers (1 2 3...)
- Product/item grids or lists that might span multiple pages

If pagination EXISTS, respond with a JSON object:
{{
  "has_pagination": true,
  "action": "paginate",
  "selector": "CSS selector for next button",
  "description": "Description of pagination"
}}

If NO pagination, respond with:
{{
  "has_pagination": false
}}

Return ONLY valid JSON, no other text.
"""

            response_text = (await self._generate_text(reanalysis_prompt)).strip()

            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)

            if result.get("has_pagination"):
                logger.info(f"Pagination detected on {current_url}")
                return {
                    "action": "paginate",
                    "selector": result.get("selector"),
                    "description": result.get("description", "Navigate through all pages")
                }
            else:
                logger.info(f"No pagination found on {current_url}")
                return None

        except Exception as e:
            logger.warning(f"Re-analysis failed for {current_url}: {str(e)}")
            return None

    async def _execute_navigation(
        self, crawler, url: str, steps: List[Dict], max_pages: int = 50,
        original_prompt: str = "",
        kafka_publisher=None,
        job_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute navigation steps (with dynamic pagination re-detection)

        Args:
            crawler: AsyncWebCrawler instance
            url: Starting URL
            steps: List of navigation steps
            max_pages: Maximum pages for pagination

        Returns:
            Dictionary with final_url, executed_steps, pages_collected
        """
        current_url = url
        executed_steps = []
        collected_pages = []
        collected_page_urls = []  # LOCAL state - no persistent instance attributes
        visited_urls = set()  # Track analyzed URLs to prevent re-analysis loops
        visited_urls.add(url)  # Mark initial URL as analyzed

        try:
            for step_index, step in enumerate(steps):
                action = step.get("action")
                selector = step.get("selector")
                value = step.get("value")

                logger.info(f"Executing step: {action} - {step.get('description')}")

                if action == "click":
                    # Click element using JavaScript
                    js_code = f"document.querySelector('{selector}').click()"
                    result = await crawler.arun(
                        url=current_url,
                        js_code=[js_code],
                        wait_for="networkidle"
                    )
                    current_url = result.url
                    executed_steps.append({
                        "action": action,
                        "selector": selector,
                        "success": True
                    })

                    # Re-analyze for pagination if we navigated to a new page
                    if self._should_reanalyze_after_step(step):
                        logger.info(f"Re-analyzing page after navigation: {current_url}")
                        pagination_step = await self._reanalyze_for_pagination(
                            crawler,
                            current_url,
                            result.html,
                            original_prompt,
                            visited_urls
                        )

                        if pagination_step:
                            logger.info(f"Found pagination on new page: {pagination_step}")

                            # Check if pagination already exists in remaining steps
                            current_index = steps.index(step)
                            remaining_steps = steps[current_index + 1:]
                            has_pagination = any(s.get("action") == "paginate" for s in remaining_steps)

                            if has_pagination:
                                logger.info("Pagination already exists in plan, skipping insertion")
                            else:
                                # Find extract action to insert before it
                                extract_index = None
                                for i, s in enumerate(remaining_steps):
                                    if s.get("action") == "extract":
                                        extract_index = i
                                        break

                                if extract_index is not None:
                                    # Insert before extract
                                    insertion_position = current_index + 1 + extract_index
                                    steps.insert(insertion_position, pagination_step)
                                    logger.info(f"Inserted pagination at position {insertion_position}")
                                else:
                                    # No extract found, append to end
                                    steps.append(pagination_step)
                                    logger.info("Appended pagination to end of plan")

                elif action == "select":
                    # Select dropdown option
                    js_code = f"document.querySelector('{selector}').value = '{value}'"
                    result = await crawler.arun(
                        url=current_url,
                        js_code=[js_code],
                        wait_for="networkidle"
                    )
                    executed_steps.append({
                        "action": action,
                        "selector": selector,
                        "value": value,
                        "success": True
                    })

                elif action == "input":
                    # Input text into field
                    js_code = f"document.querySelector('{selector}').value = '{value}'"
                    result = await crawler.arun(
                        url=current_url,
                        js_code=[js_code],
                        wait_for="networkidle"
                    )
                    executed_steps.append({
                        "action": action,
                        "selector": selector,
                        "value": value,
                        "success": True
                    })

                elif action == "paginate":
                    # Handle pagination - collect data from multiple pages
                    pagination_result = await self._handle_pagination(
                        crawler, current_url, selector, max_pages,
                        original_prompt=original_prompt,
                        kafka_publisher=kafka_publisher,
                        job_id=job_id,
                        user_id=user_id
                    )
                    # Extract html_pages and page_urls from result
                    html_pages = pagination_result.get("html_pages", [])
                    page_urls = pagination_result.get("page_urls", [])
                    
                    collected_pages.extend(html_pages)
                    
                    # Store page URLs locally (no persistent state - prevents pollution across runs)
                    collected_page_urls.extend(page_urls)
                    
                    executed_steps.append({
                        "action": action,
                        "selector": selector,
                        "pages_collected": len(html_pages),
                        "success": True
                    })

                elif action == "wait":
                    # Wait for element or time
                    await crawler.arun(url=current_url, wait_for="networkidle")
                    executed_steps.append({
                        "action": action,
                        "success": True
                    })

                elif action == "extract":
                    # Mark extraction point
                    executed_steps.append({
                        "action": action,
                        "success": True
                    })

                # PUBLISH EVENT 4: Navigation Step Completed (after each step)
                if kafka_publisher and job_id and action != "extract":
                    kafka_publisher.publish_progress(
                        "NavigationStepCompleted",
                        job_id,
                        user_id or "unknown",
                        {
                            "step_index": step_index + 1,
                            "total_steps": len(steps),
                            "action": action,
                            "description": step.get("description", ""),
                            "current_url": current_url
                        }
                    )

        except Exception as e:
            logger.error(f"Navigation execution failed: {str(e)}", exc_info=True)
            executed_steps.append({
                "action": step.get("action"),
                "error": str(e),
                "success": False
            })

        return {
            "final_url": current_url,
            "executed_steps": executed_steps,
            "pages_collected": len(collected_pages),
            "collected_pages": collected_pages,
            "page_urls": collected_page_urls  # Local variable - no state pollution
        }

    async def _extract_data_parallel(
        self,
        page_urls: List[str],
        prompt: str,
        schema: Optional[Dict] = None,
        kafka_publisher=None,
        job_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        PARALLEL EXTRACTION: Spawn one AsyncWebCrawler instance per page URL.
        
        This method solves the "lazy extraction" problem where Gemini only extracts
        from page 1 when given large concatenated HTML. By using independent browser
        tabs (crawler instances) per page, each LLM call processes small, focused HTML,
        ensuring complete extraction from every paginated page.
        
        Args:
            page_urls: List of URLs to crawl in parallel (from pagination)
            prompt: User's extraction intent
            schema: Optional extraction schema
            kafka_publisher: Publisher for progress events
            job_id: Job ID for Kafka events
            user_id: User ID for Kafka events
            
        Returns:
            List of deduplicated extracted data items from all pages
        """
        try:
            total_pages = len(page_urls)
            logger.info(f"Starting parallel extraction for {total_pages} pages from current crawl only (no persistent state)")
            logger.info(f"Max concurrent tabs: {self.max_concurrent_tabs}")
            
            # Log sample URLs for debugging (verify all from same domain/crawl)
            if page_urls:
                sample_urls = page_urls[:3]
                logger.info(f"Sample URLs: {sample_urls}")
                
                # Verify domain consistency
                from urllib.parse import urlparse
                domains = set(urlparse(u).netloc for u in page_urls)
                if len(domains) > 1:
                    logger.error(f"STATE POLLUTION DETECTED: Multiple domains in page_urls: {domains}")
                else:
                    logger.info(f"Domain consistency verified: {domains.pop() if domains else 'none'}")
            
            # PUBLISH EVENT: Data Extraction Parallel Started
            if kafka_publisher and job_id:
                kafka_publisher.publish_progress(
                    "DataExtractionParallelStarted",
                    job_id,
                    user_id or "unknown",
                    {
                        "total_pages": total_pages,
                        "max_concurrent_tabs": self.max_concurrent_tabs,
                        "page_urls": page_urls
                    }
                )
            
            # Semaphore to limit concurrent browser instances
            semaphore = asyncio.Semaphore(self.max_concurrent_tabs)
            all_extracted_items = []
            extraction_errors = []
            
            async def extract_single_page(page_url: str, page_index: int) -> List[Dict[str, Any]]:
                """Extract data from a single page URL with retry logic."""
                async with semaphore:
                    max_retries = 2
                    for attempt in range(1, max_retries + 1):
                        try:
                            logger.info(f"[Page {page_index + 1}/{total_pages}] Crawling {page_url} (attempt {attempt}/{max_retries})")
                            
                            # Create independent crawler instance for this page
                            async with AsyncWebCrawler(verbose=True, headless=True) as page_crawler:
                                # Crawl the page
                                result = await page_crawler.arun(
                                    url=page_url,
                                    wait_for="networkidle",
                                    delay_before_return_html=2.0
                                )
                                
                                page_html = result.html or result.cleaned_html
                                
                                if not page_html:
                                    logger.warning(f"[Page {page_index + 1}/{total_pages}] No HTML content from {page_url}")
                                    return []
                                
                                # Extract data from this page only (RAG-style chunking)
                                extraction_instruction = f"""
Extract data from this page based on the user's request: "{prompt}"

Return a JSON array of objects. Each object should contain relevant fields.

Example format:
[
  {{"product_name": "iPhone 15 Pro", "price_usd": 999.99, "brand": "Apple"}},
  {{"product_name": "iPhone 15", "price_usd": 799.99, "brand": "Apple"}}
]

Return ONLY the JSON array, no other text.
"""
                                
                                page_items = await self._fallback_gemini_extraction(
                                    page_html,
                                    extraction_instruction,
                                    prompt
                                )
                                
                                logger.info(f"[Page {page_index + 1}/{total_pages}] Extracted {len(page_items)} items from {page_url}")
                                
                                # PUBLISH EVENT: Parallel Page Completed (verbose mode only)
                                if kafka_publisher and job_id and self.kafka_verbose_events:
                                    kafka_publisher.publish_progress(
                                        "DataExtractionParallelPageCompleted",
                                        job_id,
                                        user_id or "unknown",
                                        {
                                            "page_index": page_index + 1,
                                            "total_pages": total_pages,
                                            "page_url": page_url,
                                            "items_extracted": len(page_items),
                                            "attempt": attempt
                                        }
                                    )
                                
                                return page_items
                                
                        except Exception as e:
                            logger.warning(f"[Page {page_index + 1}/{total_pages}] Extraction failed for {page_url} (attempt {attempt}/{max_retries}): {str(e)}")
                            
                            if attempt < max_retries:
                                # Wait before retry
                                await asyncio.sleep(2.0 * attempt)
                            else:
                                # Final failure - log and skip this page
                                error_msg = f"Failed extraction for {page_url} after {max_retries} attempts: {str(e)}"
                                logger.error(error_msg)
                                extraction_errors.append({"page_url": page_url, "error": str(e)})
                                return []
                    
                    return []
            
            # Execute parallel extraction with asyncio.gather
            logger.info(f"Spawning {total_pages} crawler tasks...")
            extraction_tasks = [
                extract_single_page(url, idx)
                for idx, url in enumerate(page_urls)
            ]
            
            results = await asyncio.gather(*extraction_tasks, return_exceptions=False)
            
            # Merge all results
            for page_items in results:
                if isinstance(page_items, list):
                    all_extracted_items.extend(page_items)
            
            # Global deduplication across all pages
            unique_items = self._deduplicate_items(all_extracted_items)
            
            logger.info(f"Parallel extraction complete: {len(all_extracted_items)} total → {len(unique_items)} unique items")
            
            # PUBLISH EVENT: Data Extraction Completed
            if kafka_publisher and job_id:
                kafka_publisher.publish_progress(
                    "DataExtractionCompleted",
                    job_id,
                    user_id or "unknown",
                    {
                        "total_items_extracted": len(unique_items),
                        "pages_processed": total_pages,
                        "extraction_successful": True,
                        "parallel_mode": True,
                        "skipped_pages": len(extraction_errors),
                        "errors": extraction_errors if extraction_errors else None
                    }
                )
            
            return unique_items
            
        except Exception as e:
            logger.error(f"Parallel extraction failed: {str(e)}", exc_info=True)
            
            # PUBLISH EVENT: Data Extraction Completed (with error)
            if kafka_publisher and job_id:
                kafka_publisher.publish_progress(
                    "DataExtractionCompleted",
                    job_id,
                    user_id or "unknown",
                    {
                        "total_items_extracted": 0,
                        "pages_processed": 0,
                        "extraction_successful": False,
                        "parallel_mode": True,
                        "error": str(e)
                    }
                )
            
            return []

    def _find_forward_pagination_link(self, soup, selector: str, current_url: str, current_page_num: int) -> Optional[Any]:
        """Find the first pagination link that goes FORWARD (page > current_page).
        
        Args:
            soup: BeautifulSoup instance
            selector: CSS selector for pagination links
            current_url: Current page URL for resolving relative links
            current_page_num: Current page number
            
        Returns:
            BeautifulSoup element of next forward link, or None if not found
        """
        from urllib.parse import urljoin
        
        try:
            links = soup.select(selector)
            logger.debug(f"_find_forward_pagination_link: Found {len(links)} links with selector '{selector}'")
            
            for link in links:
                href = link.get('href')
                if not href or href.startswith('javascript:') or href == '#':
                    continue
                    
                full_url = urljoin(current_url, href)
                link_page = self._extract_page_number(full_url)
                
                # Only accept links going FORWARD
                if link_page is not None and link_page > current_page_num:
                    logger.info(f"✅ Found forward link: page {current_page_num} → {link_page}, href={full_url}")
                    return link
                else:
                    logger.debug(f"Skipped link (not forward): page {current_page_num} → {link_page}, href={full_url}")
            
            logger.debug(f"No forward links found with selector '{selector}'")
            return None
            
        except Exception as e:
            logger.warning(f"_find_forward_pagination_link failed: {e}")
            return None

    def _extract_page_number(self, url: str) -> Optional[int]:
        """Extract page number from URL query params or path.
        
        Handles patterns:
        - ?page=2, ?p=3 (query params)
        - /page/2, /p/3 (path patterns)
        
        Returns:
            int: Page number if found, None otherwise
        """
        from urllib.parse import urlparse, parse_qs
        import re
        
        parsed = urlparse(url)
        
        # Check query parameters
        query = parse_qs(parsed.query)
        for param in ['page', 'p']:
            if param in query:
                try:
                    return int(query[param][0])
                except (ValueError, IndexError):
                    pass
        
        # Check path patterns like /page/2 or /p/3
        path_match = re.search(r'/(?:page|p)/(\d+)', parsed.path)
        if path_match:
            try:
                return int(path_match.group(1))
            except ValueError:
                pass
        
        return None

    def _verify_content_changed(self, old_html: str, new_html: str, url: str) -> bool:
        """
        Verify if page content has changed by comparing MD5 hashes.
        
        Args:
            old_html: Previous page HTML
            new_html: New page HTML
            url: Current URL for logging
            
        Returns:
            True if content changed, False if duplicate
        """
        import hashlib
        
        old_hash = hashlib.md5(old_html.encode('utf-8')).hexdigest()
        new_hash = hashlib.md5(new_html.encode('utf-8')).hexdigest()
        old_len = len(old_html)
        new_len = len(new_html)
        
        if old_hash != new_hash:
            logger.info(f"Content changed at {url}: hash {old_hash[:12]}... -> {new_hash[:12]}...")
            return True
        
        # Suspicious case: same hash but different lengths (extremely rare with MD5)
        if old_len != new_len:
            logger.warning(f"Suspicious: Hashes same but lengths differ ({old_len} vs {new_len}) at {url}")
            return True  # Conservative: treat as changed
        
        logger.warning(f"Duplicate content detected at {url} (hash: {old_hash[:12]}...)")
        return False

    async def _trigger_conditional_redetection(
        self,
        page_html: str,
        normalized_url: str,
        original_prompt: str,
        current_page: int,
        reason: str,
        soup,
        kafka_publisher=None,
        job_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Trigger AI-powered pagination re-detection and return updated selector state.
        
        Args:
            page_html: Current page HTML
            normalized_url: Current normalized URL
            original_prompt: Original extraction prompt for context
            current_page: Current page number
            reason: Trigger reason ("selector_failure", "directionality_rejection", etc.)
            soup: BeautifulSoup instance for current page
            kafka_publisher: Optional Kafka publisher
            job_id: Optional job ID
            user_id: Optional user ID
            
        Returns:
            Dict with:
                - success: bool
                - new_selector: str (if found)
                - new_confidence: float
                - next_button: BeautifulSoup element (if found and validated)
        """
        logger.info(f"🔍 Triggering AI re-detection on page {current_page} ({reason})")
        
        result = {
            "success": False,
            "new_selector": None,
            "new_confidence": 0.0,
            "next_button": None
        }
        
        try:
            detection_result = await self._detect_pagination_with_model(
                page_html,
                normalized_url,
                original_prompt
            )
            
            new_selector = detection_result.get("selector")
            new_confidence = detection_result.get("confidence", 0.0)
            has_pagination = detection_result.get("has_pagination", False)
            
            if has_pagination and new_selector:
                result["success"] = True
                result["new_selector"] = new_selector
                result["new_confidence"] = new_confidence
                
                # Try new selector immediately
                next_button = soup.select_one(new_selector)
                if next_button:
                    result["next_button"] = next_button
                
                logger.info(f"✅ Re-detected selector: '{new_selector}' (confidence: {new_confidence:.2f})")
                
                # PUBLISH EVENT: Selector Change
                if kafka_publisher and job_id:
                    kafka_publisher.publish_progress(
                        "PaginationSelectorUpdated",
                        job_id,
                        user_id or "unknown",
                        {
                            "page_number": current_page,
                            "new_selector": new_selector,
                            "new_confidence": new_confidence,
                            "trigger_reason": reason,
                            "selector_tier": 0  # AI-detected = highest tier
                        }
                    )
            else:
                logger.debug(f"Re-detection found no pagination on page {current_page}")
                
        except Exception as e:
            logger.warning(f"Re-detection failed: {e}")
        
        return result

    async def _handle_pagination(
        self, crawler, url: str, next_selector: str, max_pages: int = 50,
        original_prompt: str = "",
        kafka_publisher=None,
        job_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Collect HTML from all paginated pages with hybrid URL+content duplicate detection.

        IMPROVEMENTS:
        - Hybrid tracking: (normalized_URL, content_hash) to handle SPA pagination
        - Prefers direct URL navigation over JavaScript clicks (more reliable)
        - Comprehensive diagnostic logging for debugging
        - Retry logic with strategy switching
        - URL normalization to prevent false duplicates
        - **NEW**: Returns both HTML strings AND page URLs for parallel extraction

        Args:
            crawler: AsyncWebCrawler instance
            url: Starting URL
            next_selector: CSS selector for next/more button
            max_pages: Maximum pages to collect

        Returns:
            Dict with:
                - html_pages: List[str] - HTML content from each page
                - page_urls: List[str] - Corresponding URLs for parallel extraction
        """
        # Step 1: Import required modules
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse, urlunparse
        import hashlib

        html_pages = []  # HTML content
        page_urls = []   # Corresponding URLs for parallel extraction
        current_page = 1
        current_url = url
        visited = set()  # Set of (normalized_url, content_hash) tuples - HYBRID APPROACH
        
        # Persistent selector tracking (prevents reverting to original after finding alternatives)
        working_selector = next_selector  # Start with provided selector
        selector_tier = 1  # 1=original, 2=alternative, 3=broad fallback
        current_selector_confidence = 1.0  # Confidence of current selector (1.0 = from main AI)
        consecutive_failures = 0  # Track consecutive pages with no next button found
        directionality_failures = 0  # Track consecutive directionality rejections (separate from selector failures)

        logger.info(f"Starting pagination with max {max_pages} pages, initial selector: {next_selector}")
        logger.info(f"Re-detection strategy: {self.redetection_strategy}, milestones: {self.redetection_milestones}")

        while current_page <= max_pages:
            try:
                # Step 3: Log iteration diagnostics
                logger.info(f"Iteration {current_page}: Set sizes - visited: {len(visited)}, html_pages: {len(html_pages)}")

                # Step 1: Normalize URL (remove fragments, normalize query)
                parsed = urlparse(current_url)
                normalized_url = urlunparse(parsed._replace(fragment=''))
                
                # Crawl current page with wait
                logger.info(f"Loading page {current_page}: {normalized_url}")
                result = await crawler.arun(
                    url=current_url,
                    wait_for="networkidle",
                    delay_before_return_html=1.5  # Wait for dynamic content
                )

                # Collect page HTML and compute hash
                page_html = result.cleaned_html or result.html
                content_hash = hashlib.md5(page_html.encode('utf-8')).hexdigest()
                
                # Step 3: Log content hash
                logger.info(f"Content hash: {content_hash[:12]}... (page {len(page_html)} chars)")

                # Step 1: Check for duplicates (HYBRID: URL + Content)
                url_visited = normalized_url in [t[0] for t in visited]
                content_visited = content_hash in [t[1] for t in visited]
                
                # Step 3: Log duplicate check
                logger.info(f"Checking duplicates: URL={url_visited}, Content={content_visited}")
                
                # Break only if BOTH URL and content are duplicates (Option C)
                if url_visited and content_visited:
                    logger.warning(f"Duplicate detected (URL + Content): {normalized_url}, stopping pagination")
                    break

                # Add to visited set and pages list (skip page_urls if duplicate)
                url_content_tuple = (normalized_url, content_hash)
                visited.add(url_content_tuple)
                html_pages.append(page_html)
                
                # Only append to page_urls if not duplicate (for parallel extraction)
                if not (url_visited or content_visited):
                    page_urls.append(normalized_url)
                else:
                    logger.debug(f"Skipping duplicate URL in page_urls list: {normalized_url}")
                
                logger.info(f"Collected page {current_page} ({len(page_html)} chars) from {normalized_url}")

                # PUBLISH EVENT 5: Pagination Page Loaded
                if kafka_publisher and job_id:
                    kafka_publisher.publish_progress(
                        "PaginationPageLoaded",
                        job_id,
                        user_id or "unknown",
                        {
                            "page_number": current_page,
                            "total_pages_collected": len(html_pages),
                            "max_pages": max_pages,
                            "page_size_chars": len(page_html),
                            "url": normalized_url,
                            "content_hash": content_hash[:12]
                        }
                    )

                # Check if next button exists using BeautifulSoup
                soup = BeautifulSoup(result.html, 'html.parser')
                
                # DYNAMIC RE-DETECTION: Trigger AI re-analysis on milestones or after selector failures
                should_redetect = False
                redetect_reason = ""
                
                if self.redetection_strategy in ["always"]:
                    should_redetect = True
                    redetect_reason = "always strategy"
                elif self.redetection_strategy in ["milestones", "hybrid"] and current_page in self.redetection_milestones:
                    should_redetect = True
                    redetect_reason = f"milestone page {current_page}"
                
                if should_redetect and original_prompt:
                    logger.info(f"🔍 Triggering AI re-detection on page {current_page} ({redetect_reason})")
                    try:
                        detection_result = await self._detect_pagination_with_model(
                            page_html,
                            normalized_url,
                            original_prompt
                        )
                        
                        new_selector = detection_result.get("selector")
                        new_confidence = detection_result.get("confidence", 0.0)
                        has_pagination = detection_result.get("has_pagination", False)
                        
                        if has_pagination and new_selector:
                            # Update working_selector if new has better confidence (with margin to prevent churn)
                            confidence_threshold = current_selector_confidence + self.confidence_margin
                            
                            if new_confidence >= confidence_threshold:
                                old_selector = working_selector
                                working_selector = new_selector
                                current_selector_confidence = new_confidence
                                selector_tier = 0  # AI-detected = highest tier
                                
                                logger.info(f"✅ Updated selector on page {current_page}: '{old_selector}' → '{new_selector}' (confidence: {new_confidence:.2f})")
                                
                                # PUBLISH EVENT: Selector Change
                                if kafka_publisher and job_id:
                                    kafka_publisher.publish_progress(
                                        "PaginationSelectorUpdated",
                                        job_id,
                                        user_id or "unknown",
                                        {
                                            "page_number": current_page,
                                            "old_selector": old_selector,
                                            "new_selector": new_selector,
                                            "new_confidence": new_confidence,
                                            "trigger_reason": redetect_reason,
                                            "selector_tier": selector_tier
                                        }
                                    )
                            else:
                                logger.debug(f"Keeping current selector (new confidence {new_confidence:.2f} < threshold {confidence_threshold:.2f})")
                        else:
                            logger.debug(f"Re-detection found no pagination on page {current_page}")
                    
                    except Exception as e:
                        logger.warning(f"Re-detection failed on page {current_page}: {e}")
                
                # Use working_selector (persists across iterations) or fallback to original
                current_selector = working_selector or next_selector
                logger.info(f"Page {current_page}: Trying pagination with selector '{current_selector}' (tier {selector_tier})")
                
                next_button = soup.select_one(current_selector)

                if not next_button:
                    consecutive_failures += 1
                    logger.warning(f"Current working selector '{current_selector}' failed on page {current_page} (consecutive failures: {consecutive_failures}), checking alternatives...")
                    
                    # CONDITIONAL RE-DETECTION: Trigger on selector failure (if enabled)
                    if self.redetection_strategy in ["conditional", "hybrid"] and original_prompt and consecutive_failures == 1:
                        redetection_result = await self._trigger_conditional_redetection(
                            page_html,
                            normalized_url,
                            original_prompt,
                            current_page,
                            "selector_failure",
                            soup,
                            kafka_publisher=kafka_publisher,
                            job_id=job_id,
                            user_id=user_id
                        )
                        
                        if redetection_result["success"]:
                            old_selector = working_selector
                            working_selector = redetection_result["new_selector"]
                            current_selector_confidence = redetection_result["new_confidence"]
                            selector_tier = 0  # AI-detected
                            consecutive_failures = 0  # Reset on successful re-detection
                            
                            # Use the re-detected button if found
                            if redetection_result["next_button"]:
                                next_button = redetection_result["next_button"]
                                logger.info(f"✅ Re-detected selector after failure: '{old_selector}' → '{working_selector}'")
                    
                    # Safety check: Break if too many consecutive failures
                    if consecutive_failures >= 3:
                        logger.warning(f"Stopping pagination: {consecutive_failures} consecutive failures to find next button")
                        break
                    
                    # selector_tier already tracked from initialization

                    # Try alternative selectors (ordered by specificity)
                    alternatives = [
                        'a[rel="next"]',           # Semantic HTML (most reliable)
                        'a.next[href]',             # Class-based next link
                        '.pagination .next a',      # Container-based
                        'a.next-page',              # Alternative class patterns
                        '.pagination li.active + li a'  # Next page after active
                    ]
                    
                    for tier, alt_selector in enumerate(alternatives, start=2):
                        try:
                            next_button = soup.select_one(alt_selector)
                            if next_button and next_button.get('href'):
                                logger.info(f"✓ Found next button with tier {tier} selector: '{alt_selector}' (upgrading from tier {selector_tier})")
                                old_selector = working_selector
                                working_selector = alt_selector  # PERSIST for next iteration
                                selector_tier = tier
                                current_selector_confidence = 0.5  # Lower confidence for alternatives
                                consecutive_failures = 0  # Reset on success
                                
                                # PUBLISH EVENT: Selector upgraded to alternative
                                if kafka_publisher and job_id:
                                    kafka_publisher.publish_progress(
                                        "PaginationSelectorUpdated",
                                        job_id,
                                        user_id or "unknown",
                                        {
                                            "page_number": current_page,
                                            "old_selector": old_selector,
                                            "new_selector": alt_selector,
                                            "new_confidence": 0.5,
                                            "trigger_reason": "alternative_fallback",
                                            "selector_tier": selector_tier
                                        }
                                    )
                                break
                        except Exception as e:
                            logger.debug(f"Alternative selector {alt_selector} failed: {e}")
                            continue
                    
                    # If still not found, try broad selector with strict filtering
                    if not next_button or not next_button.get('href'):
                        logger.debug(f"All specific selectors failed (tier {selector_tier}), trying broad 'a[href*=\"page=\"]' search with semantic filtering...")
                        try:
                            page_links = soup.select('a[href*="page="]')
                            for link in page_links:
                                href = link.get('href', '')
                                text = link.get_text(strip=True).lower()
                                aria = link.get('aria-label', '').lower()
                                
                                # Semantic filtering: reject previous/back links
                                reject_keywords = ['prev', 'previous', 'back', 'trước', '‹', '<']
                                if any(kw in text or kw in aria for kw in reject_keywords):
                                    logger.debug(f"Rejected link (previous/back indicator): {href} (text: '{text}')")
                                    continue
                                
                                # Reject page 1 links (likely previous)
                                if 'page=1' in href and text == '1':
                                    logger.debug(f"Rejected link (page 1): {href}")
                                    continue
                                
                                # Accept if has next indicators or numeric increment
                                next_keywords = ['next', 'tiếp', '›', '>']
                                if any(kw in text or kw in aria for kw in next_keywords):
                                    next_button = link
                                    selector_tier = 3  # Broad selector tier
                                    filtered_selector = f"a[href*='page=']"
                                    old_selector = working_selector
                                    working_selector = filtered_selector  # PERSIST filtered selector
                                    current_selector_confidence = 0.3  # Lowest confidence for broad selectors
                                    consecutive_failures = 0  # Reset on success
                                    logger.info(f"✓ Found next button with tier 3 filtered selector: {filtered_selector} (text: '{text}')")
                                    
                                    # PUBLISH EVENT: Selector upgraded to broad
                                    if kafka_publisher and job_id:
                                        kafka_publisher.publish_progress(
                                            "PaginationSelectorUpdated",
                                            job_id,
                                            user_id or "unknown",
                                            {
                                                "page_number": current_page,
                                                "old_selector": old_selector,
                                                "new_selector": filtered_selector,
                                                "new_confidence": 0.3,
                                                "trigger_reason": "broad_fallback",
                                                "selector_tier": selector_tier
                                            }
                                        )
                                    break
                                
                                # If no explicit text, validate by page number (directionality check below)
                                if not text or text.isdigit():
                                    next_button = link  # Tentative, will validate directionality
                                    selector_tier = 3
                                    old_selector = working_selector
                                    working_selector = 'a[href*="page="]'  # PERSIST
                                    current_selector_confidence = 0.3
                                    consecutive_failures = 0  # Reset on tentative success
                                    logger.debug(f"Tentative next button (will validate directionality): {href}")
                                    break
                        except Exception as e:
                            logger.debug(f"Broad selector filtering failed: {e}")

                if not next_button:
                    # Last attempt: try to find ANY forward link from current working selector
                    logger.debug("Last attempt: searching for any forward link before stopping")
                    current_page_num_for_search = self._extract_page_number(current_url) or 1
                    next_button = self._find_forward_pagination_link(soup, working_selector, current_url, current_page_num_for_search)
                    
                    if not next_button:
                        logger.info(f"No more pages found after page {current_page} (last tried: '{current_selector}', tier {selector_tier}). Pagination complete.")
                        break
                    else:
                        logger.info(f"✅ Found forward link in last attempt")
                        consecutive_failures = 0
                else:
                    # Successfully found next button - reset consecutive failures
                    consecutive_failures = 0
                    # Note: directionality_failures reset happens in directionality validation section

                # Check if next button is disabled
                if next_button.get('disabled') or 'disabled' in next_button.get('class', []):
                    logger.info("Next button is disabled, no more pages")
                    break

                # Step 2: Get next page URL and determine navigation strategy
                next_url = next_button.get('href')
                
                # Directionality validation: ensure next link goes forward
                from urllib.parse import urljoin
                next_url_absolute = urljoin(current_url, next_url) if next_url else None
                
                if next_url_absolute:
                    current_page_num = self._extract_page_number(current_url)
                    next_page_num = self._extract_page_number(next_url_absolute)
                    
                    # Assume base URL is page 1 if no number detected
                    if current_page_num is None:
                        current_page_num = 1
                        logger.debug("Assumed base URL as page 1 for direction check")
                    
                    # Validate directionality if both numbers extracted
                    if next_page_num is not None:
                        if next_page_num <= current_page_num:
                            directionality_failures += 1
                            consecutive_failures += 1  # Treat as selector failure
                            
                            logger.info(f"Rejected link (backward/same page): current={current_page_num}, next={next_page_num}, href={next_url}, directionality_failures={directionality_failures}")
                            
                            # PUBLISH EVENT: Directionality Rejection
                            if kafka_publisher and job_id:
                                kafka_publisher.publish_progress(
                                    "PaginationDirectionalityRejected",
                                    job_id,
                                    user_id or "unknown",
                                    {
                                        "page_number": current_page,
                                        "current_page_num": current_page_num,
                                        "next_page_num": next_page_num,
                                        "rejected_href": next_url,
                                        "directionality_failures": directionality_failures,
                                        "consecutive_failures": consecutive_failures,
                                        "selector": working_selector,
                                        "selector_tier": selector_tier
                                    }
                                )
                            
                            # TRIGGER RE-DETECTION: Try to find better selector
                            if self.redetection_strategy in ["conditional", "hybrid"] and original_prompt and directionality_failures == 1:
                                logger.info(f"🔍 Triggering AI re-detection due to directionality rejection")
                                
                                redetection_result = await self._trigger_conditional_redetection(
                                    page_html,
                                    normalized_url,
                                    original_prompt,
                                    current_page,
                                    "directionality_rejection",
                                    soup,
                                    kafka_publisher=kafka_publisher,
                                    job_id=job_id,
                                    user_id=user_id
                                )
                                
                                if redetection_result["success"]:
                                    old_selector = working_selector
                                    working_selector = redetection_result["new_selector"]
                                    current_selector_confidence = redetection_result["new_confidence"]
                                    selector_tier = 0  # AI-detected
                                    directionality_failures = 0  # Reset on successful re-detection
                                    consecutive_failures = 0
                                    
                                    # Try new selector
                                    next_button = redetection_result.get("next_button")
                                    if next_button:
                                        logger.info(f"✅ Re-detected selector fixed directionality: '{old_selector}' → '{working_selector}'")
                                        # Re-validate with new button (loop will continue to directionality check)
                                        next_url = next_button.get('href')
                                        next_url_absolute = urljoin(current_url, next_url) if next_url else None
                                        
                                        if next_url_absolute:
                                            new_next_page_num = self._extract_page_number(next_url_absolute)
                                            if new_next_page_num and new_next_page_num > current_page_num:
                                                logger.info(f"✅ New selector provides forward link: page {current_page_num} → {new_next_page_num}")
                                                # Continue with navigation using new button
                                            else:
                                                logger.warning(f"Re-detected selector still provides backward/same link, trying to find forward link")
                                                # Try to find a forward link with the re-detected selector
                                                forward_button = self._find_forward_pagination_link(soup, working_selector, current_url, current_page_num)
                                                if forward_button:
                                                    next_button = forward_button
                                                    next_url = next_button.get('href')  # UPDATE next_url!
                                                    next_url_absolute = urljoin(current_url, next_url) if next_url else None
                                                    logger.info(f"✅ Using forward link from re-detected selector: {next_url}")
                                                else:
                                                    # No forward link = reached LAST PAGE
                                                    logger.info(f"No forward link found - reached last page (page {current_page_num}). Stopping pagination.")
                                                    next_button = None
                                                    break  # EXIT pagination loop
                                    else:
                                        logger.warning("Re-detection succeeded but found no button, trying to find forward link")
                                        # Try to find a forward link with the re-detected selector
                                        forward_button = self._find_forward_pagination_link(soup, working_selector, current_url, current_page_num)
                                        if forward_button:
                                            next_button = forward_button
                                            next_url = next_button.get('href')  # UPDATE next_url!
                                            next_url_absolute = urljoin(current_url, next_url) if next_url else None
                                            logger.info(f"✅ Using forward link: {next_url}")
                                        else:
                                            # No forward link = reached LAST PAGE
                                            logger.info(f"No forward link found - reached last page (page {current_page_num}). Stopping pagination.")
                                            next_button = None
                                            break  # EXIT pagination loop
                                else:
                                    # Re-detection failed, try to find forward link with current selector
                                    logger.warning("Re-detection failed, trying to find forward link with current selector")
                                    forward_button = self._find_forward_pagination_link(soup, working_selector, current_url, current_page_num)
                                    if forward_button:
                                        next_button = forward_button
                                        next_url = next_button.get('href')  # UPDATE next_url!
                                        next_url_absolute = urljoin(current_url, next_url) if next_url else None
                                        logger.info(f"✅ Using forward link: {next_url}")
                                    else:
                                        # No forward link = reached LAST PAGE
                                        logger.info(f"No forward link found - reached last page (page {current_page_num}). Stopping pagination.")
                                        next_button = None
                                        break  # EXIT pagination loop
                            else:
                                # No re-detection strategy or not first failure
                                logger.debug("No re-detection triggered, trying to find forward link with current selector")
                                forward_button = self._find_forward_pagination_link(soup, working_selector, current_url, current_page_num)
                                if forward_button:
                                    next_button = forward_button
                                    next_url = next_button.get('href')  # UPDATE next_url!
                                    next_url_absolute = urljoin(current_url, next_url) if next_url else None
                                    logger.info(f"✅ Using forward link: {next_url}")
                                else:
                                    # No forward link = reached LAST PAGE
                                    logger.info(f"No forward link found - reached last page (page {current_page_num}). Stopping pagination.")
                                    next_button = None
                                    break  # EXIT pagination loop
                            
                            # Safety check: break if too many directionality failures AND no forward link found
                            if directionality_failures >= 2 and (not next_button or not next_url):
                                logger.warning(f"Stopping pagination: {directionality_failures} consecutive directionality failures (likely reached end)")
                                break
                        else:
                            # Valid forward navigation
                            directionality_failures = 0  # Reset on successful forward link
                            logger.debug(f"Validated forward navigation: page {current_page_num} → {next_page_num}")
                    else:
                        # No page number in next link - rely on duplicate detection
                        logger.debug(f"Could not extract page number from {next_url_absolute}, relying on content duplicate detection")
                
                # Step 2: INVERTED LOGIC - Prefer direct URL navigation over clicking
                # Check if href is valid (not javascript:void(0) or empty)
                has_valid_href = (
                    next_url and 
                    next_url.strip() and 
                    not next_url.startswith('javascript:') and
                    next_url != '#'
                )

                # Determine primary and fallback strategies
                if has_valid_href:
                    primary_strategy = "url_navigation"
                    fallback_strategy = "javascript_click"
                    logger.info(f"Found valid href: {next_url}")
                else:
                    primary_strategy = "javascript_click"
                    fallback_strategy = None
                    logger.info("No valid href, will use JavaScript click")

                # Step 5: Retry logic with strategy switching (max 2 attempts)
                navigation_success = False
                previous_html = page_html  # Store current page HTML for comparison
                
                for attempt in range(1, 3):  # 2 attempts max
                    try:
                        current_strategy = primary_strategy if attempt == 1 else fallback_strategy
                        
                        if current_strategy is None:
                            logger.warning("No fallback strategy available")
                            break
                        
                        # Step 3: Log attempt
                        logger.info(f"Attempt {attempt}/2: {current_strategy.replace('_', ' ').title()}")

                        if current_strategy == "url_navigation":
                            # Step 3: Log strategy
                            logger.info("Strategy: Direct URL navigation")
                            
                            # Step 2: Resolve next URL
                            if next_url.startswith('/'):
                                resolved_url = urljoin(current_url, next_url)
                            elif next_url.startswith('http'):
                                resolved_url = next_url
                            else:
                                # Relative URL
                                resolved_url = urljoin(current_url, next_url)
                            
                            logger.info(f"Navigating to: {resolved_url}")
                            
                            # Navigate directly
                            nav_result = await crawler.arun(
                                url=resolved_url,
                                wait_for="networkidle",
                                delay_before_return_html=2.0 if attempt == 2 else 1.5
                            )
                            
                            current_url = nav_result.url
                            new_html = nav_result.cleaned_html or nav_result.html

                        elif current_strategy == "javascript_click":
                            # Step 3: Log strategy
                            logger.info("Strategy: JavaScript click")
                            logger.info(f"Clicking pagination button: {next_selector}")

                            js_click = f"document.querySelector('{next_selector}').click()"
                            
                            click_result = await crawler.arun(
                                url=current_url,
                                js_code=[js_click],
                                wait_for="networkidle",
                                delay_before_return_html=3.0 if attempt == 2 else 2.0  # Longer delay on retry
                            )

                            # Update URL if changed
                            if click_result.url != current_url:
                                current_url = click_result.url
                                logger.info(f"URL changed to: {current_url}")
                            else:
                                logger.warning("URL didn't change after click")
                            
                            new_html = click_result.cleaned_html or click_result.html

                        # Step 4: Verify content changed using helper method
                        if self._verify_content_changed(previous_html, new_html, current_url):
                            navigation_success = True
                            logger.info(f"Navigation successful via {current_strategy}")
                            break  # Success, exit retry loop
                        else:
                            # Step 3: Log unchanged content with preview
                            logger.warning(f"Content unchanged after {current_strategy}; HTML preview: {new_html[:200]}...")
                            
                            if attempt == 1 and fallback_strategy:
                                logger.warning(f"Attempt {attempt}/2: {current_strategy} failed, trying {fallback_strategy}")
                                # Add extra delay before retry
                                await asyncio.sleep(2.0)
                            else:
                                logger.error(f"Both strategies exhausted, content unchanged")
                                break

                    except Exception as nav_error:
                        # Step 5: Log navigation errors
                        logger.warning(f"Attempt {attempt}/2: {current_strategy} failed with error: {nav_error}")
                        
                        if attempt == 1 and fallback_strategy:
                            logger.warning(f"Switching to {fallback_strategy}")
                            await asyncio.sleep(2.0)  # Delay before retry
                        else:
                            logger.error(f"All navigation attempts failed")
                            raise  # Re-raise to be caught by outer try-except

                # Check if navigation succeeded
                if not navigation_success:
                    logger.warning("Navigation failed after all attempts, stopping pagination")
                    break

                current_page += 1

                # Small delay between pages to avoid rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Pagination error at page {current_page}: {str(e)}", exc_info=True)
                break

        logger.info(f"Pagination complete. Collected {len(html_pages)} pages (visited {len(visited)} unique URL+content combinations)")
        
        # Deduplicate page_urls list (preserve order)
        seen_urls = set()
        deduplicated_urls = []
        for url in page_urls:
            if url not in seen_urls:
                seen_urls.add(url)
                deduplicated_urls.append(url)
        
        if len(page_urls) != len(deduplicated_urls):
            logger.info(f"Removed {len(page_urls) - len(deduplicated_urls)} duplicate URLs from page_urls list")
        
        return {
            "html_pages": html_pages,
            "page_urls": deduplicated_urls
        }

    async def _extract_data(
        self,
        crawler,
        url: str,
        additional_pages: List[str],
        prompt: str,
        schema: Optional[Dict] = None,
        kafka_publisher=None,
        job_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract data using RAG-style Gemini (1 call per page).
        
        NOTE: This method is used for SINGLE-PAGE extraction (no pagination).
        For paginated results, use _extract_data_parallel() instead.

        Args:
            crawler: AsyncWebCrawler instance
            url: Final URL after navigation
            additional_pages: Additional pages from pagination (DEPRECATED - use parallel extraction)
            prompt: User's extraction intent
            schema: Optional extraction schema

        Returns:
            List of extracted data items
        """
        try:
            # PUBLISH EVENT 6: Data Extraction Started
            if kafka_publisher and job_id:
                kafka_publisher.publish_progress(
                    "DataExtractionStarted",
                    job_id,
                    user_id or "unknown",
                    {
                        "final_url": url,
                        "additional_pages_count": len(additional_pages),
                        "total_pages_to_process": 1 + len(additional_pages[:10]),
                        "parallel_mode": False
                    }
                )

            # Default extraction instruction
            extraction_instruction = f"""
Extract data from this page based on the user's request: "{prompt}"

Return a JSON array of objects. Each object should contain:

Example format:
[
  {{"product_name": "iPhone 15 Pro", "price_usd": 999.99, "brand": "Apple", "category": "Electronics"}},
  {{"product_name": "iPhone 15", "price_usd": 799.99, "brand": "Apple", "category": "Electronics"}}
]

Return ONLY the JSON array, no other text.
"""

            extracted_items = []

            # Build the list of HTML documents we will process (final page + pagination)
            pages_to_process: List[str] = []

            # Crawl the final navigated page (single page, no pagination)
            # NOTE: In parallel mode, this method is skipped entirely
            logger.info(f"Crawling final page (sequential mode): {url}")
            result = await crawler.arun(
                url=url,
                wait_for="networkidle",
                delay_before_return_html=2.0
            )
            final_html = result.html or result.cleaned_html

            if final_html:
                pages_to_process.append(final_html)
                logger.info("Queued final page HTML for extraction")
            else:
                logger.warning("No HTML available for final page extraction")

            # Append up to 10 paginated pages (if available)
            # NOTE: This is legacy code - pagination now uses parallel extraction
            for idx, page_html in enumerate(additional_pages[:10], start=1):
                if not page_html:
                    logger.warning(f"Paginated page {idx} has no HTML, skipping")
                    continue
                pages_to_process.append(page_html)
                logger.info(f"Queued paginated page {idx} for extraction")

            if not pages_to_process:
                logger.warning("No HTML content available to extract from")

                if kafka_publisher and job_id:
                    kafka_publisher.publish_progress(
                        "DataExtractionCompleted",
                        job_id,
                        user_id or "unknown",
                        {
                            "total_items_extracted": 0,
                            "pages_processed": 0,
                            "extraction_successful": False,
                            "error": "No HTML content available"
                        }
                    )

                return []

            logger.info(f"Processing {len(pages_to_process)} page(s) for extraction")

            # Run the RAG extraction pass per HTML document, then deduplicate once at the end
            for page_index, page_html in enumerate(pages_to_process, start=1):
                try:
                    page_items = await self._fallback_gemini_extraction(
                        page_html,
                        extraction_instruction,
                        prompt
                    )
                    extracted_items.extend(page_items)
                    logger.info(
                        f"Extraction from page {page_index}/{len(pages_to_process)}: {len(page_items)} items"
                    )
                except Exception as e:
                    logger.error(f"Failed to extract from page {page_index}: {str(e)}")

            total_unique_items = self._deduplicate_items(extracted_items)
            logger.info(f"Total extracted: {len(total_unique_items)} items")

            # PUBLISH EVENT 7: Data Extraction Completed
            if kafka_publisher and job_id:
                kafka_publisher.publish_progress(
                    "DataExtractionCompleted",
                    job_id,
                    user_id or "unknown",
                    {
                        "total_items_extracted": len(total_unique_items),
                        "pages_processed": len(pages_to_process),
                        "extraction_successful": True
                    }
                )

            return total_unique_items

        except Exception as e:
            logger.error(f"Data extraction failed: {str(e)}", exc_info=True)

            # PUBLISH EVENT 7: Data Extraction Completed (with error)
            if kafka_publisher and job_id:
                kafka_publisher.publish_progress(
                    "DataExtractionCompleted",
                    job_id,
                    user_id or "unknown",
                    {
                        "total_items_extracted": 0,
                        "pages_processed": 0,
                        "extraction_successful": False,
                        "error": str(e)
                    }
                )

            return []

    async def _fallback_gemini_extraction(self, html_content: str, extraction_instruction: str, prompt: str) -> list:
        """
        RAG-STYLE EXTRACTION: Chunk → Concatenate → **1 Gemini Call**
        
        IMPROVED: Disabled relevance filtering to prevent data loss on product pages.

        Args:
            html_content: HTML content to extract from
            extraction_instruction: Instructions for what to extract
            prompt: Original user prompt

        Returns:
            List of extracted items
        """
        try:
            logger.info(f"Starting RAG-style extraction - Total HTML: {len(html_content)} chars")

            # Step 1: Clean HTML
            clean_html = self._deep_clean_html(html_content)
            logger.info(f"Cleaned HTML: {len(html_content)} → {len(clean_html)} chars")

            # Step 2: Chunk
            chunks = self._chunk_html(clean_html, chunk_size=15000, overlap=500)
            logger.info(f"Split into {len(chunks)} chunks")

            # Step 3: DISABLED - Relevance filtering removed to prevent data loss
            # The AI relevance filter was incorrectly marking valid product chunks as irrelevant,
            # causing products to be skipped during extraction (e.g., 28 instead of 30 items).
            # Processing all chunks ensures complete extraction.
            relevant_chunks = chunks
            logger.info(f"Processing all {len(chunks)} chunks (filtering disabled to prevent data loss)")

            if not relevant_chunks:
                logger.warning("No chunks available")
                return []

            # Step 4: Concatenate all relevant chunks (with separators)
            separator = "\n\n--- END OF CHUNK ---\n\n"
            combined_content = separator.join(relevant_chunks)
            total_chars = len(combined_content)
            logger.info(f"Combined relevant chunks: {total_chars} chars")
            rag_prompt = f"""
You are an expert data extractor. Use ALL the HTML chunks below to extract structured data.

USER REQUEST: "{prompt}"

INSTRUCTION:
{extraction_instruction}

HTML CHUNKS (concatenated, separated by '--- END OF CHUNK ---'):
{combined_content}

RULES:
1. Extract EVERY unique item that matches the request.
2. Do NOT duplicate items.
3. If a field is missing, use null or omit it.
4. Return ONLY a valid JSON array of objects.
5. Use consistent field names across all items.

Example:
[
    {{"product_name": "Easydew Ex", "price": "750,000₫", "brand": "Easydew"}},
    {{"product_name": "Easydew DW-EGF", "price": "1,200,000₫", "brand": "Easydew"}}
]

Return ONLY the JSON array:
"""

            logger.info("Calling LLM ONCE with all relevant chunks (RAG-style)...")
            response_text = (await self._generate_text(rag_prompt)).strip()

            # Parse response
            items = self._parse_llm_response(response_text)
            unique_items = self._deduplicate_items(items)

            logger.info(f"RAG extraction complete: {len(items)} → {len(unique_items)} unique items")
            return unique_items

        except Exception as e:
            logger.error(f"RAG extraction failed: {str(e)}", exc_info=True)
            return []

    def _parse_llm_response(self, response_text: str) -> list:
        """
        Parse LLM response with multiple fallback strategies.

        Args:
            response_text: Raw LLM response text

        Returns:
            List of parsed items
        """
        # Strategy 1: Direct JSON parse
        try:
            items = json.loads(response_text)
            if isinstance(items, list):
                return items
            elif isinstance(items, dict):
                return [items]
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from ```json code block
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
                logger.info(f"Strategy 2: Extracted JSON string length: {len(json_str)} chars")
                items = json.loads(json_str)
                if isinstance(items, list):
                    logger.info(f"Strategy 2: Successfully parsed {len(items)} items")
                    return items
                elif isinstance(items, dict):
                    logger.info("Strategy 2: Parsed single dict, wrapping in list")
                    return [items]
        except IndexError as e:
            logger.warning(f"Strategy 2: IndexError - {str(e)}")
        except json.JSONDecodeError as e:
            logger.warning(f"Strategy 2: JSONDecodeError at line {e.lineno}, col {e.colno}: {e.msg}")
            logger.warning(f"Strategy 2: Problematic JSON snippet: {json_str[max(0, e.pos-100):e.pos+100]}")

    # Strategy 3: Extract from ``` code block
        try:
            if "```" in response_text:
                code_blocks = response_text.split("```")
                for i in range(1, len(code_blocks), 2):  # Odd indices are code blocks
                    block = code_blocks[i].strip()
                    # Skip language identifier if present
                    if '\n' in block:
                        lines = block.split('\n')
                        if len(lines) > 1 and lines[0].strip().isalpha():
                            block = '\n'.join(lines[1:]).strip()
                    try:
                        items = json.loads(block)
                        if isinstance(items, list):
                            return items
                        elif isinstance(items, dict):
                            return [items]
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        # Strategy 4: Look for JSON array pattern [...]
        try:
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                items = json.loads(json_str)
                if isinstance(items, list):
                    return items
        except (json.JSONDecodeError, AttributeError):
            pass

        # Strategy 5: Extract complete objects from incomplete/truncated JSON
        try:
            import re
            # Extract JSON content from code blocks if present
            json_content = response_text
            if "```json" in response_text:
                try:
                    json_content = response_text.split("```json")[1].split("```")[0].strip()
                except IndexError:
                    # Truncated before closing ```
                    json_content = response_text.split("```json")[1] if "```json" in response_text else response_text
            elif "```" in response_text:
                try:
                    json_content = response_text.split("```")[1].split("```")[0].strip()
                except IndexError:
                    json_content = response_text.split("```")[1] if len(response_text.split("```")) > 1 else response_text
            
            # Find all complete JSON objects (with matching opening and closing braces)
            # This regex matches complete objects even if the array is incomplete
            object_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
            matches = re.findall(object_pattern, json_content, re.DOTALL)
            
            if matches:
                parsed_objects = []
                for match in matches:
                    try:
                        obj = json.loads(match)
                        if isinstance(obj, dict) and obj:  # Valid non-empty dict
                            parsed_objects.append(obj)
                    except json.JSONDecodeError:
                        continue
                
                if parsed_objects:
                    logger.info(f"Strategy 5: Recovered {len(parsed_objects)} complete objects from incomplete JSON")
                    return parsed_objects
        except Exception as e:
            logger.warning(f"Strategy 5: Failed with error: {str(e)}")

        logger.warning(f"All JSON parsing strategies failed. Response: {response_text[:500]}...")
        return []

    def _chunk_html(self, html: str, chunk_size: int = 15000, overlap: int = 500) -> list:
        """
        Split HTML into overlapping chunks while respecting element boundaries (IMPROVED VERSION).

        Fixes:
        - Tries to chunk by complete HTML elements (products, articles, etc.)
        - Falls back to tag-aware character chunking if element detection fails
        - Prevents splitting products mid-element

        Args:
            html: HTML content to chunk
            chunk_size: Size of each chunk in characters
            overlap: Number of characters to overlap between chunks

        Returns:
            List of HTML chunks
        """
        from bs4 import BeautifulSoup

        if len(html) <= chunk_size:
            return [html]

        try:
            # Try to parse and chunk by elements (more intelligent)
            soup = BeautifulSoup(html, 'html.parser')

            # Find common container elements for products/items
            containers = (
                soup.select('.product-item') or
                soup.select('.product') or
                soup.select('[class*="product"]') or
                soup.select('article') or
                soup.select('.item') or
                soup.select('li')
            )

            if containers and len(containers) > 5:
                # Group elements into chunks
                logger.info(f"Found {len(containers)} product containers, chunking by elements")
                chunks = []
                current_chunk = ""

                for container in containers:
                    container_html = str(container)

                    if len(current_chunk) + len(container_html) > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = container_html
                    else:
                        current_chunk += container_html

                if current_chunk:
                    chunks.append(current_chunk)

                logger.info(f"Created {len(chunks)} chunks from {len(containers)} elements")
                return chunks

        except Exception as e:
            logger.warning(f"Element-based chunking failed: {e}, falling back to character chunking")

        # Fallback: Character-based chunking with tag boundary awareness
        chunks = []
        start = 0

        while start < len(html):
            end = min(start + chunk_size, len(html))

            # Try to end at a closing tag to avoid splitting elements
            if end < len(html):
                # Look for nearest closing tag before end
                search_range = html[max(0, end - 100):min(len(html), end + 100)]
                closing_tags = ['</div>', '</article>', '</li>', '</section>', '</p>']

                best_break = end
                for tag in closing_tags:
                    tag_pos = search_range.rfind(tag)
                    if tag_pos != -1:
                        # Adjust end to be after this closing tag
                        best_break = max(0, end - 100) + tag_pos + len(tag)
                        break

                end = best_break

            chunks.append(html[start:end])

            if end >= len(html):
                break

            # Overlap for continuity
            start = end - overlap

        logger.info(f"Created {len(chunks)} chunks from {len(html)} chars (chunk_size={chunk_size}, overlap={overlap})")
        return chunks

    async def _batch_check_chunk_relevance(self, chunks: list, user_prompt: str) -> list:
        """
        Check relevance of ALL chunks in a single Gemini API call (batch processing).
        This dramatically reduces API calls: N chunks → 1 API call instead of N calls.

        Args:
            chunks: List of HTML chunks to check
            user_prompt: Original user request

        Returns:
            List of booleans indicating if each chunk is relevant
        """
        try:
            # Build batch prompt with all chunk previews
            batch_prompt = f"""Based on the user's request: "{user_prompt}"

I will show you {len(chunks)} HTML chunk previews numbered 1 to {len(chunks)}.
For EACH chunk, determine if it contains relevant content that could help answer the user's request.

Respond ONLY with a JSON array of "YES" or "NO" values, one for each chunk in order.
Example format: ["YES", "NO", "YES", "NO", "YES", "NO"]

"""

            # Add all chunk previews (first 500 chars of each)
            for i, chunk in enumerate(chunks):
                preview = chunk[:500]
                batch_prompt += f"\n--- Chunk {i+1} Preview ---\n{preview}\n"

            batch_prompt += f"""\n\nRespond with a JSON array of exactly {len(chunks)} values (YES or NO):"""

            logger.info(f"Batch checking {len(chunks)} chunks in 1 API call...")
            response_text = (await self._generate_text(batch_prompt)).strip()

            logger.debug(f"Batch relevance response: {response_text}")

            # Parse JSON array response
            import re
            # Try to extract JSON array
            json_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
            if json_match:
                json_str = '[' + json_match.group(1) + ']'
                results = json.loads(json_str)

                # Convert to boolean list
                relevance_flags = []
                for r in results:
                    r_str = str(r).strip().strip('"').strip("'").upper()
                    relevance_flags.append("YES" in r_str)

                logger.info(f"Batch relevance check results: {relevance_flags}")
                return relevance_flags
            else:
                # Fallback: try direct JSON parse
                results = json.loads(response_text)
                relevance_flags = [str(r).upper() == "YES" or "YES" in str(r).upper() for r in results]
                logger.info(f"Batch relevance check results: {relevance_flags}")
                return relevance_flags

        except Exception as e:
            logger.warning(f"Batch relevance check failed: {str(e)} - treating all chunks as relevant")
            # If batch check fails, treat all chunks as relevant (safe fallback)
            return [True] * len(chunks)

    async def _is_chunk_relevant(self, preview: str, user_prompt: str) -> bool:
        """
        Quick check if HTML chunk preview is relevant to user's request.

        Args:
            preview: First 500 chars of chunk
            user_prompt: Original user request

        Returns:
            True if chunk is relevant, False otherwise
        """
        try:
            check_prompt = f"""Based on the user's request: "{user_prompt}"

Does this HTML preview contain relevant content that could help answer the user's request?

Answer ONLY "YES" or "NO" (one word).

HTML Preview:
{preview}

Answer:"""

            answer = (await self._generate_text(check_prompt)).strip().upper()

            is_relevant = "YES" in answer
            logger.debug(f"Relevance check result: {answer} -> {is_relevant}")

            return is_relevant

        except Exception as e:
            logger.warning(f"Relevance check failed: {str(e)} - treating as relevant")
            return True  # If check fails, process the chunk anyway

    def _deep_clean_html(self, html: str) -> str:
        """
        Remove JavaScript, CSS, and non-content elements to reduce token usage.
        Expected: 50-70% reduction in HTML size while preserving extractable content.

        Args:
            html: Raw HTML content

        Returns:
            Cleaned HTML with JS/CSS removed
        """
        try:
            from bs4 import BeautifulSoup, Comment

            logger.debug(f"Deep cleaning HTML ({len(html)} chars)...")
            soup = BeautifulSoup(html, 'lxml')

            # 1. Remove script tags (except JSON-LD structured data which is useful)
            script_count = 0
            for script in soup.find_all('script'):
                # Keep JSON-LD for structured data extraction
                if script.get('type') == 'application/ld+json':
                    continue
                script.decompose()
                script_count += 1

            # 2. Remove style tags
            style_count = 0
            for style in soup.find_all('style'):
                style.decompose()
                style_count += 1

            # 3. Remove SVG elements (usually large and not useful for text extraction)
            svg_count = 0
            for svg in soup.find_all('svg'):
                svg.decompose()
                svg_count += 1

            # 4. Remove HTML comments
            comment_count = 0
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
                comment_count += 1

            # 5. Remove inline styles and non-essential attributes
            for tag in soup.find_all(True):
                # Remove inline style attribute
                if tag.has_attr('style'):
                    del tag['style']

                # Remove data-* attributes (usually for JS interactions)
                data_attrs = [attr for attr in tag.attrs if attr.startswith('data-')]
                for attr in data_attrs:
                    del tag[attr]

                # Remove event handlers (onclick, onload, etc.)
                event_attrs = [attr for attr in tag.attrs if attr.startswith('on')]
                for attr in event_attrs:
                    del tag[attr]

            cleaned_html = str(soup)

            logger.debug(f"Cleaned HTML: Removed {script_count} scripts, {style_count} styles, {svg_count} SVGs, {comment_count} comments")
            logger.debug(f"Size reduction: {len(html)} -> {len(cleaned_html)} chars ({100 - int(len(cleaned_html)/len(html)*100)}% reduction)")

            return cleaned_html

        except Exception as e:
            logger.warning(f"HTML cleaning failed: {str(e)} - using original HTML")
            return html  # Fallback to original HTML if cleaning fails

    def _extract_pagination_section(self, cleaned_html: str) -> str:
        """
        Extract top and bottom sections of HTML for pagination detection.
        Pagination controls are typically at the bottom, but we include top for context.
        
        Args:
            cleaned_html: Cleaned HTML content
            
        Returns:
            Combined top + bottom sections (max 60K chars)
        """
        if len(cleaned_html) <= 30000:
            return cleaned_html  # Short page, return as-is
        
        top_section = cleaned_html[:30000]
        bottom_section = cleaned_html[-30000:]
        
        # Avoid duplication if page is exactly 30K or overlaps significantly
        if top_section == bottom_section or len(cleaned_html) <= 60000:
            return cleaned_html[:60000]
        
        combined = top_section + "\n\n--- PAGE BREAK (Middle content omitted) ---\n\n" + bottom_section
        logger.debug(f"Extracted pagination section: {len(combined)} chars from {len(cleaned_html)} total")
        return combined

    async def _detect_pagination_with_model(self, full_html: str, url: str, prompt: str) -> Dict[str, Any]:
        """
        Detect pagination using dedicated AI model + BeautifulSoup fallback.
        Focuses on finding <a> tags with valid href attributes (not JavaScript clicks).
        
        Args:
            full_html: Full cleaned HTML content
            url: Current page URL
            prompt: User's request (for context)
            
        Returns:
            Dict with has_pagination, selector, href_pattern, confidence, next_href_example
        """
        try:
            from bs4 import BeautifulSoup
            
            # Extract top + bottom sections (pagination usually at bottom)
            pagination_section = self._extract_pagination_section(full_html)
            
            # AI-based detection using dedicated pagination model
            detection_prompt = f"""You are a pagination detection expert analyzing an e-commerce webpage.

USER REQUEST: "{prompt}"
WEBPAGE URL: {url}

PAGE HTML (top + bottom sections, middle omitted):
{pagination_section}

TASK: Detect if this page has pagination for navigating through multiple pages of products/items.

CRITICAL RULES:
1. **ONLY detect pagination with <a> tags that have valid href attributes**
2. **IGNORE** buttons with onclick handlers or javascript: links (crawl4ai cannot execute JavaScript)
3. Focus on href patterns like: ?page=2, /page/2, /products?p=2, ?p=2
4. Common selectors: a.next-page[href], a[rel="next"][href], .pagination a[href]
5. Text patterns in links: "Next", "Next Page", numbered links (2, 3, 4...)

EXAMPLES OF VALID PAGINATION:
- <a href="?page=2" class="next">Next</a>
- <a href="/products/page/2" rel="next">→</a>
- <a class="page-link" href="?p=2">2</a>

EXAMPLES TO IGNORE:
- <button onclick="loadPage(2)">Next</button>
- <a href="javascript:void(0)" class="next">Next</a>
- <div class="load-more" data-page="2">Load More</div>

Respond in JSON format:
{{
    "has_pagination": true/false,
    "selector": "CSS selector for next/page links (if found)",
    "href_pattern": "Pattern in href like ?page= or /page/",
    "confidence": 0.0-1.0,
    "next_href_example": "Example href value like ?page=2",
    "reasoning": "Brief explanation"
}}

If NO valid href-based pagination found, return:
{{
    "has_pagination": false,
    "confidence": 1.0,
    "reasoning": "No <a> tags with pagination href patterns found"
}}

Return ONLY the JSON object, no other text.
"""

            # Use dedicated pagination model (external or Gemini)
            if self.use_external_pagination_model:
                # Use external LLM API (OpenAI-compatible) for pagination
                logger.info(f"Using external model for pagination detection: {self.pagination_model_name}")
                response_text = await self._generate_text_with_model(
                    detection_prompt, 
                    model_override=self.pagination_model_name,
                    temperature=0.3,
                    max_tokens=1500
                )
            elif self.pagination_model:
                # Use dedicated Gemini model
                response = await asyncio.to_thread(
                    lambda: self.pagination_model.generate_content(detection_prompt)
                )
                response_text = response.text.strip()
            else:
                # Fallback to regular model if pagination model not available
                response_text = await self._generate_text(detection_prompt)
            
            # Parse JSON response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            ai_result = json.loads(response_text)
            logger.info(f"AI pagination detection: {ai_result.get('has_pagination')} (confidence: {ai_result.get('confidence', 0):.2f})")
            
            # If AI found pagination with sufficient confidence, return it
            if ai_result.get("has_pagination") and ai_result.get("confidence", 0) >= self.min_pagination_confidence:
                return ai_result
            
            # FALLBACK: BeautifulSoup-based detection (rule-based)
            logger.info("AI detection below threshold or negative, trying BeautifulSoup fallback...")
            soup = BeautifulSoup(full_html, 'lxml')
            
            # Try common pagination patterns (href-based only)
            pagination_candidates = [
                ('a[rel="next"][href]', 'Next link with rel attribute'),
                ('a.next[href]', 'Next link with class'),
                ('a.next-page[href]', 'Next page link'),
                ('.pagination a[href*="page"]', 'Pagination with page parameter'),
                ('a[href*="?page="]', 'Query param pagination'),
                ('a[href*="/page/"]', 'Path-based pagination'),
                ('a[href*="?p="]', 'Short page parameter'),
                ('.pagination li.active + li a[href]', 'Next after active page')
            ]
            
            for selector, description in pagination_candidates:
                try:
                    elements = soup.select(selector)
                    if elements:
                        # Found candidate, extract href pattern
                        first_elem = elements[0]
                        href = first_elem.get('href', '')
                        
                        # Validate href is not javascript or empty
                        if href and not href.startswith('javascript:') and href != '#':
                            # Extract pattern
                            href_pattern = ''
                            if '?page=' in href or '&page=' in href:
                                href_pattern = '?page='
                            elif '/page/' in href:
                                href_pattern = '/page/'
                            elif '?p=' in href or '&p=' in href:
                                href_pattern = '?p='
                            
                            logger.info(f"BeautifulSoup found pagination: {selector} ({description}) - href: {href}")
                            return {
                                "has_pagination": True,
                                "selector": selector,
                                "href_pattern": href_pattern,
                                "confidence": 0.5,  # Lower confidence for rule-based
                                "next_href_example": href,
                                "reasoning": f"BeautifulSoup fallback: {description}"
                            }
                except Exception as e:
                    logger.debug(f"BeautifulSoup selector '{selector}' failed: {e}")
                    continue
            
            # No pagination found by AI or BeautifulSoup
            logger.info("No href-based pagination detected by AI or BeautifulSoup")
            return {
                "has_pagination": False,
                "confidence": 1.0,
                "reasoning": "No valid <a> tags with pagination href patterns found"
            }
            
        except Exception as e:
            logger.error(f"Pagination detection failed: {str(e)}", exc_info=True)
            return {
                "has_pagination": False,
                "confidence": 0.0,
                "reasoning": f"Detection error: {str(e)}"
            }

    async def _generate_specific_page_urls(
        self, 
        base_url: str, 
        page_numbers: List[int], 
        detected_start_page: int = 1
    ) -> List[str]:
        """
        Generate URLs for specific page numbers based on URL pattern detection.
        
        Handles common pagination patterns:
        - /page/N (WordPress style)
        - ?page=N (query parameter)
        - ?p=N (short parameter)
        
        Args:
            base_url: Original URL (may already contain page number)
            page_numbers: List of page numbers to crawl
            detected_start_page: Starting page detected from URL (default 1)
            
        Returns:
            List of URLs for specific pages
        """
        from urllib.parse import urlparse, parse_qs, urljoin, urlunparse
        import re
        
        try:
            parsed = urlparse(base_url)
            urls = []
            
            # Detect pagination pattern in base_url
            # Pattern 1: /page/N in path
            path_page_match = re.search(r'/page/(\d+)', parsed.path)
            if path_page_match:
                logger.info(f"🔍 Detected path-based pagination pattern: /page/N")
                base_path = re.sub(r'/page/\d+', '', parsed.path)
                
                for page_num in page_numbers:
                    new_path = f"{base_path}/page/{page_num}"
                    new_url = urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, parsed.query, parsed.fragment))
                    urls.append(new_url)
                    logger.info(f"  📄 Page {page_num}: {new_url}")
                
                return urls
            
            # Pattern 2: /p/N in path
            path_p_match = re.search(r'/p/(\d+)', parsed.path)
            if path_p_match:
                logger.info(f"🔍 Detected short path pagination pattern: /p/N")
                base_path = re.sub(r'/p/\d+', '', parsed.path)
                
                for page_num in page_numbers:
                    new_path = f"{base_path}/p/{page_num}"
                    new_url = urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, parsed.query, parsed.fragment))
                    urls.append(new_url)
                    logger.info(f"  📄 Page {page_num}: {new_url}")
                
                return urls
            
            # Pattern 3: ?page=N in query
            query_params = parse_qs(parsed.query)
            if 'page' in query_params:
                logger.info(f"🔍 Detected query parameter pagination: ?page=N")
                
                for page_num in page_numbers:
                    new_params = query_params.copy()
                    new_params['page'] = [str(page_num)]
                    # Reconstruct query string
                    from urllib.parse import urlencode
                    new_query = urlencode(new_params, doseq=True)
                    new_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
                    urls.append(new_url)
                    logger.info(f"  📄 Page {page_num}: {new_url}")
                
                return urls
            
            # Pattern 4: ?p=N in query
            if 'p' in query_params:
                logger.info(f"🔍 Detected short query parameter: ?p=N")
                
                for page_num in page_numbers:
                    new_params = query_params.copy()
                    new_params['p'] = [str(page_num)]
                    from urllib.parse import urlencode
                    new_query = urlencode(new_params, doseq=True)
                    new_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
                    urls.append(new_url)
                    logger.info(f"  📄 Page {page_num}: {new_url}")
                
                return urls
            
            # Fallback: Assume path-based /page/N pattern if no pattern detected
            logger.warning(f"⚠️  No pagination pattern detected in URL, assuming /page/N pattern")
            base_path = parsed.path.rstrip('/')
            
            for page_num in page_numbers:
                new_path = f"{base_path}/page/{page_num}"
                new_url = urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, parsed.query, parsed.fragment))
                urls.append(new_url)
                logger.info(f"  📄 Page {page_num} (fallback): {new_url}")
            
            return urls
            
        except Exception as e:
            logger.error(f"Failed to generate specific page URLs: {str(e)}", exc_info=True)
            # Return base URL as fallback
            return [base_url]

    async def _extract_from_chunk(self, chunk: str, extraction_instruction: str, prompt: str) -> list:
        """
        Extract data from a single HTML chunk using Gemini.

        Args:
            chunk: HTML chunk to process
            extraction_instruction: Instructions for extraction
            prompt: Original user prompt

        Returns:
            List of extracted items from this chunk
        """
        try:
            extraction_prompt = f"""Extract data from this HTML based on the user's request: "{prompt}"

HTML Content:
{chunk}

{extraction_instruction}

IMPORTANT: Return ONLY a valid JSON array, nothing else. Each item should be a JSON object.
Example: [{{"type": "product", "name": "...", "price": "..."}}, ...]
"""

            response_text = (await self._generate_text(extraction_prompt)).strip()

            logger.debug(f"Chunk extraction response length: {len(response_text)} chars")

            # Parse response
            items = self._parse_llm_response(response_text)

            return items

        except Exception as e:
            logger.error(f"Chunk extraction failed: {str(e)}")
            return []

    def _deduplicate_items(self, items: list) -> list:
        """
        Remove duplicate items based on content similarity.

        Args:
            items: List of extracted items

        Returns:
            List with duplicates removed
        """
        if not items:
            return []

        # Convert items to JSON strings for comparison
        seen = set()
        unique_items = []

        for item in items:
            # Create a hashable representation
            item_str = json.dumps(item, sort_keys=True)

            if item_str not in seen:
                seen.add(item_str)
                unique_items.append(item)

        logger.info(f"Deduplication: {len(items)} -> {len(unique_items)} items")
        return unique_items

    async def generate_summary(self, data: Any, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a summary and chart recommendations based on data and optional prompt.
        """
        try:
            data_str = json.dumps(data, default=str)[:100000] # Truncate if too large
            
            user_instruction = ""
            if prompt:
                user_instruction = f"FOCUS ON THIS USER REQUEST: '{prompt}'"
            
            system_prompt = f"""
You are a Data Analyst Agent. Analyze the provided JSON data and generate a summary and chart recommendations.
{user_instruction}

DATA:
{data_str}

OUTPUT FORMAT (JSON):
{{
    "summaryText": "A concise narrative summary of the data...",
    "insightHighlights": ["Key insight 1", "Key insight 2", ...],
    "charts": [
        {{
            "chartType": "bar|pie|line|histogram",
            "title": "Chart Title",
            "reasoning": "Why this chart is useful...",
            "xAxisFields": ["field_name"],
            "yAxisFields": ["field_name"],
            "aggregationFunction": "count|sum|avg|min|max",
            "confidence": 0.9
        }}
    ]
}}
"""
            response_text = await self._generate_text(system_prompt)
            
            # Parse JSON response
            try:
                # Clean markdown code blocks if present
                clean_text = response_text.replace("```json", "").replace("```", "").strip()
                return json.loads(clean_text)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse summary JSON: {response_text}")
                return {
                    "summaryText": "Failed to generate structured summary.",
                    "insightHighlights": [],
                    "charts": []
                }
                
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}", exc_info=True)
            return {
                "summaryText": f"Error generating summary: {str(e)}",
                "insightHighlights": [],
                "charts": []
            }

    def _detect_schema_type(self, data: List[Dict[str, Any]]) -> str:
        """
        Auto-detect schema type from crawled data structure.
        
        Args:
            data: List of extracted items
            
        Returns:
            Schema type: 'product_list', 'article', 'generic_data'
        """
        if not data or not isinstance(data, list) or len(data) == 0:
            return "generic_data"
        
        # Sample first few items to detect patterns
        sample = data[:min(10, len(data))]
        field_names = set()
        
        for item in sample:
            if isinstance(item, dict):
                field_names.update(item.keys())
        
        # Product list indicators
        product_indicators = {"price", "name", "title", "product", "rating", "reviews", "stock", "brand", "sku"}
        if len(field_names & product_indicators) >= 2:
            return "product_list"
        
        # Article indicators
        article_indicators = {"title", "content", "author", "date", "published", "body", "text", "article"}
        if len(field_names & article_indicators) >= 2:
            return "article"
        
        return "generic_data"

    def _calculate_quality_score(self, data: List[Dict[str, Any]]) -> float:
        """
        Calculate data quality score (0.0 - 1.0) based on completeness and consistency.
        
        Args:
            data: List of extracted items
            
        Returns:
            Quality score between 0.0 (poor) and 1.0 (excellent)
        """
        if not data or not isinstance(data, list) or len(data) == 0:
            return 0.0
        
        total_score = 0.0
        
        for item in data:
            if not isinstance(item, dict):
                continue
            
            item_score = 0.0
            field_count = len(item)
            
            if field_count == 0:
                continue
            
            # Calculate completeness (non-null fields)
            non_null_fields = sum(1 for v in item.values() if v not in [None, "", "N/A", "null"])
            completeness = non_null_fields / field_count if field_count > 0 else 0.0
            
            # Calculate consistency (uniform types within fields)
            type_consistency = 1.0  # Simplified - could be enhanced
            
            item_score = (completeness * 0.7) + (type_consistency * 0.3)
            total_score += item_score
        
        return min(1.0, total_score / len(data))

    def _create_embedding_text(self, data: List[Dict[str, Any]], conversation_name: str, prompt: str) -> str:
        """
        Create optimized text representation for embedding generation.
        
        Args:
            data: Extracted data items
            conversation_name: Short conversation title
            prompt: Original user request
            
        Returns:
            Formatted text for embedding (optimized for semantic search)
        """
        # Start with metadata
        text_parts = [
            f"Conversation: {conversation_name}",
            f"User Request: {prompt}",
            "\nData Summary:"
        ]
        
        # Add data preview (first 5 items, truncated)
        preview_items = data[:min(5, len(data))]
        for idx, item in enumerate(preview_items, 1):
            if isinstance(item, dict):
                # Create compact representation
                fields = []
                for k, v in list(item.items())[:5]:  # Max 5 fields per item
                    if v not in [None, "", "N/A", "null"]:
                        v_str = str(v)[:50]  # Truncate long values
                        fields.append(f"{k}={v_str}")
                
                if fields:
                    text_parts.append(f"{idx}. {', '.join(fields)}")
        
        # Add summary stats
        text_parts.append(f"\nTotal items: {len(data)}")
        
        # Limit total length for embedding API
        full_text = "\n".join(text_parts)
        return full_text[:2000]  # Truncate to 2000 chars for embedding

    async def _generate_embedding_with_stats(
        self, 
        data: List[Dict[str, Any]], 
        conversation_name: str, 
        prompt: str
    ) -> Dict[str, Any]:
        """
        OPTIMIZED: Generate embedding + analyze data quality in single operation.
        This eliminates separate API call from .NET service.
        
        Args:
            data: Extracted data items
            conversation_name: Short conversation title
            prompt: Original user request
            
        Returns:
            Dict with embedding_text, embedding_vector, schema_type, quality_score
        """
        try:
            # Detect schema type
            schema_type = self._detect_schema_type(data)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(data)
            
            # Create embedding text
            embedding_text = self._create_embedding_text(data, conversation_name, prompt)
            
            # Generate embedding using configured provider (external LLM or Gemini)
            # Uses adapter pattern to support multiple providers
            embedding_vector = await self.gemini_client.embed(embedding_text)
            
            logger.info(f"Generated embedding: {len(embedding_vector)}-dim, schema={schema_type}, quality={quality_score:.2f}")
            
            return {
                "embedding_text": embedding_text,
                "embedding_vector": embedding_vector,
                "schema_type": schema_type,
                "quality_score": quality_score
            }
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
            # Return fallback with zero vector
            return {
                "embedding_text": self._create_embedding_text(data, conversation_name, prompt),
                "embedding_vector": [0.0] * 768,  # Zero vector fallback
                "schema_type": self._detect_schema_type(data),
                "quality_score": self._calculate_quality_score(data)
            }