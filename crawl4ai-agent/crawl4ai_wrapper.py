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

        if gemini_client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info("Crawl4AIWrapper initialized with dedicated Gemini model")
        else:
            # Shared Gemini client handles its own configuration/caching
            self.model = getattr(gemini_client, "model", None)
            logger.info("Crawl4AIWrapper using shared Gemini client instance")

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

                    # Combined AI call: conversation name + navigation plan + extraction fields
                    analysis_result = await self._analyze_and_plan_with_name(crawler, url, prompt)
                    
                    conversation_name = analysis_result["conversation_name"]
                    navigation_steps = analysis_result["navigation_plan"]
                    extraction_fields = analysis_result["data_extraction_fields"]
                    
                    logger.info(f"OPTIMIZED: Generated name '{conversation_name}' + {len(navigation_steps)} steps in single call")

                    # PUBLISH EVENT 2: Navigation Planning Completed
                    if self.kafka_publisher and job_id:
                        self.kafka_publisher.publish_progress(
                            "NavigationPlanningCompleted",
                            job_id,
                            user_id or "unknown",
                            {
                                "url": url,
                                "conversation_name": conversation_name,
                                "steps_count": len(navigation_steps),
                                "steps": navigation_steps,
                                "extraction_fields": extraction_fields
                            }
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
                extracted_data = await self._extract_data(
                    crawler,
                    navigation_result["final_url"],
                    navigation_result.get("collected_pages", []),
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

            # Get first 30000 chars for comprehensive analysis
            page_html = deep_cleaned[:30000]
            logger.info(f"Analysis HTML: {len(raw_html)} -> {len(deep_cleaned)} chars ({100 - int(len(deep_cleaned)/len(raw_html)*100)}% reduction), using first {len(page_html)} chars")

            # COMBINED prompt: conversation name + navigation planning
            analysis_prompt = f"""You are analyzing a webpage to help with data extraction.

USER REQUEST: "{prompt}"
WEBPAGE URL: {url}

PAGE HTML (simplified):
{page_html}

Your tasks:
1. Create a SHORT conversation name (max 6 words) that captures what the user wants
2. Plan the navigation strategy to fulfill the user's request
3. Identify the key data fields to extract

Respond in JSON format:
{{
    "conversation_name": "Short descriptive name (max 6 words)",
    "navigation_plan": [
        {{"action": "click|select|input|scroll|paginate|extract|wait", "selector": "...", "value": "...", "description": "..."}}
    ],
    "data_extraction_fields": ["field1", "field2", "field3"]
}}

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

Example response:
{{
    "conversation_name": "iPhone Price Comparison",
    "navigation_plan": [
        {{"action": "click", "selector": "a[href*='electronics']", "description": "Navigate to electronics"}},
        {{"action": "select", "selector": "#brand-filter", "value": "iPhone", "description": "Filter by iPhone brand"}},
        {{"action": "paginate", "selector": ".pagination .next", "description": "Collect all product pages"}},
        {{"action": "extract", "description": "Extract product data"}}
    ],
    "data_extraction_fields": ["product_name", "price", "rating", "reviews_count"]
}}

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
            
            logger.info(f"Analysis complete: '{conversation_name}' with {len(navigation_plan)} steps, {len(extraction_fields)} fields")

            return {
                "conversation_name": conversation_name,
                "navigation_plan": navigation_plan,
                "data_extraction_fields": extraction_fields
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
                    pages = await self._handle_pagination(
                        crawler, current_url, selector, max_pages,
                        kafka_publisher=kafka_publisher,
                        job_id=job_id,
                        user_id=user_id
                    )
                    collected_pages.extend(pages)
                    executed_steps.append({
                        "action": action,
                        "selector": selector,
                        "pages_collected": len(pages),
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
            "collected_pages": collected_pages
        }

    async def _handle_pagination(
        self, crawler, url: str, next_selector: str, max_pages: int = 50,
        kafka_publisher=None,
        job_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[str]:
        """
        Collect HTML from all paginated pages (IMPROVED VERSION)

        Fixes:
        - Uses BeautifulSoup for reliable next-button detection
        - Tracks visited URLs to prevent infinite loops
        - Prefers URL navigation over clicking (faster)
        - Tries alternative selectors
        - Single page load per iteration

        Args:
            crawler: AsyncWebCrawler instance
            url: Starting URL
            next_selector: CSS selector for next/more button
            max_pages: Maximum pages to collect

        Returns:
            List of HTML strings from each page
        """
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        pages = []
        current_page = 1
        current_url = url
        visited_urls = set()  # Prevent infinite loops

        logger.info(f"Starting pagination with max {max_pages} pages, selector: {next_selector}")

        while current_page <= max_pages:
            try:
                # Prevent visiting same URL twice
                if current_url in visited_urls:
                    logger.warning(f"Already visited {current_url}, stopping pagination")
                    break

                visited_urls.add(current_url)

                # Crawl current page with wait
                logger.info(f"Loading page {current_page}: {current_url}")
                result = await crawler.arun(
                    url=current_url,
                    wait_for="networkidle",
                    delay_before_return_html=1.5  # Wait for dynamic content
                )

                # Collect page HTML
                page_html = result.cleaned_html or result.html
                pages.append(page_html)
                logger.info(f"Collected page {current_page} ({len(page_html)} chars)")

                # PUBLISH EVENT 5: Pagination Page Loaded
                if kafka_publisher and job_id:
                    kafka_publisher.publish_progress(
                        "PaginationPageLoaded",
                        job_id,
                        user_id or "unknown",
                        {
                            "page_number": current_page,
                            "total_pages_collected": len(pages),
                            "max_pages": max_pages,
                            "page_size_chars": len(page_html),
                            "url": current_url
                        }
                    )

                # Check if next button exists using BeautifulSoup (more reliable)
                soup = BeautifulSoup(result.html, 'html.parser')

                # Try to find next button using CSS selector
                next_button = soup.select_one(next_selector)

                if not next_button:
                    logger.info(f"No next button found with selector '{next_selector}', checking alternatives...")

                    # Try common alternative selectors
                    alternatives = [
                        'a[rel="next"]',
                        '.pagination .next a',
                        'a.next-page',
                        'button.load-more',
                        '.pagination li.active + li a'  # Next page after active
                    ]

                    for alt_selector in alternatives:
                        try:
                            next_button = soup.select_one(alt_selector)
                            if next_button:
                                logger.info(f"Found next button with alternative selector: {alt_selector}")
                                next_selector = alt_selector  # Update for next iteration
                                break
                        except:
                            continue

                if not next_button:
                    logger.info(f"No more pages found after page {current_page}")
                    break

                # Check if next button is disabled
                if next_button.get('disabled') or 'disabled' in next_button.get('class', []):
                    logger.info("Next button is disabled, no more pages")
                    break

                # Get next page URL if available
                next_url = next_button.get('href')
                
                # FIX: Prefer clicking to ensure correct context/state is maintained.
                # The previous logic prioritized manual URL construction which caused context loss 
                # (e.g. jumping from /category/wii-u to /products?page=2).
                # We now force a click unless there is no button to click.
                
                if next_url:
                    logger.info(f"Found next page URL in href: {next_url}")

                # Click the button using JavaScript
                logger.info(f"Clicking pagination button: {next_selector}")

                js_click = f"document.querySelector('{next_selector}').click()"
                
                try:
                    click_result = await crawler.arun(
                        url=current_url,
                        js_code=[js_click],
                        wait_for="networkidle",
                        delay_before_return_html=2.0  # Extra wait for AJAX/Navigation
                    )

                    # Check if URL changed
                    if click_result.url != current_url:
                        current_url = click_result.url
                        logger.info(f"URL changed to: {current_url}")
                    else:
                        logger.warning("URL didn't change after click, checking if content updated...")
                        # In some SPA cases, URL might not change but content does. 
                        # We assume success if no error, but for pagination usually URL changes.
                        
                except Exception as click_error:
                    logger.warning(f"Click failed: {click_error}. Falling back to URL navigation if available.")
                    
                    if next_url:
                        # Fallback to URL navigation only if click fails
                        if next_url.startswith('/'):
                            current_url = urljoin(current_url, next_url)
                        elif next_url.startswith('http'):
                            current_url = next_url
                        else:
                            # Relative URL
                            current_url = urljoin(current_url, next_url)
                        logger.info(f"Fallback: Navigating to next page via URL: {current_url}")
                    else:
                        raise click_error

                current_page += 1

                # Small delay between pages to avoid rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Pagination error at page {current_page}: {str(e)}", exc_info=True)
                break

        logger.info(f"Pagination complete. Collected {len(pages)} pages")
        return pages

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
        Extract data using RAG-style Gemini (1 call per page)

        Args:
            crawler: AsyncWebCrawler instance
            url: Final URL after navigation
            additional_pages: Additional pages from pagination
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
                        "total_pages_to_process": 1 + len(additional_pages[:10])
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

            # Always try to capture the final navigated page first
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
        RAG-STYLE EXTRACTION: Chunk → Filter → Concatenate → **1 Gemini Call**

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

            # Step 3: Batch relevance filter (1 API call)
            relevance_flags = await self._batch_check_chunk_relevance(chunks, prompt)
            relevant_chunks = [chunk for chunk, flag in zip(chunks, relevance_flags) if flag]
            logger.info(f"Kept {len(relevant_chunks)} relevant chunks")

            if not relevant_chunks:
                logger.warning("No relevant chunks found")
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