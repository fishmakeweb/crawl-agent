"""
Cost-optimized Gemini Client with caching, batching, and local LLM fallback
"""
import google.generativeai as genai
from cachetools import TTLCache
import asyncio
from typing import List, Optional, Dict, Any
import hashlib
import json
import aiohttp
from config import GeminiConfig


class AsyncBatcher:
    """Batch multiple requests to reduce API calls"""

    def __init__(self, max_batch: int = 8, timeout: float = 1.0, gemini_model=None):
        self.max_batch = max_batch
        self.timeout = timeout
        self.queue = []
        self.lock = asyncio.Lock()
        self.gemini_model = gemini_model
        self._processing = False

    async def submit(self, prompt: str) -> str:
        """Submit prompt and wait for batched response"""
        future = asyncio.Future()

        async with self.lock:
            self.queue.append((prompt, future))

            # Trigger batch if full
            if len(self.queue) >= self.max_batch:
                asyncio.create_task(self._process_batch())

        # Start timeout timer if not processing
        if not self._processing:
            asyncio.create_task(self._timeout_trigger())

        # Wait for result
        return await future

    async def _timeout_trigger(self):
        """Trigger batch processing after timeout"""
        await asyncio.sleep(self.timeout)
        if self.queue and not self._processing:
            await self._process_batch()

    async def _process_batch(self):
        """Process queued prompts in a single batch"""
        async with self.lock:
            if not self.queue or self._processing:
                return

            self._processing = True
            batch = self.queue[:self.max_batch]
            self.queue = self.queue[self.max_batch:]

        try:
            # Combine prompts
            combined_prompt = self._combine_prompts([p for p, _ in batch])

            # Single API call
            response = await self._call_gemini(combined_prompt)

            # Split responses
            responses = self._split_responses(response, len(batch))

            # Resolve futures
            for (_, future), resp in zip(batch, responses):
                if not future.done():
                    future.set_result(resp)

        except Exception as e:
            # Reject all futures with error
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)
        finally:
            self._processing = False

    def _combine_prompts(self, prompts: List[str]) -> str:
        """Combine multiple prompts into one request"""
        combined = "Process these requests and return as JSON array of strings:\n\n"
        for i, prompt in enumerate(prompts):
            combined += f'Request {i+1}: "{prompt[:500]}..."\n\n'
        combined += '\nReturn format: ["response1", "response2", ...]'
        return combined

    async def _call_gemini(self, prompt: str) -> str:
        """Actual Gemini API call"""
        if self.gemini_model:
            response = await self.gemini_model.generate_content_async(prompt)
            return response.text
        return ""

    def _split_responses(self, response: str, count: int) -> List[str]:
        """Split batched response back into individual responses"""
        try:
            responses = json.loads(response)
            if isinstance(responses, list):
                return responses[:count]
        except:
            pass

        # Fallback: split by delimiter or return same response for all
        return [response] * count


class GeminiClient:
    """
    Cost-optimized Gemini client with:
    - Caching (TTL-based)
    - Batching (group requests)
    - Fallback to local LLM (large/repeated tasks)
    """

    def __init__(self, config: GeminiConfig):
        self.config = config
        genai.configure(api_key=config.API_KEY)

        # Use Gemini 2.0 Flash for cost efficiency
        self.model = genai.GenerativeModel(config.MODEL)
        self.embedding_model = config.EMBEDDING_MODEL

        # Cache layer (1 hour TTL by default)
        self.cache = TTLCache(
            maxsize=config.CACHE_MAX_SIZE,
            ttl=config.CACHE_TTL_SECONDS
        ) if config.CACHE_ENABLED else {}

        # Batching layer
        self.batcher = AsyncBatcher(
            max_batch=config.MAX_BATCH_SIZE,
            timeout=config.BATCH_TIMEOUT_SECONDS,
            gemini_model=self.model
        ) if config.BATCHING_ENABLED else None

        # Local LLM fallback (optional)
        self.local_llm_endpoint = config.LOCAL_LLM_ENDPOINT if config.LOCAL_LLM_ENABLED else None
        self.local_llm_threshold = config.LOCAL_LLM_THRESHOLD_CHARS

        # Cost tracking
        self.stats = {
            "gemini_calls": 0,
            "cache_hits": 0,
            "local_llm_calls": 0,
            "batched_requests": 0,
            "total_requests": 0
        }

    def _cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key from prompt + params"""
        key_data = f"{prompt}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def generate(
        self,
        prompt: str,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text with multi-tier optimization:
        1. Check cache
        2. Route to local LLM if large/extraction task
        3. Batch with other requests
        4. Fall back to direct Gemini call
        """

        self.stats["total_requests"] += 1

        # Tier 1: Cache lookup
        if use_cache and self.config.CACHE_ENABLED:
            cache_key = self._cache_key(prompt, **kwargs)
            if cache_key in self.cache:
                self.stats["cache_hits"] += 1
                return self.cache[cache_key]

        # Tier 2: Local LLM for large extraction tasks
        if self._should_use_local_llm(prompt):
            response = await self._fallback_local_llm(prompt, **kwargs)
            if response:
                self.stats["local_llm_calls"] += 1
                if use_cache:
                    self.cache[cache_key] = response
                return response

        # Tier 3: Batching for small prompts
        if (self.batcher and
            len(prompt) < 1000 and
            not kwargs.get("response_mime_type")):
            self.stats["batched_requests"] += 1
            response = await self.batcher.submit(prompt)
        else:
            # Tier 4: Direct Gemini call
            response = await self._call_gemini(prompt, **kwargs)
            self.stats["gemini_calls"] += 1

        # Cache the response
        if use_cache and self.config.CACHE_ENABLED:
            cache_key = self._cache_key(prompt, **kwargs)
            self.cache[cache_key] = response

        return response

    def _should_use_local_llm(self, prompt: str) -> bool:
        """Decide if local LLM should be used"""
        if not self.local_llm_endpoint:
            return False

        # Use local for:
        # 1. Large prompts (> threshold chars)
        # 2. Extraction-heavy tasks (lots of HTML)

        is_large = len(prompt) > self.local_llm_threshold
        is_extraction = "extract" in prompt.lower() and "html" in prompt.lower()

        return is_large or is_extraction

    async def _fallback_local_llm(self, prompt: str, **kwargs) -> Optional[str]:
        """Use local LLM for cost savings on large tasks"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.local_llm_endpoint}/api/generate",
                    json={
                        "model": "llama2",  # or your local model
                        "prompt": prompt,
                        "stream": False,
                        **kwargs
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("response", "")
        except Exception as e:
            print(f"⚠️  Local LLM failed, falling back to Gemini: {e}")
            return None

    async def _call_gemini(self, prompt: str, **kwargs) -> str:
        """Direct Gemini API call"""
        try:
            response = await self.model.generate_content_async(prompt, **kwargs)
            return response.text
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            raise

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings (cached)"""
        cache_key = f"embed:{hashlib.sha256(text.encode()).hexdigest()}"

        if self.config.CACHE_ENABLED and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]

        result = await genai.embed_content_async(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document"
        )

        embedding = result['embedding']

        if self.config.CACHE_ENABLED:
            self.cache[cache_key] = embedding

        self.stats["gemini_calls"] += 1
        return embedding

    async def interpret_feedback(
        self,
        feedback: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Interpret user feedback with clarification validation.
        Returns structured interpretation + confidence score.
        """

        prompt = f"""
User provided this feedback on a web crawl:
"{feedback}"

Crawl context:
- URL: {context.get('url', 'N/A')}
- Extracted fields: {json.dumps(context.get('fields', {}), indent=2)}
- Errors: {context.get('errors', [])}

Interpret this feedback as structured learning signals:

{{
    "confidence": 0.0-1.0,  // How confident are you in this interpretation?
    "quality_rating": 1-5,
    "specific_issues": ["issue1", "issue2"],
    "desired_improvements": ["improvement1", "improvement2"],
    "clarification_needed": true/false,
    "clarification_question": "..."  // If clarification needed
}}

If confidence < 0.7, set clarification_needed=true and ask a specific question.

Return as JSON only.
"""

        interpretation_json = await self.generate(
            prompt,
            response_mime_type="application/json"
        )

        try:
            return json.loads(interpretation_json)
        except json.JSONDecodeError:
            # Fallback interpretation
            return {
                "confidence": 0.5,
                "quality_rating": 3,
                "specific_issues": ["Unable to parse feedback"],
                "desired_improvements": [],
                "clarification_needed": True,
                "clarification_question": "Could you please rephrase your feedback more specifically?"
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for cost monitoring"""
        total_requests = self.stats["total_requests"]
        cache_hit_rate = (
            self.stats["cache_hits"] / total_requests
            if total_requests > 0 else 0.0
        )

        gemini_cost_per_1k_tokens = 0.000125  # Gemini 2.0 Flash pricing (approx)
        estimated_tokens = self.stats["gemini_calls"] * 500  # Rough estimate
        estimated_cost = (estimated_tokens / 1000) * gemini_cost_per_1k_tokens
        estimated_savings = estimated_cost * cache_hit_rate

        return {
            **self.stats,
            "cache_hit_rate": cache_hit_rate,
            "estimated_cost_usd": round(estimated_cost, 4),
            "estimated_savings_usd": round(estimated_savings, 4)
        }

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "gemini_calls": 0,
            "cache_hits": 0,
            "local_llm_calls": 0,
            "batched_requests": 0,
            "total_requests": 0
        }
