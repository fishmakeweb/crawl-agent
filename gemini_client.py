"""
Cost-optimized Gemini Client with caching, batching, and local LLM fallback
Now supports multiple LLM providers via adapter pattern
"""
try:
    import google.generativeai as genai
except ImportError:
    genai = None

from cachetools import TTLCache
import asyncio
from typing import List, Optional, Dict, Any
import hashlib
import json
import aiohttp
from datetime import datetime, timedelta
from collections import deque
from config import LLMConfig, LLMProvider
from llm_adapters import create_adapter, LLMAdapter


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

    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = config.PROVIDER
        
        # Initialize adapter for multi-provider support
        self.adapter: LLMAdapter = create_adapter(config.get_adapter_config())
        
        # Legacy Gemini-specific initialization (only if using Gemini)
        if self.provider == LLMProvider.GEMINI and genai:
            genai.configure(api_key=config.API_KEY)
            self.model = genai.GenerativeModel(config.MODEL)
            self.embedding_model = config.EMBEDDING_MODEL
        else:
            self.model = None
            self.embedding_model = None

        # Cache layer (1 hour TTL by default)
        self.cache = TTLCache(
            maxsize=config.CACHE_MAX_SIZE,
            ttl=config.CACHE_TTL_SECONDS
        ) if config.CACHE_ENABLED else {}

        # Batching layer (only for Gemini - requires direct model access)
        self.batcher = AsyncBatcher(
            max_batch=config.MAX_BATCH_SIZE,
            timeout=config.BATCH_TIMEOUT_SECONDS,
            gemini_model=self.model
        ) if config.BATCHING_ENABLED and self.provider == LLMProvider.GEMINI else None

        # Local LLM fallback (optional)
        self.local_llm_endpoint = config.LOCAL_LLM_ENDPOINT if config.LOCAL_LLM_ENABLED else None
        self.local_llm_threshold = config.LOCAL_LLM_THRESHOLD_CHARS

        # Multi-model routing setup (only for Gemini)
        self.models = {}
        if self.provider == LLMProvider.GEMINI and config.ROUTING_ENABLED and hasattr(config, 'MODELS') and config.MODELS and genai:
            for model_id, model_config in config.MODELS.items():
                self.models[model_id] = {
                    "instance": genai.GenerativeModel(model_id),
                    "config": model_config,
                }
        
        # Rate limiting tracking (sliding window)
        self.request_timestamps = deque(maxlen=100)
        self.token_usage_window = deque(maxlen=100)

        # Cost tracking
        self.stats = {
            "gemini_calls": 0,
            "cache_hits": 0,
            "local_llm_calls": 0,
            "batched_requests": 0,
            "total_requests": 0,
            "model_usage": {},
            "rate_limit_warnings": 0,
            "model_fallbacks": 0,
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
        """LLM API call via adapter (supports multiple providers)"""
        try:
            # Use adapter for multi-provider support
            response = await self.adapter.generate_text_async(
                prompt=prompt,
                **kwargs
            )
            return response.text
        except Exception as e:
            print(f"❌ LLM API error ({self.provider.value}): {e}")
            raise

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings (cached) via adapter"""
        cache_key = f"embed:{hashlib.sha256(text.encode()).hexdigest()}"

        if self.config.CACHE_ENABLED and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]

        # Use adapter for multi-provider embedding support
        result = await self.adapter.generate_embedding_async(text=text)
        embedding = result.embedding  # Access as attribute, not dict

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

        # Extract actual data summary
        extracted_data = context.get('data', [])
        data_summary = "No data extracted"
        if extracted_data and len(extracted_data) > 0:
            item_count = len(extracted_data)
            sample_fields = list(extracted_data[0].keys()) if extracted_data[0] else []
            data_summary = f"{item_count} items extracted with fields: {sample_fields}"

        # Schema for reference
        schema = context.get('schema', {})
        schema_info = f"Expected schema: {json.dumps(schema, indent=2)}" if schema else "No schema provided"

        prompt = f"""
User provided this feedback on a web crawl:
"{feedback}"

Crawl context:
- URL: {context.get('url', 'N/A')}
- Extraction result: {data_summary}
- {schema_info}
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

    def _estimate_tokens(self, data: Any) -> int:
        """Estimate token count for input data"""
        if isinstance(data, str):
            # Rough estimate: 1 token ≈ 4 characters
            return len(data) // 4
        elif isinstance(data, dict):
            return len(json.dumps(data)) // 4
        elif isinstance(data, list):
            return sum(self._estimate_tokens(item) for item in data)
        return 0

    def _check_rate_limits(self, estimated_tokens: int) -> tuple:
        """Check if we're approaching rate limits. Returns (is_safe, warning_msg)"""
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        # Clean old timestamps
        while self.request_timestamps and self.request_timestamps[0] < one_minute_ago:
            self.request_timestamps.popleft()
        
        while self.token_usage_window and self.token_usage_window[0][0] < one_minute_ago:
            self.token_usage_window.popleft()
        
        # Calculate current usage
        rpm = len(self.request_timestamps)
        tpm = sum(tokens for _, tokens in self.token_usage_window)
        
        max_rpm = self.config.MAX_RPM
        max_tpm = self.config.MAX_TPM
        
        # Check thresholds
        rpm_pct = rpm / max_rpm if max_rpm > 0 else 0
        tpm_pct = tpm / max_tpm if max_tpm > 0 else 0
        
        if tpm_pct >= self.config.TPM_WARNING_THRESHOLD:
            return False, f"TPM at {tpm_pct*100:.1f}% ({tpm}/{max_tpm})"
        if rpm_pct >= self.config.TPM_WARNING_THRESHOLD:
            return False, f"RPM at {rpm_pct*100:.1f}% ({rpm}/{max_rpm})"
        
        return True, ""

    def _select_model_for_task(
        self,
        task_type: str,
        estimated_tokens: int,
        mode: str,
        current_cost: float
    ) -> str:
        """Select optimal model based on task type and constraints"""
        
        # Task-to-model mapping
        task_routing = {
            "crawl": {
                "production": "gemini-2.0-flash",
                "training": "gemini-2.0-flash",
            },
            "feedback": {
                "production": "gemini-2.5-flash",
                "training": "gemini-2.5-flash",
            },
            "clarification": {
                "production": "learnlm-2.0-flash",
                "training": "learnlm-2.0-flash",
            },
            "prompt_gen": {
                "production": "gemini-2.5-flash",
                "training": "gemini-2.5-flash",
            },
            "analysis": {
                "production": "gemini-2.5-flash",
                "training": "gemini-2.5-pro",  # Deep analysis in training
            },
            "rl_decide": {
                "production": "gemini-2.5-flash",
                "training": "gemini-2.5-flash",
            },
        }
        
        # Get base model
        base_model = task_routing.get(task_type, {}).get(mode, "gemini-2.5-flash")
        
        # Apply complexity escalation
        if estimated_tokens > self.config.COMPLEX_TASK_TOKEN_THRESHOLD:
            if task_type == "analysis" and mode == "training":
                base_model = "gemini-2.5-pro"
            elif mode == "training":
                base_model = "gemini-2.5-flash"
        
        # Check rate limits - fallback to lite if needed
        is_safe, warning = self._check_rate_limits(estimated_tokens)
        if not is_safe:
            print(f"⚠️  Rate limit warning: {warning}, falling back to lite model")
            self.stats["rate_limit_warnings"] += 1
            base_model = "gemini-2.0-flash-lite"
        
        # Ensure model exists
        if base_model not in self.models:
            print(f"⚠️  Model {base_model} not available, using default")
            base_model = self.config.MODEL
        
        return base_model

    async def route_model(
        self,
        task_type: str,
        input_data: dict,
        current_metrics: dict
    ) -> tuple:
        """
        Route request to optimal model based on task type and constraints.
        
        Args:
            task_type: One of 'crawl', 'feedback', 'clarification', 'prompt_gen', 'analysis', 'rl_decide'
            input_data: Dict containing 'prompt' and optional task-specific data
            current_metrics: Dict with 'tokens_used', 'cost_usd', 'mode' ('production'|'training')
        
        Returns:
            Tuple of (selected_model_id, response_text, updated_metrics)
        """
        
        if not self.config.ROUTING_ENABLED or not self.models:
            # Fallback to default behavior
            response = await self.generate(input_data.get("prompt", ""))
            return self.config.MODEL, response, current_metrics
        
        # Extract data
        prompt = input_data.get("prompt", "")
        mode = current_metrics.get("mode", "production")
        
        # Estimate tokens
        estimated_tokens = self._estimate_tokens(input_data)
        
        # Select model
        selected_model_id = self._select_model_for_task(
            task_type=task_type,
            estimated_tokens=estimated_tokens,
            mode=mode,
            current_cost=current_metrics.get("cost_usd", 0.0)
        )
        
        print(f"🎯 Routing {task_type} ({estimated_tokens} est. tokens) → {selected_model_id}")
        
        # Track model usage
        if selected_model_id not in self.stats["model_usage"]:
            self.stats["model_usage"][selected_model_id] = 0
        self.stats["model_usage"][selected_model_id] += 1
        
        # Execute with retry logic
        max_retries = 2
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # Get model instance
                model_info = self.models[selected_model_id]
                model_instance = model_info["instance"]
                
                # Track request
                self.request_timestamps.append(datetime.now())
                self.token_usage_window.append((datetime.now(), estimated_tokens))
                
                # Handle special parameters
                kwargs = {}
                if input_data.get("json_mode"):
                    kwargs["generation_config"] = genai.GenerationConfig(
                        response_mime_type="application/json"
                    )
                
                # Make API call
                response = await model_instance.generate_content_async(prompt, **kwargs)
                response_text = response.text
                
                # Estimate output tokens (rough)
                output_tokens = self._estimate_tokens(response_text)
                total_tokens = estimated_tokens + output_tokens
                
                # Calculate cost
                model_config = model_info["config"]
                input_cost = (estimated_tokens / 1_000_000) * model_config["cost_per_1m_input"]
                output_cost = (output_tokens / 1_000_000) * model_config["cost_per_1m_output"]
                request_cost = input_cost + output_cost
                
                # Update metrics
                updated_metrics = {
                    **current_metrics,
                    "tokens_used": current_metrics.get("tokens_used", 0) + total_tokens,
                    "cost_usd": current_metrics.get("cost_usd", 0.0) + request_cost,
                    "last_model": selected_model_id,
                    "last_request_cost": request_cost,
                }
                
                return selected_model_id, response_text, updated_metrics
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Handle rate limit (429) - retry with cheaper model
                if "429" in error_str or "quota" in error_str.lower():
                    print(f"⚠️  Rate limit hit on {selected_model_id}, falling back to lite")
                    selected_model_id = "gemini-2.0-flash-lite"
                    self.stats["model_fallbacks"] += 1
                    retry_count += 1
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    continue
                
                # Other errors - retry once then fail
                if retry_count < max_retries:
                    print(f"⚠️  Error on {selected_model_id}: {e}, retrying...")
                    retry_count += 1
                    await asyncio.sleep(1)
                    continue
                else:
                    print(f"❌ Failed after {max_retries} retries: {e}")
                    raise
        
        # Should not reach here, but handle gracefully
        raise Exception(f"Request failed after retries: {last_error}")

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
            "estimated_savings_usd": round(estimated_savings, 4),
            "routing_enabled": self.config.ROUTING_ENABLED if hasattr(self.config, 'ROUTING_ENABLED') else False,
        }

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "gemini_calls": 0,
            "cache_hits": 0,
            "local_llm_calls": 0,
            "batched_requests": 0,
            "total_requests": 0,
            "model_usage": {},
            "rate_limit_warnings": 0,
            "model_fallbacks": 0,
        }
