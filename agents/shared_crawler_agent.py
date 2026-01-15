from __future__ import annotations

"""
Shared Crawler Agent Core
Used by both Production and Training services
"""
from typing import Dict, Any, Optional
import sys
import os
import json
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Add crawl4ai-agent to path (inside self-learning-agent)
sys.path.append(os.path.join(os.path.dirname(__file__), '../crawl4ai-agent'))

try:
    from agentlightning import LitAgent, NamedResources, Rollout
    import agentlightning as agl
    AGENTLIGHTNING_AVAILABLE = True
except ImportError:
    print("⚠️  Agent Lightning not installed, using mock classes")
    AGENTLIGHTNING_AVAILABLE = False
    # Mock classes for development
    class LitAgent:
        @classmethod
        def __class_getitem__(cls, item):
            return cls
    class NamedResources:
        pass
    class Rollout:
        pass
    class agl:
        @staticmethod
        def emit_message(msg): print(f"📝 {msg}")
        @staticmethod
        def emit_object(obj): print(f"📦 {obj}")
        @staticmethod
        def emit_reward(r): print(f"🎯 Reward: {r}")
        @staticmethod
        def emit_exception(e): print(f"❌ {e}")

from crawl4ai_wrapper import Crawl4AIWrapper

# Determine base class without using subscript in conditional
if AGENTLIGHTNING_AVAILABLE:
    _BaseAgent = LitAgent
else:
    _BaseAgent = object


class SharedCrawlerAgent(_BaseAgent):
    """
    Shared agent core used by both Production and Training services.
    Mode determined by service context.
    """

    def __init__(self, gemini_client, mode: str = "production"):
        self.gemini_client = gemini_client
        self.mode = mode  # "production" or "training"
        self.base_crawler = Crawl4AIWrapper(gemini_client=gemini_client)
        print(f"🤖 Shared Crawler Agent initialized in {mode.upper()} mode")

    def rollout(self, task: Dict[str, Any], resources: NamedResources,
                rollout: Rollout) -> float:
        """Execute crawl with mode-specific behavior"""

        if self.mode == "production":
            return asyncio.run(self._production_rollout(task, resources, rollout))
        else:
            return asyncio.run(self._training_rollout(task, resources, rollout))

    async def _production_rollout(self, task, resources, rollout) -> float:
        """
        Production mode:
        - Use frozen resources
        - No tracing overhead
        - Optimized for speed
        - No learning
        """
        try:
            # Get frozen resources (no updates)
            frozen_prompt = resources.get("extraction_prompt", "")
            frozen_config = resources.get("crawl_config", {})
            domain_patterns = resources.get("domain_patterns", {})

            # Execute crawl (optimized path)
            domain = self._extract_domain(task["url"])

            config = {**frozen_config}
            if domain in domain_patterns:
                config.update(domain_patterns[domain])

            # Use base crawler directly
            result = await self.base_crawler.intelligent_crawl(
                url=task["url"],
                prompt=task.get("user_description", ""),
                extract_schema=task.get("extraction_schema", {}),
                max_pages=config.get("max_pages", 50)
            )

            # Return success indicator
            return 1.0 if result.get("success") else 0.0

        except Exception as e:
            print(f"❌ Production crawl error: {e}")
            return 0.0

    async def _training_rollout(self, task, resources, rollout) -> float:
        """
        Training mode:
        - Full instrumentation
        - Detailed rewards
        - Pattern learning
        - Feedback integration
        """
        try:
            # Get updatable resources
            prompt_template = resources.get("extraction_prompt", "")
            crawl_config = resources.get("crawl_config", {})
            learned_patterns = resources.get("domain_patterns", {})

            # Enhanced logging
            domain = self._extract_domain(task["url"])
            agl.emit_message(f"🎯 Training mode - Crawling: {domain}")

            # Check learned patterns
            if domain in learned_patterns:
                agl.emit_message(f"📚 Using learned patterns for {domain}")
                crawl_config.update(learned_patterns[domain])
            else:
                agl.emit_message(f"🆕 New domain: {domain}")

            # Execute crawl with full tracing
            agl.emit_message("🌐 Starting crawl...")

            crawl_result = await self.base_crawler.intelligent_crawl(
                url=task["url"],
                prompt=task.get("user_description", ""),
                extract_schema=task.get("extraction_schema", {}),
                max_pages=crawl_config.get("max_pages", 50)
            )

            agl.emit_object({
                "crawl_status": "success" if crawl_result.get("success") else "failed",
                "items_extracted": len(crawl_result.get("data", [])),
                "pages_collected": crawl_result.get("navigation_result", {}).get("pages_collected", 0),
                "execution_time_ms": crawl_result.get("execution_time_ms", 0)
            })

            # Extract data
            extracted_data = crawl_result.get("data", [])
            agl.emit_message(f"📊 Extracted {len(extracted_data)} items")

            # Validation
            validation_score = self._validate_extraction(
                extracted_data,
                task.get("extraction_schema", {})
            )
            agl.emit_reward(validation_score * 0.3)
            agl.emit_message(f"✅ Validation score: {validation_score:.2f}")

            # Check improvement from previous feedback
            if task.get("feedback_from_previous"):
                improvement = await self._assess_improvement(
                    task["feedback_from_previous"],
                    extracted_data
                )
                agl.emit_reward(improvement * 0.2)
                agl.emit_message(f"📈 Improvement: {improvement:.2f}")

            # Base reward (will be augmented by user feedback)
            base_reward = self._calculate_base_reward(
                crawl_result,
                extracted_data,
                validation_score
            )

            agl.emit_message(f"🎯 Base reward: {base_reward:.2f}")
            return base_reward

        except Exception as e:
            agl.emit_exception(e)
            agl.emit_message(f"❌ Error: {str(e)}")
            return 0.0

    def _validate_extraction(self, extracted: list, schema: dict) -> float:
        """Validate against schema"""
        if not schema or not extracted:
            return 1.0 if extracted else 0.0

        required = schema.get("required", [])
        if not required:
            return 1.0

        # Check if extracted items have required fields
        valid_items = 0
        for item in extracted:
            if all(field in item and item[field] for field in required):
                valid_items += 1

        return valid_items / len(extracted) if extracted else 0.0

    async def _assess_improvement(self, previous_feedback: str,
                                  current_extraction: list) -> float:
        """Assess if extraction improved from feedback"""

        extraction_str = json.dumps(current_extraction, indent=2)

        prompt = f"""
Previous user feedback: "{previous_feedback}"
Current extraction result: {extraction_str}

On a scale of 0.0 to 1.0, how well does the current extraction address
the issues mentioned in the previous feedback?

Return only the number (0.0-1.0).
"""

        try:
            score = await self.gemini_client.generate(prompt)
            return float(score.strip())
        except:
            return 0.5

    def _calculate_base_reward(self, crawl_result: dict,
                              extracted: list, validation: float) -> float:
        """Calculate base reward"""

        success = 1.0 if crawl_result.get("success") else 0.0
        has_data = 1.0 if extracted else 0.0
        no_errors = 1.0 if not crawl_result.get("error") else 0.5

        # Quantity bonus (diminishing returns)
        quantity_bonus = min(1.0, len(extracted) / 10) * 0.2

        return (
            success * 0.3 +
            has_data * 0.2 +
            validation * 0.3 +
            no_errors * 0.2 +
            quantity_bonus * 0.1
        )

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        return urlparse(url).netloc

    # Standalone execution methods (non-Agent Lightning)
    async def execute_crawl(self, task: Dict[str, Any],
                           resources: Optional[Dict] = None,
                           kafka_publisher=None) -> Dict[str, Any]:
        """
        Standalone execution method (without Agent Lightning).
        Used for testing or direct invocation.
        Supports Kafka real-time progress events.
        """

        if resources is None:
            resources = self._get_default_resources()

        try:
            if kafka_publisher is not None:
                self.base_crawler.kafka_publisher = kafka_publisher

            # Use 'prompt' field (preferred) or fall back to 'user_description'
            prompt = task.get("prompt") or task.get("user_description", "")
            
            result = await self.base_crawler.intelligent_crawl(
                url=task["url"],
                prompt=prompt,
                job_id=task.get("job_id"),
                user_id=task.get("user_id"),
                navigation_steps=task.get("navigation_steps"),
                extract_schema=task.get("extraction_schema", {}),
                max_pages=resources.get("crawl_config", {}).get("max_pages", 50)
            )

            # Log what will be returned to C# service
            extracted_data = result.get("data", [])
            embedding_data = result.get("embedding_data")
            conversation_name = result.get("conversation_name", "Data Collection")
            
            logger.info("=" * 80)
            logger.info("📤 RESPONSE TO .NET SERVICE:")
            logger.info(f"   ✓ success: {result.get('success', False)}")
            logger.info(f"   ✓ conversation_name: '{conversation_name}'")
            logger.info(f"   ✓ data: {len(extracted_data)} items")
            if embedding_data:
                emb_vector = embedding_data.get('embedding_vector', [])
                emb_dims = len(emb_vector) if isinstance(emb_vector, list) else 0
                logger.info(f"   ✓ embedding: {emb_dims}-dim, quality={embedding_data.get('quality_score', 0):.2f}")
            else:
                logger.info("   ✓ embedding: None")
            logger.info(f"   ✓ execution_time: {result.get('execution_time_ms', 0)}ms")
            logger.info("=" * 80)

            return {
                "success": result.get("success", False),
                "data": extracted_data,
                "navigation_result": result.get("navigation_result", {}),
                "execution_time_ms": result.get("execution_time_ms", 0),
                "conversation_name": result.get("conversation_name", "Data Collection"),
                "embedding_data": embedding_data,  # NEW: Pre-generated embeddings
                "metadata": {
                    "execution_time_ms": result.get("execution_time_ms", 0),
                    "pages_collected": result.get("navigation_result", {}).get("pages_collected", 0),
                    "domain": self._extract_domain(task["url"])
                },
                "error": result.get("error")
            }

        except Exception as e:
            return {
                "success": False,
                "data": [],
                "metadata": {},
                "error": str(e)
            }

    def _get_default_resources(self) -> Dict[str, Any]:
        """Get default resources when none provided"""
        return {
            "extraction_prompt": """
                Extract data from this webpage according to the user's request.

                URL: {url}
                User wants: {user_intent}

                HTML content:
                {html_content}

                Target schema:
                {schema}

                Extract data matching the schema. Return as JSON array.
            """,
            "crawl_config": {
                "timeout": 30,
                "wait_for": "networkidle",
                "screenshot": False,
                "max_pages": 50
            },
            "domain_patterns": {},
            "knowledge_version": 0
        }


# Test function
async def test_agent():
    """Test the shared crawler agent"""
    from config import Config
    from gemini_client import GeminiClient

    config = Config()
    config.validate()

    gemini = GeminiClient(config.gemini)
    agent = SharedCrawlerAgent(gemini, mode="training")

    task = {
        "url": "https://www.example.com",
        "user_description": "Extract all product names and prices",
        "extraction_schema": {
            "required": ["product_name", "price"]
        }
    }

    result = await agent.execute_crawl(task)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(test_agent())
