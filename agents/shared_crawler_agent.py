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
            # max_pages comes from task (UI field), not domain config
            result = await self.base_crawler.intelligent_crawl(
                url=task["url"],
                prompt=task.get("user_description", ""),
                extract_schema=task.get("extraction_schema", {}),
                max_pages=task.get("max_pages")  # None = extract from prompt, int = explicit UI value
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
                max_pages=task.get("max_pages")  # None = extract from prompt, int = explicit UI value
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
        """
        Validate extraction quality with multi-dimensional scoring
        Returns score 0.0-1.0 based on:
        - Schema compliance
        - Data completeness
        - Data quality
        """
        if not extracted:
            logger.debug("⚠️  Validation: No data extracted")
            return 0.0
        
        # If no schema provided, infer basic quality from extracted data
        if not schema or not schema.get("required"):
            logger.warning(f"⚠️  Validation: No schema provided - inferring from {len(extracted)} items")
            return self._infer_quality_without_schema(extracted)
        
        required = schema.get("required", [])
        logger.debug(f"Validating {len(extracted)} items against schema with {len(required)} required fields: {required}")
        
        total_score = 0.0
        
        for item in extracted:
            item_score = 0.0
            
            # 1. Field presence (40%)
            fields_present = sum(1 for field in required if field in item)
            field_presence_score = fields_present / len(required)
            
            # 2. Field completeness (30%) - non-null, non-empty values
            fields_complete = sum(
                1 for field in required 
                if field in item and item[field] not in [None, "", [], {}]
            )
            completeness_score = fields_complete / len(required)
            
            # 3. Data quality (30%) - type correctness, reasonable values
            quality_score = self._assess_data_quality(item, required)
            
            item_score = (
                field_presence_score * 0.4 +
                completeness_score * 0.3 +
                quality_score * 0.3
            )
            
            total_score += item_score
        
        return total_score / len(extracted)
    
    def _infer_quality_without_schema(self, extracted: list) -> float:
        """
        Infer data quality when no schema is provided
        - Check completeness of fields (no null/empty)
        - Check data diversity (not all same values)
        - Penalize for missing schema (max 0.85)
        """
        if not extracted:
            return 0.0
        
        total_score = 0.0
        
        for item in extracted:
            if not isinstance(item, dict) or not item:
                continue
            
            fields = list(item.keys())
            if not fields:
                continue
            
            # 1. Completeness: How many fields have non-null values? (50%)
            non_null_fields = sum(
                1 for field in fields
                if item[field] not in [None, "", [], {}]
            )
            completeness = non_null_fields / len(fields)
            
            # 2. Quality assessment on common fields (50%)
            quality = self._assess_data_quality(item, fields)
            
            item_score = completeness * 0.5 + quality * 0.5
            total_score += item_score
        
        avg_score = total_score / len(extracted)
        
        # Penalize for missing schema: max 0.85 instead of 1.0
        # This encourages users to provide schemas
        return min(0.85, avg_score)
    
    def _assess_data_quality(self, item: dict, required_fields: list) -> float:
        """
        Assess data quality of extracted item
        - String fields should not be generic ("N/A", "null", etc.)
        - Numeric fields should be valid numbers
        - Dates should be parseable
        """
        quality_checks = 0
        total_checks = 0
        
        for field in required_fields:
            if field not in item:
                continue
            
            value = item[field]
            total_checks += 1
            
            # Skip None/empty
            if value in [None, "", [], {}]:
                continue
            
            field_lower = field.lower()
            
            # Price/cost fields should be numeric and positive
            if any(indicator in field_lower for indicator in ["price", "cost", "amount"]):
                try:
                    num_value = float(str(value).replace("$", "").replace(",", ""))
                    if num_value > 0:
                        quality_checks += 1
                except:
                    pass  # Invalid number
            
            # Rating fields should be in valid range
            elif "rating" in field_lower or "stars" in field_lower:
                try:
                    rating = float(value)
                    if 0 <= rating <= 5:  # Assume 5-star scale
                        quality_checks += 1
                except:
                    pass
            
            # Generic string quality check
            elif isinstance(value, str):
                # Avoid generic placeholders
                if value.lower() not in ["n/a", "null", "none", "unknown", "undefined", ""]:
                    if len(value) > 1:  # At least 2 chars
                        quality_checks += 1
            
            # Any other type with non-None value
            else:
                quality_checks += 1
        
        return quality_checks / total_checks if total_checks > 0 else 0.5

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
        """
        Calculate comprehensive reward score (0.0 - 1.0)
        
        Components:
        - Success (25%): Did the crawl complete without errors?
        - Data Presence (15%): Was data extracted?
        - Validation (30%): How well does data match schema?
        - Quantity (15%): Sufficient items extracted?
        - Efficiency (15%): Pagination-aware performance (time per page)
        """
        
        # 1. Success component (25%)
        success = 1.0 if crawl_result.get("success") else 0.0
        
        # 2. Data presence (15%)
        has_data = 1.0 if extracted else 0.0
        
        # 3. Validation score (30%) - from _validate_extraction
        # Already includes schema compliance, completeness, and quality
        
        # 4. Quantity component (15%) - diminishing returns
        if extracted:
            # Different thresholds based on pattern type
            metadata = crawl_result.get("metadata", {})
            
            # Infer expected quantity from extraction type
            # E.g., product lists should have 10+, articles might have 1-5
            if "product" in str(crawl_result.get("conversation_name", "")).lower():
                target_count = 10  # E-commerce usually has many products
            elif "article" in str(crawl_result.get("conversation_name", "")).lower():
                target_count = 5   # Articles are fewer per page
            else:
                target_count = 5   # Default
            
            quantity_score = min(1.0, len(extracted) / target_count)
        else:
            quantity_score = 0.0
        
        # 5. Efficiency score (15%) - PAGINATION-AWARE performance
        no_errors = 1.0 if not crawl_result.get("error") else 0.0
        
        # Calculate time efficiency based on pages crawled
        exec_time = crawl_result.get("execution_time_ms", 0)
        navigation_result = crawl_result.get("navigation_result", {})
        pages_collected = max(1, navigation_result.get("pages_collected", 1))  # At least 1 page
        
        # Time per page (more fair for pagination crawls)
        time_per_page = exec_time / pages_collected
        
        # Dynamic penalty based on time per page
        # - Single page: penalize if > 30s per page
        # - Multi-page: penalize if > 15s per page (should be faster with pagination)
        if pages_collected == 1:
            # Single page crawl: allow more time
            if time_per_page > 30000:  # > 30s
                time_penalty = min(0.3, (time_per_page - 30000) / 60000)  # Up to 30% penalty
            else:
                time_penalty = 0.0
        else:
            # Multi-page crawl: expect efficiency
            if time_per_page > 15000:  # > 15s per page
                time_penalty = min(0.4, (time_per_page - 15000) / 45000)  # Up to 40% penalty
            else:
                # Bonus for fast pagination!
                if time_per_page < 10000:  # < 10s per page
                    time_penalty = -0.1  # 10% bonus for efficiency
                else:
                    time_penalty = 0.0
        
        efficiency_score = no_errors * max(0.0, min(1.0, 1.0 - time_penalty))
        
        # Weighted sum
        reward = (
            success * 0.25 +          # 25%: Basic success
            has_data * 0.15 +         # 15%: Data extracted
            validation * 0.30 +       # 30%: Validation quality (biggest weight)
            quantity_score * 0.15 +   # 15%: Quantity
            efficiency_score * 0.15   # 15%: Pagination-aware efficiency
        )
        
        return max(0.0, min(1.0, reward))  # Clamp to [0, 1]

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        return urlparse(url).netloc

    def _infer_pattern_type(self, extracted_data: list, extraction_schema: dict, 
                           user_description: str = "") -> str:
        """
        Infer pattern type from extraction results
        Same logic as algorithm's _classify_pattern_type but agent-side
        """
        if not extracted_data:
            return "generic_extraction"
        
        # Get fields from first extracted item
        extraction_fields = list(extracted_data[0].keys()) if extracted_data else []
        if not extraction_fields:
            return "generic_extraction"
        
        fields_lower = [f.lower() for f in extraction_fields]
        
        # E-commerce product patterns
        if any(f in fields_lower for f in ["price", "product_name", "product_title"]):
            if any(f in fields_lower for f in ["rating", "review", "stars"]):
                return "product_with_reviews"
            return "product_list"
        
        # Article/content patterns
        if any(f in fields_lower for f in ["headline", "title", "content", "article"]):
            if any(f in fields_lower for f in ["author", "date", "published"]):
                return "article_extraction"
            return "content_extraction"
        
        # Review extraction
        if any(f in fields_lower for f in ["rating", "review", "comment"]):
            return "review_extraction"
        
        # Contact info
        if any(f in fields_lower for f in ["email", "phone", "address"]):
            return "contact_info"
        
        return "generic_extraction"
    
    def _get_quality_threshold(self, pattern_type: str) -> Dict[str, float]:
        """
        Get quality thresholds for different pattern types
        
        Returns:
            {
                "min_items": minimum items for success,
                "success_threshold": reward threshold for "successful" pattern (>0.7),
                "excellent_threshold": reward threshold for "excellent" pattern (>0.9)
            }
        """
        thresholds = {
            "product_list": {
                "min_items": 5,
                "success_threshold": 0.75,
                "excellent_threshold": 0.90
            },
            "product_with_reviews": {
                "min_items": 3,
                "success_threshold": 0.70,
                "excellent_threshold": 0.85
            },
            "article_extraction": {
                "min_items": 1,
                "success_threshold": 0.80,
                "excellent_threshold": 0.95
            },
            "review_extraction": {
                "min_items": 5,
                "success_threshold": 0.70,
                "excellent_threshold": 0.85
            },
            "contact_info": {
                "min_items": 1,
                "success_threshold": 0.85,
                "excellent_threshold": 0.95
            },
            "price_extraction": {
                "min_items": 3,
                "success_threshold": 0.80,
                "excellent_threshold": 0.90
            },
            "generic_extraction": {
                "min_items": 1,
                "success_threshold": 0.70,
                "excellent_threshold": 0.85
            }
        }
        
        return thresholds.get(pattern_type, thresholds["generic_extraction"])

    # Standalone execution methods (non-Agent Lightning)
    async def execute_crawl(self, task: Dict[str, Any],
                           resources: Optional[Dict] = None,
                           kafka_publisher=None,
                           learned_patterns: Optional[list] = None) -> Dict[str, Any]:
        """
        Standalone execution method (without Agent Lightning).
        Used for testing or direct invocation.
        Supports Kafka real-time progress events.
        
        Args:
            learned_patterns: Patterns retrieved from knowledge store (Qdrant+Neo4j)
                            to enhance crawl strategy
        """

        if resources is None:
            resources = self._get_default_resources()

        try:
            if kafka_publisher is not None:
                self.base_crawler.kafka_publisher = kafka_publisher

            # Log learned patterns usage
            if learned_patterns:
                logger.info(f"🧠 Using {len(learned_patterns)} learned patterns to enhance crawl:")
                for i, pattern in enumerate(learned_patterns[:3], 1):  # Show top 3
                    logger.info(f"   Pattern {i}: {pattern.get('type', 'unknown')} "
                              f"(success={pattern.get('success_rate', 0):.2f}, "
                              f"freq={pattern.get('frequency', 0)}, "
                              f"score={pattern.get('score', 0):.2f})")
            else:
                logger.info("📦 No learned patterns - using frozen resources only")

            # Use 'prompt' field (preferred) or fall back to 'user_description'
            prompt = task.get("prompt") or task.get("user_description", "")
            
            # Enhance prompt with learned patterns knowledge
            enhanced_prompt = prompt
            if learned_patterns and len(learned_patterns) > 0:
                # Use best pattern's metadata to enhance crawl
                best_pattern = learned_patterns[0]  # Highest score
                pattern_metadata = best_pattern.get("metadata", {})
                
                # Add hints from successful patterns
                if pattern_metadata.get("successful_selectors"):
                    enhanced_prompt += f"\n\nHint: Previous successful extractions used selectors like: {pattern_metadata['successful_selectors']}"
                
                # Add pagination strategy hints
                pagination = pattern_metadata.get("pagination", {})
                if pagination.get("used_pagination"):
                    strategy = pagination.get("pagination_strategy", "unknown")
                    logger.info(f"💡 Applying learned pagination strategy: {strategy}")
                    enhanced_prompt += f"\n\nHint: This site typically uses {strategy} pagination."
            
            result = await self.base_crawler.intelligent_crawl(
                url=task["url"],
                prompt=enhanced_prompt,  # Use enhanced prompt
                job_id=task.get("job_id"),
                user_id=task.get("user_id"),
                navigation_steps=task.get("navigation_steps"),
                extract_schema=task.get("extraction_schema", {}),
                max_pages=task.get("max_pages")  # None = extract from prompt, int = explicit UI value
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

            # Calculate reward if in training mode
            calculated_reward = None
            if self.mode == "training" and result.get("success"):
                # Calculate multi-dimensional reward for training
                validation_score = self._validate_extraction(
                    extracted_data,
                    task.get("extraction_schema", {})
                )
                calculated_reward = self._calculate_base_reward(
                    result,
                    extracted_data,
                    validation_score
                )
                logger.info(f"📊 Calculated reward: {calculated_reward:.3f} (validation: {validation_score:.2f})")

            return {
                "success": result.get("success", False),
                "data": extracted_data,
                "navigation_result": result.get("navigation_result", {}),
                "execution_time_ms": result.get("execution_time_ms", 0),
                "conversation_name": result.get("conversation_name", "Data Collection"),
                "embedding_data": embedding_data,  # NEW: Pre-generated embeddings
                "calculated_reward": calculated_reward,  # NEW: For training mode
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
