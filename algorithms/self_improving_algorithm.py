"""
Self-Improving Crawler Algorithm with N-Rollout Cycles
Learns from execution results, user feedback, and prompt quality
"""
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict


try:
    from agentlightning import Algorithm
    AGENTLIGHTNING_AVAILABLE = True
except ImportError:
    print("⚠️  Agent Lightning not installed, using mock Algorithm")
    AGENTLIGHTNING_AVAILABLE = False
    class Algorithm:
        def get_store(self): return None
        def get_adapter(self): return None


class SelfImprovingCrawlerAlgorithm(Algorithm if AGENTLIGHTNING_AVAILABLE else object):
    """
    Algorithm that learns from:
    - Execution results (success/failure patterns)
    - User feedback (natural language corrections)
    - Prompt quality (A/B testing and correlation)
    """

    def __init__(self, gemini_client, knowledge_store, update_frequency: int = 5):
        self.gemini_client = gemini_client
        self.knowledge_store = knowledge_store
        self.update_frequency = update_frequency
        self.current_cycle = 0
        self.pending_rollouts = []
        self.feedback_queue = []

        # Learning history
        self.performance_history = []
        self.prompt_variants = []

        print(f"🧠 Self-Improving Algorithm initialized (update every {update_frequency} rollouts)")

    def run(self, train_dataset: List[Dict], val_dataset: List[Dict]):
        """Main training loop with N-rollout cycles"""
        return asyncio.run(self._run_async(train_dataset, val_dataset))

    async def _run_async(self, train_dataset: List[Dict], val_dataset: List[Dict]):
        """Async training loop"""
        store = self.get_store()

        # Initialize resources
        initial_resources = self.create_initial_resources()
        store.update_resources(initial_resources, version=0)
        print(f"✅ Initialized resources (version 0)")

        rollout_count = 0

        for task in train_dataset:
            # Enqueue task
            rollout = store.enqueue_rollout(task, resource_version=self.current_cycle)
            self.pending_rollouts.append(rollout.id)
            rollout_count += 1

            # Check if it's time to update
            if rollout_count % self.update_frequency == 0:
                print(f"\n{'='*60}")
                print(f"🔄 Update Cycle {self.current_cycle}: Learning from {self.update_frequency} rollouts...")
                print(f"{'='*60}")

                # Wait for completion
                completed_rollouts = store.wait_for_rollouts(
                    self.pending_rollouts,
                    timeout=300
                )

                # Query spans
                spans = store.query_spans(rollout_ids=self.pending_rollouts)

                print(f"📊 Collected {len(spans)} spans from {len(completed_rollouts)} rollouts")

                # Learn from spans + feedback
                new_resources = await self.learn_from_cycle(
                    spans,
                    completed_rollouts
                )

                # Update resources
                self.current_cycle += 1
                store.update_resources(new_resources, version=self.current_cycle)

                # Performance tracking
                avg_reward = self._calculate_average_reward(spans)
                self.performance_history.append({
                    "cycle": self.current_cycle,
                    "avg_reward": avg_reward,
                    "timestamp": datetime.now().isoformat()
                })

                print(f"✅ Resources updated to version {self.current_cycle}")
                print(f"📈 Average reward: {avg_reward:.3f}")
                print(f"{'='*60}\n")

                # Reset pending
                self.pending_rollouts = []

                # Validation check every 5 cycles
                if self.current_cycle % 5 == 0 and val_dataset:
                    await self._validate(store, val_dataset)

        # Final update if pending rollouts remain
        if self.pending_rollouts:
            print(f"\n🔄 Final update cycle...")
            await self._final_update(store)

        print(f"\n🎓 Training complete! Total cycles: {self.current_cycle}")
        self._print_learning_summary()

    async def learn_from_cycle(
        self,
        spans: List,
        rollouts: List
    ) -> Dict[str, Any]:
        """
        Learn from a batch of rollouts and generate new resources

        Learning sources:
        1. Execution results (success/failure analysis)
        2. User feedback (natural language corrections)
        3. Prompt quality (what worked well)
        """

        print(f"\n  📚 Analyzing execution results...")

        # 1. Extract patterns from successful rollouts
        successful_patterns = self._extract_successful_patterns(
            [s for s in spans if s.attributes.get("reward", 0) > 0.7]
        )
        print(f"     ✓ Found {len(successful_patterns)} successful patterns")

        # 2. Analyze failures
        failure_patterns = self._analyze_failures(
            [s for s in spans if s.attributes.get("reward", 0) < 0.3]
        )
        print(f"     ✓ Analyzed {len(failure_patterns)} failure patterns")

        # 3. Process user feedback
        print(f"\n  💬 Processing user feedback...")
        feedback_insights = await self._process_all_feedback(rollouts)
        print(f"     ✓ Processed {len(feedback_insights)} feedback items")

        # 4. Update knowledge store
        print(f"\n  💾 Updating knowledge store...")
        self.knowledge_store.add_patterns(successful_patterns)
        self.knowledge_store.add_failure_patterns(failure_patterns)
        self.knowledge_store.add_feedback_insights(feedback_insights)
        print(f"     ✓ Knowledge store updated")

        # 5. Generate improved prompts using Gemini
        print(f"\n  🧠 Generating improved prompts...")
        improved_prompts = await self._generate_improved_prompts(
            successful_patterns,
            feedback_insights
        )
        print(f"     ✓ Prompts improved")

        # 6. Generate improved crawl configs
        print(f"\n  ⚙️  Optimizing crawl configurations...")
        improved_configs = self._synthesize_crawl_configs(
            successful_patterns,
            failure_patterns
        )
        print(f"     ✓ Configs optimized")

        # 7. Retrieve learned domain patterns
        domain_patterns = self.knowledge_store.get_domain_patterns()
        print(f"     ✓ Domain patterns: {len(domain_patterns)} domains")

        return {
            "extraction_prompt": improved_prompts,
            "crawl_config": improved_configs,
            "domain_patterns": domain_patterns,
            "knowledge_version": self.current_cycle,
            "performance_metrics": {
                "successful_patterns": len(successful_patterns),
                "failure_patterns": len(failure_patterns),
                "feedback_insights": len(feedback_insights)
            }
        }

    def _extract_successful_patterns(self, spans: List) -> List[Dict]:
        """Extract patterns from successful crawls"""
        patterns = []

        for span in spans:
            attrs = span.attributes

            # Extract useful patterns
            pattern = {
                "id": span.span_id,
                "type": "successful_crawl",
                "domain": attrs.get("domain", "unknown"),
                "selectors": attrs.get("selectors", []),
                "extraction_fields": attrs.get("extracted_fields", []),
                "success_rate": attrs.get("reward", 0.0),
                "frequency": 1,
                "metadata": {
                    "execution_time": attrs.get("execution_time_ms", 0),
                    "items_extracted": attrs.get("items_extracted", 0),
                    "pages_collected": attrs.get("pages_collected", 0)
                },
                "description": f"Successful pattern for {attrs.get('domain', 'unknown')}"
            }

            patterns.append(pattern)

        return patterns

    def _analyze_failures(self, spans: List) -> List[Dict]:
        """Analyze failure patterns to learn what to avoid"""
        failure_patterns = []

        # Group failures by error type
        error_groups = defaultdict(list)

        for span in spans:
            attrs = span.attributes
            error = attrs.get("error") or attrs.get("exception")

            if error:
                error_type = type(error).__name__ if hasattr(error, '__name__') else "UnknownError"
                error_groups[error_type].append({
                    "domain": attrs.get("domain", "unknown"),
                    "url": attrs.get("url", ""),
                    "error": str(error)
                })

        # Create failure patterns
        for error_type, errors in error_groups.items():
            failure_patterns.append({
                "id": f"failure_{error_type}_{self.current_cycle}",
                "type": "failure_pattern",
                "error_type": error_type,
                "occurrences": len(errors),
                "domains": list(set([e["domain"] for e in errors])),
                "description": f"Common failure: {error_type}",
                "metadata": {"examples": errors[:5]}  # Keep first 5 examples
            })

        return failure_patterns

    async def _process_all_feedback(self, rollouts: List) -> List[Dict]:
        """Process natural language feedback from users"""
        insights = []

        for rollout in rollouts:
            feedback = rollout.metadata.get("user_feedback")
            if not feedback:
                continue

            # Try direct interpretation first
            insight = self._parse_feedback_directly(feedback)

            # If unclear, route through Gemini
            if insight.get("confidence", 0) < 0.7:
                try:
                    gemini_insight = await self.gemini_client.interpret_feedback(
                        feedback,
                        context={
                            "url": rollout.task.get("url", ""),
                            "fields": rollout.task.get("extraction_schema", {}),
                            "errors": rollout.metadata.get("errors", [])
                        }
                    )
                    insight = gemini_insight
                except Exception as e:
                    print(f"     ⚠️  Gemini feedback interpretation failed: {e}")

            insights.append({
                "rollout_id": rollout.id,
                "original_feedback": feedback,
                "interpreted": insight,
                "domain": self._extract_domain_from_url(rollout.task.get("url", "")),
                "timestamp": datetime.now().isoformat()
            })

        return insights

    def _parse_feedback_directly(self, feedback: str) -> Dict:
        """Simple rule-based feedback parsing"""
        feedback_lower = feedback.lower()

        # Detect quality rating from keywords
        quality_rating = 3  # Default neutral

        if any(word in feedback_lower for word in ["excellent", "perfect", "great", "amazing"]):
            quality_rating = 5
        elif any(word in feedback_lower for word in ["good", "nice", "well"]):
            quality_rating = 4
        elif any(word in feedback_lower for word in ["bad", "poor", "wrong", "incorrect", "missing"]):
            quality_rating = 2
        elif any(word in feedback_lower for word in ["terrible", "useless", "failed"]):
            quality_rating = 1

        # Detect issues
        issues = []
        if "missing" in feedback_lower:
            issues.append("missing_data")
        if "wrong" in feedback_lower or "incorrect" in feedback_lower:
            issues.append("incorrect_data")
        if "format" in feedback_lower:
            issues.append("format_issue")
        if "slow" in feedback_lower or "timeout" in feedback_lower:
            issues.append("performance_issue")

        # Confidence based on keyword presence
        confidence = 0.8 if (quality_rating != 3 or issues) else 0.5

        return {
            "confidence": confidence,
            "quality_rating": quality_rating,
            "specific_issues": issues,
            "desired_improvements": [],
            "clarification_needed": confidence < 0.7,
            "clarification_question": "Could you provide more specific details about what needs improvement?"
        }

    async def _generate_improved_prompts(
        self,
        patterns: List,
        feedback: List
    ) -> str:
        """Use Gemini to generate improved extraction prompts"""

        # Summarize learnings
        summary = self._summarize_learnings(patterns, feedback)

        meta_prompt = f"""
You are optimizing a web crawler's extraction prompt based on learning data.

SUCCESSFUL PATTERNS (what worked well):
{json.dumps([p for p in patterns[:10]], indent=2)}

USER FEEDBACK INSIGHTS:
{json.dumps([f.get("interpreted", {}) for f in feedback[:10]], indent=2)}

CURRENT CYCLE: {self.current_cycle}

LEARNING SUMMARY:
{summary}

Generate an improved extraction prompt template that:
1. Incorporates successful patterns discovered
2. Addresses user feedback and common issues
3. Handles edge cases better (missing fields, format issues)
4. Uses clear, specific instructions for the LLM
5. Maintains compatibility with schema-based extraction

The prompt should have placeholders for: url, html_content, schema, user_intent

IMPORTANT:
- Be more specific about field extraction based on learned patterns
- Include examples from successful extractions
- Add validation instructions to prevent common errors
- Keep prompt concise but comprehensive

Return only the improved prompt template as plain text.
"""

        try:
            improved_prompt = await self.gemini_client.generate(meta_prompt)
            return improved_prompt.strip()
        except Exception as e:
            print(f"     ⚠️  Prompt generation failed: {e}, using default")
            return self._get_default_prompt()

    def _synthesize_crawl_configs(
        self,
        successful_patterns: List,
        failure_patterns: List
    ) -> Dict[str, Any]:
        """Generate optimized crawl configurations from patterns"""

        # Analyze execution times
        execution_times = [
            p.get("metadata", {}).get("execution_time", 30000)
            for p in successful_patterns
        ]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 30000

        # Adjust timeout based on observed times (with safety margin)
        optimal_timeout = min(120, max(30, int(avg_execution_time / 1000 * 1.5)))

        # Analyze page collection patterns
        pages_collected = [
            p.get("metadata", {}).get("pages_collected", 1)
            for p in successful_patterns
        ]
        avg_pages = sum(pages_collected) / len(pages_collected) if pages_collected else 10
        optimal_max_pages = min(100, max(10, int(avg_pages * 2)))

        # Check for timeout failures
        has_timeout_failures = any(
            "timeout" in str(f.get("error_type", "")).lower()
            for f in failure_patterns
        )

        if has_timeout_failures:
            optimal_timeout = min(optimal_timeout * 1.5, 180)

        return {
            "timeout": optimal_timeout,
            "wait_for": "networkidle",
            "screenshot": False,
            "max_pages": optimal_max_pages,
            "delay_before_return": 1.5,
            "headless": True,
            "optimized_for_cycle": self.current_cycle
        }

    def _summarize_learnings(self, patterns: List, feedback: List) -> str:
        """Create a summary of key learnings"""

        # Domain statistics
        domains = defaultdict(int)
        for p in patterns:
            domains[p.get("domain", "unknown")] += 1

        # Feedback statistics
        avg_rating = sum(
            f.get("interpreted", {}).get("quality_rating", 3)
            for f in feedback
        ) / len(feedback) if feedback else 3.0

        common_issues = defaultdict(int)
        for f in feedback:
            for issue in f.get("interpreted", {}).get("specific_issues", []):
                common_issues[issue] += 1

        summary = f"""
Cycle {self.current_cycle} Learning Summary:
- Successful patterns: {len(patterns)}
- Top domains: {dict(list(domains.items())[:5])}
- Average feedback rating: {avg_rating:.2f}/5.0
- Common issues: {dict(common_issues)}
- Total feedback items: {len(feedback)}
"""
        return summary

    def _calculate_average_reward(self, spans: List) -> float:
        """Calculate average reward from spans"""
        rewards = [s.attributes.get("reward", 0.0) for s in spans if "reward" in s.attributes]
        return sum(rewards) / len(rewards) if rewards else 0.0

    async def _validate(self, store, val_dataset: List[Dict]):
        """Run validation on validation dataset"""
        print(f"\n  🧪 Running validation...")

        # Enqueue validation tasks
        val_rollout_ids = []
        for task in val_dataset[:20]:  # Validate on first 20
            rollout = store.enqueue_rollout(
                task,
                resource_version=self.current_cycle,
                mode="validation"
            )
            val_rollout_ids.append(rollout.id)

        # Wait for completion
        val_rollouts = store.wait_for_rollouts(val_rollout_ids, timeout=300)
        val_spans = store.query_spans(rollout_ids=val_rollout_ids)

        # Calculate validation metrics
        val_reward = self._calculate_average_reward(val_spans)

        print(f"     ✓ Validation reward: {val_reward:.3f}")

        # Detect overfitting
        if len(self.performance_history) > 0:
            train_reward = self.performance_history[-1]["avg_reward"]
            gap = train_reward - val_reward

            if gap > 0.2:
                print(f"     ⚠️  Potential overfitting detected (gap: {gap:.3f})")
            else:
                print(f"     ✓ Generalization good (gap: {gap:.3f})")

    async def _final_update(self, store):
        """Final update with remaining rollouts"""
        if not self.pending_rollouts:
            return

        completed = store.wait_for_rollouts(self.pending_rollouts, timeout=300)
        spans = store.query_spans(rollout_ids=self.pending_rollouts)

        new_resources = await self.learn_from_cycle(spans, completed)
        self.current_cycle += 1
        store.update_resources(new_resources, version=self.current_cycle)

        print(f"✅ Final resources updated to version {self.current_cycle}")

    def _print_learning_summary(self):
        """Print overall learning summary"""
        print(f"\n{'='*60}")
        print(f"📊 LEARNING SUMMARY")
        print(f"{'='*60}")
        print(f"Total training cycles: {self.current_cycle}")
        print(f"Performance history:")

        for record in self.performance_history[-10:]:  # Last 10 cycles
            print(f"  Cycle {record['cycle']}: {record['avg_reward']:.3f} reward")

        if len(self.performance_history) > 1:
            improvement = (
                self.performance_history[-1]["avg_reward"] -
                self.performance_history[0]["avg_reward"]
            )
            print(f"\nTotal improvement: {improvement:+.3f}")

        print(f"{'='*60}\n")

    def create_initial_resources(self) -> Dict[str, Any]:
        """Create initial resources before any learning"""
        return {
            "extraction_prompt": self._get_default_prompt(),
            "crawl_config": {
                "timeout": 30,
                "wait_for": "networkidle",
                "screenshot": False,
                "max_pages": 50,
                "headless": True
            },
            "domain_patterns": {},
            "knowledge_version": 0
        }

    def _get_default_prompt(self) -> str:
        """Get default extraction prompt"""
        return """
Extract data from this webpage according to the schema and user's request.

URL: {url}
User wants: {user_intent}

HTML content (relevant portion):
{html_content}

Target extraction schema:
{schema}

Instructions:
1. Extract ALL items that match the schema
2. Ensure field names match the schema exactly
3. If a field is missing, use null or omit it
4. Return a JSON array of objects
5. Be precise with data types (numbers as numbers, not strings)

Example output format:
[
  {{"product_name": "Example Product", "price": 99.99, "brand": "ExampleBrand"}},
  {{"product_name": "Another Product", "price": 149.99, "brand": "AnotherBrand"}}
]

Extract the data now. Return ONLY the JSON array.
"""

    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        try:
            return urlparse(url).netloc
        except:
            return "unknown"


# Standalone test
async def test_algorithm():
    """Test the self-improving algorithm"""
    from config import Config
    from gemini_client import GeminiClient
    from knowledge.hybrid_knowledge_store import HybridKnowledgeStore
    from knowledge.rl_controller import RLResourceController

    config = Config()
    gemini = GeminiClient(config.gemini)
    rl_controller = RLResourceController(gemini, config.training)
    knowledge_store = HybridKnowledgeStore(gemini, rl_controller, config.knowledge_store)

    algorithm = SelfImprovingCrawlerAlgorithm(
        gemini_client=gemini,
        knowledge_store=knowledge_store,
        update_frequency=5
    )

    # Create mock training dataset
    train_dataset = [
        {
            "url": f"https://example.com/page{i}",
            "user_description": "Extract products",
            "extraction_schema": {"required": ["name", "price"]}
        }
        for i in range(15)
    ]

    # Note: This requires a mock LightningStore
    # algorithm.run(train_dataset, [])


if __name__ == "__main__":
    asyncio.run(test_algorithm())
