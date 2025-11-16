"""
RL-Based Autonomous Resource Controller
Manages knowledge store limits, pruning, and retention policies
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List
import json
import asyncio


class RLResourceController:
    """
    RL-based controller that autonomously manages:
    - Knowledge store size limits
    - Summarization triggers
    - Pruning strategies
    - Retention vs forgetting decisions
    """

    def __init__(self, gemini_client, config):
        self.gemini_client = gemini_client
        self.config = config

        # State space (normalized 0-1)
        self.state = {
            "storage_utilization": 0.0,  # 0-1
            "redundancy_ratio": 0.0,      # 0-1
            "cache_hit_rate": 0.0,        # 0-1
            "retrieval_frequency": 0.0,   # normalized
            "growth_rate": 0.0            # MB/day
        }

        # Action space
        self.actions = {
            "no_action": 0,
            "increase_limits": 1,
            "decrease_limits": 2,
            "run_pruning": 3,
            "run_summarization": 4,
            "run_consolidation": 5,
            "archive_old": 6
        }

        # Policy parameters (learned over time)
        self.policy = {
            "storage_threshold_upper": 0.8,
            "storage_threshold_lower": 0.3,
            "redundancy_threshold": 0.2,
            "min_cache_hit_rate": 0.4,
            "retention_days": config.INITIAL_PATTERN_RETENTION_DAYS,
            "max_size_gb": config.INITIAL_KNOWLEDGE_STORE_MAX_SIZE_GB,
            "min_frequency": config.INITIAL_MIN_PATTERN_FREQUENCY
        }

        # Learning history
        self.history = []
        self.learning_rate = config.RL_LEARNING_RATE

        # Monitoring
        self.monitoring_active = False

    async def start_monitoring(self, knowledge_store, interval_hours: int = 1):
        """Start autonomous monitoring loop"""
        self.monitoring_active = True
        print(f"🤖 RL Controller: Starting autonomous monitoring (every {interval_hours}h)")

        while self.monitoring_active:
            await asyncio.sleep(interval_hours * 3600)

            try:
                # Collect metrics
                metrics = knowledge_store.get_metrics()

                # Update state
                self._update_state(metrics)

                # Decide action
                action_name, action_params = await self.decide_action(metrics)

                # Execute action
                if action_name != "no_action":
                    reward = await self.execute_action(
                        action_name,
                        action_params,
                        knowledge_store
                    )

                    # Learn from outcome
                    await self.learn_from_outcome(action_name, reward)

            except Exception as e:
                print(f"❌ RL Controller error: {e}")

    def stop_monitoring(self):
        """Stop autonomous monitoring"""
        self.monitoring_active = False
        print("🛑 RL Controller: Stopped monitoring")

    async def decide_action(self, metrics: Dict[str, Any]) -> Tuple[str, Dict]:
        """
        Use RL policy to decide what action to take.
        Returns: (action_name, action_params)
        """

        # Update state from metrics
        self._update_state(metrics)

        # Calculate state vector
        state_vector = np.array([
            self.state["storage_utilization"],
            self.state["redundancy_ratio"],
            self.state["cache_hit_rate"],
            self.state["retrieval_frequency"],
            self.state["growth_rate"]
        ])

        # Evaluate policy (rule-based initially, evolves to learned)
        action_scores = self._evaluate_policy(state_vector)

        # Select action (greedy)
        action_id = np.argmax(action_scores)
        action_name = list(self.actions.keys())[action_id]

        # Generate action parameters using Gemini
        action_params = await self._generate_action_params(action_name, metrics)

        # Log decision
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "state": self.state.copy(),
            "action": action_name,
            "params": action_params,
            "action_scores": {
                name: float(score)
                for name, score in zip(self.actions.keys(), action_scores)
            }
        }
        self.history.append(decision_record)

        # Keep only recent history (last 1000 decisions)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        if action_name != "no_action":
            print(f"🧠 RL Decision: {action_name} (state: {self.state})")

        return action_name, action_params

    def _update_state(self, metrics: Dict[str, Any]):
        """Update state from current metrics"""
        max_size_bytes = self.policy["max_size_gb"] * 1024 * 1024 * 1024
        current_size_bytes = (
            metrics.get("vector_size_mb", 0) * 1024 * 1024 +
            metrics.get("graph_nodes", 0) * 1000  # Estimate 1KB per node
        )

        self.state["storage_utilization"] = min(1.0, current_size_bytes / max_size_bytes)
        self.state["redundancy_ratio"] = min(1.0, metrics.get("pattern_redundancy", 0.0))
        self.state["cache_hit_rate"] = min(1.0, metrics.get("cache_hit_rate", 0.0))

        # Calculate retrieval frequency (normalized by expected daily volume)
        total_retrievals = sum(metrics.get("retrieval_frequency", {}).values())
        self.state["retrieval_frequency"] = min(1.0, total_retrievals / 1000)

        # Calculate growth rate
        if len(self.history) > 0:
            prev_state = self.history[-1]["state"]
            prev_time = datetime.fromisoformat(self.history[-1]["timestamp"])
            days_elapsed = max(1, (datetime.now() - prev_time).days)

            prev_util = prev_state["storage_utilization"]
            current_util = self.state["storage_utilization"]

            self.state["growth_rate"] = (current_util - prev_util) / days_elapsed
        else:
            self.state["growth_rate"] = 0.0

    def _evaluate_policy(self, state_vector: np.ndarray) -> np.ndarray:
        """
        Evaluate which action to take based on current state.
        Returns scores for each action.
        """
        scores = np.zeros(len(self.actions))

        storage_util = state_vector[0]
        redundancy = state_vector[1]
        cache_hit = state_vector[2]
        retrieval_freq = state_vector[3]
        growth_rate = state_vector[4]

        # Rule-based heuristics (will be replaced by learned Q-values over time)

        # High storage → prune or archive
        if storage_util > self.policy["storage_threshold_upper"]:
            scores[self.actions["run_pruning"]] = 0.8
            scores[self.actions["archive_old"]] = 0.7
            scores[self.actions["run_consolidation"]] = 0.6

        # High redundancy → consolidate
        if redundancy > self.policy["redundancy_threshold"]:
            scores[self.actions["run_consolidation"]] = 0.9
            scores[self.actions["run_summarization"]] = 0.5

        # Low cache hit rate → need better patterns
        if cache_hit < self.policy["min_cache_hit_rate"]:
            scores[self.actions["run_consolidation"]] = 0.6

        # Low storage but high growth → increase limits proactively
        if storage_util < 0.5 and growth_rate > 0.1:
            scores[self.actions["increase_limits"]] = 0.7

        # Low storage and low growth → decrease limits to save resources
        if storage_util < self.policy["storage_threshold_lower"] and growth_rate < 0.01:
            scores[self.actions["decrease_limits"]] = 0.5

        # Very low retrieval frequency → data not being used
        if retrieval_freq < 0.1 and storage_util > 0.5:
            scores[self.actions["run_pruning"]] = 0.7

        # Healthy state → no action
        if (0.3 < storage_util < 0.7 and
            redundancy < 0.15 and
            cache_hit > 0.5):
            scores[self.actions["no_action"]] = 1.0

        return scores

    async def _generate_action_params(self, action_name: str, metrics: Dict) -> Dict:
        """Use Gemini to generate specific parameters for the action"""

        if action_name == "no_action":
            return {}

        param_prompt = f"""
You are managing a knowledge store for a web crawler agent.
Based on these metrics, generate optimal parameters for action: {action_name}

Current metrics:
{json.dumps(metrics, indent=2)}

Current policy:
{json.dumps(self.policy, indent=2)}

Current state:
{json.dumps(self.state, indent=2)}

Generate specific parameters for this action:

For "increase_limits":
- new_max_size_gb: Increase by 20-50% based on growth rate
- new_retention_days: Extend by 10-20 days
- reasoning: Why this is needed

For "decrease_limits":
- new_max_size_gb: Decrease by 10-30%
- new_retention_days: Reduce by 5-10 days
- reasoning: Why this is safe

For "run_pruning":
- min_frequency_threshold: Patterns with frequency below this are candidates
- age_threshold_days: Patterns older than this are candidates
- max_patterns_to_remove: Safety limit
- reasoning: Criteria explanation

For "run_summarization":
- compression_ratio: Target compression (e.g., 0.5 = reduce by 50%)
- priority_threshold: Only summarize patterns below this success rate
- reasoning: Strategy explanation

For "run_consolidation":
- similarity_threshold: Merge patterns above this similarity (0.8-0.95)
- min_cluster_size: Minimum patterns to form cluster
- reasoning: Why consolidation is needed

For "archive_old":
- archive_age_days: Archive patterns older than this
- min_frequency_for_keep: Keep even old patterns if used this often
- reasoning: Archive strategy

Return as JSON only.
"""

        try:
            params_json = await self.gemini_client.generate(
                param_prompt,
                response_mime_type="application/json"
            )
            return json.loads(params_json)
        except Exception as e:
            print(f"⚠️  Gemini param generation failed: {e}")
            return self._get_default_params(action_name)

    def _get_default_params(self, action_name: str) -> Dict:
        """Fallback default parameters"""
        defaults = {
            "increase_limits": {
                "new_max_size_gb": self.policy["max_size_gb"] * 1.3,
                "new_retention_days": self.policy["retention_days"] + 15,
                "reasoning": "Default increase"
            },
            "decrease_limits": {
                "new_max_size_gb": self.policy["max_size_gb"] * 0.8,
                "new_retention_days": self.policy["retention_days"] - 10,
                "reasoning": "Default decrease"
            },
            "run_pruning": {
                "min_frequency_threshold": 2,
                "age_threshold_days": 60,
                "max_patterns_to_remove": 1000,
                "reasoning": "Default pruning"
            },
            "run_consolidation": {
                "similarity_threshold": 0.85,
                "min_cluster_size": 2,
                "reasoning": "Default consolidation"
            }
        }
        return defaults.get(action_name, {})

    async def execute_action(
        self,
        action_name: str,
        params: Dict,
        knowledge_store
    ) -> float:
        """
        Execute the decided action and return reward.
        Reward is based on improvement in metrics.
        """

        print(f"⚙️  Executing: {action_name}")
        print(f"   Parameters: {json.dumps(params, indent=2)}")

        # Record state before action
        metrics_before = knowledge_store.get_metrics()

        try:
            if action_name == "increase_limits":
                self.policy["max_size_gb"] = params.get("new_max_size_gb", self.policy["max_size_gb"])
                self.policy["retention_days"] = params.get("new_retention_days", self.policy["retention_days"])
                print(f"   ✅ Limits increased: {self.policy['max_size_gb']:.2f}GB, {self.policy['retention_days']} days")

            elif action_name == "decrease_limits":
                self.policy["max_size_gb"] = params.get("new_max_size_gb", self.policy["max_size_gb"])
                self.policy["retention_days"] = params.get("new_retention_days", self.policy["retention_days"])
                print(f"   ✅ Limits decreased: {self.policy['max_size_gb']:.2f}GB, {self.policy['retention_days']} days")

            elif action_name == "run_pruning":
                await self._prune_patterns(knowledge_store, params)

            elif action_name == "run_consolidation":
                merged_count = await knowledge_store.consolidate_patterns()
                print(f"   ✅ Consolidated {merged_count} pattern clusters")

            elif action_name == "run_summarization":
                await self._summarize_patterns(knowledge_store, params)

            elif action_name == "archive_old":
                await self._archive_old_patterns(knowledge_store, params)

            # Wait a moment for changes to propagate
            await asyncio.sleep(2)

            # Record state after action
            metrics_after = knowledge_store.get_metrics()

            # Calculate reward
            reward = self._calculate_reward(metrics_before, metrics_after)

            print(f"   📊 Action reward: {reward:.2f}")
            return reward

        except Exception as e:
            print(f"   ❌ Action failed: {e}")
            return -0.5  # Negative reward for failure

    async def _prune_patterns(self, knowledge_store, params: Dict):
        """Prune low-value patterns"""
        # This would query and delete patterns based on criteria
        # Implementation depends on knowledge store specifics
        print(f"   🧹 Pruning patterns (threshold: {params.get('min_frequency_threshold', 2)})")

    async def _summarize_patterns(self, knowledge_store, params: Dict):
        """Summarize and compress similar patterns"""
        print(f"   📝 Summarizing patterns (compression: {params.get('compression_ratio', 0.5)})")

    async def _archive_old_patterns(self, knowledge_store, params: Dict):
        """Archive old, infrequently used patterns"""
        print(f"   📦 Archiving patterns older than {params.get('archive_age_days', 90)} days")

    def _calculate_reward(
        self,
        metrics_before: Dict,
        metrics_after: Dict
    ) -> float:
        """
        Calculate reward based on improvement in metrics.
        Higher reward for better efficiency without losing quality.
        """

        reward = 0.0

        # Storage efficiency improvement
        storage_before = metrics_before.get("vector_size_mb", 0)
        storage_after = metrics_after.get("vector_size_mb", 0)
        if storage_before > 0:
            storage_improvement = (storage_before - storage_after) / storage_before
            reward += storage_improvement * 0.3

        # Redundancy reduction
        redundancy_before = metrics_before.get("pattern_redundancy", 0)
        redundancy_after = metrics_after.get("pattern_redundancy", 0)
        redundancy_reduction = redundancy_before - redundancy_after
        reward += redundancy_reduction * 0.3

        # Cache hit rate improvement
        cache_before = metrics_before.get("cache_hit_rate", 0)
        cache_after = metrics_after.get("cache_hit_rate", 0)
        cache_improvement = cache_after - cache_before
        reward += cache_improvement * 0.4

        # Penalty for excessive deletion
        patterns_before = metrics_before.get("total_patterns", 0)
        patterns_after = metrics_after.get("total_patterns", 0)
        if patterns_before > 0:
            deletion_ratio = (patterns_before - patterns_after) / patterns_before
            if deletion_ratio > 0.3:  # More than 30% deleted
                reward -= 0.2  # Penalty

        return np.clip(reward, -1.0, 1.0)

    async def learn_from_outcome(self, action_name: str, reward: float):
        """
        Update policy based on outcome of previous action.
        Simple policy gradient update.
        """

        if len(self.history) < 2:
            return

        # Get previous record
        prev_record = self.history[-2]
        prev_state = np.array([
            prev_record["state"]["storage_utilization"],
            prev_record["state"]["redundancy_ratio"],
            prev_record["state"]["cache_hit_rate"],
            prev_record["state"]["retrieval_frequency"],
            prev_record["state"]["growth_rate"]
        ])

        # Simple policy update: adjust thresholds based on reward
        if reward > 0.5:
            # Reinforce successful actions
            if action_name == "run_pruning":
                # Success with pruning → can be more aggressive
                self.policy["storage_threshold_upper"] *= (1 - self.learning_rate * 0.1)

            elif action_name == "run_consolidation":
                # Success with consolidation → adjust redundancy threshold
                self.policy["redundancy_threshold"] *= (1 - self.learning_rate * 0.1)

            elif action_name == "increase_limits":
                # Success with increase → remember this threshold
                pass  # Already updated in execute_action

            print(f"   📚 Policy updated (reward={reward:.2f})")

        elif reward < -0.3:
            # Negative reward → revert or adjust
            print(f"   ⚠️  Poor outcome, may need to adjust strategy")

    def get_policy_summary(self) -> Dict[str, Any]:
        """Get current policy for inspection"""
        return {
            "policy": self.policy.copy(),
            "current_state": self.state.copy(),
            "recent_actions": [
                {
                    "action": h["action"],
                    "timestamp": h["timestamp"]
                }
                for h in self.history[-10:]
            ],
            "total_decisions": len(self.history)
        }

    def save_policy(self, filepath: str):
        """Save learned policy to file"""
        policy_data = {
            "policy": self.policy,
            "history": self.history[-100:],  # Last 100 decisions
            "saved_at": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(policy_data, f, indent=2)

        print(f"💾 Policy saved to {filepath}")

    def load_policy(self, filepath: str):
        """Load learned policy from file"""
        try:
            with open(filepath, 'r') as f:
                policy_data = json.load(f)

            self.policy = policy_data["policy"]
            self.history = policy_data.get("history", [])

            print(f"📂 Policy loaded from {filepath}")
            print(f"   Loaded {len(self.history)} historical decisions")

        except Exception as e:
            print(f"⚠️  Failed to load policy: {e}")
