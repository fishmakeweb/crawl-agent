"""
Example usage of multi-model routing in the Self-Learning Agent
"""
import asyncio
import os
from gemini_client import GeminiClient
from config import GeminiConfig


async def example_routing():
    """Demonstrate multi-model routing for different task types"""
    
    # Initialize client with routing enabled
    config = GeminiConfig(
        API_KEY=os.getenv("GEMINI_API_KEY"),
        ROUTING_ENABLED=True,
        MAX_RPM=15,
        MAX_TPM=250000,
    )
    
    client = GeminiClient(config)
    
    print("=" * 80)
    print("Multi-Model Routing Examples")
    print("=" * 80)
    
    # Initialize metrics
    metrics = {
        "tokens_used": 0,
        "cost_usd": 0.0,
        "mode": "production"
    }
    
    # Example 1: Simple crawl extraction (uses 2.0 Flash for speed)
    print("\n1. CRAWL TASK (Production Mode)")
    print("-" * 80)
    model_id, response, metrics = await client.route_model(
        task_type="crawl",
        input_data={
            "prompt": "Extract product names from this HTML: <div class='product'>Laptop</div>"
        },
        current_metrics=metrics
    )
    print(f"✅ Model: {model_id}")
    print(f"💰 Cost: ${metrics['last_request_cost']:.6f}")
    print(f"📝 Response: {response[:100]}...")
    
    # Example 2: User feedback interpretation (uses 2.5 Flash for thinking)
    print("\n2. FEEDBACK TASK (Production Mode)")
    print("-" * 80)
    model_id, response, metrics = await client.route_model(
        task_type="feedback",
        input_data={
            "prompt": "Interpret this user feedback: 'The prices are missing from the results'",
            "json_mode": True
        },
        current_metrics=metrics
    )
    print(f"✅ Model: {model_id}")
    print(f"💰 Cost: ${metrics['last_request_cost']:.6f}")
    print(f"📝 Response: {response[:150]}...")
    
    # Example 3: Clarification with LearnLM
    print("\n3. CLARIFICATION TASK (Production Mode)")
    print("-" * 80)
    model_id, response, metrics = await client.route_model(
        task_type="clarification",
        input_data={
            "prompt": "Ask a pedagogical clarification question about: 'bad results'"
        },
        current_metrics=metrics
    )
    print(f"✅ Model: {model_id}")
    print(f"💰 Cost: ${metrics['last_request_cost']:.6f}")
    print(f"📝 Response: {response[:150]}...")
    
    # Example 4: Switch to training mode
    print("\n4. ANALYSIS TASK (Training Mode)")
    print("-" * 80)
    metrics["mode"] = "training"
    
    # Simulate complex analysis task
    complex_prompt = "Analyze and cluster these crawl patterns:\n" + "\n".join([
        f"Pattern {i}: domain{i}.com with selector .class{i}" 
        for i in range(50)
    ])
    
    model_id, response, metrics = await client.route_model(
        task_type="analysis",
        input_data={
            "prompt": complex_prompt
        },
        current_metrics=metrics
    )
    print(f"✅ Model: {model_id}")
    print(f"💰 Cost: ${metrics['last_request_cost']:.6f}")
    print(f"📝 Response: {response[:150]}...")
    
    # Example 5: RL decision making
    print("\n5. RL_DECIDE TASK (Training Mode)")
    print("-" * 80)
    model_id, response, metrics = await client.route_model(
        task_type="rl_decide",
        input_data={
            "prompt": "Given reward=-0.3, should we adjust knowledge store size? Return decision.",
            "json_mode": True
        },
        current_metrics=metrics
    )
    print(f"✅ Model: {model_id}")
    print(f"💰 Cost: ${metrics['last_request_cost']:.6f}")
    print(f"📝 Response: {response[:150]}...")
    
    # Example 6: Prompt generation for learning
    print("\n6. PROMPT_GEN TASK (Training Mode)")
    print("-" * 80)
    model_id, response, metrics = await client.route_model(
        task_type="prompt_gen",
        input_data={
            "prompt": "Generate an extraction prompt for e-commerce product pages with prices and ratings"
        },
        current_metrics=metrics
    )
    print(f"✅ Model: {model_id}")
    print(f"💰 Cost: ${metrics['last_request_cost']:.6f}")
    print(f"📝 Response: {response[:150]}...")
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Total Tokens Used: {metrics['tokens_used']:,}")
    print(f"Total Cost: ${metrics['cost_usd']:.6f}")
    
    stats = client.get_stats()
    print(f"\nModel Usage Breakdown:")
    for model, count in stats['model_usage'].items():
        print(f"  - {model}: {count} requests")
    
    print(f"\nCache Hit Rate: {stats['cache_hit_rate']:.1%}")
    print(f"Rate Limit Warnings: {stats['rate_limit_warnings']}")
    print(f"Model Fallbacks: {stats['model_fallbacks']}")
    print(f"Routing Enabled: {stats['routing_enabled']}")


async def test_rate_limit_handling():
    """Test rate limit detection and fallback"""
    
    print("\n" + "=" * 80)
    print("Rate Limit Handling Test")
    print("=" * 80)
    
    config = GeminiConfig(
        API_KEY=os.getenv("GEMINI_API_KEY"),
        ROUTING_ENABLED=True,
        MAX_RPM=5,  # Low limit to trigger warnings
        MAX_TPM=10000,
        TPM_WARNING_THRESHOLD=0.5,  # Warn at 50%
    )
    
    client = GeminiClient(config)
    metrics = {"tokens_used": 0, "cost_usd": 0.0, "mode": "production"}
    
    # Make several requests quickly
    for i in range(3):
        print(f"\nRequest {i+1}:")
        model_id, response, metrics = await client.route_model(
            task_type="crawl",
            input_data={"prompt": f"Extract data from page {i}"},
            current_metrics=metrics
        )
        print(f"  Model: {model_id}")
        await asyncio.sleep(0.5)  # Small delay
    
    stats = client.get_stats()
    print(f"\n⚠️  Rate Limit Warnings: {stats['rate_limit_warnings']}")
    print(f"🔄 Model Fallbacks: {stats['model_fallbacks']}")


if __name__ == "__main__":
    print("Starting Multi-Model Routing Examples...\n")
    
    # Run main examples
    asyncio.run(example_routing())
    
    # Run rate limit test
    asyncio.run(test_rate_limit_handling())
    
    print("\n✅ All examples completed!")
